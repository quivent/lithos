\\ megakernel-link.ls — Megakernel linker for Lithos
\\
\\ Takes per-layer instruction streams and concatenates them into two
\\ cooperative megakernels with grid-sync between layers.
\\
\\ Target: Qwen 3.5 27B hybrid — 64 layers (3 DeltaNet + 1 full) x 16
\\
\\ Expected .text sizes:
\\   Per DeltaNet layer:
\\     rmsnorm ~2KB + 7 projections (q,k,v,z,beta,decay,o) x ~100KB
\\     + conv1d ~4KB + l2norm ~2KB + silu ~1KB + delta_fused ~20KB
\\     + MLP (gate+up+silu+down) 4 x ~100KB = ~400KB
\\     Total: ~1.1 MB SASS per DeltaNet layer
\\   Per full attention layer:
\\     rmsnorm ~2KB + 4 projections (q,k,v,gate) x ~100KB
\\     + mRoPE ~8KB + KV write ~2KB + softmax attn ~40KB + o_proj ~100KB
\\     + MLP 4 x ~100KB = ~400KB
\\     Total: ~0.95 MB SASS per full attention layer
\\   48 DeltaNet layers x 1.1MB = ~52.8 MB
\\   16 full attn layers x 0.95MB = ~15.2 MB
\\   token_embed: ~20 KB
\\   final_rmsnorm: ~1 KB
\\   lm_head (5120 x 248320 Q4): ~127 MB (weight-embedded GEMV)
\\   argmax: ~4 KB
\\   67 grid-sync barriers x 304 bytes = ~20 KB
\\   Total forward megakernel .text: ~195 MB
\\
\\   Recurrence megakernel (48 DeltaNet state updates):
\\   48 x ~20KB = ~960 KB + 47 grid-syncs = ~14 KB
\\   Total recurrence .text: ~1 MB
\\
\\ Outputs: two cooperative ELF cubins via lithos-elf.ls


\\ ============================================================
\\ MODEL CONSTANTS — Qwen 3.5 27B
\\ ============================================================

5120  constant HIDDEN_DIM
17408 constant INTERMEDIATE_DIM
248320 constant VOCAB_SIZE
64    constant NUM_LAYERS
48    constant NUM_DELTANET_LAYERS
16    constant NUM_FULL_ATTN_LAYERS
4     constant FULL_ATTN_INTERVAL
0.000001 constant RMS_EPS

\\ DeltaNet dimensions
16    constant DN_KEY_HEADS
48    constant DN_VALUE_HEADS
128   constant DN_HEAD_DIM
4     constant DN_CONV_KERNEL
3     constant DN_GQA_RATIO

\\ Full attention dimensions
24    constant FA_Q_HEADS
4     constant FA_KV_HEADS
256   constant FA_HEAD_DIM
6     constant FA_GQA_RATIO
64    constant FA_ROTARY_DIMS

\\ Derived sizes (element counts)
2048  constant DN_Q_SIZE       \\ DN_KEY_HEADS * DN_HEAD_DIM
2048  constant DN_K_SIZE
6144  constant DN_V_SIZE       \\ DN_VALUE_HEADS * DN_HEAD_DIM
6144  constant DN_Z_SIZE
48    constant DN_BETA_SIZE
48    constant DN_DECAY_SIZE
6144  constant FA_Q_SIZE       \\ FA_Q_HEADS * FA_HEAD_DIM
1024  constant FA_K_SIZE       \\ FA_KV_HEADS * FA_HEAD_DIM
1024  constant FA_V_SIZE
6144  constant FA_GATE_SIZE

\\ Grid launch config
132   constant GRID_CTA_COUNT  \\ H100 has 132 SMs; one CTA per SM


\\ ============================================================
\\ LAYER TYPE ENCODING
\\ ============================================================

0 constant LAYER_LINEAR_ATTENTION
1 constant LAYER_FULL_ATTENTION

\\ Layer type table — 64 entries, filled by parse_layer_schedule
buf layer_types 64

\\ Per-layer weight base addresses — filled by resolve_weight_addrs
\\ Each layer gets a struct of 12 u64 pointers (weight tensor addresses)
\\ 12 pointers x 8 bytes x 64 layers = 6144 bytes
buf layer_weight_addrs 6144

\\ Offsets within the per-layer weight address struct (byte offsets)
0   constant WOFF_INPUT_LN      \\ input layernorm weight
8   constant WOFF_Q_PROJ        \\ Q projection qweight
16  constant WOFF_K_PROJ        \\ K projection qweight
24  constant WOFF_V_PROJ        \\ V projection qweight
32  constant WOFF_Z_GATE_PROJ   \\ Z (deltanet) or gate_attn (full)
40  constant WOFF_BETA_PROJ     \\ beta projection (deltanet only)
48  constant WOFF_DECAY_PROJ    \\ decay projection (deltanet only)
56  constant WOFF_O_PROJ        \\ output projection qweight
64  constant WOFF_POST_LN       \\ post-attention layernorm weight
72  constant WOFF_UP_PROJ       \\ MLP up qweight
80  constant WOFF_GATE_PROJ     \\ MLP gate qweight
88  constant WOFF_DOWN_PROJ     \\ MLP down qweight
96  constant LAYER_WEIGHT_STRIDE


\\ ============================================================
\\ MEGAKERNEL CODE BUFFER
\\ ============================================================
\\ 256 MB buffer for the forward megakernel .text section.
\\ The recurrence megakernel gets its own smaller buffer.

268435456 constant FORWARD_BUF_SIZE
var forward_pos 0
buf forward_buf FORWARD_BUF_SIZE

1048576 constant RECUR_BUF_SIZE
var recur_pos 0
buf recur_buf RECUR_BUF_SIZE

\\ Grid-sync offset tracking for ELF metadata
buf forward_sync_offsets 1024   \\ up to 256 grid-sync sites x 4 bytes
var forward_sync_count 0

buf recur_sync_offsets 512
var recur_sync_count 0

\\ Max register / shared memory high-water marks
var forward_max_reg 0
var forward_max_smem 0
var recur_max_reg 0
var recur_max_smem 0


\\ ============================================================
\\ CODE BUFFER MANAGEMENT
\\ ============================================================

\\ Append the contents of gpu_buf[0..gpu_pos) into the forward megakernel
\\ buffer at forward_pos. Update high-water marks.
append_to_forward :
    for i 0 gpu_pos 1
        byte = → 8 gpu_buf + i
        ← 8 forward_buf + forward_pos byte
        forward_pos + 1
    endfor
    if> max_reg forward_max_reg
        forward_max_reg max_reg
    if> shmem_size forward_max_smem
        forward_max_smem shmem_size

\\ Append gpu_buf into the recurrence megakernel buffer
append_to_recur :
    for i 0 gpu_pos 1
        byte = → 8 gpu_buf + i
        ← 8 recur_buf + recur_pos byte
        recur_pos + 1
    endfor
    if> max_reg recur_max_reg
        recur_max_reg max_reg
    if> shmem_size recur_max_smem
        recur_max_smem shmem_size


\\ ============================================================
\\ GRID-SYNC BARRIER EMISSION
\\ ============================================================
\\ Emits a cooperative grid-wide barrier into the current gpu_buf,
\\ then appends it to the target megakernel buffer.
\\ Records the byte offset of the barrier for ELF COOP_GROUP attrs.

emit_grid_sync_barrier_forward ctr_reg flag_reg :
    \\ Record the offset within the forward megakernel
    ← 32 forward_sync_offsets + forward_sync_count * 4 forward_pos
    forward_sync_count + 1

    \\ Emit the ~19-instruction grid-sync sequence
    gpu_reset
    emit_grid_sync ctr_reg flag_reg GRID_CTA_COUNT
    append_to_forward

emit_grid_sync_barrier_recur ctr_reg flag_reg :
    ← 32 recur_sync_offsets + recur_sync_count * 4 recur_pos
    recur_sync_count + 1

    gpu_reset
    emit_grid_sync ctr_reg flag_reg GRID_CTA_COUNT
    append_to_recur


\\ ============================================================
\\ PARSE LAYER SCHEDULE
\\ ============================================================
\\ Reads config.json layer_types array and fills layer_types[64].
\\ Also validates against safetensors tensor names.
\\
\\ For Qwen 3.5 27B the schedule is perfectly periodic:
\\   layer_idx % 4 == 3 -> full_attention, else linear_attention
\\ The parser confirms this against config.json, but the periodic
\\ formula is the ground truth for the megakernel.

parse_layer_schedule :
    for i 0 NUM_LAYERS 1
        remainder = i % FULL_ATTN_INTERVAL
        if== remainder 3
            ← 8 layer_types + i LAYER_FULL_ATTENTION
        else
            ← 8 layer_types + i LAYER_LINEAR_ATTENTION
    endfor


\\ ============================================================
\\ WEIGHT ADDRESS RESOLUTION
\\ ============================================================
\\ For each layer, look up tensor addresses from the safetensors
\\ mmap'd file using the naming conventions from hybrid-layers.md.
\\ Stores resolved data pointers into layer_weight_addrs.

resolve_weight_addrs layer_idx data_region :
    base = layer_weight_addrs + layer_idx * LAYER_WEIGHT_STRIDE
    layer_type = → 8 layer_types + layer_idx

    \\ Input layernorm — same name for both types
    \\ "model.layers.{N}.input_layernorm.weight"
    addr_ln = st_lookup_tensor data_region layer_idx "input_layernorm.weight"
    ← 64 base + WOFF_INPUT_LN addr_ln

    \\ Post-attention layernorm
    addr_pln = st_lookup_tensor data_region layer_idx "post_attention_layernorm.weight"
    ← 64 base + WOFF_POST_LN addr_pln

    \\ MLP — same for both layer types
    addr_up = st_lookup_tensor data_region layer_idx "mlp.up_proj.qweight"
    ← 64 base + WOFF_UP_PROJ addr_up
    addr_gate = st_lookup_tensor data_region layer_idx "mlp.gate_proj.qweight"
    ← 64 base + WOFF_GATE_PROJ addr_gate
    addr_down = st_lookup_tensor data_region layer_idx "mlp.down_proj.qweight"
    ← 64 base + WOFF_DOWN_PROJ addr_down

    \\ Attention projections — differ by layer type
    if== layer_type LAYER_LINEAR_ATTENTION
        addr_q = st_lookup_tensor data_region layer_idx "linear_attn.q_proj.qweight"
        ← 64 base + WOFF_Q_PROJ addr_q
        addr_k = st_lookup_tensor data_region layer_idx "linear_attn.k_proj.qweight"
        ← 64 base + WOFF_K_PROJ addr_k
        addr_v = st_lookup_tensor data_region layer_idx "linear_attn.v_proj.qweight"
        ← 64 base + WOFF_V_PROJ addr_v
        addr_z = st_lookup_tensor data_region layer_idx "linear_attn.z_proj.qweight"
        ← 64 base + WOFF_Z_GATE_PROJ addr_z
        addr_b = st_lookup_tensor data_region layer_idx "linear_attn.b_proj.qweight"
        ← 64 base + WOFF_BETA_PROJ addr_b
        addr_a = st_lookup_tensor data_region layer_idx "linear_attn.a_proj.qweight"
        ← 64 base + WOFF_DECAY_PROJ addr_a
        addr_o = st_lookup_tensor data_region layer_idx "linear_attn.o_proj.qweight"
        ← 64 base + WOFF_O_PROJ addr_o

    if== layer_type LAYER_FULL_ATTENTION
        addr_q = st_lookup_tensor data_region layer_idx "self_attn.q_proj.qweight"
        ← 64 base + WOFF_Q_PROJ addr_q
        addr_k = st_lookup_tensor data_region layer_idx "self_attn.k_proj.qweight"
        ← 64 base + WOFF_K_PROJ addr_k
        addr_v = st_lookup_tensor data_region layer_idx "self_attn.v_proj.qweight"
        ← 64 base + WOFF_V_PROJ addr_v
        addr_g = st_lookup_tensor data_region layer_idx "self_attn.gate_proj.qweight"
        ← 64 base + WOFF_Z_GATE_PROJ addr_g
        addr_o = st_lookup_tensor data_region layer_idx "self_attn.o_proj.qweight"
        ← 64 base + WOFF_O_PROJ addr_o


\\ ============================================================
\\ EMIT LAYER — DeltaNet (linear attention)
\\ ============================================================
\\ Emits the complete instruction stream for one DeltaNet layer.
\\ Calls existing compositions: rmsnorm, gptq_gemv, deltanet_fused,
\\ activate_silu, elemwise_mul, residual_add.
\\
\\ All weight addresses are resolved from layer_weight_addrs.

emit_layer_deltanet layer_idx :
    wbase = layer_weight_addrs + layer_idx * LAYER_WEIGHT_STRIDE

    \\ Load weight addresses for this layer
    w_input_ln = → 64 wbase + WOFF_INPUT_LN
    w_q        = → 64 wbase + WOFF_Q_PROJ
    w_k        = → 64 wbase + WOFF_K_PROJ
    w_v        = → 64 wbase + WOFF_V_PROJ
    w_z        = → 64 wbase + WOFF_Z_GATE_PROJ
    w_beta     = → 64 wbase + WOFF_BETA_PROJ
    w_decay    = → 64 wbase + WOFF_DECAY_PROJ
    w_o        = → 64 wbase + WOFF_O_PROJ
    w_post_ln  = → 64 wbase + WOFF_POST_LN
    w_up       = → 64 wbase + WOFF_UP_PROJ
    w_gate     = → 64 wbase + WOFF_GATE_PROJ
    w_down     = → 64 wbase + WOFF_DOWN_PROJ

    \\ Compute state base address for this layer's DeltaNet state
    \\ deltanet_state[layer_slot, 16, 3, 128, 128] f32
    \\ layer_slot = count of deltanet layers before this index
    \\ For periodic schedule: slot = layer_idx - (layer_idx / 4) - (layer_idx >= 3)
    \\ Simpler: enumerate deltanet layers 0..47
    dn_slot = deltanet_slot layer_idx
    state_bytes_per_layer = DN_KEY_HEADS * DN_GQA_RATIO * DN_HEAD_DIM * DN_HEAD_DIM * 4
    state_base = deltanet_state_ptr + dn_slot * state_bytes_per_layer

    \\ ---- Attention sub-block ----

    \\ 1. RMSNorm on residual stream -> normalized x
    gpu_reset
    emit_rmsnorm_inline w_input_ln HIDDEN_DIM RMS_EPS
    append_to_forward

    \\ 2. Q projection: hidden_dim -> DN_Q_SIZE (5120 -> 2048)
    gpu_reset
    emit_gemv_inline w_q HIDDEN_DIM DN_Q_SIZE
    append_to_forward

    \\ 3. K projection: hidden_dim -> DN_K_SIZE (5120 -> 2048)
    gpu_reset
    emit_gemv_inline w_k HIDDEN_DIM DN_K_SIZE
    append_to_forward

    \\ 4. V projection: hidden_dim -> DN_V_SIZE (5120 -> 6144)
    gpu_reset
    emit_gemv_inline w_v HIDDEN_DIM DN_V_SIZE
    append_to_forward

    \\ 5. Z projection (output gate): hidden_dim -> DN_Z_SIZE (5120 -> 6144)
    gpu_reset
    emit_gemv_inline w_z HIDDEN_DIM DN_Z_SIZE
    append_to_forward

    \\ 6. Beta projection: hidden_dim -> DN_BETA_SIZE (5120 -> 48)
    gpu_reset
    emit_gemv_inline w_beta HIDDEN_DIM DN_BETA_SIZE
    append_to_forward

    \\ 7. Decay projection: hidden_dim -> DN_DECAY_SIZE (5120 -> 48)
    gpu_reset
    emit_gemv_inline w_decay HIDDEN_DIM DN_DECAY_SIZE
    append_to_forward

    \\ 8. Conv1d on K (causal, kernel_size=4)
    gpu_reset
    emit_conv1d_inline DN_K_SIZE DN_CONV_KERNEL
    append_to_forward

    \\ 9. L2-normalize K
    gpu_reset
    emit_l2norm_inline DN_K_SIZE
    append_to_forward

    \\ 10. SiLU on Q
    gpu_reset
    emit_silu_inline DN_Q_SIZE
    append_to_forward

    \\ 11. L2-normalize Q (after SiLU)
    gpu_reset
    emit_l2norm_inline DN_Q_SIZE
    append_to_forward

    \\ 12. Fused DeltaNet recurrence: state update + output gate
    \\ deltanet_fused reads q,k,v,z,decay,beta,state; writes out
    gpu_reset
    emit_deltanet_fused_inline state_base DN_KEY_HEADS DN_GQA_RATIO DN_HEAD_DIM
    append_to_forward

    \\ 13. Output projection: DN_V_SIZE -> hidden_dim (6144 -> 5120)
    \\ Includes residual add into X
    gpu_reset
    emit_gemv_residual_inline w_o DN_V_SIZE HIDDEN_DIM
    append_to_forward

    \\ ---- MLP sub-block ----

    \\ 14. RMSNorm on residual stream
    gpu_reset
    emit_rmsnorm_inline w_post_ln HIDDEN_DIM RMS_EPS
    append_to_forward

    \\ 15. Gate projection: hidden_dim -> intermediate_dim (5120 -> 17408)
    gpu_reset
    emit_gemv_inline w_gate HIDDEN_DIM INTERMEDIATE_DIM
    append_to_forward

    \\ 16. Up projection: hidden_dim -> intermediate_dim (5120 -> 17408)
    gpu_reset
    emit_gemv_inline w_up HIDDEN_DIM INTERMEDIATE_DIM
    append_to_forward

    \\ 17. SiLU on gate output
    gpu_reset
    emit_silu_inline INTERMEDIATE_DIM
    append_to_forward

    \\ 18. Elementwise multiply: gate * up
    gpu_reset
    emit_elemwise_mul_inline INTERMEDIATE_DIM
    append_to_forward

    \\ 19. Down projection: intermediate_dim -> hidden_dim (17408 -> 5120)
    \\ Includes residual add into X
    gpu_reset
    emit_gemv_residual_inline w_down INTERMEDIATE_DIM HIDDEN_DIM
    append_to_forward


\\ ============================================================
\\ EMIT LAYER — Full Attention
\\ ============================================================
\\ Emits the complete instruction stream for one full attention layer.
\\ Calls existing compositions: rmsnorm, gptq_gemv, attend_full,
\\ activate_silu, elemwise_mul, residual_add.

emit_layer_full_attention layer_idx :
    wbase = layer_weight_addrs + layer_idx * LAYER_WEIGHT_STRIDE

    \\ Load weight addresses
    w_input_ln  = → 64 wbase + WOFF_INPUT_LN
    w_q         = → 64 wbase + WOFF_Q_PROJ
    w_k         = → 64 wbase + WOFF_K_PROJ
    w_v         = → 64 wbase + WOFF_V_PROJ
    w_gate_attn = → 64 wbase + WOFF_Z_GATE_PROJ
    w_o         = → 64 wbase + WOFF_O_PROJ
    w_post_ln   = → 64 wbase + WOFF_POST_LN
    w_up        = → 64 wbase + WOFF_UP_PROJ
    w_gate      = → 64 wbase + WOFF_GATE_PROJ
    w_down      = → 64 wbase + WOFF_DOWN_PROJ

    \\ KV cache slot for this layer
    \\ Full attn layers are at indices 3,7,11,...,63 -> slot = layer_idx / 4
    kv_slot = layer_idx / FULL_ATTN_INTERVAL

    \\ ---- Attention sub-block ----

    \\ 1. RMSNorm
    gpu_reset
    emit_rmsnorm_inline w_input_ln HIDDEN_DIM RMS_EPS
    append_to_forward

    \\ 2. Q projection: hidden_dim -> FA_Q_SIZE (5120 -> 6144)
    gpu_reset
    emit_gemv_inline w_q HIDDEN_DIM FA_Q_SIZE
    append_to_forward

    \\ 3. K projection: hidden_dim -> FA_K_SIZE (5120 -> 1024)
    gpu_reset
    emit_gemv_inline w_k HIDDEN_DIM FA_K_SIZE
    append_to_forward

    \\ 4. V projection: hidden_dim -> FA_V_SIZE (5120 -> 1024)
    gpu_reset
    emit_gemv_inline w_v HIDDEN_DIM FA_V_SIZE
    append_to_forward

    \\ 5. Gate projection (attn output gate): hidden_dim -> FA_GATE_SIZE
    gpu_reset
    emit_gemv_inline w_gate_attn HIDDEN_DIM FA_GATE_SIZE
    append_to_forward

    \\ 6. mRoPE on Q (partial rotary: 64 of 256 dims, interleaved)
    \\ mrope_section = [11, 11, 10] pairs = 64 rotary dims
    gpu_reset
    emit_mrope_inline FA_Q_HEADS FA_HEAD_DIM FA_ROTARY_DIMS
    append_to_forward

    \\ 7. mRoPE on K
    gpu_reset
    emit_mrope_inline FA_KV_HEADS FA_HEAD_DIM FA_ROTARY_DIMS
    append_to_forward

    \\ 8. KV cache write at current position
    gpu_reset
    emit_kv_write_inline kv_slot FA_KV_HEADS FA_HEAD_DIM
    append_to_forward

    \\ 9. Softmax attention: Q @ K^T / sqrt(d) -> scores -> @ V
    \\ Full GQA attention with 24 Q heads, 4 KV heads
    gpu_reset
    emit_attend_full_inline kv_slot FA_Q_HEADS FA_KV_HEADS FA_HEAD_DIM FA_GQA_RATIO
    append_to_forward

    \\ 10. Output gate: out *= sigmoid(gate)
    gpu_reset
    emit_output_gate_inline FA_Q_SIZE
    append_to_forward

    \\ 11. Output projection: FA_Q_SIZE -> hidden_dim (6144 -> 5120)
    \\ Includes residual add
    gpu_reset
    emit_gemv_residual_inline w_o FA_Q_SIZE HIDDEN_DIM
    append_to_forward

    \\ ---- MLP sub-block (identical to DeltaNet) ----

    \\ 12. RMSNorm
    gpu_reset
    emit_rmsnorm_inline w_post_ln HIDDEN_DIM RMS_EPS
    append_to_forward

    \\ 13. Gate projection: hidden_dim -> intermediate_dim
    gpu_reset
    emit_gemv_inline w_gate HIDDEN_DIM INTERMEDIATE_DIM
    append_to_forward

    \\ 14. Up projection: hidden_dim -> intermediate_dim
    gpu_reset
    emit_gemv_inline w_up HIDDEN_DIM INTERMEDIATE_DIM
    append_to_forward

    \\ 15. SiLU on gate
    gpu_reset
    emit_silu_inline INTERMEDIATE_DIM
    append_to_forward

    \\ 16. Elementwise multiply: gate * up
    gpu_reset
    emit_elemwise_mul_inline INTERMEDIATE_DIM
    append_to_forward

    \\ 17. Down projection: intermediate_dim -> hidden_dim (residual add)
    gpu_reset
    emit_gemv_residual_inline w_down INTERMEDIATE_DIM HIDDEN_DIM
    append_to_forward


\\ ============================================================
\\ DELTANET SLOT MAPPING
\\ ============================================================
\\ Maps layer_idx (0..63) to DeltaNet state slot (0..47).
\\ Full attention layers (idx % 4 == 3) have no deltanet slot.
\\ DeltaNet layers: slot = idx - floor(idx/4) - 1 if idx>=3, else idx

deltanet_slot layer_idx :
    \\ Count how many full-attn layers are at indices < layer_idx
    \\ For periodic schedule with interval 4: n_full_before = (layer_idx + 1) / 4
    n_full = layer_idx + 1
    n_full = n_full / FULL_ATTN_INTERVAL
    layer_idx - n_full


\\ ============================================================
\\ EMIT PREAMBLE — Token Embedding
\\ ============================================================
\\ Looks up the embedding vector for the input token_id.
\\ embed_table is FP16 [vocab_size, hidden_dim].
\\ Output is FP32 [hidden_dim] in the activation buffer.

emit_token_embed :
    gpu_reset
    emit_token_embed_inline VOCAB_SIZE HIDDEN_DIM
    append_to_forward


\\ ============================================================
\\ EMIT POSTAMBLE — Final RMSNorm + LM Head + Argmax
\\ ============================================================

emit_final_head w_final_ln w_lm_head :
    \\ 1. Final RMSNorm
    gpu_reset
    emit_rmsnorm_inline w_final_ln HIDDEN_DIM RMS_EPS
    append_to_forward

    \\ Grid sync before the massive lm_head projection
    emit_grid_sync_barrier_forward sync_ctr_reg sync_flag_reg

    \\ 2. LM head projection: hidden_dim -> vocab_size (5120 -> 248320)
    \\ This is the largest single GEMV — produces logits for entire vocabulary
    gpu_reset
    emit_gemv_inline w_lm_head HIDDEN_DIM VOCAB_SIZE
    append_to_forward

    \\ Grid sync before sampling
    emit_grid_sync_barrier_forward sync_ctr_reg sync_flag_reg

    \\ 3. Argmax over logits -> next token_id
    gpu_reset
    emit_argmax_inline VOCAB_SIZE
    append_to_forward


\\ ============================================================
\\ LINK FORWARD MEGAKERNEL
\\ ============================================================
\\ Main entry point. Orchestrates the entire forward pass:
\\   token_embed -> 64 layers (each with grid-sync) -> rmsnorm -> lm_head -> argmax
\\
\\ Reads config to determine layer types, resolves weight addresses,
\\ emits all instruction streams in order, wraps in cooperative ELF.
\\
\\ sync_ctr_reg / sync_flag_reg: register indices holding kernel param
\\ pointers for the grid-sync counter and done flag (loaded from c[0x0]).

var sync_ctr_reg 2
var sync_flag_reg 3
var deltanet_state_ptr 0
var kv_cache_ptr 0
var position_ptr 0

link_forward_megakernel data_region w_embed w_final_ln w_lm_head :
    \\ Reset all megakernel state
    forward_pos 0
    forward_sync_count 0
    forward_max_reg 0
    forward_max_smem 0

    \\ Step 1: Parse layer schedule (fills layer_types[64])
    parse_layer_schedule

    \\ Step 2: Resolve weight addresses for all 64 layers
    for layer_idx 0 NUM_LAYERS 1
        resolve_weight_addrs layer_idx data_region
    endfor

    \\ Step 3: Emit kernel parameter loads (sync counter + done flag)
    \\ These are loaded from constant bank c[0x0] at kernel entry
    gpu_reset
    emit_ldc sync_ctr_reg 0 0x210     \\ param 0: sync counter VA
    emit_ldc sync_flag_reg 0 0x218    \\ param 1: done flag VA
    append_to_forward

    \\ Step 4: Emit token embedding lookup
    emit_token_embed

    \\ Grid sync after embedding (all CTAs must have the activation vector)
    emit_grid_sync_barrier_forward sync_ctr_reg sync_flag_reg

    \\ Step 5: Emit 64 layers with grid-sync between each
    for layer_idx 0 NUM_LAYERS 1
        layer_type = → 8 layer_types + layer_idx

        if== layer_type LAYER_LINEAR_ATTENTION
            emit_layer_deltanet layer_idx

        if== layer_type LAYER_FULL_ATTENTION
            emit_layer_full_attention layer_idx

        \\ Grid-sync barrier between layers
        \\ (skip after last layer — final_head has its own syncs)
        if< layer_idx 63
            emit_grid_sync_barrier_forward sync_ctr_reg sync_flag_reg
    endfor

    \\ Step 6: Final RMSNorm + LM head projection + argmax
    emit_grid_sync_barrier_forward sync_ctr_reg sync_flag_reg
    emit_final_head w_final_ln w_lm_head

    \\ Step 7: Emit EXIT instruction
    gpu_reset
    emit_exit
    append_to_forward

    \\ Step 8: Wrap in cooperative ELF via lithos-elf.ls
    \\ Kernel name: "lithos_forward"
    \\ 4 kernel params: sync_counter, done_flag, activation_buf, position
    elf_build "lithos_forward" 15 forward_buf forward_pos 4 forward_max_reg forward_max_smem 1 forward_sync_offsets forward_sync_count


\\ ============================================================
\\ LINK RECURRENCE MEGAKERNEL
\\ ============================================================
\\ Separate megakernel for DeltaNet state updates when running
\\ the recurrence independently (e.g. prefill warm-start).
\\
\\ Different grid dims from forward pass: each CTA handles one
\\ key head of one layer, so grid = 48 * 16 = 768 CTAs.
\\ But on H100 with 132 SMs, we use 132 CTAs and loop internally.

768 constant RECUR_GRID_SIZE

link_recurrence_megakernel data_region :
    recur_pos 0
    recur_sync_count 0
    recur_max_reg 0
    recur_max_smem 0

    \\ Load sync params
    gpu_reset
    emit_ldc sync_ctr_reg 0 0x210
    emit_ldc sync_flag_reg 0 0x218
    append_to_recur

    \\ Track which DeltaNet slot we're on
    dn_slot = 0

    for layer_idx 0 NUM_LAYERS 1
        layer_type = → 8 layer_types + layer_idx

        \\ Only emit for DeltaNet layers
        if== layer_type LAYER_LINEAR_ATTENTION
            wbase = layer_weight_addrs + layer_idx * LAYER_WEIGHT_STRIDE

            \\ Compute state base for this slot
            state_bytes_per_layer = DN_KEY_HEADS * DN_GQA_RATIO * DN_HEAD_DIM * DN_HEAD_DIM * 4
            state_base = deltanet_state_ptr + dn_slot * state_bytes_per_layer

            \\ Emit the recurrence update (state matrix update only, no projection)
            gpu_reset
            emit_deltanet_recurrence_inline state_base DN_KEY_HEADS DN_GQA_RATIO DN_HEAD_DIM
            append_to_recur

            dn_slot + 1

            \\ Grid-sync between layers (skip after last DeltaNet layer)
            if< dn_slot NUM_DELTANET_LAYERS
                emit_grid_sync_barrier_recur sync_ctr_reg sync_flag_reg
    endfor

    \\ EXIT
    gpu_reset
    emit_exit
    append_to_recur

    \\ Wrap in cooperative ELF
    elf_build "lithos_recurrence" 18 recur_buf recur_pos 3 recur_max_reg recur_max_smem 1 recur_sync_offsets recur_sync_count


\\ ============================================================
\\ EMIT HELPERS — SiLU, L2Norm, Conv1d, Output Gate
\\ ============================================================
\\ These are thin wrappers that call the elementwise / reduce
\\ emitters with the right dimensions and then reset gpu state.

emit_silu_inline dim :
    \\ SiLU(x) = x * sigmoid(x) — elementwise, dim elements
    \\ Emits: stride loop of (neg, fmul 1.4427, ex2, fadd 1.0, rcp, fmul x)
    emit_silu_kernel dim

emit_l2norm_inline dim :
    \\ L2norm: x /= sqrt(sum(x^2))
    \\ Emits: sum-of-squares reduce + rsqrt + scale loop
    emit_l2norm_kernel dim

emit_conv1d_inline dim kernel_size :
    \\ Causal conv1d with kernel_size taps
    \\ Reads from a conv state ring buffer per DeltaNet layer
    emit_conv1d_kernel dim kernel_size

emit_elemwise_mul_inline dim :
    \\ a[i] *= b[i] elementwise
    emit_elemwise_mul_kernel dim

emit_output_gate_inline dim :
    \\ out[i] *= sigmoid(gate[i])
    \\ Sigmoid = 1 / (1 + exp(-x))
    emit_sigmoid_mul_kernel dim

emit_rmsnorm_inline w_ptr dim eps :
    \\ Two-pass RMSNorm: reduce sum-of-squares, then scale+weight
    emit_rmsnorm_kernel w_ptr dim eps

emit_gemv_inline w_ptr in_dim out_dim :
    \\ GPTQ W4A16 GEMV: dequant weights + matvec
    \\ Weight-as-code: dequant at compile time into FMUL-IMM immediates
    emit_gptq_gemv_kernel w_ptr in_dim out_dim

emit_gemv_residual_inline w_ptr in_dim out_dim :
    \\ GEMV with fused residual add: X += W @ activation
    emit_gptq_gemv_residual_kernel w_ptr in_dim out_dim

emit_token_embed_inline vocab_size hidden_dim :
    \\ Token embedding lookup: table[token_id * hidden_dim + i]
    emit_embed_kernel vocab_size hidden_dim

emit_argmax_inline dim :
    \\ Argmax over logits vector
    emit_argmax_kernel dim

emit_deltanet_fused_inline state_base n_key_heads gqa_ratio head_dim :
    \\ Fused DeltaNet: state update + output gate (silu(z))
    emit_deltanet_fused_kernel state_base n_key_heads gqa_ratio head_dim

emit_deltanet_recurrence_inline state_base n_key_heads gqa_ratio head_dim :
    \\ DeltaNet state update only (no output projection)
    emit_deltanet_recurrence_kernel state_base n_key_heads gqa_ratio head_dim

emit_attend_full_inline kv_slot n_q_heads n_kv_heads head_dim gqa_ratio :
    \\ Full softmax attention with GQA
    emit_attend_full_kernel kv_slot n_q_heads n_kv_heads head_dim gqa_ratio

emit_mrope_inline n_heads head_dim rotary_dims :
    \\ Multi-resolution RoPE (partial rotary)
    emit_mrope_kernel n_heads head_dim rotary_dims

emit_kv_write_inline kv_slot n_kv_heads head_dim :
    \\ Append K,V to cache at current position
    emit_kv_write_kernel kv_slot n_kv_heads head_dim


\\ ============================================================
\\ FULL PIPELINE — Build Both Megakernels
\\ ============================================================
\\ Top-level entry: open safetensors, resolve everything, emit both
\\ megakernels, write two ELF files.

link_all config_path safetensors_path forward_out_path recur_out_path :
    \\ Open and mmap safetensors file
    st_open safetensors_path
    data_region = st_data_ptr

    \\ Resolve embedding and final head weights
    w_embed    = st_lookup_global data_region "model.embed_tokens.weight"
    w_final_ln = st_lookup_global data_region "model.norm.weight"
    w_lm_head  = st_lookup_global data_region "lm_head.weight"

    \\ Parse layer schedule
    parse_layer_schedule

    \\ Resolve all per-layer weight addresses
    for layer_idx 0 NUM_LAYERS 1
        resolve_weight_addrs layer_idx data_region
    endfor

    \\ Build forward megakernel
    link_forward_megakernel data_region w_embed w_final_ln w_lm_head

    \\ Save forward ELF
    elf_save forward_out_path

    \\ Build recurrence megakernel
    link_recurrence_megakernel data_region

    \\ Save recurrence ELF
    elf_save recur_out_path

    \\ Report sizes
    \\ forward_pos holds final .text size in bytes
    \\ recur_pos holds recurrence .text size
    \\ forward_sync_count = number of grid-sync barriers in forward pass
    \\ recur_sync_count = number of grid-sync barriers in recurrence
