// launcher.s -- Native ARM64 launcher for Lithos inference engine
//
// Replaces the Python/ctypes dispatch layer entirely.
// Links dynamically against libcuda.so.1 (CUDA driver API).
//
// Performance:
//   Python dispatch: 314ms per token (97% overhead on 9ms GPU work)
//   Native dispatch: ~13us per token (130 BLR calls * 100ns each)
//   Speedup: ~24,000x dispatch, bringing total to ~9ms/token = ~111 tok/s
//
// Build:
//   as -o launcher.o launcher.s
//   ld -o lithos-launch launcher.o -lc -lcuda -dynamic-linker /lib/ld-linux-aarch64.so.1
//   OR: use the build-launcher.fs Forth script
//
// Usage:
//   ./lithos-launch <model_dir> <kernel_dir> [max_tokens]
//
// Architecture:
//   _start -> init CUDA -> load cubins -> mmap weights -> decode loop
//   decode loop: embed -> 64 layers -> final_norm -> lm_head -> sample -> print
//
// Register allocation in decode hot loop:
//   x19 = CUstream handle
//   x20 = activation buffer A base
//   x21 = activation buffer B base
//   x22 = residual stream pointer
//   x23 = weight base pointer (mmap'd safetensors)
//   x24 = KV cache base
//   x25 = DeltaNet state base
//   x26 = sequence position
//   x27 = act buffer selector (0=A input, 1=B input)
//   x28 = kernel function table base

.equ SYS_READ,      63
.equ SYS_WRITE,     64
.equ SYS_OPENAT,    56
.equ SYS_CLOSE,     57
.equ SYS_LSEEK,     62
.equ SYS_MMAP,      222
.equ SYS_MADVISE,   233
.equ SYS_MPROTECT,  226
.equ MADV_WILLNEED,   3
.equ SYS_EXIT,      93
.equ SYS_BRK,       214
.equ SYS_PREAD64,    67
.equ SYS_FSTAT,      80

.equ AT_FDCWD,      -100
.equ O_RDONLY,       0
.equ SEEK_SET,       0
.equ SEEK_END,       2

.equ PROT_READ,      1
.equ PROT_WRITE,     2
.equ PROT_RW,        3
.equ MAP_PRIVATE,    2
.equ MAP_ANONYMOUS,  0x20
.equ MAP_PRIV_ANON,  0x22
.equ CU_MEMHOSTREGISTER_DEVICEMAP, 0x2  // cuMemHostRegister flag: create device mapping

// ---- Model constants (Qwen 3.5-27B -- from config.json) ----
.equ NUM_LAYERS,       64
.equ HIDDEN_DIM,       5120      // was 3584, corrected from config.json
.equ NUM_HEADS,        24        // was 28, corrected (num_attention_heads)
.equ NUM_KV_HEADS,     4
.equ HEAD_DIM,         256       // was 128, corrected (head_dim for full attn)
.equ INTERMEDIATE_DIM, 17408     // was 18944, corrected
.equ VOCAB_SIZE,       248320    // was 152064, corrected
.equ DTYPE_BYTES,      2
.equ DELTANET_PERIOD,  4
.equ DELTANET_FA_IDX,  3
// DeltaNet-specific dimensions:
// linear_key_head_dim=128, linear_num_key_heads=16, linear_num_value_heads=48
// linear_value_head_dim=128, linear_conv_kernel_dim=4

// ---- Kernel function table offsets ----
.equ KF_EMBED,        0
.equ KF_NORM,         8
.equ KF_PROJ,         16
.equ KF_ATTN,         24
.equ KF_RECUR,        32
.equ KF_ACTIVATE,     40
.equ KF_ROPE,         48
.equ KF_SAMPLE,       56
.equ KF_GATE_SIG,     64
.equ KF_CONV1D,       72
.equ KF_L2NORM,       80
.equ KF_LM_HEAD,      88
.equ KF_TABLE_SIZE,   96

// ---- Per-layer weight pointer slot indices ----
// Access via: ldr x_reg, [x23, #(WS_xxx * 8)]
// where x23 = weight_ptrs base for current layer
//
// DeltaNet layers (layer_idx % 4 != 3):
.equ WS_DN_QKV_QWEIGHT,    0    // linear_attn.in_proj_qkv.qweight
.equ WS_DN_QKV_SCALES,     1    // linear_attn.in_proj_qkv.scales
.equ WS_DN_QKV_QZEROS,     2    // linear_attn.in_proj_qkv.qzeros
.equ WS_DN_Z_QWEIGHT,      3    // linear_attn.in_proj_z.qweight
.equ WS_DN_Z_SCALES,       4    // linear_attn.in_proj_z.scales
.equ WS_DN_Z_QZEROS,       5    // linear_attn.in_proj_z.qzeros
.equ WS_DN_OUT_QWEIGHT,    6    // linear_attn.out_proj.qweight
.equ WS_DN_OUT_SCALES,     7    // linear_attn.out_proj.scales
.equ WS_DN_OUT_QZEROS,     8    // linear_attn.out_proj.qzeros
.equ WS_DN_PROJ_A,         9    // linear_attn.in_proj_a.weight (bf16)
.equ WS_DN_PROJ_B,        10    // linear_attn.in_proj_b.weight (bf16)
.equ WS_DN_A_LOG,         11    // linear_attn.A_log (f32)
.equ WS_DN_DT_BIAS,       12    // linear_attn.dt_bias (bf16)
.equ WS_DN_CONV1D,        13    // linear_attn.conv1d.weight (bf16)
.equ WS_DN_NORM,          14    // linear_attn.norm.weight (f32)
//
// Full-attention layers (layer_idx % 4 == 3):
.equ WS_FA_Q_QWEIGHT,      0    // self_attn.q_proj.qweight
.equ WS_FA_Q_SCALES,       1    // self_attn.q_proj.scales
.equ WS_FA_Q_QZEROS,       2    // self_attn.q_proj.qzeros
.equ WS_FA_K_QWEIGHT,      3    // self_attn.k_proj.qweight
.equ WS_FA_K_SCALES,       4    // self_attn.k_proj.scales
.equ WS_FA_K_QZEROS,       5    // self_attn.k_proj.qzeros
.equ WS_FA_V_QWEIGHT,      6    // self_attn.v_proj.qweight
.equ WS_FA_V_SCALES,       7    // self_attn.v_proj.scales
.equ WS_FA_V_QZEROS,       8    // self_attn.v_proj.qzeros
.equ WS_FA_O_QWEIGHT,      9    // self_attn.o_proj.qweight
.equ WS_FA_O_SCALES,      10    // self_attn.o_proj.scales
.equ WS_FA_O_QZEROS,      11    // self_attn.o_proj.qzeros
.equ WS_FA_Q_NORM,        12    // self_attn.q_norm.weight (bf16)
.equ WS_FA_K_NORM,        13    // self_attn.k_norm.weight (bf16)
// Slot 14 unused for FA layers
//
// Common (same index for both layer types):
.equ WS_GATE_QWEIGHT,     15    // mlp.gate_proj.qweight
.equ WS_GATE_SCALES,      16    // mlp.gate_proj.scales
.equ WS_GATE_QZEROS,      17    // mlp.gate_proj.qzeros
.equ WS_UP_QWEIGHT,       18    // mlp.up_proj.qweight
.equ WS_UP_SCALES,        19    // mlp.up_proj.scales
.equ WS_UP_QZEROS,        20    // mlp.up_proj.qzeros
.equ WS_DOWN_QWEIGHT,     21    // mlp.down_proj.qweight
.equ WS_DOWN_SCALES,      22    // mlp.down_proj.scales
.equ WS_DOWN_QZEROS,      23    // mlp.down_proj.qzeros
.equ WS_INPUT_NORM,       24    // input_layernorm.weight (bf16)
.equ WS_POST_NORM,        25    // post_attention_layernorm.weight (bf16)

// ---- Activation buffer size ----
// Must hold the largest intermediate tensor that lands in an act buffer.
// DeltaNet QKV combined output = 10240 * sizeof(f32) = 40960 bytes.
// (HIDDEN_DIM * 4 = 20480 is too small; QKV output is 2x hidden.)
.equ ACT_BUF_SIZE,    40960
// Residual buffer = same
.equ RES_BUF_SIZE,    20480

// ---- KV cache per full-attention layer ----
// 2 * max_seq * kv_heads * head_dim * sizeof(f32) = 2 * 32768 * 4 * 256 * 4 = 256MB per layer
// 16 FA layers * 256MB = 4GB total
.equ MAX_SEQ_LEN,     32768
.equ KV_PER_LAYER,    268435456   // 256MB

// ---- DeltaNet state per layer ----
// 48 value_heads * 128 * 128 * sizeof(f32) = 48 * 128 * 128 * 4 = 3145728
.equ DN_PER_LAYER,    3145728

// ---- DeltaNet output gate (z projection) buffer ----
// value_dim = 48 * 128 = 6144, sizeof(f32) = 4 => 24576 bytes
.equ VALUE_DIM,        6144
.equ Z_BUF_SIZE,       24576

// ---- Conv1d shift register state per layer ----
// 10240 channels * 4 taps * sizeof(f32) = 10240 * 4 * 4 = 163840
.equ CONV1D_CHANNELS, 10240
.equ CONV1D_PER_LAYER, 163840

// ============================================================
// Data section
// ============================================================

.data

// ---- Strings ----
msg_init:     .asciz "lithos: CUDA driver initialized\n"
msg_init_len = . - msg_init - 1
msg_ctx:      .asciz "lithos: context created on device 0\n"
msg_ctx_len = . - msg_ctx - 1
msg_load:     .asciz "lithos: loading cubins...\n"
msg_load_len = . - msg_load - 1
msg_ready:    .asciz "lithos: native launcher ready\n"
msg_ready_len = . - msg_ready - 1
msg_tok:      .asciz "lithos: generating tokens...\n"
msg_tok_len = . - msg_tok - 1
msg_err:      .asciz "lithos: CUDA error, code="
msg_err_len = . - msg_err - 1
msg_dbg:      .asciz "lithos: dbg layer="
msg_dbg_len = . - msg_dbg - 1
msg_dbg2:     .asciz " tag="
msg_dbg2_len = . - msg_dbg2 - 1
msg_dbg3:     .asciz " err="
msg_dbg3_len = . - msg_dbg3 - 1
msg_nl:       .asciz "\n"
msg_done:     .asciz "\nlithos: done\n"
msg_done_len = . - msg_done - 1
msg_exit:     .asciz "lithos: exiting now\n"
msg_exit_len = . - msg_exit - 1
msg_toks:     .asciz "lithos: "
msg_toks_len = . - msg_toks - 1
msg_toks2:    .asciz " tokens in "
msg_toks2_len = . - msg_toks2 - 1
msg_us:       .asciz "us = "
msg_us_len = . - msg_us - 1
msg_speed:    .asciz " tok/s\n"
msg_speed_len = . - msg_speed - 1

msg_weights:  .asciz "lithos: loading model weights...\n"
msg_weights_len = . - msg_weights - 1
msg_shard:    .asciz "lithos: mmap shard "
msg_shard_len = . - msg_shard - 1
msg_bytes:    .asciz " bytes\n"
msg_bytes_len = . - msg_bytes - 1
msg_wdone:    .asciz "lithos: weight table built (1648 tensors from 4 shards)\n"
msg_wdone_len = . - msg_wdone - 1
msg_werr:     .asciz "lithos: ERROR: failed to mmap shard\n"
msg_werr_len = . - msg_werr - 1
msg_hostreg:  .asciz "lithos: cuMemHostRegister shard "
msg_hostreg_len = . - msg_hostreg - 1
msg_hostreg_ok: .asciz " -> OK (GPU-pinned)\n"
msg_hostreg_ok_len = . - msg_hostreg_ok - 1
msg_hostreg_fail: .asciz " -> FAILED rc="
msg_hostreg_fail_len = . - msg_hostreg_fail - 1
msg_idxerr:   .asciz "lithos: ERROR: weight_index.bin not found\n"
msg_idxerr_len = . - msg_idxerr - 1

// Weight index filename
widx_filename: .asciz "weight_index.bin"

// Cubin filenames (relative to kernel_dir arg)
cubin_embed:       .asciz "embed_f16.cubin"
cubin_norm:        .asciz "norm.cubin"
cubin_proj:        .asciz "gptq_gemv_safe.cubin"    // was projection.cubin
cubin_attn:        .asciz "attention_score.cubin"
cubin_recur:       .asciz "recurrence.cubin"
cubin_activate:    .asciz "activate.cubin"
cubin_rope:        .asciz "rotate.cubin"
cubin_gate_sig:    .asciz "gate_sigmoid.cubin"
cubin_conv1d:      .asciz "recur.cubin"
cubin_l2norm:      .asciz "reduce.cubin"

// Kernel function names (must match symbols inside the cubins)
kname_embed:       .asciz "embed_f16"
kname_norm:        .asciz "norm"
kname_proj:        .asciz "gptq_gemv_safe"           // was "projection"
kname_attn:        .asciz "attention_score"
kname_recur:       .asciz "recurrence"
kname_activate:    .asciz "activate"
kname_rope:        .asciz "rotate"
kname_gate_sig:    .asciz "gate_sigmoid"
kname_conv1d:      .asciz "conv1d_infer"
kname_l2norm:      .asciz "l2norm"
cubin_lm_head:     .asciz "lm_head.cubin"
kname_lm_head:     .asciz "lm_head"

// Path buffer
.align 3
path_buf:     .space 512

// ---- State variables ----
.align 3
cuda_device:  .quad 0
cuda_context: .quad 0
cuda_stream:  .quad 0
cuda_stream_q: .quad 0          // Q projection stream (multi-stream FA QKV)
cuda_stream_k: .quad 0          // K projection stream
cuda_stream_v: .quad 0          // V projection stream
cuda_event_q:  .quad 0          // completion event for Q projection
cuda_event_k:  .quad 0          // completion event for K projection
cuda_event_v:  .quad 0          // completion event for V projection
seq_position: .quad 0
last_token:   .quad 0
num_tokens_generated: .quad 0

// Kernel function table (8 CUfunction handles)
.align 3
kern_table:   .space KF_TABLE_SIZE

// Module handles
.align 3
modules:      .space 64      // 8 CUmodule handles

// Launch parameter scratch area (used to build kernelParams)
.align 3
launch_params: .space 256    // 32 param slots * 8 bytes
param_ptrs:    .space 256    // 32 void* pointers into launch_params
// Per-stream launch param scratch for parallel Q/K/V projection
.align 3
launch_params_q: .space 256
param_ptrs_q:    .space 256
launch_params_k: .space 256
param_ptrs_k:    .space 256
launch_params_v: .space 256
param_ptrs_v:    .space 256

// Per-layer weight POINTER table (resolved virtual addresses)
// 64 layers * 26 slots per layer * 8 bytes = 13312 bytes
// Each entry is a direct pointer into mmap'd shard data
.equ WEIGHT_SLOTS_PER_LAYER, 26
.equ WEIGHT_TABLE_STRIDE, (WEIGHT_SLOTS_PER_LAYER * 8)   // 208 bytes per layer
.align 3
weight_ptrs:   .space 13312  // 64 * 26 * 8

// Global weight pointers (embed, lm_head, final_norm)
.equ GLOBAL_EMBED, 0
.equ GLOBAL_LMHEAD, 8
.equ GLOBAL_FINALNORM, 16
.align 3
global_weight_ptrs: .space 24   // 3 * 8 bytes

// Shard mmap base pointers (up to 8 shards)
.align 3
shard_bases:   .space 64     // 8 * 8 bytes
shard_sizes:   .space 64     // 8 * 8 bytes
num_shards_loaded: .quad 0

// Weight index mmap
.align 3
widx_base:     .quad 0       // mmap'd weight_index.bin base
widx_size:     .quad 0       // size of weight_index.bin

// Activation buffers, residual, KV cache, DN state base pointers
.align 3
act_buf_a:     .quad 0
act_buf_b:     .quad 0
residual_buf:  .quad 0
kv_cache_ptr:  .quad 0
dn_state_ptr:  .quad 0
act_selector:  .quad 0       // 0 = A is input, 1 = B is input
z_buf:         .quad 0       // output gate projection buffer (value_dim=6144 * 4 = 24576 bytes)
conv1d_state:  .quad 0       // conv1d shift register state (all layers)

// ---- Per-token layer counters (contiguous; reset once per token with stp xzr,xzr) ----
dn_layer_counter:  .quad 0   // counts DeltaNet layers (layer_idx % 4 != 3)
fa_layer_counter:  .quad 0   // counts full-attention layers (layer_idx % 4 == 3)

// ---- Per-layer state counters (contiguous; reset per layer with 2x stp xzr,xzr) ----
proj_state:         .quad 0  // which projection within the layer (0..6)
norm_state:         .quad 0  // which norm within the layer (0..1)
conv1d_call_counter:.quad 0  // which conv1d call within the layer (0=Q, 1=K, 2=V)
l2norm_call_counter:.quad 0  // which l2norm call within the layer (0=K, 1=Q)

// ---- RoPE cos/sin precomputed tables ----
rope_cos_table: .quad 0      // pointer to precomputed cos table
rope_sin_table: .quad 0      // pointer to precomputed sin table

// ---- MLP scratch buffers ----
mlp_gate_buf:  .quad 0       // gate projection output (INTERMEDIATE_DIM * 4 bytes)
mlp_up_buf:    .quad 0       // up projection output (INTERMEDIATE_DIM * 4 bytes)

// ---- Logits buffer ----
// VOCAB_SIZE * sizeof(f32) = 248320 * 4 = 993280 bytes (~970KB)
.equ LOGITS_BUF_SIZE, 993280
logits_buf:    .quad 0

// ---- GPU logits device buffer (CUdeviceptr) ----
// Allocated via cuMemAlloc; lm_head kernel writes here, then DtoH copy to logits_buf
cuda_logits_dev: .quad 0

// Shard allocation temp pointer (for cuMemAllocHost output)
.align 3
shard_alloc_ptr: .quad 0

// Timing
.align 3
time_start:    .quad 0
time_end:      .quad 0

// Saved argc/argv
.align 3
saved_argc:    .quad 0
saved_argv:    .quad 0

// ============================================================
// BSS section (large allocations)
// ============================================================

.bss
.align 12     // 4KB page alignment

// File read buffer (1MB for cubin loading)
file_buffer:  .space 1048576

// ============================================================
// Text section
// ============================================================

.text
.global _start
.align 4

// ============================================================
// _start -- entry point
// ============================================================
_start:
    // Save argc, argv
    ldr     x0, [sp]              // argc
    add     x1, sp, #8            // argv
    adrp    x2, saved_argc
    add     x2, x2, :lo12:saved_argc
    stp     x0, x1, [x2]

    // Set up frame
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    // Save callee-saved regs
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!

    // ---- cuInit(0) ----
    mov     x0, #0
    bl      cuInit
    cbnz    x0, cuda_error

    // Print init message
    adrp    x1, msg_init
    add     x1, x1, :lo12:msg_init
    mov     x2, msg_init_len
    bl      print_msg

    // ---- cuDeviceGet(&device, 0) ----
    adrp    x0, cuda_device
    add     x0, x0, :lo12:cuda_device
    mov     x1, #0
    bl      cuDeviceGet
    cbnz    x0, cuda_error

    // ---- cuCtxCreate_v2(&context, 0, device) ----
    adrp    x0, cuda_context
    add     x0, x0, :lo12:cuda_context
    mov     x1, #0                 // flags
    adrp    x2, cuda_device
    add     x2, x2, :lo12:cuda_device
    ldr     w2, [x2]               // device (int)
    bl      cuCtxCreate_v2
    cbnz    x0, cuda_error

    adrp    x1, msg_ctx
    add     x1, x1, :lo12:msg_ctx
    mov     x2, msg_ctx_len
    bl      print_msg

    // ---- cuStreamCreate(&stream, 0) ----
    adrp    x0, cuda_stream
    add     x0, x0, :lo12:cuda_stream
    mov     x1, #0
    bl      cuStreamCreate
    cbnz    x0, cuda_error

    // ---- Create Q/K/V projection streams ----
    adrp    x0, cuda_stream_q
    add     x0, x0, :lo12:cuda_stream_q
    mov     x1, #0
    bl      cuStreamCreate
    cbnz    x0, cuda_error

    adrp    x0, cuda_stream_k
    add     x0, x0, :lo12:cuda_stream_k
    mov     x1, #0
    bl      cuStreamCreate
    cbnz    x0, cuda_error

    adrp    x0, cuda_stream_v
    add     x0, x0, :lo12:cuda_stream_v
    mov     x1, #0
    bl      cuStreamCreate
    cbnz    x0, cuda_error

    // ---- Create completion events for Q/K/V (CU_EVENT_DISABLE_TIMING=0x2) ----
    // Using flag 0 (default) for simplicity; disable timing for lower overhead
    adrp    x0, cuda_event_q
    add     x0, x0, :lo12:cuda_event_q
    mov     x1, #2                  // CU_EVENT_DISABLE_TIMING
    bl      cuEventCreate
    cbnz    x0, cuda_error

    adrp    x0, cuda_event_k
    add     x0, x0, :lo12:cuda_event_k
    mov     x1, #2
    bl      cuEventCreate
    cbnz    x0, cuda_error

    adrp    x0, cuda_event_v
    add     x0, x0, :lo12:cuda_event_v
    mov     x1, #2
    bl      cuEventCreate
    cbnz    x0, cuda_error

    // ---- Allocate activation buffers via mmap ----
    bl      alloc_buffers

    // ---- Load and mmap model weights ----
    adrp    x1, msg_weights
    add     x1, x1, :lo12:msg_weights
    mov     x2, msg_weights_len
    bl      print_msg

    bl      load_weights

    // ---- Load cubins ----
    adrp    x1, msg_load
    add     x1, x1, :lo12:msg_load
    mov     x2, msg_load_len
    bl      print_msg

    bl      load_cubins

    // ---- Print ready message ----
    adrp    x1, msg_ready
    add     x1, x1, :lo12:msg_ready
    mov     x2, msg_ready_len
    bl      print_msg

    // ---- Decode loop ----
    // For the initial version, we demonstrate the hot loop structure
    // with a configurable number of tokens.
    // Default: 32 tokens (enough to measure dispatch overhead)

    mov     x26, #0                 // seq_pos = 0
    mov     x27, #0                 // act_selector = 0

    adrp    x1, msg_tok
    add     x1, x1, :lo12:msg_tok
    mov     x2, msg_tok_len
    bl      print_msg

    // Get max_tokens from argv[3] or default to 32
    adrp    x0, saved_argc
    add     x0, x0, :lo12:saved_argc
    ldr     x0, [x0]
    cmp     x0, #4
    b.lt    .use_default_tokens
    adrp    x0, saved_argv
    add     x0, x0, :lo12:saved_argv
    ldr     x0, [x0]
    ldr     x0, [x0, #24]          // argv[3]
    bl      atoi
    mov     x28, x0
    cbnz    x28, .have_token_count
.use_default_tokens:
    mov     x28, #32
.have_token_count:

    // ---- Read start time (CLOCK_MONOTONIC) ----
    // clock_gettime(CLOCK_MONOTONIC=1, &timespec)
    mov     x0, #1                  // CLOCK_MONOTONIC
    adrp    x1, time_start
    add     x1, x1, :lo12:time_start
    mov     x8, #113                // SYS_CLOCK_GETTIME
    svc     #0

    // ---- Main decode loop ----
.decode_loop:
    cbz     x28, .decode_done

    // ---- One token: embed + 64 layers + final_norm + lm_head + sample ----
    bl      decode_one_token

    // Advance
    add     x26, x26, #1           // seq_pos++
    sub     x28, x28, #1           // tokens_remaining--

    // Count generated tokens
    adrp    x0, num_tokens_generated
    add     x0, x0, :lo12:num_tokens_generated
    ldr     x1, [x0]
    add     x1, x1, #1
    str     x1, [x0]

    b       .decode_loop

.decode_done:
    // ---- Read end time ----
    mov     x0, #1                  // CLOCK_MONOTONIC
    adrp    x1, time_end
    add     x1, x1, :lo12:time_end
    mov     x8, #113
    svc     #0

    // ---- Compute and print elapsed time ----
    adrp    x0, time_end
    add     x0, x0, :lo12:time_end
    ldp     x2, x3, [x0]           // x2 = end_sec, x3 = end_nsec
    adrp    x0, time_start
    add     x0, x0, :lo12:time_start
    ldp     x4, x5, [x0]           // x4 = start_sec, x5 = start_nsec

    sub     x6, x2, x4             // delta_sec
    sub     x7, x3, x5             // delta_nsec
    // Handle nsec borrow
    cmp     x7, #0
    b.ge    .no_borrow
    sub     x6, x6, #1
    movz    x0, #0x3B9A, lsl #16    // 0x3B9ACA00 = 1000000000
    movk    x0, #0xCA00
    add     x7, x7, x0
.no_borrow:

    // Compute total_usec = sec * 1000000 + nsec / 1000
    movz    x0, #0x000F, lsl #16    // 0x000F4240 = 1000000
    movk    x0, #0x4240
    mul     x6, x6, x0             // sec * 1M
    mov     x0, #1000
    udiv    x7, x7, x0             // nsec / 1000
    add     x6, x6, x7             // x6 = total_usec

    // Load num_tokens
    adrp    x0, num_tokens_generated
    add     x0, x0, :lo12:num_tokens_generated
    ldr     x7, [x0]               // x7 = num_tokens

    // Print "lithos: N tokens in Xus = Y tok/s"
    adrp    x1, msg_toks
    add     x1, x1, :lo12:msg_toks
    mov     x2, msg_toks_len
    bl      print_msg

    mov     x0, x7
    bl      print_int

    adrp    x1, msg_toks2
    add     x1, x1, :lo12:msg_toks2
    mov     x2, msg_toks2_len
    bl      print_msg

    mov     x0, x6
    bl      print_int

    adrp    x1, msg_us
    add     x1, x1, :lo12:msg_us
    mov     x2, msg_us_len
    bl      print_msg

    // Compute tok/s = num_tokens * 1000000 / total_usec
    movz    x0, #0x000F, lsl #16    // 1000000
    movk    x0, #0x4240
    mul     x0, x7, x0             // tokens * 1M
    udiv    x0, x0, x6             // / total_usec
    bl      print_int

    adrp    x1, msg_speed
    add     x1, x1, :lo12:msg_speed
    mov     x2, msg_speed_len
    bl      print_msg

    // ---- Exit immediately ----
    mov     x0, #0
    mov     x8, #94            // SYS_EXIT_GROUP (kills all threads)
    svc     #0

// ============================================================
// cuda_error -- print error and exit
// ============================================================
cuda_error:
    mov     x19, x0                 // save error code
    adrp    x1, msg_err
    add     x1, x1, :lo12:msg_err
    mov     x2, msg_err_len
    bl      print_msg

    // Print error code as decimal
    mov     x0, x19
    bl      print_int
    bl      print_newline

    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0

// ============================================================
// decode_one_token -- run the full transformer for one token
// ============================================================
// This is the HOT PATH that replaces 314ms of Python dispatch.
// Each kernel launch is a direct BLR to cuLaunchKernel.
// Total: ~130 launches * ~100ns = ~13us.
.align 4
decode_one_token:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!  // x22 = layer type flag (survives bl calls)

    // Reset per-token layer counters (dn_layer_counter + fa_layer_counter contiguous)
    adrp    x0, dn_layer_counter
    add     x0, x0, :lo12:dn_layer_counter
    stp     xzr, xzr, [x0]             // zero dn_layer_counter and fa_layer_counter

    // Load buffer pointers
    adrp    x20, act_buf_a
    add     x20, x20, :lo12:act_buf_a
    ldr     x20, [x20]              // x20 = act_buf_a

    adrp    x21, act_buf_b
    add     x21, x21, :lo12:act_buf_b
    ldr     x21, [x21]              // x21 = act_buf_b

    // Hoist launch_params and param_ptrs addresses into x11/x12.
    // All launch_* helpers treat these as implicit arguments (do not clobber
    // x11/x12 before consuming them as x1/x2 respectively).
    adrp    x11, launch_params
    add     x11, x11, :lo12:launch_params
    adrp    x12, param_ptrs
    add     x12, x12, :lo12:param_ptrs

    // ---- Embed lookup ----
    // cuLaunchKernel(embed_func, grid, block, 0, stream, params, NULL)
    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_EMBED]    // CUfunction
    cbz     x0, .skip_embed         // skip if not loaded

    // Build kernelParams for embed:
    //   param[0] = &token_id  (u32)
    //   param[1] = &embed_ptr (u64)
    //   param[2] = &output_ptr (u64)
    mov     x1, x11                 // launch_params (hoisted above)
    adrp    x2, last_token
    add     x2, x2, :lo12:last_token
    ldr     w3, [x2]
    str     w3, [x1]                // param[0] = token_id
    adrp    x2, global_weight_ptrs
    add     x2, x2, :lo12:global_weight_ptrs
    ldr     x3, [x2, #GLOBAL_EMBED]
    str     x3, [x1, #8]           // param[1] = embed_table_ptr (from mmap'd weights)
    // output goes to current input buffer
    str     x20, [x1, #16]         // param[2] = output_ptr

    // Build param_ptrs
    mov     x2, x12                 // param_ptrs (hoisted above)
    str     x1, [x2]               // param_ptrs[0] = &param[0]
    add     x3, x1, #8
    str     x3, [x2, #8]           // param_ptrs[1] = &param[1]
    add     x3, x1, #16
    str     x3, [x2, #16]          // param_ptrs[2] = &param[2]

    // cuLaunchKernel(func, gridX, gridY, gridZ, blockX, blockY, blockZ,
    //                sharedMem, stream, paramPtrs, extra)
    // x0 = func (already set)
    mov     x1, #5                  // gridDimX = ceil(5120 / 1024) = 5 (5*1024=5120 covers full HIDDEN_DIM)
    mov     x2, #1                  // gridDimY
    mov     x3, #1                  // gridDimZ
    mov     x4, #256                // blockDimX
    mov     x5, #1                  // blockDimY
    mov     x6, #1                  // blockDimZ
    mov     x7, #0                  // sharedMemBytes
    // Stack args: stream, kernelParams, extra
    adrp    x9, cuda_stream
    add     x9, x9, :lo12:cuda_stream
    ldr     x9, [x9]
    mov     x10, x12                // param_ptrs (hoisted above)
    stp     x9, x10, [sp, #-32]!   // push stream, paramPtrs
    str     xzr, [sp, #16]         // push extra = NULL
    bl      cuLaunchKernel
    add     sp, sp, #32

.skip_embed:

    // Re-hoist launch_params/param_ptrs: cuLaunchKernel (called above for embed)
    // is a C function and clobbers x11/x12 (caller-saved per ABI).
    adrp    x11, launch_params
    add     x11, x11, :lo12:launch_params
    adrp    x12, param_ptrs
    add     x12, x12, :lo12:param_ptrs

    // ---- Layer loop: 64 layers ----
    mov     x19, #0                 // layer counter

.layer_loop:
    cmp     x19, #NUM_LAYERS
    b.ge    .layers_done

    // ---- Reset per-layer state counters (all 4 contiguous; 2x stp xzr,xzr) ----
    adrp    x0, proj_state
    add     x0, x0, :lo12:proj_state
    stp     xzr, xzr, [x0]             // zero proj_state + norm_state
    stp     xzr, xzr, [x0, #16]        // zero conv1d_call_counter + l2norm_call_counter

    // ---- Set up weight pointer base for this layer ----
    // x23 = &weight_ptrs[layer_idx * WEIGHT_SLOTS_PER_LAYER]
    adrp    x23, weight_ptrs
    add     x23, x23, :lo12:weight_ptrs
    mov     x0, #WEIGHT_TABLE_STRIDE   // 26 * 8 = 208
    mul     x0, x19, x0
    add     x23, x23, x0              // x23 = base of this layer's weight pointers

    // ---- Determine layer type ----
    // layer_type = (layer_idx % 4 == 3) ? full_attention : deltanet
    // Save to x22 (callee-saved) so it survives bl calls.
    // x22 == 0 means full-attention layer, x22 != 0 means DeltaNet.
    and     x22, x19, #3
    sub     x22, x22, #DELTANET_FA_IDX

    // ---- Pre-attention RMSNorm ----
    bl      launch_norm
    mov     x0, #1; bl debug_sync_tag

    // ---- QKV projection ----
    // FA layers: Q, K, V are independent — launch on separate streams, sync before attention.
    // DeltaNet layers: combined QKV single kernel, use serial path.
    cbz     x22, .qkv_parallel
    // DeltaNet: single fused QKV projection, serial
    bl      launch_proj              // QKV combined
    mov     x0, #2; bl debug_sync_tag
    b       .qkv_done
.qkv_parallel:
    // FA: Q, K, V on dedicated streams, then barrier on cuda_stream before attention
    bl      launch_proj_fa_qkv_parallel  // launches Q/K/V on stream_q/k/v, records events,
                                         // waits on cuda_stream
    mov     x0, #3; bl debug_sync_tag
    // Re-hoist launch_params/param_ptrs: launch_proj_fa_qkv_parallel clobbers x11/x12.
    adrp    x11, launch_params
    add     x11, x11, :lo12:launch_params
    adrp    x12, param_ptrs
    add     x12, x12, :lo12:param_ptrs
.qkv_done:

    // ---- Attention or DeltaNet ----
    // x22 survives all bl calls above; no flag recomputation needed.
    cbz     x22, .do_full_attention

    // ---- DeltaNet path ----
    // Conv1d on Q, K, V (causal, 4-tap shift register per channel)
    // x19 = layer index, used by launch_conv1d for state offset
    bl      launch_conv1d            // conv1d on Q
    mov     x0, #4; bl debug_sync_tag
    bl      launch_conv1d            // conv1d on K
    mov     x0, #5; bl debug_sync_tag
    bl      launch_l2norm            // steps 10-13: L2-norm on K (dim=2048)
    mov     x0, #6; bl debug_sync_tag
    bl      launch_conv1d            // conv1d on V
    mov     x0, #7; bl debug_sync_tag
    bl      launch_activate          // steps 14-18: SiLU on Q
    mov     x0, #8; bl debug_sync_tag
    bl      launch_l2norm            // steps 19-22: L2-norm on Q (dim=2048)
    mov     x0, #9; bl debug_sync_tag
    bl      launch_recurrence
    mov     x0, #10; bl debug_sync_tag
    // Output gate (steps 29-34): z = W_z @ x, output *= sigmoid(z)
    // z projection uses ORIGINAL input x (still in act buffer, not overwritten by QKV/recur)
    bl      launch_z_proj            // step 29: z = W_z @ x -> z_buf
    mov     x0, #11; bl debug_sync_tag
    bl      launch_gate_sigmoid      // steps 30-34: output *= sigmoid(z)
    mov     x0, #12; bl debug_sync_tag
    // Post-attention RMSNorm on gated output, THEN output projection
    bl      launch_norm              // steps 35-40: RMSNorm on gated output
    mov     x0, #13; bl debug_sync_tag
    bl      launch_proj              // step 41: output projection
    mov     x0, #14; bl debug_sync_tag
    // Increment DN layer counter
    adrp    x0, dn_layer_counter
    add     x0, x0, :lo12:dn_layer_counter
    ldr     x1, [x0]
    add     x1, x1, #1
    str     x1, [x0]
    b       .post_proj

.do_full_attention:
    // ---- Full attention path (RoPE required) ----
    bl      launch_rope
    mov     x0, #15; bl debug_sync_tag
    bl      launch_attention
    mov     x0, #16; bl debug_sync_tag
    // Full attention: output projection, then residual norm
    bl      launch_proj              // output projection
    mov     x0, #17; bl debug_sync_tag
    bl      launch_norm              // residual add + norm
    mov     x0, #18; bl debug_sync_tag
    // Increment FA layer counter
    adrp    x0, fa_layer_counter
    add     x0, x0, :lo12:fa_layer_counter
    ldr     x1, [x0]
    add     x1, x1, #1
    str     x1, [x0]

.post_proj:

    // ---- Swap activation buffers ----
    eor     x27, x27, #1

    // ---- Pre-MLP RMSNorm ----
    bl      launch_norm
    mov     x0, #19; bl debug_sync_tag

    // ---- Gate projection ----
    bl      launch_proj
    mov     x0, #20; bl debug_sync_tag

    // ---- Up projection ----
    bl      launch_proj
    mov     x0, #21; bl debug_sync_tag

    // ---- SwiGLU: silu(gate) * up ----
    bl      launch_activate
    mov     x0, #22; bl debug_sync_tag

    // ---- Down projection ----
    bl      launch_proj
    mov     x0, #23; bl debug_sync_tag

    // ---- Residual add ----
    bl      launch_norm
    mov     x0, #24; bl debug_sync_tag

    // ---- Swap activation buffers ----
    eor     x27, x27, #1

    // Next layer
    add     x19, x19, #1
    b       .layer_loop

.layers_done:
    // ---- Final RMSNorm (uses global_weight_ptrs[GLOBAL_FINALNORM]) ----
    bl      launch_final_norm

    // ---- LM head projection (uses global_weight_ptrs[GLOBAL_LMHEAD]) ----
    // Output: cuda_logits_dev (GPU device buffer; DtoH copy follows)
    bl      launch_lm_head

    // ---- Synchronize GPU before DtoH copy ----
    bl      cuCtxSynchronize

    // ---- DEBUG: check cuCtxSync return ----
    cbnz    x0, .sync_err
    b       .sync_ok
.sync_err:
    // Print sync error
    stp     x19, x20, [sp, #-16]!
    mov     x19, x0
    adrp    x1, msg_err
    add     x1, x1, :lo12:msg_err
    mov     x2, msg_err_len
    bl      print_msg
    mov     x0, x19
    bl      print_int
    bl      print_newline
    ldp     x19, x20, [sp], #16
.sync_ok:

    // ---- DtoH copy: cuda_logits_dev -> logits_buf (CPU) ----
    // cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
    adrp    x0, logits_buf
    add     x0, x0, :lo12:logits_buf
    ldr     x0, [x0]               // x0 = CPU logits_buf base addr
    adrp    x1, cuda_logits_dev
    add     x1, x1, :lo12:cuda_logits_dev
    ldr     x1, [x1]               // x1 = GPU device ptr
    movz    x2, #0x000F, lsl #16   // 0x000F2800 = 993280
    movk    x2, #0x2800
    bl      cuMemcpyDtoH

    // ---- CPU argmax over logits ----
    adrp    x0, logits_buf
    add     x0, x0, :lo12:logits_buf
    ldr     x0, [x0]               // x0 = logits base pointer
    movz    x1, #0x0003, lsl #16   // 0x3C980 = 248320
    movk    x1, #0xC980            // x1 = VOCAB_SIZE

    // NEON argmax: 4 floats/cycle, 4 parallel max+idx streams
    // V0.4S = best vals (lanes 0-3), V1.4S = best idxs (as f32)
    // V2.4S = current idx vector, V3.4S = increment {4,4,4,4}
    // V4.4S = candidate vals, V5.4S = mask from fcmeq, V6.4S = candidate idxs

    // Seed: first 4 elements
    ld1     {v0.4s}, [x0], #16         // v0 = logits[0..3], advance ptr
    // idx vector: {0.0, 1.0, 2.0, 3.0}
    // build {0.0, 1.0, 2.0, 3.0} index seed in v1
    movi    v1.4s, #0                  // v1 = {0,0,0,0}
    fmov    s16, #1.0                  // s16 = 1.0
    fmov    s17, #2.0
    fmov    s18, #3.0
    ins     v1.s[1], v16.s[0]         // v1.s[1] = 1.0
    ins     v1.s[2], v17.s[0]         // v1.s[2] = 2.0
    ins     v1.s[3], v18.s[0]         // v1.s[3] = 3.0
    // stride vector: {4.0, 4.0, 4.0, 4.0}
    fmov    v3.4s, #4.0
    // current leading index for the 4-wide window
    fadd    v2.4s, v1.4s, v3.4s        // v2 = {4,5,6,7} — next window idxs

    mov     x4, #4                     // x4 = current scalar idx
    sub     x5, x1, #3                 // loop until x4 >= VOCAB_SIZE-3

.argmax_neon:
    cmp     x4, x5
    b.ge    .argmax_neon_done
    ld1     {v4.4s}, [x0], #16         // load next 4 logits
    fcmgt   v5.4s, v4.4s, v0.4s        // v5[i]=0xFFFFFFFF where v4 > v0
    bsl     v5.16b, v4.16b, v0.16b     // v0[i] = max(v4[i], v0[i])
    // same mask selects new idx
    fcmgt   v5.4s, v4.4s, v0.4s        // recompute (v0 already updated — redo)
    // fix: compute mask before updating v0
    // unroll the update:
    // a) mask = v4 > old_v0 => update v0 to max, v1 to v2 where mask
    // redo cleanly:
    fcmgt   v5.4s, v4.4s, v0.4s        // v5 = mask (v4 > v0)
    fmax    v0.4s, v0.4s, v4.4s        // v0 = elementwise max
    bsl     v5.16b, v2.16b, v1.16b     // v1 = idx where v4 won, else old idx
    mov     v1.16b, v5.16b             // commit new best idx vector
    fadd    v2.4s, v2.4s, v3.4s        // advance candidate idx window by 4
    add     x4, x4, #4
    b       .argmax_neon

.argmax_neon_done:
    // Horizontal reduce v0/v1: find max across 4 lanes
    // lane-by-lane comparison to find the winning lane
    dup     v6.4s, v0.s[0]
    dup     v7.4s, v0.s[1]
    dup     v8.4s, v0.s[2]
    dup     v9.4s, v0.s[3]
    // find max of all 4
    fmaxp   v10.4s, v0.4s, v0.4s       // pairwise max: [max(0,1), max(2,3), ...]
    fmaxp   v10.4s, v10.4s, v10.4s     // [max_all, max_all, ...]
    // which lane matches?
    fcmeq   v11.4s, v0.4s, v10.4s      // lane mask: 1 where v0 == global max
    // pick lowest set lane index from v1
    // extract each candidate, compare mask
    umov    w5, v11.s[0]
    cbnz    w5, .argmax_lane0
    umov    w5, v11.s[1]
    cbnz    w5, .argmax_lane1
    umov    w5, v11.s[2]
    cbnz    w5, .argmax_lane2
.argmax_lane3:
    mov     v12.16b, v1.16b
    ins     v13.s[0], v1.s[3]
    fcvtzs  x3, s13                    // idx from lane 3
    b       .argmax_scalar_tail
.argmax_lane2:
    ins     v13.s[0], v1.s[2]
    fcvtzs  x3, s13                    // idx from lane 2
    b       .argmax_scalar_tail
.argmax_lane1:
    ins     v13.s[0], v1.s[1]
    fcvtzs  x3, s13                    // idx from lane 1
    b       .argmax_scalar_tail
.argmax_lane0:
    fcvtzs  x3, s1                     // idx from lane 0 (s1 = v1.s[0])
    // fall through

    // Scalar tail for remaining < 4 elements
.argmax_scalar_tail:
    umov    w2, v10.s[0]               // current max bits

.argmax_tail_loop:
    cmp     x4, x1
    b.ge    .argmax_done
    ldr     w5, [x0], #4
    fmov    s0, w2
    fmov    s1, w5
    fcmp    s1, s0
    b.le    .argmax_tail_next
    mov     w2, w5
    mov     x3, x4
.argmax_tail_next:
    add     x4, x4, #1
    b       .argmax_tail_loop

.argmax_done:
    // x3 = argmax token ID
    // Store as last_token for next iteration's embed
    adrp    x0, last_token
    add     x0, x0, :lo12:last_token
    str     w3, [x0]

    // Print token ID (so we can verify output)
    mov     x0, x3
    bl      print_int
    // Print space separator
    sub     sp, sp, #16
    mov     w0, #' '
    strb    w0, [sp]
    mov     x0, #1
    mov     x1, sp
    mov     x2, #1
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #16

    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// launch_norm -- launch RMSNorm kernel
// ============================================================
// Uses norm_state counter to select which norm weight to load:
//   norm_state=0: input_layernorm (WS_INPUT_NORM)
//   norm_state=1: post_attention_layernorm (WS_POST_NORM, or WS_DN_NORM for DeltaNet)
//
// Kernel signature: norm(input, residual, weight, output, hidden_dim, epsilon)
//   param[0]: .u64 input_ptr   -- current activation buffer
//   param[1]: .u64 residual_ptr -- residual_buf
//   param[2]: .u64 weight_ptr  -- norm weight from weight table
//   param[3]: .u64 output_ptr  -- other activation buffer
//   param[4]: .u32 hidden_dim  -- HIDDEN_DIM=5120
//   param[5]: .f32 epsilon     -- 1e-6 = 0x358637BD
.align 4
launch_norm:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x11, x12, [sp, #-16]!  // preserve hoisted launch_params/param_ptrs

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_NORM]
    cbz     x0, .norm_skip

    // Capture hoisted launch_params into x1 before x11 is clobbered below.
    mov     x1, x11                 // x1 = launch_params (hoisted from decode_one_token)

    // ---- Read and increment norm_state ----
    adrp    x9, norm_state
    add     x9, x9, :lo12:norm_state
    ldr     x10, [x9]
    add     x11, x10, #1
    str     x11, [x9]
    // x10 = current norm_state (0 or 1)

    // ---- Select norm weight pointer ----
    // norm_state=0: WS_INPUT_NORM (slot 24)
    // norm_state=1: WS_POST_NORM (slot 25) for FA, WS_DN_NORM (slot 14) for DN
    cbnz    x10, .norm_post
    // norm_state=0: input_layernorm
    ldr     x11, [x23, #(WS_INPUT_NORM * 8)]
    b       .norm_weight_ready
.norm_post:
    // norm_state=1: post_attention_layernorm
    // x22 == 0 means full-attention, nonzero means DeltaNet
    cbz     x22, .norm_post_fa
    // DeltaNet: use WS_DN_NORM (slot 14)
    ldr     x11, [x23, #(WS_DN_NORM * 8)]
    b       .norm_weight_ready
.norm_post_fa:
    // Full attention: use WS_POST_NORM (slot 25)
    ldr     x11, [x23, #(WS_POST_NORM * 8)]
.norm_weight_ready:
    // x11 = norm weight pointer; x1 = launch_params (set above)

    // ---- Build launch_params (x1 already set) ----

    // param[0] = input_ptr (current activation buffer)
    // x27=0 -> act_buf_a is input; x27=1 -> act_buf_b is input
    cbz     x27, .norm_input_a
    str     x21, [x1, #0]           // act_buf_b
    b       .norm_input_done
.norm_input_a:
    str     x20, [x1, #0]           // act_buf_a
.norm_input_done:

    // param[1] = residual_ptr
    adrp    x9, residual_buf
    add     x9, x9, :lo12:residual_buf
    ldr     x9, [x9]
    str     x9, [x1, #8]

    // param[2] = weight_ptr
    str     x11, [x1, #16]

    // param[3] = output_ptr (other activation buffer)
    cbz     x27, .norm_output_b
    str     x20, [x1, #24]          // output to act_buf_a
    b       .norm_output_done
.norm_output_b:
    str     x21, [x1, #24]          // output to act_buf_b
.norm_output_done:

    // param[4] = hidden_dim (u32)
    movz    w9, #(HIDDEN_DIM & 0xFFFF)
    movk    w9, #(HIDDEN_DIM >> 16), lsl #16
    str     w9, [x1, #32]

    // param[5] = epsilon = 1e-6 as f32 = 0x358637BD
    movz    w9, #0x37BD
    movk    w9, #0x3586, lsl #16
    str     w9, [x1, #36]

    // ---- Build param_ptrs array ----
    mov     x2, x12                 // param_ptrs (hoisted from decode_one_token prologue)
    str     x1, [x2, #0]            // &param[0]
    add     x3, x1, #8
    str     x3, [x2, #8]            // &param[1]
    add     x3, x1, #16
    str     x3, [x2, #16]           // &param[2]
    add     x3, x1, #24
    str     x3, [x2, #24]           // &param[3]
    add     x3, x1, #32
    str     x3, [x2, #32]           // &param[4]
    add     x3, x1, #36
    str     x3, [x2, #40]           // &param[5]

    // ---- Launch kernel ----
    // x0 still holds KF_NORM CUfunction from the null-check load above
    mov     x1, #1                  // gridX
    mov     x2, #1                  // gridY
    mov     x3, #1                  // gridZ
    mov     x4, #256                // blockX
    mov     x5, #1                  // blockY
    mov     x6, #1                  // blockZ
    mov     x7, #128                // sharedMem

    adrp    x9, cuda_stream
    add     x9, x9, :lo12:cuda_stream
    ldr     x9, [x9]
    mov     x10, x12               // param_ptrs (hoisted from decode_one_token prologue)
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

.norm_skip:
    ldp     x11, x12, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// launch_proj -- launch projection (GEMV/GEMM) kernel
// ============================================================
// Uses proj_state counter to determine which weight slots and dimensions.
//
// DeltaNet layer sequence:
//   proj_state=0: QKV combined (DN_QKV slots, N=10240, K=5120)
//   proj_state=1: output proj  (DN_OUT slots, N=5120, K=6144)
//   proj_state=2: gate proj    (GATE slots, N=17408, K=5120)
//   proj_state=3: up proj      (UP slots, N=17408, K=5120)
//   proj_state=4: down proj    (DOWN slots, N=5120, K=17408)
//
// Full-attention layer sequence:
//   proj_state=0: Q proj (FA_Q slots, N=6144, K=5120)
//   proj_state=1: K proj (FA_K slots, N=1024, K=5120)
//   proj_state=2: V proj (FA_V slots, N=1024, K=5120)
//   proj_state=3: O proj (FA_O slots, N=5120, K=6144)
//   proj_state=4: gate proj (GATE slots, N=17408, K=5120)
//   proj_state=5: up proj (UP slots, N=17408, K=5120)
//   proj_state=6: down proj (DOWN slots, N=5120, K=17408)
//
// Kernel: gptq_gemv_safe(qweight, scales, input, output, N, K, K_SPLITS, k_packed_per_split)
//   param[0]: .u64 qweight_ptr
//   param[1]: .u64 scales_ptr
//   param[2]: .u64 input_ptr
//   param[3]: .u64 output_ptr
//   param[4]: .u32 param_N (output cols)
//   param[5]: .u32 param_K (input rows)
//   param[6]: .u32 param_K_SPLITS (1 for all projections)
//   param[7]: .u32 param_k_packed_per_split (K / 8 for 4-bit packing)
.align 4
launch_proj:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x11, x12, [sp, #-16]!  // preserve hoisted launch_params/param_ptrs
    stp     x19, x20, [sp, #-16]!  // save outer x19/x20 on our own frame

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_PROJ]
    cbz     x0, .proj_skip

    // ---- Read and increment proj_state ----
    adrp    x9, proj_state
    add     x9, x9, :lo12:proj_state
    ldr     x10, [x9]               // x10 = current proj_state
    add     x11, x10, #1
    str     x11, [x9]

    // ---- Determine weight slots, N, K based on layer type and proj_state ----
    // x22 == 0 means full-attention, nonzero means DeltaNet
    // We'll set: x11=qweight_ptr, x12=scales_ptr, x13=N, x14=K
    // Input and output pointers are computed separately.

    cbz     x22, .proj_fa_dispatch

    // ---- DeltaNet dispatch ----
    cmp     x10, #0
    b.eq    .proj_dn_qkv
    cmp     x10, #1
    b.eq    .proj_dn_out
    cmp     x10, #2
    b.eq    .proj_gate
    cmp     x10, #3
    b.eq    .proj_up
    b       .proj_down

.proj_dn_qkv:
    // QKV combined: N=10240, K=5120
    ldr     x11, [x23, #(WS_DN_QKV_QWEIGHT * 8)]
    ldr     x12, [x23, #(WS_DN_QKV_SCALES * 8)]
    ldr     x15, [x23, #(WS_DN_QKV_QZEROS * 8)]
    movz    x13, #10240              // N = 10240
    movz    x14, #5120               // K = 5120
    b       .proj_slots_ready

.proj_dn_out:
    // Output proj: N=5120, K=6144
    ldr     x11, [x23, #(WS_DN_OUT_QWEIGHT * 8)]
    ldr     x12, [x23, #(WS_DN_OUT_SCALES * 8)]
    ldr     x15, [x23, #(WS_DN_OUT_QZEROS * 8)]
    movz    x13, #5120               // N = 5120
    movz    x14, #6144               // K = 6144
    b       .proj_slots_ready

.proj_fa_dispatch:
    // ---- Full-attention dispatch ----
    cmp     x10, #0
    b.eq    .proj_fa_q
    cmp     x10, #1
    b.eq    .proj_fa_k
    cmp     x10, #2
    b.eq    .proj_fa_v
    cmp     x10, #3
    b.eq    .proj_fa_o
    cmp     x10, #4
    b.eq    .proj_gate
    cmp     x10, #5
    b.eq    .proj_up
    b       .proj_down

.proj_fa_q:
    // Q proj: N=6144, K=5120
    ldr     x11, [x23, #(WS_FA_Q_QWEIGHT * 8)]
    ldr     x12, [x23, #(WS_FA_Q_SCALES * 8)]
    ldr     x15, [x23, #(WS_FA_Q_QZEROS * 8)]
    movz    x13, #6144
    movz    x14, #5120
    b       .proj_slots_ready

.proj_fa_k:
    // K proj: N=1024, K=5120
    ldr     x11, [x23, #(WS_FA_K_QWEIGHT * 8)]
    ldr     x12, [x23, #(WS_FA_K_SCALES * 8)]
    ldr     x15, [x23, #(WS_FA_K_QZEROS * 8)]
    movz    x13, #1024
    movz    x14, #5120
    b       .proj_slots_ready

.proj_fa_v:
    // V proj: N=1024, K=5120
    ldr     x11, [x23, #(WS_FA_V_QWEIGHT * 8)]
    ldr     x12, [x23, #(WS_FA_V_SCALES * 8)]
    ldr     x15, [x23, #(WS_FA_V_QZEROS * 8)]
    movz    x13, #1024
    movz    x14, #5120
    b       .proj_slots_ready

.proj_fa_o:
    // O proj: N=5120, K=6144
    ldr     x11, [x23, #(WS_FA_O_QWEIGHT * 8)]
    ldr     x12, [x23, #(WS_FA_O_SCALES * 8)]
    ldr     x15, [x23, #(WS_FA_O_QZEROS * 8)]
    movz    x13, #5120
    movz    x14, #6144
    b       .proj_slots_ready

.proj_gate:
    // Gate proj: N=17408, K=5120
    ldr     x11, [x23, #(WS_GATE_QWEIGHT * 8)]
    ldr     x12, [x23, #(WS_GATE_SCALES * 8)]
    ldr     x15, [x23, #(WS_GATE_QZEROS * 8)]
    movz    x13, #17408
    movz    x14, #5120
    b       .proj_slots_ready

.proj_up:
    // Up proj: N=17408, K=5120
    ldr     x11, [x23, #(WS_UP_QWEIGHT * 8)]
    ldr     x12, [x23, #(WS_UP_SCALES * 8)]
    ldr     x15, [x23, #(WS_UP_QZEROS * 8)]
    movz    x13, #17408
    movz    x14, #5120
    b       .proj_slots_ready

.proj_down:
    // Down proj: N=5120, K=17408
    ldr     x11, [x23, #(WS_DOWN_QWEIGHT * 8)]
    ldr     x12, [x23, #(WS_DOWN_SCALES * 8)]
    ldr     x15, [x23, #(WS_DOWN_QZEROS * 8)]
    movz    x13, #5120
    movz    x14, #17408

.proj_slots_ready:
    // x11 = qweight_ptr, x12 = scales_ptr, x15 = qzeros_ptr
    // x13 = N (output dim), x14 = K (input dim)

    // ---- Build launch_params ----
    adrp    x1, launch_params
    add     x1, x1, :lo12:launch_params

    // param[0] = qweight_ptr
    str     x11, [x1, #0]

    // param[1] = scales_ptr
    str     x12, [x1, #8]

    // param[2] = input_ptr (current activation buffer)
    // For MLP down projection (last proj), input comes from mlp_gate_buf (activate output)
    // For all others, input is current act buffer (x27 selects a/b)
    // Determine input source based on proj_state:
    //   DN: state 4 (down) reads from mlp_gate_buf
    //   FA: state 6 (down) reads from mlp_gate_buf
    //   DN: state 2 (gate), state 3 (up) read from current act buffer
    //   FA: state 4 (gate), state 5 (up) read from current act buffer
    // Gate output goes to mlp_gate_buf, Up output goes to mlp_up_buf.

    // Check if this is the "down" projection (reads from mlp_gate_buf where activate wrote)
    cbz     x22, .proj_input_fa_check
    // DeltaNet: down = proj_state 4
    cmp     x10, #4
    b.eq    .proj_input_from_gate_buf
    b       .proj_input_normal
.proj_input_fa_check:
    // FA: down = proj_state 6
    cmp     x10, #6
    b.eq    .proj_input_from_gate_buf

.proj_input_normal:
    cbz     x27, .proj_input_a
    str     x21, [x1, #16]          // act_buf_b
    b       .proj_input_done
.proj_input_a:
    str     x20, [x1, #16]          // act_buf_a
    b       .proj_input_done
.proj_input_from_gate_buf:
    // Down projection reads from mlp_gate_buf (where activate wrote its output)
    adrp    x9, mlp_gate_buf
    add     x9, x9, :lo12:mlp_gate_buf
    ldr     x9, [x9]
    str     x9, [x1, #16]
.proj_input_done:

    // param[3] = output_ptr
    // Gate projection outputs to mlp_gate_buf
    // Up projection outputs to mlp_up_buf
    // Down projection outputs to current output buffer (other act buf)
    // All other projections output to the other act buffer
    cbz     x22, .proj_output_fa_check
    // DeltaNet: gate=2, up=3
    cmp     x10, #2
    b.eq    .proj_output_gate_buf
    cmp     x10, #3
    b.eq    .proj_output_up_buf
    b       .proj_output_other_act
.proj_output_fa_check:
    // FA: gate=4, up=5
    cmp     x10, #4
    b.eq    .proj_output_gate_buf
    cmp     x10, #5
    b.eq    .proj_output_up_buf

.proj_output_other_act:
    // Output to the other activation buffer
    cbz     x27, .proj_output_b
    str     x20, [x1, #24]          // output to act_buf_a
    b       .proj_output_done
.proj_output_b:
    str     x21, [x1, #24]          // output to act_buf_b
    b       .proj_output_done
.proj_output_gate_buf:
    adrp    x9, mlp_gate_buf
    add     x9, x9, :lo12:mlp_gate_buf
    ldr     x9, [x9]
    str     x9, [x1, #24]
    b       .proj_output_done
.proj_output_up_buf:
    adrp    x9, mlp_up_buf
    add     x9, x9, :lo12:mlp_up_buf
    ldr     x9, [x9]
    str     x9, [x1, #24]
.proj_output_done:

    // param[4] = N (u32)
    str     w13, [x1, #32]

    // param[5] = K (u32)
    str     w14, [x1, #36]

    // param[6] = K_SPLITS (u32) = 1
    mov     w9, #1
    str     w9, [x1, #40]

    // param[7] = k_packed_per_split (u32) = K / 8 (4-bit packing: 8 values per u32)
    lsr     w9, w14, #3
    str     w9, [x1, #44]

    // ---- Build param_ptrs array ----
    adrp    x2, param_ptrs
    add     x2, x2, :lo12:param_ptrs
    str     x1, [x2, #0]            // &param[0] = qweight
    add     x3, x1, #8
    str     x3, [x2, #8]            // &param[1] = scales
    add     x3, x1, #16
    str     x3, [x2, #16]           // &param[2] = input
    add     x3, x1, #24
    str     x3, [x2, #24]           // &param[3] = output
    add     x3, x1, #32
    str     x3, [x2, #32]           // &param[4] = N
    add     x3, x1, #36
    str     x3, [x2, #40]           // &param[5] = K
    add     x3, x1, #40
    str     x3, [x2, #48]           // &param[6] = K_SPLITS
    add     x3, x1, #44
    str     x3, [x2, #56]           // &param[7] = k_packed_per_split

    // ---- Compute gridX = ceil(N / 256) ----
    // gptq_gemv_safe: 64 threads/block, each handles 4 cols -> 256 cols/block
    add     w9, w13, #255
    lsr     w9, w9, #8              // gridX = (N + 255) / 256

    // ---- Launch kernel ----
    // x0 still holds KF_PROJ CUfunction from null-check load above
    mov     x1, x9                  // gridX = ceil(N/256)
    mov     x2, #1
    mov     x3, #1
    mov     x4, #64                 // blockX = 64 (kernel requires exactly 64 threads/block)
    mov     x5, #1
    mov     x6, #1
    lsl     x7, x14, #2             // sharedMem = K * 4 (input vector in smem)

    adrp    x9, cuda_stream
    add     x9, x9, :lo12:cuda_stream
    ldr     x9, [x9]
    adrp    x10, param_ptrs
    add     x10, x10, :lo12:param_ptrs
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

.proj_skip:
    ldp     x19, x20, [sp], #16
    ldp     x11, x12, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// launch_proj_fa_qkv_parallel
// ============================================================
// FA-only: launches Q, K, V projections on cuda_stream_q/k/v in parallel.
// All three read the same (read-only) input buffer; outputs are non-overlapping.
// After launching all three, records events and makes cuda_stream wait on all
// three events so the attention kernel sees a consistent QKV state.
//
// proj_state is read and advanced 3 times (0→Q, 1→K, 2→V).
// Uses launch_params_q/k/v and param_ptrs_q/k/v for independent param buffers.
//
// Register allocation (callee-saved by our own push/pop):
//   x19, x20 used internally; outer x19 (layer idx) preserved via stp/ldp.
// ============================================================
.align 4
launch_proj_fa_qkv_parallel:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x19, x20, [sp, #-16]!   // preserve outer layer-loop registers (act_buf_a in x20)
    stp     x11, x12, [sp, #-16]!   // preserve hoisted launch_params/param_ptrs
    // Snapshot act_buf_a to stack before x20 is reused; act_buf_b (x21), buf_sel (x27),
    // weight_ptrs (x23) are callee-saved through bl calls and not clobbered by us.
    sub     sp, sp, #16             // [sp+0]=act_buf_a_snap
    str     x20, [sp, #0]

    // ---- Check function handle is loaded ----
    adrp    x19, kern_table
    add     x19, x19, :lo12:kern_table
    ldr     x19, [x19, #KF_PROJ]
    cbz     x19, .pqkv_skip         // no kernel loaded, skip

    // ---- Macro-like helper: fill one param buffer and launch on a given stream ----
    // We inline Q, K, V cases. x19=KF_PROJ, x20=scratch, x9=scratch.
    // Input/output pointers derived from: snap=[sp+0]=act_buf_a, x21=act_buf_b, x27=selector.

    // ------ Q projection (proj_state=0: N=6144, K=5120) ------
    adrp    x9, proj_state
    add     x9, x9, :lo12:proj_state
    ldr     x0, [x9]               // should be 0
    add     x0, x0, #1
    str     x0, [x9]

    ldr     x11, [x23, #(WS_FA_Q_QWEIGHT * 8)]
    ldr     x12, [x23, #(WS_FA_Q_SCALES * 8)]
    ldr     x15, [x23, #(WS_FA_Q_QZEROS * 8)]

    adrp    x1, launch_params_q
    add     x1, x1, :lo12:launch_params_q
    str     x11, [x1, #0]           // param[0] = qweight
    str     x12, [x1, #8]           // param[1] = scales
    ldr     x20, [sp, #0]           // reload act_buf_a
    cbz     x27, .pqkv_q_in_a
    str     x21, [x1, #16]          // x27!=0: input = act_buf_b
    b       .pqkv_q_in_done
.pqkv_q_in_a:
    str     x20, [x1, #16]          // x27==0: input = act_buf_a
.pqkv_q_in_done:
    cbz     x27, .pqkv_q_out_b
    str     x20, [x1, #24]          // x27!=0 → output to act_buf_a
    b       .pqkv_q_out_done
.pqkv_q_out_b:
    str     x21, [x1, #24]          // x27==0 → output to act_buf_b
.pqkv_q_out_done:
    movz    w9, #6144
    str     w9, [x1, #32]           // param[4] = N=6144
    movz    w9, #5120
    str     w9, [x1, #36]           // param[5] = K=5120
    mov     w9, #1
    str     w9, [x1, #40]           // param[6] = K_SPLITS=1
    movz    w9, #640                // 5120/8=640
    str     w9, [x1, #44]           // param[7] = k_packed_per_split

    adrp    x2, param_ptrs_q
    add     x2, x2, :lo12:param_ptrs_q
    str     x1,  [x2, #0]
    add     x3, x1, #8;  str x3, [x2, #8]
    add     x3, x1, #16; str x3, [x2, #16]
    add     x3, x1, #24; str x3, [x2, #24]
    add     x3, x1, #32; str x3, [x2, #32]
    add     x3, x1, #36; str x3, [x2, #40]
    add     x3, x1, #40; str x3, [x2, #48]
    add     x3, x1, #44; str x3, [x2, #56]

    // gridX = ceil(6144/256) = 24 (gptq_gemv_safe: 64 threads, 4 cols each -> 256 cols/block)
    mov     x0, x19                 // KF_PROJ CUfunction
    mov     x1, #24                 // gridX
    mov     x2, #1; mov x3, #1
    mov     x4, #64; mov x5, #1; mov x6, #1; mov x7, #20480  // sharedMem = K*4 = 5120*4
    adrp    x9, cuda_stream_q
    add     x9, x9, :lo12:cuda_stream_q
    ldr     x9, [x9]
    adrp    x10, param_ptrs_q
    add     x10, x10, :lo12:param_ptrs_q
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

    // cuEventRecord(cuda_event_q, cuda_stream_q)
    adrp    x0, cuda_event_q
    add     x0, x0, :lo12:cuda_event_q
    ldr     x0, [x0]
    adrp    x1, cuda_stream_q
    add     x1, x1, :lo12:cuda_stream_q
    ldr     x1, [x1]
    bl      cuEventRecord

    // ------ K projection (proj_state=1: N=1024, K=5120) ------
    adrp    x20, proj_state
    add     x20, x20, :lo12:proj_state
    ldr     x0, [x20]               // should be 1
    add     x1, x0, #1
    str     x1, [x20]

    ldr     x11, [x23, #(WS_FA_K_QWEIGHT * 8)]
    ldr     x12, [x23, #(WS_FA_K_SCALES * 8)]
    ldr     x15, [x23, #(WS_FA_K_QZEROS * 8)]

    adrp    x1, launch_params_k
    add     x1, x1, :lo12:launch_params_k
    str     x11, [x1, #0]
    str     x12, [x1, #8]
    cbz     x27, .pqkv_k_in_a
    str     x21, [x1, #16]
    b       .pqkv_k_in_done
.pqkv_k_in_a:
    ldr     x9, [sp, #0]            // reload act_buf_a from stack snap (x20 = proj_state ptr)
    str     x9, [x1, #16]
.pqkv_k_in_done:
    // K output: other act buffer (serial path writes K after Q to same "other" buf region)
    // K output offset into other buf = Q_N * DTYPE_BYTES = 6144 * 2 = 12288 bytes past Q output
    cbz     x27, .pqkv_k_out_b
    ldr     x3, [sp, #0]            // reload act_buf_a
    add     x3, x3, #12288          // act_buf_a + Q_output_size
    b       .pqkv_k_out_done
.pqkv_k_out_b:
    add     x3, x21, #12288         // act_buf_b + Q_output_size
.pqkv_k_out_done:
    str     x3, [x1, #24]
    movz    w9, #1024
    str     w9, [x1, #32]           // N=1024
    movz    w9, #5120
    str     w9, [x1, #36]           // K=5120
    mov     w9, #1; str w9, [x1, #40]
    movz    w9, #640; str w9, [x1, #44]

    adrp    x2, param_ptrs_k
    add     x2, x2, :lo12:param_ptrs_k
    str     x1,  [x2, #0]
    add     x3, x1, #8;  str x3, [x2, #8]
    add     x3, x1, #16; str x3, [x2, #16]
    add     x3, x1, #24; str x3, [x2, #24]
    add     x3, x1, #32; str x3, [x2, #32]
    add     x3, x1, #36; str x3, [x2, #40]
    add     x3, x1, #40; str x3, [x2, #48]
    add     x3, x1, #44; str x3, [x2, #56]

    // gridX = ceil(1024/256) = 4 (gptq_gemv_safe: 64 threads, 4 cols each -> 256 cols/block)
    mov     x0, x19
    mov     x1, #4
    mov     x2, #1; mov x3, #1
    mov     x4, #64; mov x5, #1; mov x6, #1; mov x7, #20480  // sharedMem = K*4 = 5120*4
    adrp    x9, cuda_stream_k
    add     x9, x9, :lo12:cuda_stream_k
    ldr     x9, [x9]
    adrp    x10, param_ptrs_k
    add     x10, x10, :lo12:param_ptrs_k
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

    // cuEventRecord(cuda_event_k, cuda_stream_k)
    adrp    x0, cuda_event_k
    add     x0, x0, :lo12:cuda_event_k
    ldr     x0, [x0]
    adrp    x1, cuda_stream_k
    add     x1, x1, :lo12:cuda_stream_k
    ldr     x1, [x1]
    bl      cuEventRecord

    // ------ V projection (proj_state=2: N=1024, K=5120) ------
    adrp    x20, proj_state
    add     x20, x20, :lo12:proj_state
    ldr     x0, [x20]               // should be 2
    add     x1, x0, #1
    str     x1, [x20]

    ldr     x11, [x23, #(WS_FA_V_QWEIGHT * 8)]
    ldr     x12, [x23, #(WS_FA_V_SCALES * 8)]
    ldr     x15, [x23, #(WS_FA_V_QZEROS * 8)]

    adrp    x1, launch_params_v
    add     x1, x1, :lo12:launch_params_v
    str     x11, [x1, #0]
    str     x12, [x1, #8]
    cbz     x27, .pqkv_v_in_a
    str     x21, [x1, #16]
    b       .pqkv_v_in_done
.pqkv_v_in_a:
    ldr     x9, [sp, #0]            // reload act_buf_a from stack snap (x20 = proj_state ptr)
    str     x9, [x1, #16]
.pqkv_v_in_done:
    // V output: other buf + Q_N*2 + K_N*2 = 12288 + 2048 = 14336 bytes
    cbz     x27, .pqkv_v_out_b
    ldr     x3, [sp, #0]            // reload act_buf_a
    mov     x9, #14336
    add     x3, x3, x9              // act_buf_a + V_output_offset
    b       .pqkv_v_out_done
.pqkv_v_out_b:
    mov     x3, #14336
    add     x3, x21, x3
.pqkv_v_out_done:
    str     x3, [x1, #24]
    movz    w9, #1024
    str     w9, [x1, #32]           // N=1024
    movz    w9, #5120
    str     w9, [x1, #36]           // K=5120
    mov     w9, #1; str w9, [x1, #40]
    movz    w9, #640; str w9, [x1, #44]

    adrp    x2, param_ptrs_v
    add     x2, x2, :lo12:param_ptrs_v
    str     x1,  [x2, #0]
    add     x3, x1, #8;  str x3, [x2, #8]
    add     x3, x1, #16; str x3, [x2, #16]
    add     x3, x1, #24; str x3, [x2, #24]
    add     x3, x1, #32; str x3, [x2, #32]
    add     x3, x1, #36; str x3, [x2, #40]
    add     x3, x1, #40; str x3, [x2, #48]
    add     x3, x1, #44; str x3, [x2, #56]

    // gridX = ceil(1024/256) = 4 (gptq_gemv_safe: 64 threads, 4 cols each -> 256 cols/block)
    mov     x0, x19
    mov     x1, #4
    mov     x2, #1; mov x3, #1
    mov     x4, #64; mov x5, #1; mov x6, #1; mov x7, #20480  // sharedMem = K*4 = 5120*4
    adrp    x9, cuda_stream_v
    add     x9, x9, :lo12:cuda_stream_v
    ldr     x9, [x9]
    adrp    x10, param_ptrs_v
    add     x10, x10, :lo12:param_ptrs_v
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

    // cuEventRecord(cuda_event_v, cuda_stream_v)
    adrp    x0, cuda_event_v
    add     x0, x0, :lo12:cuda_event_v
    ldr     x0, [x0]
    adrp    x1, cuda_stream_v
    add     x1, x1, :lo12:cuda_stream_v
    ldr     x1, [x1]
    bl      cuEventRecord

    // ---- Barrier: make cuda_stream wait for Q, K, V events ----
    // cuStreamWaitEvent(cuda_stream, event, 0) for each of q/k/v
    adrp    x9, cuda_stream
    add     x9, x9, :lo12:cuda_stream
    ldr     x9, [x9]                // x9 = main stream handle (callee-saved not needed, tmp)

    adrp    x0, cuda_stream         // x0 = &cuda_stream
    add     x0, x0, :lo12:cuda_stream
    ldr     x0, [x0]               // x0 = main stream
    adrp    x1, cuda_event_q
    add     x1, x1, :lo12:cuda_event_q
    ldr     x1, [x1]
    mov     x2, #0
    bl      cuStreamWaitEvent

    adrp    x0, cuda_stream
    add     x0, x0, :lo12:cuda_stream
    ldr     x0, [x0]
    adrp    x1, cuda_event_k
    add     x1, x1, :lo12:cuda_event_k
    ldr     x1, [x1]
    mov     x2, #0
    bl      cuStreamWaitEvent

    adrp    x0, cuda_stream
    add     x0, x0, :lo12:cuda_stream
    ldr     x0, [x0]
    adrp    x1, cuda_event_v
    add     x1, x1, :lo12:cuda_event_v
    ldr     x1, [x1]
    mov     x2, #0
    bl      cuStreamWaitEvent
    // After this, any kernel launched on cuda_stream (e.g. RoPE, attention) is guaranteed
    // to execute only after Q/K/V projections complete on their respective streams.

.pqkv_skip:
    add     sp, sp, #16             // drop act_buf_a snap slot
    ldp     x11, x12, [sp], #16     // restore hoisted launch_params/param_ptrs
    ldp     x19, x20, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// launch_attention -- launch full attention decode kernel
// ============================================================
// Kernel signature: attention_score(q, k_cache, v_cache, output, seq_len, head_dim, num_heads)
//   param[0]: .u64 q_ptr       -- Q vector in "other" act buffer (after RoPE)
//   param[1]: .u64 k_cache_ptr -- kv_cache_ptr + fa_layer_idx * KV_PER_LAYER
//   param[2]: .u64 v_cache_ptr -- k_cache_ptr + KV_PER_LAYER / 2
//   param[3]: .u64 output_ptr  -- "other" act buffer (overwrite Q region)
//   param[4]: .u32 seq_len     -- x26 (current sequence position)
//   param[5]: .u32 head_dim    -- 256
//   param[6]: .u32 num_heads   -- 24
//
// fa_layer_idx is read from fa_layer_counter.
// KV_PER_LAYER = 268435456 (256MB), half = 134217728 (128MB)
//
// Grid: 24 blocks (one per head), Block: 256 threads, SharedMem: 2048
.align 4
launch_attention:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x11, x12, [sp, #-16]!  // preserve hoisted launch_params/param_ptrs

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_ATTN]
    cbz     x0, .attn_skip

    // ---- Build launch_params ----
    mov     x1, x11                 // launch_params (hoisted from decode_one_token)

    // Compute QKV buffer base (the "other" act buffer)
    cbz     x27, .attn_qkv_b
    mov     x9, x20                    // x27=1 -> other is act_buf_a
    b       .attn_qkv_done
.attn_qkv_b:
    mov     x9, x21                    // x27=0 -> other is act_buf_b
.attn_qkv_done:

    // param[0] = q_ptr: "other" act buffer + 0 (Q after RoPE)
    str     x9, [x1, #0]              // param[0] = q_ptr

    // param[1] = k_cache_ptr: kv_cache_ptr + fa_layer_counter * KV_PER_LAYER
    adrp    x2, kv_cache_ptr
    add     x2, x2, :lo12:kv_cache_ptr
    ldr     x2, [x2]                  // kv_cache base
    adrp    x3, fa_layer_counter
    add     x3, x3, :lo12:fa_layer_counter
    ldr     x3, [x3]                  // fa_layer_idx
    // KV_PER_LAYER = 268435456 = 0x10000000
    movz    x4, #0x1000, lsl #16      // 0x10000000
    mul     x4, x3, x4
    add     x2, x2, x4                // k_cache for this FA layer
    str     x2, [x1, #8]              // param[1] = k_cache_ptr

    // param[2] = v_cache_ptr: k_cache + KV_PER_LAYER / 2
    // KV_PER_LAYER / 2 = 134217728 = 0x08000000
    movz    x4, #0x0800, lsl #16      // 0x08000000
    add     x3, x2, x4
    str     x3, [x1, #16]             // param[2] = v_cache_ptr

    // param[3] = output_ptr: "other" act buffer (overwrite Q region)
    str     x9, [x1, #24]             // param[3] = output_ptr

    // param[4] = seq_len (u32) = x26 (current sequence position)
    str     w26, [x1, #32]            // param[4] = seq_len

    // param[5] = head_dim (u32) = 256
    mov     w3, #256
    str     w3, [x1, #36]             // param[5] = head_dim

    // param[6] = num_heads (u32) = 24
    mov     w3, #24
    str     w3, [x1, #40]             // param[6] = num_heads

    // ---- Build param_ptrs array ----
    mov     x2, x12                 // param_ptrs (hoisted from decode_one_token)
    str     x1, [x2, #0]              // &param[0] = q_ptr
    add     x3, x1, #8
    str     x3, [x2, #8]              // &param[1] = k_cache
    add     x3, x1, #16
    str     x3, [x2, #16]             // &param[2] = v_cache
    add     x3, x1, #24
    str     x3, [x2, #24]             // &param[3] = output
    add     x3, x1, #32
    str     x3, [x2, #32]             // &param[4] = seq_len
    add     x3, x1, #36
    str     x3, [x2, #40]             // &param[5] = head_dim
    add     x3, x1, #40
    str     x3, [x2, #48]             // &param[6] = num_heads

    // ---- Launch kernel ----
    // x0 still holds KF_ATTN CUfunction from null-check load above
    mov     x1, #24                 // gridX = num_heads = 24
    mov     x2, #1
    mov     x3, #1
    mov     x4, #256                // blockX = head_dim = 256
    mov     x5, #1
    mov     x6, #1
    mov     x7, #2048               // sharedMem for softmax scratch

    adrp    x9, cuda_stream
    add     x9, x9, :lo12:cuda_stream
    ldr     x9, [x9]
    mov     x10, x12               // param_ptrs (hoisted)
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

.attn_skip:
    ldp     x11, x12, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// launch_recurrence -- launch DeltaNet recurrence kernel
// ============================================================
// Kernel signature: recurrence(state, k, v, q, gate, beta, output,
//                               num_heads, key_dim, value_dim)
//   param[0]: .u64 state_ptr   -- dn_state_ptr + dn_layer_idx * DN_PER_LAYER
//   param[1]: .u64 k_ptr       -- QKV buffer + 8192 (K slice)
//   param[2]: .u64 v_ptr       -- QKV buffer + 16384 (V slice)
//   param[3]: .u64 q_ptr       -- QKV buffer + 0 (Q slice)
//   param[4]: .u64 gate_ptr    -- WS_DN_A_LOG (decay gate from model)
//   param[5]: .u64 beta_ptr    -- WS_DN_DT_BIAS (dt bias from model)
//   param[6]: .u64 output_ptr  -- "other" act buffer (recurrence output)
//   param[7]: .u32 num_heads   -- 16 (linear_num_key_heads)
//   param[8]: .u32 key_dim     -- 128 (linear_key_head_dim)
//   param[9]: .u32 value_dim   -- 128 (linear_value_head_dim)
//
// dn_layer_idx is read from dn_layer_counter (NOT the same as layer_idx).
// Grid: 16 blocks (one per head), Block: 128 threads, SharedMem: 32768
.align 4
launch_recurrence:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x11, x12, [sp, #-16]!  // preserve hoisted launch_params/param_ptrs

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_RECUR]
    cbz     x0, .recur_skip

    // ---- Build launch_params ----
    mov     x1, x11                 // launch_params (hoisted from decode_one_token)

    // param[0] = state_ptr: dn_state_ptr + dn_layer_counter * DN_PER_LAYER
    adrp    x9, dn_state_ptr
    add     x9, x9, :lo12:dn_state_ptr
    ldr     x9, [x9]                  // dn_state base
    adrp    x2, dn_layer_counter
    add     x2, x2, :lo12:dn_layer_counter
    ldr     x3, [x2]                  // dn_layer_idx
    movz    x4, #0x0030, lsl #16       // DN_PER_LAYER = 3145728 = 0x300000
    mul     x4, x3, x4
    add     x9, x9, x4
    str     x9, [x1, #0]              // param[0] = state_ptr

    // Compute QKV buffer base (the "other" act buffer where proj wrote)
    cbz     x27, .recur_qkv_b
    mov     x9, x20                    // x27=1 -> other is act_buf_a
    b       .recur_qkv_done
.recur_qkv_b:
    mov     x9, x21                    // x27=0 -> other is act_buf_b
.recur_qkv_done:
    // x9 = QKV buffer base

    // param[1] = k_ptr: QKV base + 8192
    add     x3, x9, #8192
    str     x3, [x1, #8]              // param[1] = k_ptr

    // param[2] = v_ptr: QKV base + 16384
    add     x3, x9, #16384
    str     x3, [x1, #16]             // param[2] = v_ptr

    // param[3] = q_ptr: QKV base + 0
    str     x9, [x1, #24]             // param[3] = q_ptr

    // param[4] = gate_ptr: WS_DN_A_LOG (decay gate)
    ldr     x3, [x23, #(WS_DN_A_LOG * 8)]
    str     x3, [x1, #32]             // param[4] = gate_ptr

    // param[5] = beta_ptr: WS_DN_DT_BIAS
    ldr     x3, [x23, #(WS_DN_DT_BIAS * 8)]
    str     x3, [x1, #40]             // param[5] = beta_ptr

    // param[6] = output_ptr: the "other" act buffer
    // NOTE: recurrence output is VALUE_DIM*4=24576 bytes, which exceeds
    // ACT_BUF_SIZE=20480. Buffer allocation needs to be enlarged to >=24576.
    // For now, we write to the same "other" buffer (QKV data is consumed).
    str     x9, [x1, #48]             // param[6] = output_ptr (reuse QKV buffer)

    // param[7] = num_heads (u32) = 16
    mov     w3, #16
    str     w3, [x1, #56]             // param[7] = num_heads

    // param[8] = key_dim (u32) = 128
    mov     w3, #128
    str     w3, [x1, #60]             // param[8] = key_dim

    // param[9] = value_dim (u32) = 128
    mov     w3, #128
    str     w3, [x1, #64]             // param[9] = value_dim

    // ---- Build param_ptrs array ----
    mov     x2, x12                 // param_ptrs (hoisted from decode_one_token)
    str     x1, [x2, #0]              // &param[0] = state
    add     x3, x1, #8
    str     x3, [x2, #8]              // &param[1] = k
    add     x3, x1, #16
    str     x3, [x2, #16]             // &param[2] = v
    add     x3, x1, #24
    str     x3, [x2, #24]             // &param[3] = q
    add     x3, x1, #32
    str     x3, [x2, #32]             // &param[4] = gate
    add     x3, x1, #40
    str     x3, [x2, #40]             // &param[5] = beta
    add     x3, x1, #48
    str     x3, [x2, #48]             // &param[6] = output
    add     x3, x1, #56
    str     x3, [x2, #56]             // &param[7] = num_heads
    add     x3, x1, #60
    str     x3, [x2, #64]             // &param[8] = key_dim
    add     x3, x1, #64
    str     x3, [x2, #72]             // &param[9] = value_dim

    // ---- Launch kernel ----
    // x0 still holds KF_RECUR CUfunction from null-check load above
    mov     x1, #16                 // gridX = 16 (linear_num_key_heads)
    mov     x2, #1
    mov     x3, #1
    mov     x4, #128                // blockX = 128 (key_dim)
    mov     x5, #1
    mov     x6, #1
    mov     x7, #32768              // sharedMem = 128 * 128 * 2 = 32768

    adrp    x9, cuda_stream
    add     x9, x9, :lo12:cuda_stream
    ldr     x9, [x9]
    mov     x10, x12               // param_ptrs (hoisted)
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

.recur_skip:
    ldp     x11, x12, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// launch_activate -- launch SwiGLU activation: silu(gate) * up
// ============================================================
// Kernel: activate(gate_ptr, up_ptr, output_ptr, size)
//   param[0]: .u64 gate_ptr   -- mlp_gate_buf (gate projection output)
//   param[1]: .u64 up_ptr     -- mlp_up_buf (up projection output)
//   param[2]: .u64 output_ptr -- mlp_gate_buf (reuse for down proj input)
//   param[3]: .u32 size       -- INTERMEDIATE_DIM = 17408
.align 4
launch_activate:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x11, x12, [sp, #-16]!  // preserve hoisted launch_params/param_ptrs

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_ACTIVATE]
    cbz     x0, .act_skip

    // ---- Build launch_params ----
    mov     x1, x11                 // launch_params (hoisted from decode_one_token)

    // param[0] = gate_ptr (mlp_gate_buf)
    adrp    x9, mlp_gate_buf
    add     x9, x9, :lo12:mlp_gate_buf
    ldr     x9, [x9]
    str     x9, [x1, #0]

    // param[1] = up_ptr (mlp_up_buf)
    adrp    x9, mlp_up_buf
    add     x9, x9, :lo12:mlp_up_buf
    ldr     x9, [x9]
    str     x9, [x1, #8]

    // param[2] = output_ptr (write back to mlp_gate_buf for down proj to read)
    adrp    x9, mlp_gate_buf
    add     x9, x9, :lo12:mlp_gate_buf
    ldr     x9, [x9]
    str     x9, [x1, #16]

    // param[3] = size (u32) = INTERMEDIATE_DIM = 17408
    movz    w9, #17408
    str     w9, [x1, #24]

    // ---- Build param_ptrs array ----
    mov     x2, x12                 // param_ptrs (hoisted from decode_one_token)
    str     x1, [x2, #0]            // &param[0] = gate_ptr
    add     x3, x1, #8
    str     x3, [x2, #8]            // &param[1] = up_ptr
    add     x3, x1, #16
    str     x3, [x2, #16]           // &param[2] = output_ptr
    add     x3, x1, #24
    str     x3, [x2, #24]           // &param[3] = size

    // ---- Launch kernel ----
    // x0 still holds KF_ACTIVATE CUfunction from null-check load above
    // gridX = ceil(17408 / 256) = 68
    mov     x1, #68
    mov     x2, #1
    mov     x3, #1
    mov     x4, #256
    mov     x5, #1
    mov     x6, #1
    mov     x7, #0

    adrp    x9, cuda_stream
    add     x9, x9, :lo12:cuda_stream
    ldr     x9, [x9]
    mov     x10, x12               // param_ptrs (hoisted)
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

.act_skip:
    ldp     x11, x12, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// launch_rope -- launch RoPE kernel (full-attention layers only)
// ============================================================
// Applies rotary position embeddings to Q and K vectors.
// Called after QKV projection in full-attention layers.
//
// FA QKV layout in "other" act buffer (after separate Q/K/V projections):
//   Q: offset 0,     size 24*256*4 = 24576 bytes
//   K: offset 24576, size 4*256*4  = 4096 bytes
//   V: offset 28672, size 4*256*4  = 4096 bytes
//
// Kernel signature: rotate(q_ptr, k_ptr, cos_ptr, sin_ptr, seq_pos, head_dim, num_heads)
//   param[0]: .u64 q_ptr      -- Q in "other" act buffer
//   param[1]: .u64 k_ptr      -- K in "other" act buffer + 24576
//   param[2]: .u64 cos_ptr    -- precomputed cos table
//   param[3]: .u64 sin_ptr    -- precomputed sin table
//   param[4]: .u32 seq_pos    -- x26 (current sequence position)
//   param[5]: .u32 head_dim   -- 256
//   param[6]: .u32 num_heads  -- 24
//
// Grid: ceil(24 * 256 / 256) = 24 blocks of 256 threads
.align 4
launch_rope:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x11, x12, [sp, #-16]!  // preserve hoisted launch_params/param_ptrs

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_ROPE]
    cbz     x0, .rope_skip

    // ---- Build launch_params ----
    mov     x1, x11                 // launch_params (hoisted from decode_one_token)

    // Compute QKV buffer base (the "other" act buffer)
    cbz     x27, .rope_qkv_b
    mov     x9, x20                    // x27=1 -> other is act_buf_a
    b       .rope_qkv_done
.rope_qkv_b:
    mov     x9, x21                    // x27=0 -> other is act_buf_b
.rope_qkv_done:

    // param[0] = q_ptr: QKV base + 0
    str     x9, [x1, #0]              // param[0] = q_ptr

    // param[1] = k_ptr: QKV base + 24576
    add     x3, x9, #24576
    str     x3, [x1, #8]              // param[1] = k_ptr

    // param[2] = cos_ptr: precomputed cos table
    adrp    x3, rope_cos_table
    add     x3, x3, :lo12:rope_cos_table
    ldr     x3, [x3]
    str     x3, [x1, #16]             // param[2] = cos_ptr

    // param[3] = sin_ptr: precomputed sin table
    adrp    x3, rope_sin_table
    add     x3, x3, :lo12:rope_sin_table
    ldr     x3, [x3]
    str     x3, [x1, #24]             // param[3] = sin_ptr

    // param[4] = seq_pos (u32) = x26
    str     w26, [x1, #32]            // param[4] = seq_pos

    // param[5] = head_dim (u32) = 256
    mov     w3, #256
    str     w3, [x1, #36]             // param[5] = head_dim

    // param[6] = num_heads (u32) = 24
    mov     w3, #24
    str     w3, [x1, #40]             // param[6] = num_heads

    // ---- Build param_ptrs array ----
    mov     x2, x12                 // param_ptrs (hoisted from decode_one_token)
    str     x1, [x2, #0]              // &param[0] = q_ptr
    add     x3, x1, #8
    str     x3, [x2, #8]              // &param[1] = k_ptr
    add     x3, x1, #16
    str     x3, [x2, #16]             // &param[2] = cos_ptr
    add     x3, x1, #24
    str     x3, [x2, #24]             // &param[3] = sin_ptr
    add     x3, x1, #32
    str     x3, [x2, #32]             // &param[4] = seq_pos
    add     x3, x1, #36
    str     x3, [x2, #40]             // &param[5] = head_dim
    add     x3, x1, #40
    str     x3, [x2, #48]             // &param[6] = num_heads

    // ---- Launch kernel ----
    // x0 still holds KF_ROPE CUfunction from null-check load above
    mov     x1, #24                 // gridX = num_heads = 24
    mov     x2, #1
    mov     x3, #1
    mov     x4, #256                // blockX = head_dim = 256
    mov     x5, #1
    mov     x6, #1
    mov     x7, #0                  // no shared mem

    adrp    x9, cuda_stream
    add     x9, x9, :lo12:cuda_stream
    ldr     x9, [x9]
    mov     x10, x12               // param_ptrs (hoisted)
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

.rope_skip:
    ldp     x11, x12, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// launch_l2norm -- launch L2-norm kernel (call 0=K, call 1=Q)
// ============================================================
// L2-norm: x / sqrt(sum(x^2)), in-place
// Kernel signature: l2norm(x_ptr, y_ptr, hidden_dim)
//   param[0]: .u64 x_ptr         -- input pointer (QKV buffer + slice offset)
//   param[1]: .u64 y_ptr         -- output pointer (same as input, in-place)
//   param[2]: .u32 hidden_dim    -- 2048 (16 heads * 128 dim)
//
// Call 0 (K): offset = 8192 bytes (after Q slice)
// Call 1 (Q): offset = 0 bytes (Q slice)
// Both operate on 16*128 = 2048 elements.
//
// Grid: ceil(2048/256) = 8 blocks of 256 threads
// SharedMem: 1024 bytes for reduction
.align 4
launch_l2norm:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x11, x12, [sp, #-16]!  // preserve hoisted launch_params/param_ptrs

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_L2NORM]
    cbz     x0, .l2norm_skip

    // Capture hoisted launch_params into x1 before x11 is clobbered below.
    mov     x1, x11                 // x1 = launch_params (hoisted from decode_one_token)

    // ---- Read and increment l2norm_call_counter ----
    adrp    x9, l2norm_call_counter
    add     x9, x9, :lo12:l2norm_call_counter
    ldr     x10, [x9]               // x10 = call index (0=K, 1=Q)
    add     x11, x10, #1
    str     x11, [x9]

    // ---- Determine data offset ----
    // Call 0 (K): offset = 8192
    // Call 1 (Q): offset = 0
    cbnz    x10, .l2norm_q
    mov     x11, #8192               // K slice offset
    b       .l2norm_offset_ready
.l2norm_q:
    mov     x11, #0                  // Q slice offset
.l2norm_offset_ready:

    // ---- Build launch_params (x1 already set above) ----

    // param[0] = x_ptr: QKV output buffer + slice offset
    // QKV output is in the "other" act buffer
    cbz     x27, .l2norm_input_b
    add     x9, x20, x11              // act_buf_a + offset (other when x27=1)
    b       .l2norm_input_done
.l2norm_input_b:
    add     x9, x21, x11              // act_buf_b + offset (other when x27=0)
.l2norm_input_done:
    str     x9, [x1, #0]              // param[0] = x_ptr

    // param[1] = y_ptr (same as x_ptr, in-place)
    str     x9, [x1, #8]              // param[1] = y_ptr

    // param[2] = hidden_dim (u32) = 2048 (16 heads * 128 dim)
    mov     w9, #2048
    str     w9, [x1, #16]             // param[2] = hidden_dim

    // ---- Build param_ptrs array ----
    mov     x2, x12                 // param_ptrs (hoisted from decode_one_token)
    str     x1, [x2, #0]              // &param[0] = x_ptr
    add     x3, x1, #8
    str     x3, [x2, #8]              // &param[1] = y_ptr
    add     x3, x1, #16
    str     x3, [x2, #16]             // &param[2] = hidden_dim

    // ---- Launch kernel ----
    // x0 still holds KF_L2NORM CUfunction from null-check load above
    mov     x1, #1                  // gridX = 1 (single block reduction)
    mov     x2, #1                  // gridY
    mov     x3, #1                  // gridZ
    mov     x4, #256                // blockX
    mov     x5, #1                  // blockY
    mov     x6, #1                  // blockZ
    mov     x7, #1024               // sharedMem

    adrp    x9, cuda_stream
    add     x9, x9, :lo12:cuda_stream
    ldr     x9, [x9]
    mov     x10, x12               // param_ptrs (hoisted)
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

.l2norm_skip:
    ldp     x11, x12, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// launch_gate_sigmoid -- fused sigmoid gate: output *= sigmoid(z)
// ============================================================
// Used for DeltaNet output gating (STEPS 29-34).
// The z projection (W_z @ x) has written z to z_buf.
// The recurrence output is in the "other" act buffer.
// This kernel applies: output[i] *= sigmoid(z[i])
// Kernel signature: gate_sigmoid(output_ptr, z_ptr, n)
//   param[0]: .u64 output_ptr -- recurrence output ("other" act buffer), modified in-place
//   param[1]: .u64 z_ptr      -- z_buf (from z projection)
//   param[2]: .u32 n          -- VALUE_DIM = 6144
.align 4
launch_gate_sigmoid:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x11, x12, [sp, #-16]!  // preserve hoisted launch_params/param_ptrs

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_GATE_SIG]
    cbz     x0, .gate_sig_skip

    // ---- Build launch_params ----
    mov     x1, x11                 // launch_params (hoisted from decode_one_token)

    // param[0] = output_ptr: recurrence output in "other" act buffer
    cbz     x27, .gate_sig_output_b
    str     x20, [x1, #0]             // x27=1 -> other is act_buf_a
    b       .gate_sig_output_done
.gate_sig_output_b:
    str     x21, [x1, #0]             // x27=0 -> other is act_buf_b
.gate_sig_output_done:

    // param[1] = z_ptr: z_buf
    adrp    x9, z_buf
    add     x9, x9, :lo12:z_buf
    ldr     x9, [x9]
    str     x9, [x1, #8]              // param[1] = z_ptr

    // param[2] = n (u32) = VALUE_DIM = 6144
    movz    w9, #6144
    str     w9, [x1, #16]             // param[2] = n

    // ---- Build param_ptrs array ----
    mov     x2, x12                 // param_ptrs (hoisted from decode_one_token)
    str     x1, [x2, #0]              // &param[0] = output_ptr
    add     x3, x1, #8
    str     x3, [x2, #8]              // &param[1] = z_ptr
    add     x3, x1, #16
    str     x3, [x2, #16]             // &param[2] = n

    // ---- Launch kernel ----
    // x0 still holds KF_GATE_SIG CUfunction from null-check load above
    // gridX = ceil(6144 / 256) = 24
    mov     x1, #24                 // gridX
    mov     x2, #1                  // gridY
    mov     x3, #1                  // gridZ
    mov     x4, #256                // blockX
    mov     x5, #1                  // blockY
    mov     x6, #1                  // blockZ
    mov     x7, #0                  // no shared mem

    adrp    x9, cuda_stream
    add     x9, x9, :lo12:cuda_stream
    ldr     x9, [x9]
    mov     x10, x12               // param_ptrs (hoisted)
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

.gate_sig_skip:
    ldp     x11, x12, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// launch_z_proj -- launch z projection GEMV for output gate
// ============================================================
// Projects original input x -> z (value_dim=6144) using in_proj_z weights.
// Uses the same GEMV kernel as other projections but with different dimensions:
//   output_dim = 6144 (value_dim), input_dim = 5120 (hidden_dim)
//   gridX = ceil(6144 / 128) = 48
.align 4
launch_z_proj:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x11, x12, [sp, #-16]!  // preserve hoisted launch_params/param_ptrs

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_PROJ]     // same GEMV kernel
    cbz     x0, .z_proj_skip

    // ---- Build launch_params ----
    // Uses gptq_gemv_safe kernel with Z weight slots.
    // Projects ORIGINAL input x (current input buffer) -> z_buf
    // N = VALUE_DIM = 6144, K = HIDDEN_DIM = 5120
    mov     x1, x11                 // launch_params (hoisted from decode_one_token)

    // param[0] = qweight_ptr (WS_DN_Z_QWEIGHT)
    ldr     x9, [x23, #(WS_DN_Z_QWEIGHT * 8)]
    str     x9, [x1, #0]

    // param[1] = scales_ptr (WS_DN_Z_SCALES)
    ldr     x9, [x23, #(WS_DN_Z_SCALES * 8)]
    str     x9, [x1, #8]

    // param[2] = input_ptr (current input buffer -- the original x before QKV)
    // The original input is in the current input buffer (x27 selects)
    cbz     x27, .z_proj_input_a
    str     x21, [x1, #16]          // x27=1 -> input is act_buf_b
    b       .z_proj_input_done
.z_proj_input_a:
    str     x20, [x1, #16]          // x27=0 -> input is act_buf_a
.z_proj_input_done:

    // param[3] = output_ptr (z_buf)
    adrp    x9, z_buf
    add     x9, x9, :lo12:z_buf
    ldr     x9, [x9]
    str     x9, [x1, #24]

    // param[4] = N (u32) = VALUE_DIM = 6144
    movz    w9, #6144
    str     w9, [x1, #32]

    // param[5] = K (u32) = HIDDEN_DIM = 5120
    movz    w9, #5120
    str     w9, [x1, #36]

    // param[6] = K_SPLITS (u32) = 1
    mov     w9, #1
    str     w9, [x1, #40]

    // param[7] = k_packed_per_split (u32) = K / 8 = 640
    mov     w9, #640
    str     w9, [x1, #44]

    // ---- Build param_ptrs array ----
    mov     x2, x12                 // param_ptrs (hoisted from decode_one_token)
    str     x1, [x2, #0]            // &param[0] = qweight
    add     x3, x1, #8
    str     x3, [x2, #8]            // &param[1] = scales
    add     x3, x1, #16
    str     x3, [x2, #16]           // &param[2] = input
    add     x3, x1, #24
    str     x3, [x2, #24]           // &param[3] = output
    add     x3, x1, #32
    str     x3, [x2, #32]           // &param[4] = N
    add     x3, x1, #36
    str     x3, [x2, #40]           // &param[5] = K
    add     x3, x1, #40
    str     x3, [x2, #48]           // &param[6] = K_SPLITS
    add     x3, x1, #44
    str     x3, [x2, #56]           // &param[7] = k_packed_per_split

    // ---- Launch kernel ----
    // x0 still holds KF_PROJ CUfunction from null-check load above
    // gridX = ceil(6144 / 256) = 24 (gptq_gemv_safe: 64 threads, 4 cols -> 256/block)
    mov     x1, #24                 // gridX
    mov     x2, #1                  // gridY
    mov     x3, #1                  // gridZ
    mov     x4, #64                 // blockX = 64 (kernel requires exactly 64 threads/block)
    mov     x5, #1                  // blockY
    mov     x6, #1                  // blockZ
    mov     x7, #0                  // no shared mem

    adrp    x9, cuda_stream
    add     x9, x9, :lo12:cuda_stream
    ldr     x9, [x9]
    mov     x10, x12               // param_ptrs (hoisted)
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

.z_proj_skip:
    ldp     x11, x12, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// launch_conv1d -- launch causal conv1d kernel (one Q/K/V slice)
// ============================================================
// Called 3 times per DeltaNet layer: call 0=Q, 1=K, 2=V.
// Kernel signature: conv1d_infer(input, weight, hist, output, num_heads, channels)
//   param[0]: .u64 input_ptr    -- QKV output buffer + slice offset
//   param[1]: .u64 weight_ptr   -- conv1d weight from model (WS_DN_CONV1D)
//   param[2]: .u64 hist_ptr     -- conv1d_state + layer*CONV1D_PER_LAYER + hist_offset
//   param[3]: .u64 output_ptr   -- same as input (in-place)
//   param[4]: .u32 num_heads    -- Q/K: 16, V: 48
//   param[5]: .u32 channels     -- 128 (per-head channel dim)
//
// DeltaNet QKV layout in output buffer (after combined QKV projection):
//   Q: offset 0,     size 16*128*4 = 8192,  hist_off=0
//   K: offset 8192,  size 16*128*4 = 8192,  hist_off=32768
//   V: offset 16384, size 48*128*4 = 24576, hist_off=65536
//
// History offsets: Q = 16*128 channels * 4 taps * 4 bytes = 32768
//                  K = 16*128 channels * 4 taps * 4 bytes = 32768
//                  V = 48*128 channels * 4 taps * 4 bytes = 98304
//
// Grid: ceil(total_channels / 256), where total_channels = num_heads * channels
.align 4
launch_conv1d:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x11, x12, [sp, #-16]!  // preserve hoisted launch_params/param_ptrs

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_CONV1D]
    cbz     x0, .conv1d_skip

    // Capture hoisted launch_params into x1 before x11/x12 are clobbered below.
    mov     x1, x11                 // x1 = launch_params (hoisted from decode_one_token)

    // ---- Read and increment conv1d_call_counter ----
    adrp    x9, conv1d_call_counter
    add     x9, x9, :lo12:conv1d_call_counter
    ldr     x10, [x9]               // x10 = call index (0=Q, 1=K, 2=V)
    add     x11, x10, #1
    str     x11, [x9]

    // ---- Determine slice offset, hist offset, num_heads based on call ----
    // x11 = data_offset, x12 = hist_offset, x13 = num_heads
    cmp     x10, #0
    b.eq    .conv1d_q
    cmp     x10, #1
    b.eq    .conv1d_k
    b       .conv1d_v

.conv1d_q:
    // Q: data_offset=0, hist_offset=0, num_heads=16
    mov     x11, #0
    mov     x12, #0
    mov     x13, #16
    b       .conv1d_slice_ready

.conv1d_k:
    // K: data_offset=8192, hist_offset=32768, num_heads=16
    mov     x11, #8192
    mov     x12, #32768
    mov     x13, #16
    b       .conv1d_slice_ready

.conv1d_v:
    // V: data_offset=16384, hist_offset=65536, num_heads=48
    mov     x11, #16384
    movz    x12, #0x0001, lsl #16
    // 65536 = 0x10000
    mov     x13, #48

.conv1d_slice_ready:
    // x11 = data byte offset into QKV output buffer
    // x12 = history byte offset within this layer's conv1d state
    // x13 = num_heads for this slice

    // ---- Build launch_params (x1 already set above) ----

    // param[0] = input_ptr: QKV output buffer + data_offset
    // QKV output is in the "other" act buffer (proj wrote to other)
    cbz     x27, .conv1d_input_b
    // x27=1 -> input is act_buf_b, so output (other) is act_buf_a
    add     x9, x20, x11              // act_buf_a + offset
    b       .conv1d_input_done
.conv1d_input_b:
    // x27=0 -> input is act_buf_a, so output (other) is act_buf_b
    add     x9, x21, x11              // act_buf_b + offset
.conv1d_input_done:
    str     x9, [x1, #0]              // param[0] = input_ptr

    // param[1] = weight_ptr from model weights (WS_DN_CONV1D)
    ldr     x9, [x23, #(WS_DN_CONV1D * 8)]
    str     x9, [x1, #8]              // param[1] = weight_ptr

    // param[2] = hist_ptr: conv1d_state + layer_idx * CONV1D_PER_LAYER + hist_offset
    adrp    x2, conv1d_state
    add     x2, x2, :lo12:conv1d_state
    ldr     x3, [x2]                  // conv1d_state base
    movz    x4, #0x0002, lsl #16
    movk    x4, #0x8000               // CONV1D_PER_LAYER = 163840 = 0x28000
    mul     x4, x19, x4               // layer_idx * per_layer
    add     x3, x3, x4                // base for this layer
    add     x3, x3, x12               // + hist_offset for this slice
    str     x3, [x1, #16]             // param[2] = hist_ptr

    // param[3] = output_ptr (same as input for in-place)
    ldr     x9, [x1, #0]              // reload input_ptr
    str     x9, [x1, #24]             // param[3] = output_ptr

    // param[4] = num_heads (u32)
    str     w13, [x1, #32]            // param[4] = num_heads

    // param[5] = channels (u32) = 128
    mov     w9, #128
    str     w9, [x1, #36]             // param[5] = channels

    // ---- Build param_ptrs array ----
    adrp    x2, param_ptrs
    add     x2, x2, :lo12:param_ptrs
    str     x1, [x2, #0]              // &param[0] = input
    add     x3, x1, #8
    str     x3, [x2, #8]              // &param[1] = weight
    add     x3, x1, #16
    str     x3, [x2, #16]             // &param[2] = hist
    add     x3, x1, #24
    str     x3, [x2, #24]             // &param[3] = output
    add     x3, x1, #32
    str     x3, [x2, #32]             // &param[4] = num_heads
    add     x3, x1, #36
    str     x3, [x2, #40]             // &param[5] = channels

    // ---- Compute gridX = ceil(num_heads * 128 / 256) ----
    // Q/K: 16*128=2048, ceil(2048/256)=8
    // V: 48*128=6144, ceil(6144/256)=24
    lsl     w9, w13, #7               // num_heads * 128
    add     w9, w9, #255
    lsr     w9, w9, #8                // ceil(total / 256)

    // ---- Launch kernel ----
    // x0 still holds KF_CONV1D CUfunction from null-check load above
    mov     x1, x9                     // gridDimX
    mov     x2, #1                     // gridDimY
    mov     x3, #1                     // gridDimZ
    mov     x4, #256                   // blockDimX
    mov     x5, #1                     // blockDimY
    mov     x6, #1                     // blockDimZ
    mov     x7, #0                     // sharedMemBytes

    adrp    x9, cuda_stream
    add     x9, x9, :lo12:cuda_stream
    ldr     x9, [x9]
    adrp    x10, param_ptrs
    add     x10, x10, :lo12:param_ptrs
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

.conv1d_skip:
    ldp     x11, x12, [sp], #16
    ldp     x29, x30, [sp], #16
    ret


// ============================================================
// launch_final_norm -- RMSNorm using global final_norm weight
// ============================================================
// Same kernel as launch_norm but uses global_weight_ptrs[GLOBAL_FINALNORM]
// instead of per-layer weight. Input is current act buffer, output to same.
.align 4
launch_final_norm:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x11, x12, [sp, #-16]!  // preserve hoisted launch_params/param_ptrs

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_NORM]
    cbz     x0, .fn_skip

    mov     x1, x11                 // launch_params (hoisted from decode_one_token)

    // param[0] = input (current act buf)
    cbz     x27, .fn_input_a
    str     x21, [x1, #0]
    b       .fn_input_done
.fn_input_a:
    str     x20, [x1, #0]
.fn_input_done:

    // param[1] = residual_buf
    adrp    x9, residual_buf
    add     x9, x9, :lo12:residual_buf
    ldr     x9, [x9]
    str     x9, [x1, #8]

    // param[2] = weight from global_weight_ptrs[GLOBAL_FINALNORM]
    adrp    x9, global_weight_ptrs
    add     x9, x9, :lo12:global_weight_ptrs
    ldr     x9, [x9, #GLOBAL_FINALNORM]
    str     x9, [x1, #16]

    // param[3] = output (same as input — in-place for final norm)
    cbz     x27, .fn_output_a
    str     x21, [x1, #24]
    b       .fn_output_done
.fn_output_a:
    str     x20, [x1, #24]
.fn_output_done:

    // param[4] = hidden_dim = 5120
    movz    w9, #5120
    str     w9, [x1, #32]

    // param[5] = epsilon = 1e-6 (0x358637BD as f32 bits)
    movz    w9, #0x3586, lsl #16
    movk    w9, #0x37BD
    str     w9, [x1, #36]

    // Build param_ptrs
    mov     x2, x12                 // param_ptrs (hoisted from decode_one_token)
    str     x1, [x2, #0]
    add     x3, x1, #8
    str     x3, [x2, #8]
    add     x3, x1, #16
    str     x3, [x2, #16]
    add     x3, x1, #24
    str     x3, [x2, #24]
    add     x3, x1, #32
    str     x3, [x2, #32]
    add     x3, x1, #36
    str     x3, [x2, #40]

    // Launch: 1 block, 256 threads, 128B shared
    // x0 still holds KF_NORM CUfunction from null-check load above
    mov     x1, #1
    mov     x2, #1
    mov     x3, #1
    mov     x4, #256
    mov     x5, #1
    mov     x6, #1
    mov     x7, #128

    adrp    x9, cuda_stream
    add     x9, x9, :lo12:cuda_stream
    ldr     x9, [x9]
    mov     x10, x12               // param_ptrs (hoisted)
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

.fn_skip:
    ldp     x11, x12, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// launch_lm_head -- project hidden state to vocab logits
// ============================================================
// Kernel: lm_head (FP16 dense GEMV, NOT GPTQ)
// Signature: lm_head(weight_ptr u64, input_ptr u64, output_ptr u64)
// weight: [248320, 5120] FP16 row-major
// input:  [5120] F32 (hidden state after final_norm)
// output: [248320] F32 (logits)
// Launch: grid=(62080, 1, 1), block=(128, 1, 1)   // 248320/4 = 62080 rows/block=4
.align 4
launch_lm_head:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x11, x12, [sp, #-16]!  // preserve hoisted launch_params/param_ptrs

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_LM_HEAD]
    cbz     x0, .lmh_skip

    mov     x1, x11                 // launch_params (hoisted from decode_one_token)

    // param[0] = weight_ptr (FP16 lm_head matrix)
    adrp    x9, global_weight_ptrs
    add     x9, x9, :lo12:global_weight_ptrs
    ldr     x9, [x9, #GLOBAL_LMHEAD]
    str     x9, [x1, #0]

    // param[1] = input_ptr (F32 hidden state, current act buf after final norm)
    cbz     x27, .lmh_input_a
    str     x21, [x1, #8]
    b       .lmh_input_done
.lmh_input_a:
    str     x20, [x1, #8]
.lmh_input_done:

    // param[2] = output_ptr = cuda_logits_dev (GPU device buffer for logits)
    adrp    x9, cuda_logits_dev
    add     x9, x9, :lo12:cuda_logits_dev
    ldr     x9, [x9]
    str     x9, [x1, #16]

    // Build param_ptrs (3 params)
    mov     x2, x12                 // param_ptrs (hoisted from decode_one_token)
    str     x1, [x2, #0]
    add     x3, x1, #8
    str     x3, [x2, #8]
    add     x3, x1, #16
    str     x3, [x2, #16]

    // grid=(62080, 1, 1), block=(128, 1, 1)
    // x0 still holds KF_LM_HEAD CUfunction from null-check load above
    movz    x1, #62080
    mov     x2, #1
    mov     x3, #1
    mov     x4, #128
    mov     x5, #1
    mov     x6, #1
    mov     x7, #0

    adrp    x9, cuda_stream
    add     x9, x9, :lo12:cuda_stream
    ldr     x9, [x9]
    mov     x10, x12               // param_ptrs (hoisted)
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

.lmh_skip:
    ldp     x11, x12, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// alloc_buffers -- mmap activation buffers, residual, KV cache
// ============================================================
.align 4
alloc_buffers:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    // ---- Activation buffer A ----
    mov     x0, #0                  // addr = NULL
    mov     x1, #ACT_BUF_SIZE
    mov     x2, #PROT_RW
    mov     x3, #MAP_PRIV_ANON
    mvn     x4, xzr                 // fd = -1
    mov     x5, #0
    mov     x8, #SYS_MMAP
    svc     #0
    adrp    x1, act_buf_a
    add     x1, x1, :lo12:act_buf_a
    str     x0, [x1]
    // GH200: PAGEABLE_MEMORY_ACCESS=1, anonymous mmap GPU-accessible directly.
    // cuMemHostRegister returns 801 (NOT_SUPPORTED) on GH200 - omitted.

    // ---- Activation buffer B ----
    mov     x0, #0
    mov     x1, #ACT_BUF_SIZE
    mov     x2, #PROT_RW
    mov     x3, #MAP_PRIV_ANON
    mvn     x4, xzr
    mov     x5, #0
    mov     x8, #SYS_MMAP
    svc     #0
    adrp    x1, act_buf_b
    add     x1, x1, :lo12:act_buf_b
    str     x0, [x1]

    // ---- Residual buffer ----
    mov     x0, #0
    mov     x1, #RES_BUF_SIZE
    mov     x2, #PROT_RW
    mov     x3, #MAP_PRIV_ANON
    mvn     x4, xzr
    mov     x5, #0
    mov     x8, #SYS_MMAP
    svc     #0
    adrp    x1, residual_buf
    add     x1, x1, :lo12:residual_buf
    str     x0, [x1]

    // ---- Conv1d shift register state ----
    // 64 layers * 10240 channels * 4 taps * 4 bytes = 10485760 (10MB)
    // Zero-initialized by MAP_ANONYMOUS (history starts empty)
    mov     x0, #0
    movz    x1, #0x00A0, lsl #16       // 0x00A00000 = 10485760
    mov     x2, #PROT_RW
    mov     x3, #MAP_PRIV_ANON
    mvn     x4, xzr
    mov     x5, #0
    mov     x8, #SYS_MMAP
    svc     #0
    adrp    x1, conv1d_state
    add     x1, x1, :lo12:conv1d_state
    str     x0, [x1]

    // ---- Z projection output buffer (output gate) ----
    // value_dim * sizeof(f32) = 6144 * 4 = 24576 bytes
    mov     x0, #0
    mov     x1, #Z_BUF_SIZE
    mov     x2, #PROT_RW
    mov     x3, #MAP_PRIV_ANON
    mvn     x4, xzr
    mov     x5, #0
    mov     x8, #SYS_MMAP
    svc     #0
    adrp    x1, z_buf
    add     x1, x1, :lo12:z_buf
    str     x0, [x1]

    // ---- MLP gate scratch buffer ----
    // INTERMEDIATE_DIM * sizeof(f32) = 17408 * 4 = 69632 bytes
    mov     x0, #0
    movz    x1, #0x0001, lsl #16       // 0x11000 = 69632
    movk    x1, #0x1000
    mov     x2, #PROT_RW
    mov     x3, #MAP_PRIV_ANON
    mvn     x4, xzr
    mov     x5, #0
    mov     x8, #SYS_MMAP
    svc     #0
    adrp    x1, mlp_gate_buf
    add     x1, x1, :lo12:mlp_gate_buf
    str     x0, [x1]

    // ---- MLP up scratch buffer ----
    mov     x0, #0
    movz    x1, #0x0001, lsl #16       // 0x11000 = 69632
    movk    x1, #0x1000
    mov     x2, #PROT_RW
    mov     x3, #MAP_PRIV_ANON
    mvn     x4, xzr
    mov     x5, #0
    mov     x8, #SYS_MMAP
    svc     #0
    adrp    x1, mlp_up_buf
    add     x1, x1, :lo12:mlp_up_buf
    str     x0, [x1]

    // ---- KV cache (16 FA layers * 256MB = 4GB) ----
    mov     x0, #0
    movz    x1, #0x0001, lsl #32       // 0x100000000 = 4GB
    mov     x2, #PROT_RW
    mov     x3, #MAP_PRIV_ANON
    mvn     x4, xzr
    mov     x5, #0
    mov     x8, #SYS_MMAP
    svc     #0
    adrp    x1, kv_cache_ptr
    add     x1, x1, :lo12:kv_cache_ptr
    str     x0, [x1]

    // ---- DeltaNet recurrent state (48 DN layers * 3145728 = ~150MB) ----
    // 48 * 3145728 = 150994944 = 0x08FD0000 (round up to 0x09000000 = 150994944)
    mov     x0, #0
    movz    x1, #0x0900, lsl #16       // 0x09000000 = ~150MB
    mov     x2, #PROT_RW
    mov     x3, #MAP_PRIV_ANON
    mvn     x4, xzr
    mov     x5, #0
    mov     x8, #SYS_MMAP
    svc     #0
    adrp    x1, dn_state_ptr
    add     x1, x1, :lo12:dn_state_ptr
    str     x0, [x1]

    // ---- Logits buffer (VOCAB_SIZE * 4 = 993280 bytes) ----
    mov     x0, #0
    movz    x1, #0x000F, lsl #16       // 0x000F2800 = 993280
    movk    x1, #0x2800
    mov     x2, #PROT_RW
    mov     x3, #MAP_PRIV_ANON
    mvn     x4, xzr
    mov     x5, #0
    mov     x8, #SYS_MMAP
    svc     #0
    adrp    x1, logits_buf
    add     x1, x1, :lo12:logits_buf
    str     x0, [x1]

    // ---- GPU logits device buffer via cuMemAlloc ----
    // cuMemAlloc(CUdeviceptr *dptr, size_t bytesize)
    // x0 = pointer to cuda_logits_dev (receives the device ptr)
    // x1 = LOGITS_BUF_SIZE = 993280
    adrp    x0, cuda_logits_dev
    add     x0, x0, :lo12:cuda_logits_dev
    movz    x1, #0x000F, lsl #16       // 0x000F2800 = 993280
    movk    x1, #0x2800
    bl      cuMemAlloc

    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// load_weights -- mmap safetensors shards and build weight pointer table
// ============================================================
// Reads weight_index.bin (generated by gen_weight_index.py) from model_dir,
// mmaps each shard file, then resolves all weight pointers.
//
// Weight index format:
//   [0]  u32 magic (0x4C575449)
//   [4]  u32 version (1)
//   [8]  u32 num_shards
//   [12] u32 num_layers
//   [16] u32 slots_per_layer
//   [20] u32 num_globals
//   [24] u32 reserved[2]
//   [32] shard_table: num_shards * 264 bytes (256-char filename + u64 file_size)
//   then global_table: num_globals * 24 bytes
//       each: u32 shard_idx, u32 reserved, u64 file_offset, u64 size
//   then layer_table: num_layers * slots_per_layer * 24 bytes
//       each: u32 shard_idx, u32 reserved, u64 file_offset, u64 size
.align 4
load_weights:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!

    // Get model_dir from argv[1]
    adrp    x0, saved_argc
    add     x0, x0, :lo12:saved_argc
    ldr     x19, [x0]              // argc
    cmp     x19, #2
    b.lt    .weights_done          // no model dir, skip

    adrp    x0, saved_argv
    add     x0, x0, :lo12:saved_argv
    ldr     x0, [x0]
    ldr     x20, [x0, #8]         // argv[1] = model_dir

    // ---- Step 1: Open and mmap weight_index.bin ----
    // Build path: model_dir / weight_index.bin
    adrp    x4, path_buf
    add     x4, x4, :lo12:path_buf
    mov     x5, x4                 // save path start
    mov     x0, x20
.wi_copy_dir:
    ldrb    w6, [x0], #1
    cbz     w6, .wi_dir_done
    strb    w6, [x4], #1
    b       .wi_copy_dir
.wi_dir_done:
    mov     w6, #'/'
    strb    w6, [x4], #1
    adrp    x0, widx_filename
    add     x0, x0, :lo12:widx_filename
.wi_copy_name:
    ldrb    w6, [x0], #1
    strb    w6, [x4], #1
    cbnz    w6, .wi_copy_name

    // Open weight_index.bin
    mov     x0, #AT_FDCWD
    mov     x1, x5                 // path
    mov     x2, #O_RDONLY
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    .widx_not_found
    mov     x21, x0                // fd

    // Get file size via lseek(fd, 0, SEEK_END)
    mov     x1, #0
    mov     x2, #SEEK_END
    mov     x8, #SYS_LSEEK
    svc     #0
    mov     x22, x0                // widx_size

    // mmap the index file
    mov     x0, #0                 // addr = NULL
    mov     x1, x22                // length
    mov     x2, #PROT_READ         // prot = PROT_READ
    mov     x3, #MAP_PRIVATE       // flags
    mov     x4, x21                // fd
    mov     x5, #0                 // offset
    mov     x8, #SYS_MMAP
    svc     #0
    cmn     x0, #1
    b.eq    .widx_mmap_fail
    mov     x23, x0                // x23 = widx_base (kept throughout)

    // Store widx pointers
    adrp    x1, widx_base
    add     x1, x1, :lo12:widx_base
    stp     x23, x22, [x1]

    // Close the index fd (mmap persists)
    mov     x0, x21
    mov     x8, #SYS_CLOSE
    svc     #0

    // Validate magic = 0x4C575449
    ldr     w0, [x23]
    mov     w1, #0x5449             // low half
    movk    w1, #0x4C57, lsl #16   // high half -> 0x4C575449
    cmp     w0, w1
    b.ne    .widx_not_found

    // Read header fields
    ldr     w24, [x23, #8]         // num_shards

    // ---- Step 2: Open and mmap each shard file ----
    // Shard table starts at offset 32 in the index
    add     x25, x23, #32         // x25 = shard_table_ptr
    mov     x26, #0                // shard_idx counter

.shard_loop:
    cmp     w26, w24
    b.ge    .shards_done

    // Build path: model_dir / shard_filename
    adrp    x4, path_buf
    add     x4, x4, :lo12:path_buf
    mov     x5, x4
    mov     x0, x20                // model_dir
.sh_copy_dir:
    ldrb    w6, [x0], #1
    cbz     w6, .sh_dir_done
    strb    w6, [x4], #1
    b       .sh_copy_dir
.sh_dir_done:
    mov     w6, #'/'
    strb    w6, [x4], #1
    // Copy shard filename from index (256 bytes at shard_table + shard_idx*264)
    mov     x0, #264
    mul     x0, x26, x0
    add     x0, x25, x0           // ptr to this shard entry
    mov     x27, x0               // save shard entry ptr
.sh_copy_name:
    ldrb    w6, [x0], #1
    strb    w6, [x4], #1
    cbnz    w6, .sh_copy_name

    // Get expected file size from index (at shard_entry + 256)
    ldr     x28, [x27, #256]      // file_size

    // Print "lithos: mmap shard N BYTES bytes"
    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!

    adrp    x1, msg_shard
    add     x1, x1, :lo12:msg_shard
    mov     x2, msg_shard_len
    bl      print_msg

    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16

    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!

    mov     x0, x26
    bl      print_int

    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16

    // Print " "
    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!

    sub     sp, sp, #16
    mov     w0, #' '
    strb    w0, [sp]
    mov     x1, sp
    mov     x2, #1
    mov     x0, #1
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #16

    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16

    // Print file size
    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!

    mov     x0, x28
    bl      print_int

    adrp    x1, msg_bytes
    add     x1, x1, :lo12:msg_bytes
    mov     x2, msg_bytes_len
    bl      print_msg

    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16

    // Open the shard file (reload path_buf since x5 is caller-saved)
    mov     x0, #AT_FDCWD
    adrp    x1, path_buf
    add     x1, x1, :lo12:path_buf
    mov     x2, #O_RDONLY
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    .shard_open_fail
    mov     x21, x0                // fd

    // ---- Allocate pinned host memory via cuMemAllocHost (iter-17 fix) ----
    // Replaces file-mmap + cuMemHostRegister which returned NOT_PERMITTED (801).
    // cuMemAllocHost(void **pp, size_t bytesize) -- allocates page-locked host memory.
    // GPU kernels can read pinned host memory directly without registration.
    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!

    adrp    x0, shard_alloc_ptr
    add     x0, x0, :lo12:shard_alloc_ptr  // pp = &shard_alloc_ptr
    mov     x1, x28                         // bytesize = file_size
    bl      cuMemAllocHost
    cbnz    x0, .shard_alloc_fail_restore   // non-zero return = CUDA error

    // Load the allocated host pointer
    adrp    x22, shard_alloc_ptr
    add     x22, x22, :lo12:shard_alloc_ptr
    ldr     x22, [x22]                      // x22 = pinned host buffer base

    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16

    // ---- pread loop: fill pinned buffer from shard file ----
    // x21 = fd, x22 = buf_base, x28 = file_size
    // x9 = offset into file, x10 = remaining bytes, x12 = write cursor
    mov     x12, x22                        // write cursor = buffer start
    mov     x9, #0                          // file offset = 0
    mov     x10, x28                        // remaining = file_size

.shard_pread_loop:
    cbz     x10, .shard_pread_done
    // chunk = min(remaining, 256MB)
    mov     x11, #(256 * 1024 * 1024)
    cmp     x10, x11
    csel    x2, x10, x11, lo               // x2 = this chunk size
    // pread64(fd, buf, count, offset)
    mov     x0, x21                         // fd
    mov     x1, x12                         // destination
    // x2 already set (chunk size)
    mov     x3, x9                          // file offset
    mov     x8, #SYS_PREAD64
    svc     #0
    cmp     x0, #0
    b.le    .shard_pread_fail               // 0=EOF or negative=error
    add     x12, x12, x0                    // advance write cursor
    add     x9, x9, x0                      // advance file offset
    sub     x10, x10, x0                    // decrement remaining
    b       .shard_pread_loop

.shard_pread_done:
    // Store shard base pointer (pinned host memory)
    adrp    x1, shard_bases
    add     x1, x1, :lo12:shard_bases
    str     x22, [x1, x26, lsl #3]

    // Store shard size
    adrp    x1, shard_sizes
    add     x1, x1, :lo12:shard_sizes
    str     x28, [x1, x26, lsl #3]

    // Close fd (buffer is independent)
    mov     x0, x21
    mov     x8, #SYS_CLOSE
    svc     #0
    b       .shard_loop_next

.shard_alloc_fail_restore:
    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16
    b       .shard_mmap_fail

.shard_pread_fail:
    // Close fd and fall through to error
    mov     x0, x21
    mov     x8, #SYS_CLOSE
    svc     #0
    b       .shard_mmap_fail

.shard_loop_next:
    // Next shard
    add     x26, x26, #1
    b       .shard_loop

.shards_done:
    // Store num_shards_loaded
    adrp    x0, num_shards_loaded
    add     x0, x0, :lo12:num_shards_loaded
    str     x26, [x0]

    // ---- Step 3: Resolve global weight pointers ----
    // Global table offset = 32 + num_shards * 264
    mov     x0, #264
    umull   x0, w24, w0           // num_shards * 264
    add     x0, x0, #32
    add     x25, x23, x0          // x25 = global_table_ptr

    ldr     w26, [x23, #20]       // num_globals
    adrp    x27, global_weight_ptrs
    add     x27, x27, :lo12:global_weight_ptrs

    mov     x21, #0                // global_idx
.global_resolve_loop:
    cmp     w21, w26
    b.ge    .globals_resolve_done

    // Each global entry: u32 shard_idx, u32 pad, u64 file_offset, u64 size
    mov     x0, #24
    mul     x0, x21, x0
    add     x0, x25, x0           // ptr to this global entry

    ldr     w1, [x0]              // shard_idx
    ldr     x2, [x0, #8]          // file_offset
    ldr     x3, [x0, #16]         // size

    // If offset and size are both 0, this slot is unused -> NULL
    orr     x4, x2, x3
    cbz     x4, .global_store_null

    // Resolve: ptr = shard_bases[shard_idx] + file_offset
    adrp    x4, shard_bases
    add     x4, x4, :lo12:shard_bases
    ldr     x4, [x4, x1, lsl #3]  // shard_base
    add     x4, x4, x2            // resolved pointer

    str     x4, [x27, x21, lsl #3]
    b       .global_resolve_next
.global_store_null:
    str     xzr, [x27, x21, lsl #3]
.global_resolve_next:
    add     x21, x21, #1
    b       .global_resolve_loop

.globals_resolve_done:

    // ---- Step 4: Resolve per-layer weight pointers ----
    // Layer table offset = global_table_ptr + num_globals * 24
    mov     x0, #24
    umull   x0, w26, w0           // num_globals * 24
    add     x25, x25, x0          // x25 = layer_table_ptr

    ldr     w21, [x23, #12]       // num_layers (should be 64)
    ldr     w22, [x23, #16]       // slots_per_layer (should be 26)

    adrp    x27, weight_ptrs
    add     x27, x27, :lo12:weight_ptrs

    mov     x19, #0                // layer_idx
.layer_resolve_loop:
    cmp     w19, w21
    b.ge    .layers_resolve_done

    // For this layer, resolve all slots
    mov     x28, #0                // slot_idx

.slot_resolve_loop:
    cmp     w28, w22
    b.ge    .slots_resolve_done

    // entry offset = (layer_idx * slots_per_layer + slot_idx) * 24
    umull   x0, w19, w22          // layer_idx * slots_per_layer
    add     x0, x0, x28           // + slot_idx
    mov     x1, #24
    mul     x0, x0, x1            // * 24
    add     x0, x25, x0           // ptr to entry

    ldr     w1, [x0]              // shard_idx
    ldr     x2, [x0, #8]          // file_offset
    ldr     x3, [x0, #16]         // size

    // If offset and size are both 0, this slot is unused -> NULL
    orr     x4, x2, x3
    cbz     x4, .slot_store_null

    // Resolve: ptr = shard_bases[shard_idx] + file_offset
    adrp    x4, shard_bases
    add     x4, x4, :lo12:shard_bases
    ldr     x4, [x4, x1, lsl #3]
    add     x4, x4, x2

    // Store in weight_ptrs table
    // dest index = layer_idx * 26 + slot_idx
    mov     x5, #WEIGHT_SLOTS_PER_LAYER
    umull   x5, w19, w5
    add     x5, x5, x28
    str     x4, [x27, x5, lsl #3]
    b       .slot_resolve_next

.slot_store_null:
    mov     x5, #WEIGHT_SLOTS_PER_LAYER
    umull   x5, w19, w5
    add     x5, x5, x28
    str     xzr, [x27, x5, lsl #3]

.slot_resolve_next:
    add     x28, x28, #1
    b       .slot_resolve_loop

.slots_resolve_done:
    add     x19, x19, #1
    b       .layer_resolve_loop

.layers_resolve_done:
    // Print success
    adrp    x1, msg_wdone
    add     x1, x1, :lo12:msg_wdone
    mov     x2, msg_wdone_len
    bl      print_msg

    b       .weights_done

.widx_not_found:
    adrp    x1, msg_idxerr
    add     x1, x1, :lo12:msg_idxerr
    mov     x2, msg_idxerr_len
    bl      print_msg
    b       .weights_done

.widx_mmap_fail:
    mov     x0, x21
    mov     x8, #SYS_CLOSE
    svc     #0
    b       .widx_not_found

.shard_open_fail:
.shard_mmap_fail:
    adrp    x1, msg_werr
    add     x1, x1, :lo12:msg_werr
    mov     x2, msg_werr_len
    bl      print_msg
    // Continue to next shard instead of hard-failing
    add     x26, x26, #1
    b       .shard_loop

.weights_done:
    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// get_layer_weight_ptr -- return weight pointer for layer/slot
// ============================================================
// x0 = layer_idx, x1 = slot_idx
// Returns x0 = pointer (or 0 if missing)
.align 4
get_layer_weight_ptr:
    adrp    x2, weight_ptrs
    add     x2, x2, :lo12:weight_ptrs
    mov     x3, #WEIGHT_SLOTS_PER_LAYER
    madd    x3, x0, x3, x1
    ldr     x0, [x2, x3, lsl #3]
    ret

// ============================================================
// load_cubins -- load cubin files and resolve kernel functions
// ============================================================
// For each kernel type:
//   1. Open cubin file
//   2. Read into buffer
//   3. cuModuleLoadData(&module, buffer)
//   4. cuModuleGetFunction(&func, module, name)
.align 4
load_cubins:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!

    // Get kernel_dir from argv[2]
    adrp    x0, saved_argc
    add     x0, x0, :lo12:saved_argc
    ldr     x19, [x0]              // argc
    cmp     x19, #3
    b.lt    .cubins_done           // no kernel dir specified, skip

    adrp    x0, saved_argv
    add     x0, x0, :lo12:saved_argv
    ldr     x0, [x0]
    ldr     x20, [x0, #16]        // argv[2] = kernel_dir

    // Load embed cubin
    adrp    x1, cubin_embed
    add     x1, x1, :lo12:cubin_embed
    mov     x0, x20
    adrp    x2, kern_table
    add     x2, x2, :lo12:kern_table
    add     x2, x2, #KF_EMBED
    adrp    x3, kname_embed
    add     x3, x3, :lo12:kname_embed
    bl      load_one_cubin

    // Load norm/activate cubin
    adrp    x1, cubin_norm
    add     x1, x1, :lo12:cubin_norm
    mov     x0, x20
    adrp    x2, kern_table
    add     x2, x2, :lo12:kern_table
    add     x2, x2, #KF_NORM
    adrp    x3, kname_norm
    add     x3, x3, :lo12:kname_norm
    bl      load_one_cubin

    // Load projection cubin
    adrp    x1, cubin_proj
    add     x1, x1, :lo12:cubin_proj
    mov     x0, x20
    adrp    x2, kern_table
    add     x2, x2, :lo12:kern_table
    add     x2, x2, #KF_PROJ
    adrp    x3, kname_proj
    add     x3, x3, :lo12:kname_proj
    bl      load_one_cubin

    // Load attention cubin
    adrp    x1, cubin_attn
    add     x1, x1, :lo12:cubin_attn
    mov     x0, x20
    adrp    x2, kern_table
    add     x2, x2, :lo12:kern_table
    add     x2, x2, #KF_ATTN
    adrp    x3, kname_attn
    add     x3, x3, :lo12:kname_attn
    bl      load_one_cubin

    // Load recurrence cubin
    adrp    x1, cubin_recur
    add     x1, x1, :lo12:cubin_recur
    mov     x0, x20
    adrp    x2, kern_table
    add     x2, x2, :lo12:kern_table
    add     x2, x2, #KF_RECUR
    adrp    x3, kname_recur
    add     x3, x3, :lo12:kname_recur
    bl      load_one_cubin

    // Load activate cubin
    adrp    x1, cubin_activate
    add     x1, x1, :lo12:cubin_activate
    mov     x0, x20
    adrp    x2, kern_table
    add     x2, x2, :lo12:kern_table
    add     x2, x2, #KF_ACTIVATE
    adrp    x3, kname_activate
    add     x3, x3, :lo12:kname_activate
    bl      load_one_cubin

    // Load rope/rotate cubin
    adrp    x1, cubin_rope
    add     x1, x1, :lo12:cubin_rope
    mov     x0, x20
    adrp    x2, kern_table
    add     x2, x2, :lo12:kern_table
    add     x2, x2, #KF_ROPE
    adrp    x3, kname_rope
    add     x3, x3, :lo12:kname_rope
    bl      load_one_cubin

    // Load gate_sigmoid cubin (output gate for DeltaNet)
    adrp    x1, cubin_gate_sig
    add     x1, x1, :lo12:cubin_gate_sig
    mov     x0, x20
    adrp    x2, kern_table
    add     x2, x2, :lo12:kern_table
    add     x2, x2, #KF_GATE_SIG
    adrp    x3, kname_gate_sig
    add     x3, x3, :lo12:kname_gate_sig
    bl      load_one_cubin

    // Load conv1d cubin (from recur.cubin which contains conv1d_infer)
    adrp    x1, cubin_conv1d
    add     x1, x1, :lo12:cubin_conv1d
    mov     x0, x20
    adrp    x2, kern_table
    add     x2, x2, :lo12:kern_table
    add     x2, x2, #KF_CONV1D
    adrp    x3, kname_conv1d
    add     x3, x3, :lo12:kname_conv1d
    bl      load_one_cubin

    // Load l2norm from reduce.cubin
    adrp    x1, cubin_l2norm
    add     x1, x1, :lo12:cubin_l2norm
    mov     x0, x20
    adrp    x2, kern_table
    add     x2, x2, :lo12:kern_table
    add     x2, x2, #KF_L2NORM
    adrp    x3, kname_l2norm
    add     x3, x3, :lo12:kname_l2norm
    bl      load_one_cubin

    // Load lm_head cubin (FP16 dense GEMV for vocab projection)
    adrp    x1, cubin_lm_head
    add     x1, x1, :lo12:cubin_lm_head
    mov     x0, x20
    adrp    x2, kern_table
    add     x2, x2, :lo12:kern_table
    add     x2, x2, #KF_LM_HEAD
    adrp    x3, kname_lm_head
    add     x3, x3, :lo12:kname_lm_head
    bl      load_one_cubin

.cubins_done:
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// load_one_cubin -- load a single cubin and resolve function
// ============================================================
// x0 = kernel_dir (null-terminated string)
// x1 = cubin filename (null-terminated string)
// x2 = address to store CUfunction
// x3 = kernel name (null-terminated string)
.align 4
load_one_cubin:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!

    mov     x19, x2                 // save func_dest
    mov     x20, x3                 // save kernel_name

    // Build path: kernel_dir / cubin_filename
    adrp    x4, path_buf
    add     x4, x4, :lo12:path_buf
    mov     x5, x4                  // save start

    // Copy kernel_dir
.copy_dir:
    ldrb    w6, [x0], #1
    cbz     w6, .dir_done
    strb    w6, [x4], #1
    b       .copy_dir
.dir_done:
    // Add '/'
    mov     w6, #'/'
    strb    w6, [x4], #1
    // Copy filename
.copy_fname:
    ldrb    w6, [x1], #1
    strb    w6, [x4], #1
    cbnz    w6, .copy_fname

    // Open the cubin file
    mov     x0, #AT_FDCWD
    mov     x1, x5                  // path
    mov     x2, #O_RDONLY
    mov     x3, #0
    mov     x8, #SYS_OPENAT
    svc     #0
    cmp     x0, #0
    b.lt    .cubin_not_found
    mov     x21, x0                 // fd

    // Get file size via lseek
    mov     x1, #0
    mov     x2, #SEEK_END
    mov     x8, #SYS_LSEEK
    svc     #0
    mov     x22, x0                 // file_size

    // Seek back to start
    mov     x0, x21
    mov     x1, #0
    mov     x2, #SEEK_SET
    mov     x8, #SYS_LSEEK
    svc     #0

    // Read file
    mov     x0, x21                 // fd
    adrp    x1, file_buffer
    add     x1, x1, :lo12:file_buffer
    mov     x2, x22                 // count
    mov     x8, #SYS_READ
    svc     #0

    // Close file
    mov     x0, x21
    mov     x8, #SYS_CLOSE
    svc     #0

    // cuModuleLoadData(&module, buffer)
    sub     sp, sp, #16
    mov     x0, sp                  // &module on stack
    adrp    x1, file_buffer
    add     x1, x1, :lo12:file_buffer
    bl      cuModuleLoadData
    cbnz    x0, .cubin_load_failed

    // cuModuleGetFunction(&func, module, name)
    ldr     x1, [sp]               // module
    mov     x0, x19                 // &func destination
    mov     x2, x20                 // kernel name
    bl      cuModuleGetFunction
    add     sp, sp, #16
    cbnz    x0, .cubin_func_failed

    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

.cubin_not_found:
    // File not found -- store 0 (kernel will be skipped)
    str     xzr, [x19]
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

.cubin_load_failed:
    add     sp, sp, #16
.cubin_func_failed:
    str     xzr, [x19]
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// Utility functions
// ============================================================

// print_msg -- write string to stdout
// x1 = string address, x2 = length
.align 4
print_msg:
    mov     x0, #1                  // stdout
    mov     x8, #SYS_WRITE
    svc     #0
    ret

// print_newline
.align 4
print_newline:
    stp     x29, x30, [sp, #-16]!
    adrp    x1, msg_nl
    add     x1, x1, :lo12:msg_nl
    mov     x2, #1
    bl      print_msg
    ldp     x29, x30, [sp], #16
    ret

// print_int -- print x0 as unsigned decimal
.align 4
print_int:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #32             // digit buffer
    mov     x1, sp
    add     x1, x1, #31            // end of buffer
    mov     x2, #0                  // digit count
    mov     x3, #10

    cbz     x0, .print_zero
.digit_loop:
    cbz     x0, .print_digits
    udiv    x4, x0, x3
    msub    x5, x4, x3, x0         // x5 = x0 % 10
    add     x5, x5, #'0'
    strb    w5, [x1], #-1
    add     x2, x2, #1
    mov     x0, x4
    b       .digit_loop
.print_digits:
    add     x1, x1, #1             // back to first digit
    mov     x0, #1                  // stdout
    mov     x8, #SYS_WRITE
    svc     #0
    add     sp, sp, #32
    ldp     x29, x30, [sp], #16
    ret
.print_zero:
    mov     w5, #'0'
    strb    w5, [x1], #-1
    mov     x2, #1
    b       .print_digits

// atoi -- simple string to integer
// x0 = string address, returns x0 = integer
.align 4
atoi:
    mov     x1, #0                  // result
    mov     x2, #10
.atoi_loop:
    ldrb    w3, [x0], #1
    cbz     w3, .atoi_done
    sub     w3, w3, #'0'
    cmp     w3, #9
    b.hi    .atoi_done
    madd    x1, x1, x2, x3
    b       .atoi_loop
.atoi_done:
    mov     x0, x1
    ret

// debug_sync_tag -- call cuCtxSynchronize, print tag, exit if error
// x0 = tag number, x19 = layer index
// Preserves x11, x12, x19-x28.
.align 4
debug_sync_tag:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    stp     x11, x12, [sp, #-16]!
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!
    mov     x25, x0                 // save tag
    mov     x26, x19                // save layer

    bl      cuCtxSynchronize
    mov     x24, x0                 // save err

    // Print "lithos: dbg layer=<layer> tag=<tag> err=<err>\n"
    adrp    x1, msg_dbg
    add     x1, x1, :lo12:msg_dbg
    mov     x2, msg_dbg_len
    bl      print_msg
    mov     x0, x26
    bl      print_int
    adrp    x1, msg_dbg2
    add     x1, x1, :lo12:msg_dbg2
    mov     x2, msg_dbg2_len
    bl      print_msg
    mov     x0, x25
    bl      print_int
    adrp    x1, msg_dbg3
    add     x1, x1, :lo12:msg_dbg3
    mov     x2, msg_dbg3_len
    bl      print_msg
    mov     x0, x24
    bl      print_int
    bl      print_newline

    cbnz    x24, .dbg_sync_exit
    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ldp     x11, x12, [sp], #16
    ldp     x29, x30, [sp], #16
    ret
.dbg_sync_exit:
    mov     x0, #1
    mov     x8, #SYS_EXIT
    svc     #0
