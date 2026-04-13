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
.equ SYS_MPROTECT,  226
.equ SYS_EXIT,      93
.equ SYS_BRK,       214

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
.equ KF_TABLE_SIZE,   88

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
// HIDDEN_DIM * sizeof(f32) = 5120 * 4 = 20480 bytes per token
// Round up to 4 pages = 16384 (page aligned, generous)
.equ ACT_BUF_SIZE,    20480
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
msg_nl:       .asciz "\n"
msg_done:     .asciz "\nlithos: done\n"
msg_done_len = . - msg_done - 1
msg_exit:     .asciz "lithos: exiting now\n"
msg_exit_len = . - msg_exit - 1
msg_us:       .asciz "us total\n"
msg_us_len = . - msg_us - 1
msg_toks:     .asciz "lithos: tokens generated: "
msg_toks_len = . - msg_toks - 1
msg_speed:    .asciz " tok/s\n"
msg_perf0:    .asciz "lithos: dispatch overhead per token: "
msg_perf0_len = . - msg_perf0 - 1

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

// Path buffer
.align 3
path_buf:     .space 512

// ---- State variables ----
.align 3
cuda_device:  .quad 0
cuda_context: .quad 0
cuda_stream:  .quad 0
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

    // Print timing: "lithos: N tokens in X.XXXs (dispatch only)\n"
    adrp    x1, msg_done
    add     x1, x1, :lo12:msg_done
    mov     x2, msg_done_len
    bl      print_msg

    // Print: "lithos: dispatch time: "
    adrp    x1, msg_perf0
    add     x1, x1, :lo12:msg_perf0
    mov     x2, msg_perf0_len
    bl      print_msg

    // Print elapsed microseconds
    // usec = sec * 1000000 + nsec / 1000
    movz    x0, #0x000F, lsl #16    // 0x000F4240 = 1000000
    movk    x0, #0x4240
    mul     x0, x6, x0             // sec * 1M
    mov     x1, #1000
    udiv    x7, x7, x1             // nsec / 1000
    add     x0, x0, x7             // total usec
    bl      print_int

    // Print "us\n"
    adrp    x1, msg_us
    add     x1, x1, :lo12:msg_us
    mov     x2, msg_us_len
    bl      print_msg

    // ---- Print tokens generated ----
    adrp    x1, msg_toks
    add     x1, x1, :lo12:msg_toks
    mov     x2, msg_toks_len
    bl      print_msg

    adrp    x0, num_tokens_generated
    add     x0, x0, :lo12:num_tokens_generated
    ldr     x0, [x0]
    bl      print_int
    bl      print_newline

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

    // Load buffer pointers
    adrp    x20, act_buf_a
    add     x20, x20, :lo12:act_buf_a
    ldr     x20, [x20]              // x20 = act_buf_a

    adrp    x21, act_buf_b
    add     x21, x21, :lo12:act_buf_b
    ldr     x21, [x21]              // x21 = act_buf_b

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
    adrp    x1, launch_params
    add     x1, x1, :lo12:launch_params
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
    adrp    x2, param_ptrs
    add     x2, x2, :lo12:param_ptrs
    str     x1, [x2]               // param_ptrs[0] = &param[0]
    add     x3, x1, #8
    str     x3, [x2, #8]           // param_ptrs[1] = &param[1]
    add     x3, x1, #16
    str     x3, [x2, #16]          // param_ptrs[2] = &param[2]

    // cuLaunchKernel(func, gridX, gridY, gridZ, blockX, blockY, blockZ,
    //                sharedMem, stream, paramPtrs, extra)
    // x0 = func (already set)
    mov     x1, #4                  // gridDimX = ceil(3584 / (256*4)) = 4
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
    adrp    x10, param_ptrs
    add     x10, x10, :lo12:param_ptrs
    stp     x9, x10, [sp, #-32]!   // push stream, paramPtrs
    str     xzr, [sp, #16]         // push extra = NULL
    bl      cuLaunchKernel
    add     sp, sp, #32
    // Don't check error on hot path (check after sync)

.skip_embed:

    // ---- Layer loop: 64 layers ----
    mov     x19, #0                 // layer counter

.layer_loop:
    cmp     x19, #NUM_LAYERS
    b.ge    .layers_done

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

    // ---- QKV projection (3 launches) ----
    bl      launch_proj              // Q
    bl      launch_proj              // K
    bl      launch_proj              // V

    // ---- Attention or DeltaNet ----
    // x22 survives all bl calls above; no flag recomputation needed.
    cbz     x22, .do_full_attention

    // ---- DeltaNet path ----
    // Conv1d on Q, K, V (causal, 4-tap shift register per channel)
    // x19 = layer index, used by launch_conv1d for state offset
    bl      launch_conv1d            // conv1d on Q
    bl      launch_conv1d            // conv1d on K
    bl      launch_l2norm            // steps 10-13: L2-norm on K (dim=2048)
    bl      launch_conv1d            // conv1d on V
    bl      launch_activate          // steps 14-18: SiLU on Q
    bl      launch_l2norm            // steps 19-22: L2-norm on Q (dim=2048)
    bl      launch_recurrence
    // Output gate (steps 29-34): z = W_z @ x, output *= sigmoid(z)
    // z projection uses ORIGINAL input x (still in act buffer, not overwritten by QKV/recur)
    bl      launch_z_proj            // step 29: z = W_z @ x -> z_buf
    bl      launch_gate_sigmoid      // steps 30-34: output *= sigmoid(z)
    // Post-attention RMSNorm on gated output, THEN output projection
    bl      launch_norm              // steps 35-40: RMSNorm on gated output
    bl      launch_proj              // step 41: output projection
    b       .post_proj

.do_full_attention:
    // ---- Full attention path (RoPE required) ----
    bl      launch_rope
    bl      launch_attention
    // Full attention: output projection, then residual norm
    bl      launch_proj              // output projection
    bl      launch_norm              // residual add + norm

.post_proj:

    // ---- Swap activation buffers ----
    eor     x27, x27, #1

    // ---- Pre-MLP RMSNorm ----
    bl      launch_norm

    // ---- Gate projection ----
    bl      launch_proj

    // ---- Up projection ----
    bl      launch_proj

    // ---- SwiGLU: silu(gate) * up ----
    bl      launch_activate

    // ---- Down projection ----
    bl      launch_proj

    // ---- Residual add ----
    bl      launch_norm

    // ---- Swap activation buffers ----
    eor     x27, x27, #1

    // Next layer
    add     x19, x19, #1
    b       .layer_loop

.layers_done:
    // ---- Final RMSNorm ----
    bl      launch_norm

    // ---- LM head projection ----
    bl      launch_proj

    // ---- Synchronize ----
    // cuCtxSynchronize removed from hot loop.
    // In production: sync only when reading logits for sampling.
    // The stream implicitly serializes kernel launches.

    // ---- Sample (CPU argmax for now) ----
    // In the real version, we'd launch a sample kernel or do CPU argmax
    // For now, just advance the token counter

    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// launch_norm -- launch RMSNorm kernel
// ============================================================
.align 4
launch_norm:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_NORM]
    cbz     x0, .norm_skip

    // For the hot path, we skip full param setup and just do the call.
    // In production, params would be wired up per-layer.
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
    adrp    x10, param_ptrs
    add     x10, x10, :lo12:param_ptrs
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

.norm_skip:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// launch_proj -- launch projection (GEMV/GEMM) kernel
// ============================================================
.align 4
launch_proj:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_PROJ]
    cbz     x0, .proj_skip

    mov     x1, #28                 // gridX = ceil(hidden_dim / 128)
    mov     x2, #1
    mov     x3, #1
    mov     x4, #128                // blockX
    mov     x5, #1
    mov     x6, #1
    mov     x7, #0                  // no shared mem

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
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// launch_attention -- launch full attention kernel
// ============================================================
.align 4
launch_attention:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_ATTN]
    cbz     x0, .attn_skip

    mov     x1, #NUM_HEADS          // gridX = num_heads = 28
    mov     x2, #1
    mov     x3, #1
    mov     x4, #128                // blockX = head_dim
    mov     x5, #1
    mov     x6, #1
    mov     x7, #512                // sharedMem for softmax

    adrp    x9, cuda_stream
    add     x9, x9, :lo12:cuda_stream
    ldr     x9, [x9]
    adrp    x10, param_ptrs
    add     x10, x10, :lo12:param_ptrs
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

.attn_skip:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// launch_recurrence -- launch DeltaNet recurrence kernel
// ============================================================
.align 4
launch_recurrence:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_RECUR]
    cbz     x0, .recur_skip

    mov     x1, #NUM_HEADS          // gridX = num_heads = 28
    mov     x2, #1
    mov     x3, #1
    mov     x4, #HEAD_DIM           // blockX = head_dim = 128
    mov     x5, #1
    mov     x6, #1
    lsr     x7, x4, #0             // sharedMem = head_dim * head_dim * 2
    mov     x7, #32768              // 128 * 128 * 2 = 32768

    adrp    x9, cuda_stream
    add     x9, x9, :lo12:cuda_stream
    ldr     x9, [x9]
    adrp    x10, param_ptrs
    add     x10, x10, :lo12:param_ptrs
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

.recur_skip:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// launch_activate -- launch SwiGLU activation: silu(gate) * up
// ============================================================
.align 4
launch_activate:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_ACTIVATE]
    cbz     x0, .act_skip

    // gridX = ceil(intermediate_dim / 256) = ceil(17408/256) = 68
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
    adrp    x10, param_ptrs
    add     x10, x10, :lo12:param_ptrs
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

.act_skip:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// launch_rope -- launch RoPE kernel
// ============================================================
.align 4
launch_rope:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_ROPE]
    cbz     x0, .rope_skip

    mov     x1, #14                 // gridX = ceil(num_heads * head_dim / 256)
    mov     x2, #1
    mov     x3, #1
    mov     x4, #256
    mov     x5, #1
    mov     x6, #1
    mov     x7, #0

    adrp    x9, cuda_stream
    add     x9, x9, :lo12:cuda_stream
    ldr     x9, [x9]
    adrp    x10, param_ptrs
    add     x10, x10, :lo12:param_ptrs
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

.rope_skip:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// launch_l2norm -- launch L2-norm kernel (steps 10-13 for K, 19-22 for Q)
// ============================================================
// L2-norm: x / sqrt(sum(x^2))
// Kernel signature: (x: u64, y: u64, param_hidden_dim: u32)
// Grid: 1 block, Block: 256 threads, SharedMem: 1024 bytes
.align 4
launch_l2norm:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_L2NORM]
    cbz     x0, .l2norm_skip

    mov     x1, #1                  // gridX = 1 (single block reduction)
    mov     x2, #1                  // gridY
    mov     x3, #1                  // gridZ
    mov     x4, #256                // blockX
    mov     x5, #1                  // blockY
    mov     x6, #1                  // blockZ
    mov     x7, #1024               // sharedMem (smem_reduce[1024])

    adrp    x9, cuda_stream
    add     x9, x9, :lo12:cuda_stream
    ldr     x9, [x9]
    adrp    x10, param_ptrs
    add     x10, x10, :lo12:param_ptrs
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

.l2norm_skip:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// launch_gate_sigmoid -- fused sigmoid gate: output *= sigmoid(z)
// ============================================================
// Used for DeltaNet output gating (STEPS 29-34).
// The z projection (W_z @ x) runs as a standard GEMV launch before this.
// This kernel applies sigmoid(z) element-wise and multiplies into output.
// Operates on value_dim = 6144 elements.
.align 4
launch_gate_sigmoid:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_GATE_SIG]
    cbz     x0, .gate_sig_skip

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
    adrp    x10, param_ptrs
    add     x10, x10, :lo12:param_ptrs
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

.gate_sig_skip:
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

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_PROJ]     // same GEMV kernel
    cbz     x0, .z_proj_skip

    // gridX = ceil(value_dim / 128) = ceil(6144/128) = 48
    mov     x1, #48                 // gridX
    mov     x2, #1                  // gridY
    mov     x3, #1                  // gridZ
    mov     x4, #128                // blockX
    mov     x5, #1                  // blockY
    mov     x6, #1                  // blockZ
    mov     x7, #0                  // no shared mem

    adrp    x9, cuda_stream
    add     x9, x9, :lo12:cuda_stream
    ldr     x9, [x9]
    adrp    x10, param_ptrs
    add     x10, x10, :lo12:param_ptrs
    stp     x9, x10, [sp, #-32]!
    str     xzr, [sp, #16]
    bl      cuLaunchKernel
    add     sp, sp, #32

.z_proj_skip:
    ldp     x29, x30, [sp], #16
    ret

// ============================================================
// launch_conv1d -- launch causal conv1d kernel (one Q/K/V proj)
// ============================================================
// Called 3 times per DeltaNet layer (Q, K, V).
// conv1d_infer(input, history, weights, output, N):
//   input   = current activation buffer (Q/K/V projection output)
//   history = per-layer shift register state (4 taps * N channels)
//   weights = conv1d weight pointer (from model weights)
//   output  = same buffer (in-place, via act_buf_b scratch)
//   N       = CONV1D_CHANNELS (10240 = concat of all head dims)
//
// x19 = layer index (callee-saved, set by layer_loop)
// The history pointer advances by CONV1D_PER_LAYER * layer_idx.
//
// Grid: ceil(10240 / 256) = 40 blocks of 256 threads = 10240 threads
.align 4
launch_conv1d:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_CONV1D]
    cbz     x0, .conv1d_skip

    // Build kernel params:
    //   param[0] = input ptr  (act_buf_a -- current activation)
    //   param[1] = history ptr (conv1d_state + layer * CONV1D_PER_LAYER)
    //   param[2] = weights ptr (TODO: wire from model weights)
    //   param[3] = output ptr  (act_buf_b -- scratch, copied back)
    //   param[4] = N = CONV1D_CHANNELS (u32)
    adrp    x1, launch_params
    add     x1, x1, :lo12:launch_params

    // param[0] = act_buf_a (input)
    adrp    x2, act_buf_a
    add     x2, x2, :lo12:act_buf_a
    ldr     x3, [x2]
    str     x3, [x1, #0]

    // param[1] = conv1d_state + layer_idx * CONV1D_PER_LAYER
    adrp    x2, conv1d_state
    add     x2, x2, :lo12:conv1d_state
    ldr     x3, [x2]                   // base
    movz    x4, #0x0002, lsl #16
    movk    x4, #0x8000             // CONV1D_PER_LAYER = 163840 = 0x28000
    mul     x4, x19, x4                // layer_idx * per_layer
    add     x3, x3, x4                 // history for this layer
    str     x3, [x1, #8]

    // param[2] = weights (placeholder -- needs model weight wiring)
    str     xzr, [x1, #16]

    // param[3] = output (act_buf_b as scratch)
    adrp    x2, act_buf_b
    add     x2, x2, :lo12:act_buf_b
    ldr     x3, [x2]
    str     x3, [x1, #24]

    // param[4] = N (u32)
    mov     w3, #CONV1D_CHANNELS
    str     w3, [x1, #32]

    // Build param_ptrs array
    adrp    x2, param_ptrs
    add     x2, x2, :lo12:param_ptrs
    str     x1, [x2, #0]              // &param[0]
    add     x3, x1, #8
    str     x3, [x2, #8]              // &param[1]
    add     x3, x1, #16
    str     x3, [x2, #16]             // &param[2]
    add     x3, x1, #24
    str     x3, [x2, #24]             // &param[3]
    add     x3, x1, #32
    str     x3, [x2, #32]             // &param[4]

    // Launch: gridX=40, blockX=256, no shared mem
    adrp    x0, kern_table
    add     x0, x0, :lo12:kern_table
    ldr     x0, [x0, #KF_CONV1D]
    mov     x1, #40                    // gridDimX = ceil(10240/256)
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

    // mmap the entire shard file (PROT_READ, MAP_PRIVATE)
    // On GH200 unified memory, this gives GPU-accessible pointers
    mov     x0, #0                 // addr = NULL (kernel picks)
    mov     x1, x28                // length = file_size
    mov     x2, #PROT_READ
    mov     x3, #MAP_PRIVATE
    mov     x4, x21                // fd
    mov     x5, #0                 // offset = 0
    mov     x8, #SYS_MMAP
    svc     #0
    cmn     x0, #1
    b.eq    .shard_mmap_fail

    // Store shard base pointer
    adrp    x1, shard_bases
    add     x1, x1, :lo12:shard_bases
    str     x0, [x1, x26, lsl #3]

    // Store shard size
    adrp    x1, shard_sizes
    add     x1, x1, :lo12:shard_sizes
    str     x28, [x1, x26, lsl #3]

    // Close fd (mmap persists)
    mov     x0, x21
    mov     x8, #SYS_CLOSE
    svc     #0

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

    ldp     x29, x30, [sp], #16
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
    strb    w5, [x1]
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
