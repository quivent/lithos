// LITHOS CLI — GPU compute language toolkit (Rust implementation)

use clap::{Parser, Subcommand};

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

struct Kernel {
    key: &'static str,
    name: &'static str,
    summary: &'static str,
    fusions: &'static [&'static str],
    per_layer: u32,
    pct_compute: &'static str,
    inputs: &'static str,
    outputs: &'static str,
    description: &'static str,
    what_it_looks_like: &'static str,
    ptx_instructions: &'static [&'static str],
}

struct FusedKernel {
    key: &'static str,
    name: &'static str,
    summary: &'static str,
    components: &'static [&'static str],
    description: &'static str,
}

// ---------------------------------------------------------------------------
// Kernel database — compile-time constants
// ---------------------------------------------------------------------------

static KERNELS: &[Kernel] = &[
    Kernel {
        key: "projection",
        name: "PROJECTION",
        summary: "Multiply a vector by a weight matrix",
        fusions: &["dequantize"],
        per_layer: 7,
        pct_compute: "~90%",
        inputs: "activation vector (hidden_dim), weight matrix (hidden_dim x out_dim), [scale, zero_point for quantized]",
        outputs: "result vector (out_dim)",
        description: "\
The core operation of inference. A vector of hidden_dim numbers goes in,
the kernel reads a weight matrix of hidden_dim x out_dim, multiplies,
a vector of out_dim numbers comes out.

Every attention layer does this 4 times:
  Q projection: (5120) -> (24 heads x 256 head_dim) = (6144)
  K projection: (5120) -> (4 kv_heads x 256) = (1024)
  V projection: (5120) -> (4 kv_heads x 256) = (1024)
  O projection: (6144) -> (5120)

Every MLP layer does this 3 times:
  Gate projection: (5120) -> (17408)
  Up projection:   (5120) -> (17408)
  Down projection:  (17408) -> (5120)

7 projections x 64 layers = 448 per forward pass.
This single kernel type accounts for ~90% of all compute and memory bandwidth.",
        what_it_looks_like: "\
32 warps cooperate to tile the matrix multiplication.
Each warp:
  1. Loads a tile of the weight matrix into shared memory (async copy, no register use)
  2. Loads the corresponding input vector chunk into registers
  3. Feeds both to tensor cores (mma.sync.m16n8k16)
  4. Accumulates partial results in f32 registers
  5. Repeats until the full K dimension is consumed
  6. Writes the output tile back to HBM

For W4A16 quantized weights: dequantization is fused.
  - Load packed u32 (8 x 4-bit weights)
  - Extract nibbles via shift+mask
  - Multiply by per-group scale factor
  - Feed dequantized f16 values directly to tensor cores
  - Dequantized values never touch HBM -- registers only

Memory traffic: read each weight once, read input once, write output once.
At batch=1 this is entirely memory-bandwidth-bound.",
        ptx_instructions: &[
            "cp.async.cg.shared.global  -- async weight tile load",
            "ldmatrix.sync.aligned.x4   -- shared->register for tensor core",
            "mma.sync.aligned.m16n8k16  -- tensor core multiply-accumulate",
            "ld.global.v4.f16           -- input vector load (vectorized)",
            "st.global.v4.f32           -- output write",
            "bar.sync                   -- tile synchronization",
        ],
    },
    Kernel {
        key: "attention-score",
        name: "ATTENTION-SCORE",
        summary: "Compute attention: how relevant is each past token to the current one",
        fusions: &["rotate", "norm"],
        per_layer: 1,
        pct_compute: "~5% (decode), ~40% (long prefill)",
        inputs: "query vector (head_dim), KV cache (seq_len x head_dim x 2), attention mask",
        outputs: "context vector (head_dim) per head",
        description: "\
The \"looking back at context\" operation. For each attention head:

  1. Dot product the current query with every past key -> one score per past token
  2. Scale by 1/sqrt(head_dim)
  3. Softmax: exp(score - max) / sum(exp(score - max))
  4. Multiply each past value by its attention score
  5. Sum the weighted values -> output vector

For Qwen 3.5-27B:
  24 query heads, 4 KV heads (grouped query attention = 6:1 ratio)
  head_dim = 256
  At sequence length 4096: each head does 4096 dot products

This kernel scales with sequence length -- O(seq_len) per token during decode.
At short sequences it's cheap. At 100K+ context it dominates.",
        what_it_looks_like: "\
Flash attention pattern -- never materializes the full attention matrix:

  1. Load query tile into registers (one head's query, 256 elements)
  2. Stream through KV cache in tiles:
     a. Load K tile from HBM into shared memory
     b. Compute QK^T dot products in shared memory (warp-level)
     c. Track running max and running sum (online softmax)
     d. Load V tile from HBM into shared memory
     e. Multiply scores x V, accumulate partial output
  3. After all tiles: rescale output by final softmax denominator
  4. Write output vector back to HBM

The key insight: the full seq_len x seq_len attention matrix never exists.
Only one tile of scores lives in shared memory at a time.
Memory traffic: read KV cache once per head. No intermediate writes.

For GQA (grouped query attention): 6 query heads share 1 KV head.
The K and V tiles are loaded once and reused across 6 query heads.",
        ptx_instructions: &[
            "ld.global.v4.f16           -- KV cache tile load",
            "fma.rn.f32                 -- dot product accumulation",
            "shfl.sync.bfly.b32         -- warp reduction for max/sum",
            "ex2.approx.f32             -- exp for softmax",
            "rcp.approx.f32             -- 1/sum for softmax normalization",
            "bar.sync                   -- tile boundary sync",
        ],
    },
    Kernel {
        key: "norm",
        name: "NORM",
        summary: "RMSNorm -- normalize a vector by its root mean square",
        fusions: &["residual"],
        per_layer: 2,
        pct_compute: "<1%",
        inputs: "vector (hidden_dim), learned weight vector (hidden_dim), epsilon",
        outputs: "normalized vector (hidden_dim)",
        description: "\
RMSNorm: x_out = (x / RMS(x)) * weight
where RMS(x) = sqrt(mean(x^2) + epsilon)

Two per layer: one before attention, one before MLP.
2 x 64 layers = 128 per forward pass.

Tiny computation -- the vector is 5120 elements.
But it touches every element twice (read + write), so at batch=1
the memory traffic matters more than the math.",
        what_it_looks_like: "\
One warp handles the entire vector:

  1. Each thread loads multiple elements (5120 / 32 = 160 per thread)
  2. Each thread squares its elements and accumulates locally
  3. Warp shuffle butterfly reduction: sum of squares across all 32 lanes
  4. Lane 0 computes: rsqrt(sum / 5120 + epsilon)
  5. Broadcast scale factor to all lanes via warp shuffle
  6. Each thread multiplies its elements by scale * learned_weight
  7. Write back

5 instructions of real work per element. Memory-bound at batch=1.
Usually fused with RESIDUAL to avoid a separate read/write pass.",
        ptx_instructions: &[
            "ld.global.v4.f32           -- load input vector",
            "fma.rn.f32                 -- square and accumulate",
            "shfl.sync.bfly.b32         -- warp reduction (5 rounds)",
            "rsqrt.approx.f32           -- inverse RMS",
            "mul.f32                    -- scale * weight",
            "st.global.v4.f32           -- store normalized output",
        ],
    },
    Kernel {
        key: "activate",
        name: "ACTIVATE",
        summary: "SiLU activation: x * sigmoid(x)",
        fusions: &[],
        per_layer: 1,
        pct_compute: "<1%",
        inputs: "vector (intermediate_dim = 17408)",
        outputs: "activated vector (intermediate_dim)",
        description: "\
SiLU (Sigmoid Linear Unit): f(x) = x * sigma(x) = x / (1 + exp(-x))

Applied elementwise in the MLP between gate and down projections.
Also called \"swish\". Used by most modern LLMs (LLaMA, Qwen, Mistral).

In the MLP:
  gate = PROJECTION(input)        -> 17408
  up   = PROJECTION(input)        -> 17408
  hidden = SiLU(gate) * up        -> 17408    <- this kernel
  output = PROJECTION(hidden)     -> 5120",
        what_it_looks_like: "\
Pure elementwise -- every thread handles one or more elements independently:

  1. Load x
  2. neg.f32: compute -x
  3. ex2.approx.f32: compute 2^(-x) (approx e^(-x) with scale adjustment)
  4. add.f32: compute 1 + exp(-x)
  5. rcp.approx.f32: compute 1 / (1 + exp(-x)) = sigmoid(x)
  6. mul.f32: compute x * sigmoid(x)
  7. Store

6 instructions per element. Completely parallel. Memory-bound.
Often fused with the elementwise multiply with the up projection output.",
        ptx_instructions: &[
            "ld.global.f32              -- load input",
            "neg.f32                    -- negate",
            "ex2.approx.f32            -- fast exp via 2^x",
            "add.f32                    -- 1 + exp(-x)",
            "rcp.approx.f32            -- reciprocal",
            "mul.f32                    -- x * sigmoid(x)",
            "st.global.f32             -- store output",
        ],
    },
    Kernel {
        key: "rotate",
        name: "ROTATE",
        summary: "Rotary position embedding (RoPE) -- encode sequence position into Q and K",
        fusions: &[],
        per_layer: 1,
        pct_compute: "<1%",
        inputs: "Q vector, K vector, position index, precomputed sin/cos tables",
        outputs: "rotated Q vector, rotated K vector",
        description: "\
RoPE encodes absolute position by rotating pairs of elements:

  (x0, x1) -> (x0*cos(theta) - x1*sin(theta),  x0*sin(theta) + x1*cos(theta))

where theta depends on the element pair index and the token position.
This gives the model a sense of order -- token 5 gets different rotations than token 50.

Applied to Q and K vectors after projection, before attention scoring.
The rotations are precomputed into sin/cos tables indexed by position.

Qwen 3.5 uses partial rotary (25% of head_dim) with interleaved M-RoPE
for multimodal position encoding.",
        what_it_looks_like: "\
Each thread handles one (x0, x1) pair:

  1. Load x0, x1 from Q (or K)
  2. Load cos(theta), sin(theta) from precomputed table
  3. fma.rn.f32: x0*cos - x1*sin -> new x0
  4. fma.rn.f32: x0*sin + x1*cos -> new x1
  5. Store

4 FMA instructions per pair. Completely parallel.
Memory-bound -- the sin/cos table is small and cached in L1.",
        ptx_instructions: &[
            "ld.global.v2.f32           -- load element pair",
            "ld.const.v2.f32            -- load sin/cos from constant memory",
            "fma.rn.f32                 -- rotation (2 FMAs per pair)",
            "st.global.v2.f32           -- store rotated pair",
        ],
    },
    Kernel {
        key: "residual",
        name: "RESIDUAL",
        summary: "Skip connection -- add layer input to layer output",
        fusions: &["norm"],
        per_layer: 2,
        pct_compute: "<0.1%",
        inputs: "input vector (hidden_dim), output vector (hidden_dim)",
        outputs: "sum vector (hidden_dim)",
        description: "\
residual = input + output

That's it. One addition per element. Two per layer (after attention, after MLP).
128 per forward pass.

The simplest kernel. Almost always fused with NORM to avoid a
separate memory round-trip. The fused RESIDUAL+NORM reads input and
output, adds them, normalizes the sum, and writes once.",
        what_it_looks_like: "\
  1. Load input element
  2. Load output element
  3. add.f32
  4. Store

One instruction of real work. Memory traffic dominates.
Never launched alone -- always fused.",
        ptx_instructions: &[
            "ld.global.f32              -- load input",
            "ld.global.f32              -- load output",
            "add.f32                    -- add",
            "st.global.f32             -- store sum",
        ],
    },
    Kernel {
        key: "dequantize",
        name: "DEQUANTIZE",
        summary: "Convert packed 4-bit weights to f16 for computation",
        fusions: &["projection"],
        per_layer: 7,
        pct_compute: "fused into PROJECTION",
        inputs: "packed u32 (8 x 4-bit weights), scale (f16), zero_point (u4)",
        outputs: "8 x f16 dequantized weights",
        description: "\
W4A16 quantization: weights stored as 4-bit integers with per-group
scale factors. A group is typically 32 or 128 weights sharing one scale.

One u32 holds 8 weights: bits [3:0], [7:4], [11:8], ..., [31:28].

Dequantize: weight_f16 = (nibble - zero_point) * scale

This is ALWAYS fused into PROJECTION. The dequantized values exist
only in registers -- they go from packed integer to f16 to tensor core
input without ever touching memory.

For Qwen 3.5-27B W4A16: 16B parameters x 0.5 bytes = ~8GB total weights.
Without quantization it would be ~32GB (f16) or ~54GB (f32).",
        what_it_looks_like: "\
Inside the PROJECTION kernel's inner loop:

  1. Load one u32 from weight matrix (contains 8 weights)
  2. Load one f16 scale for this group
  3. For each of 8 nibbles:
     a. shr.b32 + and.b32: extract 4-bit value
     b. sub: subtract zero_point
     c. cvt.f16.s32: convert to f16
     d. mul.f16: multiply by scale
  4. Pack into f16x2 pairs for tensor core input
  5. Feed to mma.sync

Never a standalone kernel. Never writes to memory.
The dequantized values exist for microseconds in registers.",
        ptx_instructions: &[
            "ld.global.b32              -- load packed weights",
            "shr.b32 + and.b32          -- extract 4-bit nibble",
            "cvt.rn.f16.s32             -- int to f16",
            "mul.rn.f16                 -- scale",
            "// feeds directly to mma.sync",
        ],
    },
    Kernel {
        key: "sample",
        name: "SAMPLE",
        summary: "Convert logits to a token -- temperature, top-k, top-p, random selection",
        fusions: &[],
        per_layer: 0,
        pct_compute: "<0.1%",
        inputs: "logit vector (vocab_size = 248320), temperature, top_k, top_p",
        outputs: "one token ID",
        description: "\
The final step. The model produces 248,320 logits -- one score per
vocabulary token. SAMPLE turns this into a single token ID.

  1. Divide all logits by temperature (higher = more random)
  2. Find top-k largest logits (partial sort)
  3. Softmax over the top-k candidates
  4. Cumulative sum for top-p (nucleus) filtering
  5. Cut off where cumsum exceeds top_p threshold
  6. Random sample from remaining candidates

Runs once per generated token. Not per layer.
Can run entirely on CPU -- the vocabulary scan is one pass over 248K floats
(~1MB at f32), which fits in Grace's L2 cache.

Running on CPU frees the GPU to start the next token's prefill immediately.",
        what_it_looks_like: "\
On CPU (72 Grace cores):
  1. Find max logit (single pass, vectorized with NEON/SVE2)
  2. Subtract max, exponentiate, sum (fused single pass)
  3. Partial sort for top-k (std::partial_sort or radix select)
  4. Cumulative sum for top-p filtering
  5. Random number -> binary search -> token ID

On GPU (if needed for very large vocab):
  Block-level parallel reduction for max.
  Warp-level top-k via tournament sort.
  Single-pass softmax + cumsum.

Total: ~50us on CPU, ~10us on GPU. Negligible either way.",
        ptx_instructions: &[
            "// Typically runs on CPU",
            "// GPU variant would use:",
            "shfl.sync.bfly.b32         -- warp reduction for max",
            "ex2.approx.f32            -- softmax exp",
            "atom.global.add.f32        -- parallel prefix sum",
        ],
    },
    Kernel {
        key: "embed",
        name: "EMBED",
        summary: "Look up token ID in the embedding table",
        fusions: &[],
        per_layer: 0,
        pct_compute: "<0.01%",
        inputs: "token ID (integer), embedding table (vocab_size x hidden_dim)",
        outputs: "embedding vector (hidden_dim = 5120)",
        description: "\
Token 4217 -> row 4217 of the embedding table -> a vector of 5120 numbers.

One indexed memory read. The embedding table is 248320 x 5120 x 2 bytes
(f16) = ~2.4GB. But each lookup reads only one row = 10KB.

Runs once at the start of each forward pass (decode) or once per prompt
token (prefill).",
        what_it_looks_like: "\
  1. Compute offset: token_id * hidden_dim * sizeof(f16)
  2. Add to embedding table base address
  3. Load 5120 x f16 = 10KB (vectorized, coalesced)
  4. Done

One memory transaction. Cached in L2 for frequent tokens.",
        ptx_instructions: &[
            "mad.lo.u64                 -- address calculation",
            "ld.global.v4.f16           -- vectorized embedding load",
        ],
    },
];

static FUSED_KERNELS: &[FusedKernel] = &[
    FusedKernel {
        key: "attention",
        name: "ATTENTION (fused)",
        summary: "Full attention layer: NORM + ROTATE + PROJECTION(Q,K,V) + ATTENTION-SCORE + PROJECTION(O) + RESIDUAL",
        components: &["norm", "projection", "rotate", "attention-score", "residual"],
        description: "One kernel replaces 6+ separate kernels. Intermediate activations stay in registers/shared memory.",
    },
    FusedKernel {
        key: "mlp",
        name: "MLP (fused)",
        summary: "Full MLP layer: NORM + PROJECTION(gate) + PROJECTION(up) + ACTIVATE + PROJECTION(down) + RESIDUAL",
        components: &["norm", "projection", "activate", "residual"],
        description: "One kernel replaces 5+ separate kernels. Gate and up projections can share the input read.",
    },
    FusedKernel {
        key: "decoder",
        name: "DECODER (fused)",
        summary: "Full transformer decoder layer: ATTENTION + MLP",
        components: &["attention", "mlp"],
        description: "The ultimate fusion -- one kernel per layer. Only reads weights and KV cache from HBM, only writes updated KV cache and output activation.",
    },
];

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "lithos", about = "lithos -- GPU compute language toolkit")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Kernel library commands
    Kernels {
        #[command(subcommand)]
        action: KernelAction,
    },
    /// Show GPU status and loaded models
    Status,
}

#[derive(Subcommand)]
enum KernelAction {
    /// List all kernels
    List,
    /// Detailed kernel description
    Inspect {
        /// Kernel name
        name: String,
    },
    /// Same as inspect
    Explain {
        /// Kernel name
        name: String,
    },
    /// Same as inspect
    Read {
        /// Kernel name
        name: String,
    },
}

// ---------------------------------------------------------------------------
// Lookup helpers
// ---------------------------------------------------------------------------

fn find_kernel(name: &str) -> Option<&'static Kernel> {
    let key = name.to_lowercase();
    KERNELS.iter().find(|k| {
        k.key == key
            || k.name.to_lowercase() == key
            || k.name.to_lowercase().replace('-', "") == key.replace('-', "")
    })
}

fn find_fused(name: &str) -> Option<&'static FusedKernel> {
    let key = name.to_lowercase();
    FUSED_KERNELS.iter().find(|f| f.key == key)
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

fn cmd_list() {
    println!("LITHOS KERNEL LIBRARY");
    println!("{}", "=".repeat(60));
    println!();
    println!("PRIMITIVE KERNELS");
    println!("{}", "-".repeat(60));
    for k in KERNELS.iter() {
        let fuse_note = if k.fusions.is_empty() {
            String::new()
        } else {
            format!("  [fuses with: {}]", k.fusions.join(", "))
        };
        println!("  {:<20} {}{}", k.name, k.summary, fuse_note);
    }
    println!();
    println!("FUSED KERNELS (Lithos targets)");
    println!("{}", "-".repeat(60));
    for f in FUSED_KERNELS.iter() {
        println!("  {:<20} {}", f.name, f.summary);
    }
    println!();
    println!(
        "Total: {} primitive, {} fused",
        KERNELS.len(),
        FUSED_KERNELS.len()
    );
}

fn cmd_inspect(name: &str) {
    // Check fused kernels first
    if let Some(f) = find_fused(name) {
        println!();
        println!("{}", "=".repeat(60));
        println!("  {}", f.name);
        println!("{}", "=".repeat(60));
        println!();
        println!("  {}", f.summary);
        println!();
        println!("  Components: {}", f.components.join(", "));
        println!();
        println!("{}", f.description);
        println!();
        println!("  Use 'lithos kernels inspect <component>' to see each part.");
        return;
    }

    // Check primitive kernels
    if let Some(k) = find_kernel(name) {
        println!();
        println!("{}", "=".repeat(60));
        println!("  {}", k.name);
        println!("{}", "=".repeat(60));
        println!();
        println!("  {}", k.summary);
        println!(
            "  Per layer: {}    Compute share: {}",
            k.per_layer, k.pct_compute
        );
        if !k.fusions.is_empty() {
            println!("  Fuses with: {}", k.fusions.join(", "));
        }
        println!();
        println!("  Inputs:  {}", k.inputs);
        println!("  Outputs: {}", k.outputs);
        println!();
        println!("DESCRIPTION");
        println!("{}", k.description);
        println!();
        println!("WHAT IT LOOKS LIKE ON THE GPU");
        println!("{}", k.what_it_looks_like);
        println!();
        println!("PTX INSTRUCTIONS USED");
        for inst in k.ptx_instructions {
            println!("  {}", inst);
        }
        println!();
        return;
    }

    // Not found
    eprintln!("Unknown kernel: {}", name);
    let all_keys: Vec<&str> = KERNELS
        .iter()
        .map(|k| k.key)
        .chain(FUSED_KERNELS.iter().map(|f| f.key))
        .collect();
    eprintln!("Available: {}", all_keys.join(", "));
    std::process::exit(1);
}

fn cmd_status() {
    println!("lithos status");
    println!();
    println!("  GPU:    not connected");
    println!("  Model:  none loaded");
    println!();
    println!("Use 'lithos kernels list' to see available kernel definitions.");
}

fn cmd_help() {
    println!("lithos -- GPU compute language toolkit");
    println!();
    println!("COMMANDS");
    println!("  lithos kernels list                 List all kernels");
    println!("  lithos kernels inspect <name>       Detailed kernel description");
    println!("  lithos kernels explain <name>       Same as inspect");
    println!("  lithos kernels read <name>          Same as inspect");
    println!("  lithos status                       Show GPU status, loaded models");
    println!();
    println!("KERNELS");
    for k in KERNELS.iter() {
        println!("  {:<20} {}", k.key, k.name);
    }
    println!();
    println!("FUSED KERNELS");
    for f in FUSED_KERNELS.iter() {
        println!("  {:<20} {}", f.key, f.name);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    match cli.command {
        None => cmd_help(),
        Some(Commands::Status) => cmd_status(),
        Some(Commands::Kernels { action }) => match action {
            KernelAction::List => cmd_list(),
            KernelAction::Inspect { name } => cmd_inspect(&name),
            KernelAction::Explain { name } => cmd_inspect(&name),
            KernelAction::Read { name } => cmd_inspect(&name),
        },
    }
}
