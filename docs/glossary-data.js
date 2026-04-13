// Lithos Glossary Data — shared across all documentation pages
var LITHOS_GLOSSARY = {
  "ATen": "PyTorch's tensor library, written in C++. The layer below Python. Every tensor operation (add, mul, matmul) hits ATen before reaching CUDA.",
  "Autograd": "PyTorch's automatic differentiation machinery. Tracks operations on tensors to compute gradients for training. Disabled during inference (torch.no_grad or inference_mode) but the checking for whether it's disabled runs on every operation.",
  "AWQ": "Activation-aware Weight Quantization. A quantization method that preserves important weights by observing activation distributions. Used alongside GPTQ as a technique to shrink model size for faster inference.",
  "Bank Conflict": "When multiple threads in a warp access different addresses in the same shared memory bank simultaneously. The accesses serialize, reducing throughput. Shared memory has 32 banks of 4 bytes each; conflict-free access requires each thread to hit a different bank.",
  "BF16": "Brain Floating Point 16. A 16-bit floating point format with the same 8-bit exponent as FP32 but only 7 bits of mantissa. Trades precision for range, making it well-suited for deep learning where dynamic range matters more than exact precision.",
  "Coalescing": "Combining memory accesses from threads in a warp into fewer, larger memory transactions. When 32 threads read 32 consecutive bytes aligned to a 32-byte sector boundary, the hardware serves it in one transaction. Misalignment wastes bandwidth proportionally.",
  "Conv1D": "A 1D causal convolution applied before the DeltaNet recurrence in Qwen 3.5. Uses a kernel size of 4 to give the recurrence short-range local context. A simple FIR filter, trivially parallel across channels.",
  "CTA": "Cooperative Thread Array. NVIDIA's internal name for a thread block — a group of threads that can share shared memory and synchronize via barriers. Maps to CUDA's threadBlock concept.",
  "CUDA": "Compute Unified Device Architecture. NVIDIA's parallel computing platform and programming model for GPU computing. Provides APIs for launching kernels, managing memory, and synchronizing execution.",
  "CUDA Core": "A scalar floating-point execution unit on an NVIDIA GPU. Each Hopper SM has 128 FP32 CUDA cores. They handle general arithmetic; for matrix math, tensor cores are far more efficient.",
  "cuBLAS": "NVIDIA's GPU-accelerated Basic Linear Algebra Subroutines library. Provides optimized matrix multiplication (GEMM) kernels selected from a pre-compiled atlas. The atlas is ~800 MB because it contains kernels for every possible matrix shape.",
  "cubin": "A compiled GPU binary in ELF format containing SASS machine code. Lithos generates cubins either by compiling PTX through ptxas or (in future) by emitting SASS directly into the ELF structure. A Lithos cubin is ~4 KB; the cuBLAS atlas is ~800 MB.",
  "CUDA graph": "A recorded sequence of kernel launches that can be replayed as a single operation. Reduces per-kernel launch overhead from 2-5\u00b5s to ~0.3-1\u00b5s amortized. Used by vLLM to amortize the cost of thousands of small kernel launches. Lithos doesn't need graphs because it fuses kernels at the language level — one large kernel instead of many small ones.",
  "cuLaunchKernel": "CUDA driver API function that launches a kernel on the GPU. Takes a function handle, grid/block dimensions, and parameter pointers. The lowest-level launch API — TensorRT, vLLM, PyTorch all eventually call this or cudaLaunchKernel (which wraps it).",
  "cudaMalloc": "CUDA runtime API for allocating GPU memory. 10-500 μs per call. Goes through ioctl to the NVIDIA kernel driver. What PyTorch's CUDACachingAllocator wraps to avoid calling directly.",
  "cudaMemcpy": "CUDA runtime API for copying memory between CPU and GPU. Required on discrete GPUs. Unnecessary on GH200 with unified memory (which is why Lithos uses mmap).",
  "CUDACachingAllocator": "PyTorch's GPU memory manager. Maintains pools of free blocks to avoid the 10-500\u00b5s cost of cudaMalloc on every allocation. Adds bookkeeping overhead on every tensor operation. Lithos replaces this with mmap (0 overhead, OS handles page management).",
  "cuDNN": "NVIDIA's Deep Neural Network library. Provides optimized primitives for convolutions, normalization, attention, and activations. Hundreds of MB of pre-compiled kernels. Lithos replaces it with purpose-built fused kernels.",
  "Cycle": "One GPU clock tick. On Hopper at 1.83 GHz, a cycle is ~0.55 nanoseconds. All hardware latencies on this site — memory access, instruction issue, tensor core throughput, SASS stall counts — are measured in cycles, not instructions or wall-clock time. Reference: 1 cycle ≈ 0.5 ns (register), 23 cycles ≈ 12 ns (shared memory), 33 cycles ≈ 18 ns (L1), 200 cycles ≈ 110 ns (L2), 400 cycles ≈ 220 ns (HBM3e). Each of an SM's four sub-partitions can issue one instruction per cycle, so a 400-cycle HBM stall costs ~400 forgone issue slots per sub-partition — the budget latency hiding spends.",
  "DeltaNet": "A linear attention mechanism with recurrent state, used in 48 of 64 layers in Qwen 3.5. Maintains a state matrix S that is updated per token in O(1) time, unlike full attention which costs O(seq_len). The state requires FP32 precision to avoid catastrophic drift.",
  "FlashAttention": "A tiled attention kernel implementation by Tri Dao that avoids the O(N\u00b2) memory cost of naive attention by computing attention in blocks that fit in shared memory, using online softmax. Used by vLLM, SGLang, TensorRT-LLM. Not an inference engine — just the attention kernel. Build fragility is a persistent issue: compilation depends on specific CUDA toolkit, PyTorch, and GPU architecture versions. When it fails (common), the fallback is naive attention with O(N\u00b2) memory and dramatically slower performance.",
  "FLOP": "Floating Point Operation. One multiply or one add. An FMA (fused multiply-add) counts as 2 FLOPs because it does both in one instruction. FLOPs are the unit for measuring compute work and GPU throughput.",
  "FLOPs/clock": "Floating point operations completed per GPU clock cycle. Per tensor core on Hopper at FP16: 1,024 FLOPs/clock effective throughput. Per SM (4 tensor cores): ~1,024 effective. Per GPU (132 SMs \u00d7 1.83 GHz): ~247 TFLOPS effective. Marketing numbers often cite higher values by counting differently.",
  "FMA": "Fused Multiply-Add. One instruction that computes a \u00d7 b + c with a single rounding step. Counts as 2 FLOPs. The single most important instruction for GEMM kernels — every dot product is a chain of FMAs.",
  "Forth": "A stack-based programming language invented by Charles Moore. Lithos is Forth-derived: programs are sequences of words that manipulate a data stack. Each Lithos word emits PTX or SASS instructions. Compilation is string concatenation, taking microseconds.",
  "FP8": "8-bit floating point, available in E4M3 and E5M2 formats on Hopper. Doubles tensor core throughput compared to FP16. Used for inference where reduced precision is acceptable. Hopper delivers ~1979 TFLOPS at FP8.",
  "FP16": "IEEE 754 half-precision floating point (16 bits). The standard precision for inference compute. Hopper tensor cores deliver ~990 TFLOPS at FP16. Insufficient for DeltaNet state accumulation due to limited dynamic range.",
  "FP32": "IEEE 754 single-precision floating point (32 bits). Required for DeltaNet state matrices to prevent drift. Used as accumulator precision in mixed-precision kernels where FP16/BF16 handles the throughput path.",
  "GH200": "NVIDIA's Grace Hopper Superchip. Combines a Grace ARM64 CPU with a Hopper GPU connected via NVLink-C2C at 900 GB/s. Features hardware-coherent unified memory across 96 GB HBM3e and 480 GB LPDDR5X. The primary target for Lithos.",
  "GPTQ": "Post-training quantization method that uses approximate second-order information to minimize quantization error. Compresses model weights (e.g., to 4-bit) with calibration data. Enables running larger models on limited GPU memory.",
  "GQA": "Grouped Query Attention. An attention variant where multiple query heads share a single key-value head, reducing KV cache size. Qwen 3.5 uses GQA in its 16 full attention layers, with fewer KV heads than query heads.",
  "Grace": "NVIDIA's ARM64 (Neoverse V2) CPU, the CPU component of the GH200 superchip. Has 72 cores and is paired with 480 GB of LPDDR5X memory. Connected to the Hopper GPU via NVLink-C2C. Lithos runs sampling and orchestration on Grace.",
  "Grid": "The complete set of thread blocks launched for a single kernel invocation. The grid dimensions determine how many thread blocks execute in total. For Lithos, grid size is calculated from model dimensions at compile time.",
  "HBM": "High Bandwidth Memory. Stacked DRAM chips connected to the GPU via a wide silicon interposer bus. GH200 has 6 HBM3e stacks providing 96 GB at 4 TB/s total bandwidth. This is where weights, activations, and hot KV cache reside.",
  "HBM3e": "The latest generation of High Bandwidth Memory, used in the GH200. Six stacks provide 96 GB capacity at 4 TB/s aggregate bandwidth (~667 GB/s per stack). Access latency ~400 GPU cycles (~220 ns at Hopper's 1.83 GHz) end-to-end from ld.global issue to data in a register. The fastest memory tier available to Lithos kernels.",
  "Hopper": "NVIDIA's GPU architecture (sm_90), the compute die in the GH200. Features 132 active SMs, 4th-generation tensor cores, TMA (Tensor Memory Accelerator), and 50 MB L2 cache. Every Lithos kernel targets Hopper specifically.",
  "Kernel": "A function that runs on the GPU, executed by thousands of threads in parallel. In Lithos, kernels are compiled from Forth words into PTX or SASS. A fused kernel combines multiple operations (e.g., norm + projection + activation) into a single launch.",
  "KV Cache": "Key-Value Cache. Stored key and value vectors from all previous tokens, used by attention layers to score against the current token. Grows linearly with sequence length. On GH200, the KV cache can span HBM3e (hot/recent) and LPDDR5X (cold/old).",
  "L1 Cache": "Per-SM on-chip cache, sharing 256 KB with shared memory on Hopper. Access latency ~33 GPU cycles (~18 ns at 1.83 GHz). Automatically caches global memory accesses. Configurable: up to 228 KB can be allocated as shared memory.",
  "L2 Cache": "GPU-wide shared cache. 50 MB on GH200, split across 12 partitions with 128-byte cache lines. Sits between the SMs and HBM3e. Latency ~200 GPU cycles (~110 ns at 1.83 GHz). The full DeltaNet state checkpoint (12.3 MB) fits in L2.",
  "Latency Hiding": "The GPU technique of switching between warps while one waits for memory. When a warp stalls on an HBM load (~400 GPU cycles, ~220 ns at Hopper's 1.83 GHz), the scheduler runs another ready warp. Cycle-budget view: while one warp waits 400 cycles, each of the SM's four sub-partitions could issue one instruction per cycle — so the SM could have issued ~400 other instructions per sub-partition in that window. Filling the SM with many warps lets it keep issuing useful work during memory stalls instead of sitting idle. More concurrent warps means more opportunities to hide latency, but requires fewer registers per thread.",
  "Lithos": "A Forth-derived GPU compute language that emits PTX and SASS for NVIDIA GPUs. Built for inference on GH200. Replaces PyTorch, cuBLAS, cuDNN, and the CUDA runtime with direct kernel compilation in microseconds. Named from lithography — writing on silicon.",
  "LPDDR5X": "Low-Power Double Data Rate 5X memory. The CPU-side memory on GH200, providing 480 GB at ~500 GB/s. Connected to HBM3e via NVLink-C2C. Used for cold KV cache, weight overflow, and OS workspace. Accessible by GPU kernels through unified addressing.",
  "mma.sync.m16n8k16": "The workhorse tensor core instruction for FP16 inference. One warp-wide instruction that multiplies a 16\u00d716 tile of matrix A by a 16\u00d78 tile of matrix B (both FP16), and adds to a 16\u00d78 accumulator (FP32). Executes 4,096 FLOPs in one instruction. Every PROJECTION kernel's inner loop is built around keeping this instruction fed with data. The entire inference engine's performance depends on this single instruction.",
  "mmap": "Memory-mapped file I/O. Maps files directly into the process virtual address space. On GH200, mmap'd memory is accessible to both CPU and GPU via unified addressing. Lithos uses mmap to load safetensors weights (lazy, fault-on-access) and to allocate KV cache and activations (eager, pre-touched).",
  "MTP": "Multi-Token Prediction. A speculative decoding technique where the model predicts multiple future tokens at once. In Lithos, the DeltaNet-only forward pass (skipping attention layers) serves as a fast draft model for self-speculative decoding at ~3x speed.",
  "nn.Module": "PyTorch's base class for neural network layers. Holds parameters (weights), sub-modules, and forward logic. Used for model definitions.",
  "NVLink-C2C": "Chip-to-Chip NVLink interconnect on the GH200, connecting the Grace CPU to the Hopper GPU at 900 GB/s bidirectional. Provides hardware-coherent unified memory — both processors see the same address space with no explicit transfers needed.",
  "Occupancy": "The ratio of active warps to the maximum possible warps on an SM. Higher occupancy provides more warps for latency hiding. Lower occupancy allows more registers per thread. The optimal balance depends on whether a kernel is memory-bound or compute-bound.",
  "Online Softmax": "Computing softmax in a streaming fashion as tiles of the attention matrix are processed, tracking running max and running sum. Enables tiled attention without materializing the full matrix. The key mathematical trick that makes FlashAttention possible.",
  "PyTorch dispatcher": "The C++ mechanism that routes tensor operations through \"dispatch keys\" (Autograd, CUDA, CPU, etc.) to the right backend implementation. 1-3μs overhead per operation.",
  "PTX": "Parallel Thread Execution. NVIDIA's virtual assembly language. Stable text ISA. Gets compiled to SASS by the driver (ptxas) at load time. This is the layer Lithos emits directly, bypassing CUDA C++ compilation.",
  "ptxas": "NVIDIA's PTX assembler. Compiles PTX to SASS. Runs inside the driver when you call cuModuleLoadData with PTX. Typical compile time: 100ms-2s per kernel. The compilation Lithos will eventually eliminate by emitting SASS directly.",
  "Register reuse flags": "Fields in the SASS control word (bits [63:58]) that hint which source registers will be reused by the next instruction. Helps the hardware avoid register bank conflicts. Set automatically by ptxas, can be set manually in direct SASS emission.",
  "Register File": "Per-thread fastest storage on the GPU. Each Hopper SM has 65,536 32-bit registers (256 KB). Access takes 1 GPU cycle (~0.5 ns at 1.83 GHz). Each thread can use up to 255 registers. More registers per thread means fewer concurrent threads, creating the fundamental GPU optimization tradeoff.",
  "RMSNorm": "Root Mean Square Normalization. A simplification of LayerNorm that normalizes by the RMS of activations: x * rsqrt(mean(x^2) + eps) * weight. Used throughout Qwen 3.5. Lithos fuses RMSNorm with residual addition into a single kernel using warp shuffles for the reduction.",
  "RoPE": "Rotary Position Embedding. Encodes token position by rotating query and key vectors in pairs of dimensions. Only needed for the 16 full attention layers in Qwen 3.5; DeltaNet layers encode position implicitly through their sequential state updates.",
  "scaled_dot_product_attention": "PyTorch's built-in attention function (torch.nn.functional.scaled_dot_product_attention). Added in PyTorch 2.0 as a one-call API that internally dispatches to FlashAttention, memory-efficient attention, or naive attention depending on inputs and hardware. The official \"you don't have to write your own attention anymore\" API. Still opaque — you don't control what kernel runs, you don't control fusion with surrounding operations, you can't see the PTX it produces.",
  "Safetensors": "A weight file format by Hugging Face. Simple structure: a JSON header describing tensor names, shapes, and dtypes, followed by flat binary data. Lithos mmap's safetensors files directly — pointer arithmetic gives access to any tensor with zero deserialization overhead.",
  "SASS": "Shader ASSembly. The actual GPU machine code that runs on NVIDIA hardware. Each instruction is 16 bytes: an 8-byte instruction word (opcode, operands) and an 8-byte control word (stall counts, barriers, register reuse). Lithos aims to emit SASS directly, bypassing ptxas.",
  "Scoreboard": "The hardware structure that tracks which registers have pending writes. The warp scheduler uses it to determine which warps are ready to issue instructions. When the scoreboard blocks all warps, the SM stalls.",
  "Stall count": "A field in the SASS control word (bits [57:53]) that specifies the minimum number of cycles the scheduler must wait before issuing the next instruction from the same warp. Normally set by ptxas based on dependency analysis. Lithos SASS emitter can set these directly for fine-grained scheduling control.",
  "SFU": "Special Function Unit. Hardware unit on each SM that computes transcendental functions: sin, cos, reciprocal (rcp), reciprocal square root (rsqrt), ex2, lg2. Each Hopper SM has 16 SFUs. Used by RMSNorm (rsqrt) and SiLU (ex2) in Lithos kernels.",
  "Shared Memory": "Per-SM fast scratchpad memory, configurable up to 228 KB on Hopper. Access latency ~23 GPU cycles (~12 ns at 1.83 GHz). Organized in 32 banks of 4 bytes each. Used as a staging area between HBM and registers. Threads in a thread block can communicate through shared memory.",
  "SiLU": "Sigmoid Linear Unit. The activation function x * sigmoid(x), computed as x / (1 + exp(-x)). Used in every MLP layer of Qwen 3.5. Lithos computes it in 6 PTX instructions entirely in registers using ex2.approx, with no intermediate memory access.",
  "SM": "Streaming Multiprocessor. The fundamental compute unit of an NVIDIA GPU. Each Hopper SM contains 128 CUDA cores, 4 tensor cores, 256 KB register file, 256 KB configurable L1/shared memory, and 4 warp schedulers. The GH200 has 132 active SMs.",
  "Tensor Core": "Dedicated matrix-multiply-accumulate hardware unit on NVIDIA GPUs. Each Hopper SM has 4 tensor cores. At FP16, one mma.sync.m16n8k16 instruction delivers 1,024 FLOPs/clock/SM. The heavy lifter for projection and attention kernels.",
  "Tiled Attention": "The general algorithm. Compute attention by loading tiles of Q, K, V into shared memory, processing tile-by-tile, using online softmax to avoid materializing the full N\u00d7N attention matrix. FlashAttention is the most popular implementation. Key insight: attention is O(N\u00b2) in compute but can be O(N) in memory if you process it in tiles and never store the intermediate attention matrix.",
  "torch.nn.functional": "PyTorch's stateless function API (as opposed to torch.nn.Module classes). The layer operations: linear, relu, softmax, etc. F.linear, F.scaled_dot_product_attention, etc.",
  "torch.compile": "PyTorch's JIT compiler (introduced in 2.0, also called Inductor). Traces Python model code, generates fused CUDA kernels via Triton. Adds 30-60 seconds of warmup time at startup. Creates 2-10 GB of disk cache at ~/.cache/torch/inductor/ per model configuration. CPU RAM usage spikes 5-10 GB during compilation — OOM risk on machines where model fits but compilation overhead doesn't. Per-config: different batch size or sequence length regenerates the cache.",
  "Triton": "OpenAI's GPU kernel compiler. A Python-embedded DSL for writing GPU kernels. Used by torch.compile/Inductor to generate fused kernels. Simpler than CUDA C++ but still requires compilation (Triton IR \u2192 PTX \u2192 ptxas \u2192 SASS). Not an inference engine — a compiler for writing individual kernels.",
  "Thread Block": "A group of threads (up to 1,024 on Hopper) that execute on the same SM and share shared memory. Also called CTA (Cooperative Thread Array). Multiple thread blocks can run on one SM simultaneously. Lithos sizes thread blocks to match model dimensions.",
  "Unified Memory": "A memory model where CPU and GPU share the same virtual address space. On GH200, this is hardware-native via NVLink-C2C — not the software-managed CUDA Unified Memory that requires driver hints and page migration. mmap returns a pointer both processors can use.",
  "W4A16": "A quantization scheme: 4-bit weights with 16-bit activations. Reduces model memory from ~54 GB (FP16) to ~18 GB for Qwen 27B, while keeping activation precision for accurate computation. The standard quantization target for Lithos inference.",
  "Warp": "A group of 32 threads that execute in lockstep on an SM. The fundamental scheduling unit of the GPU. All threads in a warp execute the same instruction simultaneously. Warp-level operations like shuffles enable fast reductions without shared memory.",
  "Tokenizer": "The component that converts text into token IDs (integers). Each token maps to a word, subword, or character fragment. Qwen 3.5 uses a vocabulary of 248,320 tokens. The tokenizer runs on CPU before any GPU work begins.",
  "Vector": "An ordered list of numbers that represents something in the model's internal space. In Qwen 3.5, the hidden state vector is 5,120 numbers. Vectors get transformed by matrix multiplications and nonlinearities as they pass through layers.",
  "Attention": "The mechanism by which a token gathers information from all previous tokens. Computes a weighted sum of past tokens' value vectors, where the weights (attention scores) are determined by comparing the current token's query against each past token's key. Cost grows linearly with sequence length.",
  "MLP": "Multi-Layer Perceptron. A feed-forward network within each transformer layer that transforms each token's vector independently (no cross-token interaction). Consists of an expansion (gate + up projections), a nonlinearity (SiLU), and a compression (down projection).",
  "Nonlinearity": "A mathematical function applied between linear transformations (matrix multiplications) that allows a neural network to learn complex patterns. Without nonlinearities, stacking layers would be mathematically equivalent to a single layer. SiLU is the nonlinearity used in Qwen 3.5's MLP.",
  "Residual": "A skip connection that adds a layer's input directly to its output. Ensures that each layer only needs to learn a refinement (delta) rather than a complete re-representation. Critical for training deep networks — without residuals, gradients vanish over 64 layers.",
  "GEMM": "General Matrix Multiply. The core computational operation in transformer inference: multiplying a vector or matrix by a weight matrix. Every projection (Q, K, V, output, gate, up, down) is a GEMM. At batch=1 decode, these become GEMVs (matrix-vector multiplies), which are memory-bandwidth-bound."
};

// Tooltip system initialization
(function() {
  // Inject tooltip CSS
  var style = document.createElement('style');
  style.textContent = [
    '.term { color: inherit; border-bottom: 1px dotted rgba(212, 160, 83, 0.4); cursor: help; }',
    '.term:hover { border-bottom-color: rgba(212, 160, 83, 0.9); }',
    '.term-tooltip { position: absolute; background: #1a1a2e; border: 1px solid #d4a053; color: #e0e0e8; padding: 8px 12px; font-size: 13px; max-width: 300px; z-index: 1000; pointer-events: none; border-radius: 2px; line-height: 1.5; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }'
  ].join('\n');
  document.head.appendChild(style);

  // Tooltip element
  var tooltip = null;

  function showTooltip(e) {
    var term = e.target.getAttribute('data-term');
    var def = LITHOS_GLOSSARY[term];
    if (!def) return;
    if (!tooltip) {
      tooltip = document.createElement('div');
      tooltip.className = 'term-tooltip';
      document.body.appendChild(tooltip);
    }
    tooltip.textContent = def;
    tooltip.style.display = 'block';
    positionTooltip(e);
  }

  function positionTooltip(e) {
    if (!tooltip) return;
    var x = e.pageX + 12;
    var y = e.pageY - 40;
    if (x + 320 > document.documentElement.scrollWidth) {
      x = e.pageX - 320;
    }
    if (y < window.scrollY + 10) {
      y = e.pageY + 20;
    }
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
  }

  function hideTooltip() {
    if (tooltip) tooltip.style.display = 'none';
  }

  function clickTerm(e) {
    var term = e.target.getAttribute('data-term');
    if (term) {
      var anchor = term.toLowerCase().replace(/[\s\/\.]+/g, '-').replace(/[^a-z0-9\-]/g, '');
      window.location.href = 'glossary.html#term-' + anchor;
    }
  }

  document.addEventListener('mouseover', function(e) {
    if (e.target.classList && e.target.classList.contains('term')) showTooltip(e);
  });
  document.addEventListener('mousemove', function(e) {
    if (e.target.classList && e.target.classList.contains('term')) positionTooltip(e);
  });
  document.addEventListener('mouseout', function(e) {
    if (e.target.classList && e.target.classList.contains('term')) hideTooltip();
  });
  document.addEventListener('click', function(e) {
    if (e.target.classList && e.target.classList.contains('term')) clickTerm(e);
  });
})();
