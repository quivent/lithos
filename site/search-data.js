var SEARCH_INDEX = [
  {
    title: "Lithos — Writing on Silicon",
    url: "/",
    sections: ["What Lithos Is", "The Problems Lithos Solves", "Target Silicon — Hopper SM (GH200)", "Architecture", "Vocabulary", "SASS Encoding — Hopper Reverse Engineering", "Inference Engine", "Kernel Guide", "Quantitative Targets", "Current Status", "Code Examples"],
    terms: "lithos inference engine gpu cuda sass ptx hopper gh200 sm streaming multiprocessor tensor core register file shared memory kernel cubin warp thread block grid quantization fp16 bf16 fp8 int8 fp32 fma bandwidth hbm lpddr5x nvlink grace unified memory mmap safetensors forth compiler opcode encoding 16-byte instruction format register encoding emit-sass design rules lithos patterns token work per token engine decisions kernel structure memory model fusion 3-kernel target"
  },
  {
    title: "Architecture — From Model File to Generated Token",
    url: "/architecture",
    sections: ["The Full Pipeline", "Memory Map on GH200", "Kernel Compilation Pipeline", "One Token's Journey", "vLLM Stack vs Lithos Stack"],
    terms: "architecture pipeline model file token generation memory map gh200 kernel compilation weight loading inference path vllm stack comparison removed layers overhead latency startup cubin ptx sass safetensors mmap hbm lpddr5x l2 cache"
  },
  {
    title: "Cores — CUDA Core, Tensor Core, FP64 Core, SFU",
    url: "/arch-cores",
    sections: ["Cells in a Sub-Partition", "CUDA Core", "Tensor Core", "FP64 Core", "SFU — Special Function Unit", "LD/ST Unit", "What Each Is Used For"],
    terms: "cuda core tensor core fp64 sfu special function unit ld/st load store sub-partition warp scheduler register file execution units fp32 fma ffma fadd fmul iadd3 imad shf lop3 matrix multiply accumulate mma wgmma m16n8k16 fp16 bf16 fp8 int8 hopper ampere 990 tflops transcendental reciprocal rsqrt sine cosine exp2 log2 mufu softmax rmsnorm layernorm swiglu sigmoid rope inference decode step"
  },
  {
    title: "GPU — The GH200 From 30,000 Feet",
    url: "/arch-gpu",
    sections: ["The Chip", "132 SMs", "50 MB L2", "6 HBM3e Stacks", "NVLink-C2C", "Data Flow", "What Lithos Sees"],
    terms: "gh200 grace hopper superchip gpu 132 sm streaming multiprocessor l2 cache 50mb hbm3e 96gb 4tb/s nvlink-c2c 900gb/s coherent bandwidth lpddr5x 480gb neoverse v2 72 arm cores data flow memory hierarchy register l1 shared memory safetensors mmap cudamalloc cudamemcpy kv cache weight activation tensor gridDim block size tile size access pattern"
  },
  {
    title: "Kernel — Anatomy of a Cubin",
    url: "/arch-kernel",
    sections: ["What a Kernel Is", "The Cubin File", "Metadata — What the Driver Needs", "The SASS Inside", "cuLaunchKernel", "Life of a Block", "Grid and Block Dims"],
    terms: "kernel cubin elf sass metadata driver culaunchkernel thread block grid dims warp occupancy registers shared memory compute work distributor cwd block scheduling launch overhead ptx nvcc fatbinary .nv.info section"
  },
  {
    title: "Registers — Register File, Banks, Reuse, Occupancy",
    url: "/arch-registers",
    sections: ["The File", "Register Allocation", "Bank Conflicts", "Occupancy Tradeoff", "Fewer Threads More Registers", "Spilling", "What Lithos Controls"],
    terms: "register file 256kb 65536 entries bank conflicts occupancy tradeoff allocation spilling reuse flags predicate registers uniform registers barrier registers r0-r255 register pressure thread count warps per sm lithos control sass encoding stall counts yield hints"
  },
  {
    title: "Kernels — Beyond the Vanilla Transformer",
    url: "/kernels",
    sections: ["Why Custom Kernels Matter", "The Qwen 3.5-27B Architecture Map", "Lithos Kernel Types for Qwen 3.5", "DeltaNet Optimization Opportunities", "When to Create a Custom Kernel", "Ideas for Future Custom Kernels", "Building Lithos Around Qwen 3.5"],
    terms: "kernels custom qwen 3.5-27b deltanet hybrid attention recurrence transformer layer attention specific deltanet specific kernel fusion state precision optimization modal mtp self-speculative decoding recurrent rollback fused kernel target rmsnorm rope qkv projection mlp swiglu softmax kv cache"
  },
  {
    title: "Hardware-Matched Quantization",
    url: "/quantization",
    sections: ["The Premise", "Fitting a Layer On-Chip", "Utilizing Every Compute Unit", "What Has Not Been Tried", "Feasibility"],
    terms: "quantization hardware-matched on-chip fitting layer compute unit tensor core cuda core sfu ld/st pipeline balance format design per-layer adaptive precision fp16 bf16 fp8 int8 int4 sm shared memory register file 228kb 256kb hopper weight compression inference accuracy"
  },
  {
    title: "The Lithos Compiler — One Language Three Backends",
    url: "/compiler",
    sections: ["The Decision", "Three Backends", "Why Not Other Languages", "The Host Chain", "Language Surface", "File Organization"],
    terms: "compiler eighth e1 forth ptx backend sass backend arm64 backend gpu kernel host code backend routing language surface file organization ideological practical future self-hosting bootstrap"
  },
  {
    title: "Performance — Resources Techniques and Targets",
    url: "/performance",
    sections: ["The Three Resources That Determine Performance", "Existing Engines Optimizations and Limits", "Lithos Optimizations", "Single-Launch Architecture", "Quantitative Targets", "Measured Results", "What Lithos Cannot Do Yet"],
    terms: "performance registers memory bandwidth compute units single-launch architecture zero launch overhead zero intermediate hbm traffic quantitative targets measured results vllm comparison forward pass timing time breakdown startup"
  },
  {
    title: "Bandwidth Utilization — The Only Number That Matters at Batch=1",
    url: "/bandwidth",
    sections: ["Claim", "Defining Bandwidth Utilization Precisely", "The Arithmetic", "Reconciling 13% vs 35.8%", "Why vLLM Is Below the Ceiling", "How Lithos Closes the Gap", "The Single-Launch Double Win", "Empirical Measurement", "Summary"],
    terms: "bandwidth utilization batch=1 hbm sustained time-averaged bytes per token ceiling vllm gap single-launch zero launch overhead zero intermediate hbm traffic engineering constraint registers code size 2.8tb/s effective 4tb/s theoretical"
  },
  {
    title: "Multicore — The 72 Grace Cores Next to the GPU",
    url: "/multicore",
    sections: ["The Premise", "What the 72 Grace Cores Actually Are", "Memory Coherence NVLink-C2C", "The Pipeline Model", "Request-Level Parallelism", "Speculative Decoding CPU Drafts GPU Verifies", "Why Frameworks Can't Do This", "What This Buys in Practice"],
    terms: "multicore grace cpu 72 arm cores neoverse v2 nvlink-c2c memory coherence pipeline model request-level parallelism speculative decoding cpu drafts gpu verifies vllm frameworks lpddr5x unified memory tokenizer sampling kv cache management"
  },
  {
    title: "Inference Engine Landscape",
    url: "/comparison",
    sections: ["What Makes Inference Fast", "Engine Deep Dives", "Component Libraries", "Comparison Table"],
    terms: "inference engine comparison vllm tensorrt-llm sglang llamacpp exllamav2 mlc-llm mlx flashattention cublas pytorch landscape fast batch throughput latency memory"
  },
  {
    title: "TensorRT-LLM — Why It's the Fastest",
    url: "/tensorrt",
    sections: ["What TensorRT-LLM Actually Is", "The Build Phase", "The Optimizations", "The Runtime", "The Tradeoffs", "Benchmarks", "What Lithos Learns From TensorRT", "The Key Insight"],
    terms: "tensorrt-llm nvidia fastest build phase optimizations runtime tradeoffs benchmarks xqa kernel llama 70b fp8 h100 mlperf inference overhead complexity deployment graph optimization kernel fusion quantization"
  },
  {
    title: "Lithos vs PyTorch",
    url: "/pytorch",
    sections: ["What PyTorch Is", "What PyTorch Does During Inference", "What PyTorch Does That Is Waste", "PyTorch Inference Path", "Lithos Inference Path", "What Lithos Replaces", "What Lithos Keeps", "Memory Comparison", "Startup Comparison", "Summary"],
    terms: "pytorch training framework inference waste overhead autograd dispatch dynamic graph memory allocation cuda runtime cudnn cublas startup time memory comparison lithos replaces keeps eager mode torch.compile"
  },
  {
    title: "What Happens to a Token — One Token Through 64 Layers",
    url: "/token-journey",
    sections: ["Before the Layers", "One Full Attention Layer", "The MLP", "One DeltaNet Layer", "After All 64 Layers", "The Numbers"],
    terms: "token journey 64 layers qwen 3.5-27b normalize rmsnorm qkv projection rope rotary position embedding kv cache attention softmax output projection residual add mlp gate up swiglu activation down projection deltanet recurrent hybrid lm head sampling"
  },
  {
    title: "Language Design — Lithos Syntax Proposal",
    url: "/language-design",
    sections: ["Examples", "Design Decisions", "Vocabulary", "Status"],
    terms: "language design syntax proposal vector add rmsnorm tensor core tile assembly ptx mnemonics draft forth-derived gpu programming stack-based"
  },
  {
    title: "Glossary — Technical Terms Reference",
    url: "/glossary",
    sections: ["Glossary"],
    terms: "glossary technical terms reference definitions vocabulary cuda gpu sm warp thread block register tensor core sass ptx kernel cubin hbm lpddr5x nvlink fp16 bf16 fp8 fp32 int8 fma mma quantization inference attention kv cache activation softmax rmsnorm rope deltanet"
  },
  {
    title: "The Kernel Recipe — How config.json Becomes Cubins",
    url: "/recipe",
    sections: ["The Recipe Book Is config.json", "The Derivation Rules", "Total for Qwen 3.5-27B: 11 Kernels", "When Does This Happen?", "Different Model Different Recipe"],
    terms: "kernel recipe config.json cubin derivation rules qwen 3.5-27b 11 kernels model architecture hidden_size num_attention_heads num_key_value_heads intermediate_size sass emission ptx compilation"
  },
  {
    title: "Hopper SM Architecture",
    url: "/sm-architecture",
    sections: ["The Critical Distinction", "Full SM Overview", "Memory Hierarchy Scaled by Latency", "GPU-Wide View GH200", "One Sub-Partition in Detail", "Why This Matters for Lithos"],
    terms: "hopper sm streaming multiprocessor sub-partition warp scheduler register file shared memory l1 cache execution units tensor core cuda core sfu ld/st memory hierarchy bandwidth latency gpu-wide gh200 132 sm data flow"
  },
  {
    title: "Memory — Why mmap Replaces cudaMalloc on GH200",
    url: "/memory",
    sections: ["The Old Model", "The GH200 Reality", "Lithos Memory Strategy", "Why mmap + Pre-allocate", "Why No One Else Does This", "Concrete Comparison", "The KV Cache Problem Solved", "Page Placement on GH200"],
    terms: "memory mmap cudamalloc gh200 pytorch cudacachingallocator vllm kv cache cudamemcpy weight loading pre-allocate eager lazy hbm lpddr5x page placement unified virtual address space zero-copy"
  },
  {
    title: "Five Minds on Hardware-Matched Quantization",
    url: "/minds",
    sections: ["The Constraint", "Side-by-Side Comparison", "Synthesis", "Where They Agree", "Where They Differ"],
    terms: "five minds hardware-matched quantization constraint 180m weights 66mb on-chip gpu memory compute unit types perspectives solutions synthesis agreement differences"
  },
  {
    title: "GPU Warp Scheduler",
    url: "/warp-scheduler",
    sections: ["What the Warp Scheduler Does", "How It Decides", "What Lithos Can Influence", "The SASS Control Word", "Why This Matters for Inference"],
    terms: "warp scheduler gpu dispatch cycle eligible warps scoreboard stall dependency latency hiding sass control word yield stall counts barrier nanosleep lithos influence inference decode occupancy"
  }
];
