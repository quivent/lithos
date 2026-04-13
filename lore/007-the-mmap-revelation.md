# The mmap Revelation

## The moment we proved CUDA's memory model is unnecessary

The GH200 has a feature that NVIDIA markets but nobody trusts: unified memory. The CPU and GPU share the same physical memory through hardware coherence. The same address, the same bytes, the same page tables. No copy required.

Every CUDA program ever written does this:

```
cudaMalloc(&gpu_ptr, size);          // allocate GPU memory
cudaMemcpy(gpu_ptr, cpu_ptr, size);  // copy data to GPU
kernel<<<blocks, threads>>>(gpu_ptr); // use GPU pointer
cudaFree(gpu_ptr);                   // free GPU memory
```

Four API calls to move data from where it is to where the GPU can see it. On discrete GPUs — where CPU and GPU have separate memory connected by a PCIe bus — this is necessary. The GPU literally cannot read host memory at speed. You have to copy.

On the GH200, CPU and GPU memory are the same memory. Connected by NVLink-C2C at 900 GB/s. Hardware-coherent. The GPU can read any CPU address. The CPU can read any GPU address. There is no bus bottleneck. There is no separate address space.

So we tried the obvious thing.

## The experiment

```python
fd = os.open(model_path, os.O_RDONLY)
host_ptr = mmap.mmap(fd, length, mmap.MAP_PRIVATE, mmap.PROT_READ)
# Pass host_ptr directly to GPU kernel
kernel(host_ptr, output_ptr, ...)
```

No cudaMalloc. No cudaMemcpy. No CUDACachingAllocator. No memory pool. No allocation tracking. No fragmentation. No out-of-memory errors from the allocator running out of its pool while physical memory sits unused.

The host mmap pointer — the one returned by the operating system's virtual memory system, the one that points to a file on disk that the kernel pages in on demand — passed directly to a GPU kernel.

It worked.

## What "it worked" means

Not "it didn't crash." Not "it produced output." It produced the correct output. The same bytes. The same logits. The same " Paris" at rank 1.

The model file sits on an NVMe drive. The OS maps it into the process's virtual address space. The GPU kernel reads from that address space. The OS pages the data from NVMe into physical memory on first access. The GPU's NVLink-C2C reads it from physical memory at 900 GB/s. The caches warm up. Subsequent accesses hit cache.

No copy. No allocation. No framework memory management.

Model load time: 9 milliseconds.

Not 9 seconds. Nine milliseconds. The time it takes to call `mmap()` and set up page table entries. No data moves. The OS doesn't read the file. The GPU doesn't allocate memory. Nothing happens until the first kernel reads the first byte, and then data flows on demand.

PyTorch loads the same model in 30-120 seconds. It reads the entire file into CPU memory, converts formats, copies to GPU memory, organizes into tensors, registers with the allocator. Thirty seconds of work that, on this machine, is unnecessary.

## What we eliminated

The entire CUDA memory management stack:

**cudaMalloc / cudaFree.** Allocation and deallocation of GPU memory regions. On discrete GPUs, this manages the GPU's separate memory. On GH200, there's nothing to manage. The memory is already there.

**cudaMemcpy.** Data transfer between host and device. On discrete GPUs, this moves bytes across PCIe or NVLink at 32-64 GB/s. On GH200, there's nothing to transfer. The data is already visible.

**CUDACachingAllocator.** PyTorch's memory pool that sits on top of cudaMalloc to avoid the overhead of repeated allocation. It tracks blocks, splits them, merges them, defragments them. A complex state machine managing a problem that doesn't exist on GH200.

**cudaMemPool.** CUDA's native memory pool API. Another layer managing the same non-problem.

**cudaMallocManaged / cudaMemPrefetchAsync.** CUDA's own unified memory API, which adds a software migration layer on top of the hardware coherence. We don't use it. The hardware coherence is sufficient. The OS mmap is sufficient.

We replaced all of it with `mmap()`. A Unix system call from 1983.

## Why nobody else has done this

Three reasons.

**Discrete GPU inertia.** Every CUDA tutorial, every framework, every library assumes discrete GPU memory. The entire ecosystem is built around the copy model. When you have a GH200, you still use the copy model because that's what the code does. Nobody rewrites the memory layer.

**NVIDIA's own messaging.** NVIDIA markets unified memory as a convenience feature — "you don't have to manage transfers manually." They don't market it as "you can delete your entire memory management stack." Their own documentation still shows cudaMallocManaged with prefetch hints. They add software complexity on top of hardware that doesn't need it.

**Nobody tested it.** Passing an mmap pointer to a GPU kernel is the kind of thing that feels like it shouldn't work. It feels like the GPU driver would reject it, or the kernel would segfault, or the data would be wrong. You have to try it and verify the output. We tried it. It works.

## The numbers

| Operation | Traditional CUDA | mmap on GH200 |
|-----------|-----------------|----------------|
| Model load (27B params) | 30-120 seconds | 9 milliseconds |
| Memory overhead | 2x (host + device copies) | 1x (file pages) |
| Allocation failures | Yes (pool exhaustion) | No (OS virtual memory) |
| Memory management code | ~2000 lines in PyTorch | 3 lines |
| Cold start | 30-120 seconds | 228 milliseconds |

The 228 ms cold start includes mmap (9 ms), kernel compilation (ptxas, ~200 ms for all kernels), and engine initialization. The model is ready to generate tokens in under a quarter of a second.

vLLM takes 30-120 seconds to start. Most of that time is loading and copying weights.

## What it means for the architecture

If you don't need cudaMalloc, you don't need a memory allocator. If you don't need a memory allocator, you don't need an allocation strategy. If you don't need an allocation strategy, you don't need a framework to manage it.

The GH200 doesn't need PyTorch's memory layer. It doesn't need CUDA's memory API. It needs `mmap()` and a kernel launch.

This is why Lithos can exist. A language that emits PTX and launches kernels doesn't need to manage memory because the hardware manages memory. The file IS the tensor. The address IS the pointer. The compiler generates code that reads from addresses, and the OS and hardware handle the rest.

On a discrete GPU, you need a framework because memory management is genuinely hard. On the GH200, you need a compiler because memory management is trivially easy and the only hard problem left is generating good kernels.

We proved it. The model loads in 9 milliseconds, runs through 64 layers, and produces the correct answer. No copies. No allocations. No framework.

That's the lore.
