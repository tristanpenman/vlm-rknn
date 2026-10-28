# Memory Mappping

A model file can be larger than a device's physical memory and still be loaded successfully. Loading does not necessarily mean copying the complete file into anonymous RAM. A runtime can instead memory-map the file and let the operating system bring individual pages into memory as they are accessed.

This distinction is particularly important on edge devices, where storage capacity may comfortably accommodate a model but physical memory cannot hold all of its weights, caches, and runtime buffers at once. Memory mapping can make such a model usable, although it does not eliminate the underlying capacity constraint. It trades memory pressure for storage traffic and can have a substantial performance cost.

## Virtual Memory and Resident Memory

`mmap` associates a region of a process's virtual address space with a file. Creating a mapping reserves address-space ranges, but it normally does not read every mapped byte into physical memory immediately. The first access to a page causes a page fault, at which point the kernel reads that page from storage and places it in the page cache.

The relevant quantities are therefore different:

- File size is the amount of storage occupied by the model artifact.
- Virtual size is the address space covered by the mapping and other allocations.
- Resident set size (RSS) is the subset of pages currently held in physical memory for the process.
- Proportional set size (PSS) apportions shared resident pages between the processes that map them.

A process can consequently have a virtual size larger than installed RAM without exhausting memory. A 10 GB file mapping on an 8 GB system is valid on a 64-bit platform because only the pages needed at a particular time must be resident.

File-backed pages also differ from anonymous allocations. When memory is needed elsewhere, the kernel can discard a clean page from a read-only model mapping and read it from the model file again later. It does not need to preserve that page in swap. Anonymous data, such as KV caches and runtime workspaces, must instead remain in RAM, be compressed or swapped, or be released.

Memory mapping is therefore not equivalent to increasing the device's memory. It gives the operating system a way to maintain a smaller resident working set backed by a larger file.

## Why Access Patterns Matter

Whether this approach performs well depends on the model's working set and access pattern.

Some large tables are accessed sparsely. An embedding lookup, for example, needs only the rows selected by the current token IDs. Memory mapping is effective in this case because untouched parts of the table never need to become resident. Pages that are revisited frequently tend to remain in the page cache.

Transformer weights present a harder case. Generating a token generally evaluates every decoder layer, so a large fraction of the ordinary layer weights may be visited repeatedly. If those frequently used pages fit alongside the KV cache, runtime buffers, operating system, and other applications, the first pass can warm the page cache and later tokens can reuse resident data. If they do not fit, the kernel repeatedly evicts and reloads weight pages. The result is page-cache thrashing, with inference throughput increasingly determined by storage bandwidth and page-fault latency rather than NPU compute.

Sequential access can benefit from kernel read-ahead, while scattered access produces more independent faults and poorer storage utilization. The exact outcome is runtime- and architecture-dependent: file size alone cannot determine the resident working set.

## Gemma 4 E2B Example

The `gemma-4-E2B-it-fp16-rk3588.rkllm` artifact created by this repository is 10,119,231,532 bytes, or approximately 9.42 GiB. That is larger than the approximately 7.7 GiB of physical memory exposed by an 8 GB Khadas Edge2.

The name E2B does not mean that the checkpoint stores only two billion parameters. The `E` means *effective*. Google's [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) describes E2B as approximately 2.3 billion effective parameters and 5.1 billion parameters when its embeddings are included. The model uses per-layer embeddings: each decoder layer has token-specific embedding data, which adds many stored parameters while requiring only selected rows for a particular token.

The artifact size is consistent with an FP16 export:

```text
5.1 billion parameters * 2 bytes per FP16 value
    = approximately 10.2 GB
```

RKLLM 1.3.1 contains a memory-mapped model-loading path and reports memory mapping as its default behavior. The application passes the model path to `rkllm_init`; it does not read the complete file into an application-owned buffer. A successful initialization therefore establishes that the runtime opened, parsed, and mapped the model, not that every byte of the model became resident simultaneously.

Gemma 4 E2B's structure makes the difference especially visible. The large per-layer embedding tables contribute to the on-disk FP16 size, but an inference request touches only the rows associated with its tokens. Those sparse accesses allow much of the mapped embedding data to remain non-resident. The decoder weights, active embedding rows, KV cache, runtime buffers, and operating-system memory still have to fit within the practical working set.

The repository's initial Gemma 4 path is also text-only. It does not load the separate vision or audio encoders, which avoids adding their weights and intermediate buffers. The converted model has a maximum context of 4096 tokens by default, while the C++ application defaults to a 2048-token runtime context. Limiting context length reduces KV-cache and context-dependent buffer pressure, although it does not reduce the stored decoder weights.

An inspection of the Edge2 used during development reported 7.7 GiB of physical memory and four zram swap devices totalling approximately 3.9 GiB. Zram can provide useful headroom for compressible anonymous pages, but it is not an extra 3.9 GiB of physical memory: its compressed contents are themselves stored in RAM. It also cannot be expected to compress FP16 weights enough to explain the result. File-backed demand paging and sparse access to the per-layer embeddings are the central mechanisms.

## Performance Impact

Memory mapping can reduce startup work because the runtime does not have to copy the complete artifact before returning from initialization. It can also avoid allocating RAM for model regions that an inference request never touches. These benefits do not imply that inference will be fast.

A cold request incurs page faults while required weights are read from storage. This can increase time to first token substantially. If the active working set fits after that first pass, subsequent tokens or requests may be faster because the relevant pages remain cached. Available memory used by the page cache is therefore useful memory, even when tools such as `free` do not label it as unused.

When the active working set exceeds available RAM, performance degrades more severely:

- Weight pages are evicted and read again during later layers or tokens.
- Token latency becomes dependent on eMMC or other storage performance.
- Throughput can fall well below the NPU's nominal compute capability.
- Latency becomes less predictable when other processes compete for memory or storage.
- Longer contexts consume more KV-cache memory and leave less room for mapped weight pages.
- Concurrent models or requests make page-cache thrashing more likely.

In the worst case, the model may initialize successfully but become impractically slow during generation, or later fail when the runtime requests a non-pageable buffer that cannot be allocated. Successful loading is therefore a weaker validation than successfully generating representative prompts under sustained memory pressure.

Quantization remains preferable when available. Reducing weights from FP16 to INT8 approximately halves their storage and bandwidth requirements, makes it more likely that the active weight set remains resident, and leaves more memory for the KV cache and runtime. Memory mapping and quantization are complementary: a quantized model can still be mapped, while benefiting from fewer page faults and less data read per inference pass.

## Measuring the Real Working Set

File size and process virtual size are insufficient for evaluating a deployment. Measurements should cover initialization, the first prompt, repeated generation, and the longest intended context. Useful Linux interfaces include:

```bash
# System-wide RAM and swap state.
free -h
cat /proc/swaps

# Mapping-level resident and proportional memory totals.
cat /proc/<pid>/smaps_rollup

# Identify the model's file-backed mapping.
grep -F '.rkllm' /proc/<pid>/maps

# Observe major faults and storage activity while generating.
pidstat -r -p <pid> 1
iostat -xz 1
```

The most useful comparison is between a cold run and a warm repeated run. A high major-fault count and sustained storage reads during every generated token indicate that the working set is not remaining resident. Low storage activity after the first prompt suggests that demand paging is primarily helping with startup and unused model regions rather than continuously substituting storage for RAM.

For capacity planning, record peak RSS or PSS, available system memory, swap or zram use, major page faults, storage throughput, time to first token, and tokens per second. These measurements reveal whether an oversized mapped model is genuinely practical or merely able to pass initialization.
