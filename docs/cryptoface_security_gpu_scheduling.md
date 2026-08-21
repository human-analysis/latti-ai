# CryptoFace security and GPU scheduling plan

Updated 2026-07-24. A candidate is promoted only after security, numerical
correctness, repeatability, memory, single-GPU latency, and multi-GPU gates.
Each scheduling experiment changes one runtime variable and is preserved in a
separate repository.

The complete cross-folder inventory of historical, rejected, diagnostic, and
promoted variants is maintained in
[`cryptoface_variant_registry.md`](cryptoface_variant_registry.md).

## Preserved secure controls

### Immutable secure reproduction

- Path: `../latti-ai-8boot-q44p61-h640-secure-repro`
- Branch: `codex/q44p61-h640-secure-repro`
- Commit: `bf51ce6b94c674edf14198cd816a38f4ed06c7a3`
- Runtime SHA-256: `208e430c9295e0cb71143c2434f48c765ab00e3dd9bd65fc70d51daed41d53cb`
- Parameters: Q44/P61, 32 Q limbs, 3 P limbs, degree 65,536
- Secrets: main H=640, bootstrap ephemeral H=32
- Physical bootstraps: eight per image
- Classical security: 128.3186816 bits
- Limiting attack: sparse-secret BDD MITM hybrid

The security estimate uses `malb/lattice-estimator` commit
`3e48ef421ec256afddb3e7d2249a77eab6e9ba12` and the complete relevant
classical attack suite. The frozen H192 Q44/P61 candidate remains separately
preserved as a security failure at 121.723606 bits.

### Promoted optimized control

- Path: `../latti-ai-8boot-q44p61-h640-task-cache-candidate`
- Branch: `codex/q44p61-h640-task-cache-candidate`
- Commit: `c1c8b0d020405f2910910105e40ceb5deb1c3305`
- Binary SHA-256: `7610a560eac1cd7a4226ac7d7d5630d91e30c1fb53bb44baa0677ac421d58866`
- Runtime SHA-256: unchanged at `208e430c9295e0cb71143c2434f48c765ab00e3dd9bd65fc70d51daed41d53cb`

The only code change retains one parsed `FheTaskGpu`/MegaAG per persistent
worker thread and project path. Parameters, arithmetic, bootstraps, model,
streams, synchronization, inputs, outputs, keys, and contexts are unchanged.

Idle full 128-pair/256-image result on GPUs 1, 2, and 3:

- all embeddings finite; relative-L2 max 0.025998; max-absolute 0.005957
- mean core inference 13.695 s; p50 13.704 s; p95 14.841 s
- first request per worker mean 17.178 s
- remaining 253 cached requests mean 13.653 s
- processing time 1,318.613 s; aggregate pair interval 10.302 s
- worker balance 84/89/83 images
- mean utilization 58.0% to 63.2%; peak memory 87,964 MiB

The controlled same-GPU idle one-pair A/B saved 3.572 s (9.75%) versus the
secure baseline, with identical 87,868 MiB peak memory.

### Promoted GPU evaluation-key cache

- Runnable path: `../latti-ai-8boot-q44p61-h640-gpu-key-cache-repro`
- Bundle branch: `codex/q44p61-h640-gpu-key-cache-repro`
- Bundle commit: `4cff63a`
- LattiSense source path: `.codex/experiments/lattisense-gpu-key-cache`
- LattiSense source branch: `codex/q44p61-h640-gpu-key-cache`
- LattiSense source commit: `4233877`
- Parent: rejected ABI diagnostic `a7ff96df56fb`, which also records the
  required Q44/P61/H640 ABI/arithmetic source deltas
- Runtime SHA-256:
  `3199749eaf8e6423a1a01b6a6a7360d8cbe6e384b38a851fe9ba776443f6ffb3`

The opt-in `LATTISENSE_GPU_KEY_CACHE=1` path retains only GPU-resident
evaluation keys and the HEonGPU context/operator/upload stream required for
their validity. Its cache namespace binds task identity, parameter
fingerprint, original evaluation-context pointer, a backend-owned generation,
CUDA device, key type, source node, level, special-prime level, and Galois
element. Context or device changes synchronously invalidate the runtime.
Host ABI structs, plaintexts, ciphertext inputs, intermediates, outputs, and
request streams remain request-scoped. The rejected host ABI cache stays
disabled.

The enabled one-pair gate recorded 53 misses on image 1 and 53 hits with zero
misses on image 2. Core GPU task time fell from 15.276 to 11.306 s; numerical
error passed and peak memory remained 87,868 MiB.

The persistent 16-pair gate passed all 32 embeddings. The first request took
14.744 s core inference; the remaining 31 averaged 10.464 s. Telemetry ended
at 53 misses and 1,643 hits. Relative-L2 max was 0.024102, max-absolute
0.005995, and peak memory 87,900 MiB.

The complete idle 128-pair/256-image gate on GPUs 1, 2, and 3 passed:

- all embeddings finite; relative-L2 max 0.026955; max-absolute 0.006123
- mean core inference 10.534 s; p50 10.344 s
- three cold requests at 15.518/17.260/18.639 s
- remaining 253 reused requests mean 10.456 s
- processing time 1,048.822 s; aggregate pair interval 8.194 s
- worker balance 87/87/82 images
- mean utilization 70.1%/72.4%/67.6%
- peak memory 87,962/87,962/87,956 MiB
- exactly three cold telemetry records and 253 hit-only records; no generation
  change
- all devices returned to idle memory after worker shutdown

Against the promoted task-cache control, mean core inference improved 23.1%,
reused mean improved 23.4%, and aggregate pair interval improved 20.5%.
Parameters, security, model arithmetic, and bootstrap schedule are unchanged.
See the candidate `EXPERIMENT.md` and its untracked `results/` directory
for complete commands, hashes, logs, GPU samples, metrics, and embeddings.

### Validated request-scoped pinned H2D staging

- LattiSense source path:
  `.codex/experiments/lattisense-pinned-request-buffers`
- Source branch: `codex/q44p61-h640-pinned-request-buffers`
- Source commit: `0226a75`
- Parent GPU-key-cache source: `4233877`
- Runtime SHA-256:
  `27b2a872bf29d295f67900882967fb9fb9577ee9fe02d245f974febc9305c852`

The opt-in `LATTISENSE_GPU_PIN_REQUEST_H2D=1` path changes only plaintext and
ciphertext H2D handling. It packs each object's discontiguous ABI components
into a stream-local pinned buffer, performs one contiguous asynchronous copy,
and uses a completion event before reuse. Buffers exist for one request only,
are zeroed before release, and never retain ABI objects or request data across
requests. Evaluation keys remain under the separately validated device cache.
Allocation failure falls back to the original pageable path.

The same-GPU one-pair A/B improved cold core time from 15.364 to 14.796 s and
hot core time from 11.661 to 11.034 s. The 16-pair gate passed all embeddings;
31 reused requests averaged 9.805 s versus 10.464 s for the key-cache control,
and processing fell from 393.362 to 372.153 s.

The complete idle 128-pair/256-image gate on GPUs 1, 2, and 3 passed:

- all embeddings finite; relative-L2 max 0.028450; max-absolute 0.005878
- mean core inference 10.190 s; p50 10.249 s
- three cold requests at 16.128/14.632/16.257 s
- remaining 253 reused requests 10.125 s mean and 10.242 s p50
- processing time 1,020.360 s; aggregate pair interval 7.972 s
- worker balance 83/90/83 images
- mean utilization 75.0%/73.1%/73.2%
- peak memory 87,958/87,964/87,958 MiB
- all 256 requests staged 20,827 objects / 11.389 GB from 21,723 original
  segments, with zero allocation failures and zero fallbacks
- exactly three cold key-cache telemetry records and 253 hot records
- all devices returned to idle memory after worker shutdown

Against the GPU-key-cache control, mean core inference improved 3.3%, reused
mean improved 3.2%, processing and pair interval improved 2.7%, and average GPU
utilization increased. Parameters, security, model arithmetic, and bootstrap
schedule are unchanged. This is the fastest validated secure source candidate;
a minimal standalone freeze remains future work.

### Paired pinned-path attribution and write-combined rejection

Nsight Systems 2025.3.2 captured one pair twice on idle GPU 1 with the same
GPU-key-cache runtime; only request-scoped pinned staging was toggled.

| Metric, two images | Pinned off | Pinned on | Delta |
| --- | ---: | ---: | ---: |
| H2D bytes | 43.620 GB | 43.620 GB | unchanged |
| H2D copies | 85,545 | 83,753 | -1,792 |
| H2D device time | 1.655 s | 1.991 s | +0.337 s |
| `cudaMemcpyAsync` host API time | 9.367 s | 5.760 s | -3.608 s |
| `cudaEventSynchronize` calls | 95,025 | 136,679 | +41,654 |
| `cudaEventSynchronize` host API time | 9.477 s | 9.738 s | +0.261 s |
| Kernel instances | 679,626 | 679,626 | unchanged |

Almost every target transfer is one 512 KiB component. Staging coalesces only
896 segments/image, so a size threshold is not useful. The speedup comes from
replacing about 4.034 s/pair of pageable `cudaMemcpyAsync` host blocking with
about 0.424 s/pair of pinned API time, while paying roughly 2.4-2.7 s/pair for
explicit CPU packing, 0.337 s more device H2D time, and 0.261 s more staging
event waits. Bytes and cryptographic work do not change.

The follow-up source at
`.codex/experiments/lattisense-pinned-write-combined`, commit `e3cbaca`,
changed only `cudaHostAllocDefault` to `cudaHostAllocWriteCombined`. An
immediate one-pair A/B improved processing 8.8%, but the persistent 16-pair
gate rejected it:

- ordinary-pinned overall/hot means: 9.939/9.805 s;
- write-combined overall/hot means: 10.140/9.995 s;
- processing: 372.153 vs 380.702 s, a 2.3% regression;
- packing mean: 1.167 vs 1.484 s/image, a 27.2% regression;
- accuracy and 87,900 MiB peak memory passed; zero allocation failures or
  fallbacks.

No 128-pair run was performed. The validated ordinary-pinned commit `0226a75`
remains preferred. See the rejected candidate `EXPERIMENT.md` for trace and
result hashes.

### Validated node-0 CPU-affinity candidate

The next experiment changed deployment only: the validated ordinary-pinned
runtime `0226a75` ran under `taskset -c 0-31,64-95` on GPUs 1, 2, and 3. Those
GPUs and CPUs are all attached to NUMA node 0. Explicit memory binding was not
used, and active inference threads were observed only on the requested CPUs.

The one-pair and persistent 16-pair gates favored affinity: processing improved
10.5% and 1.25%, while packing improved 17.2% and 9.9%, respectively. Both
accuracy gates passed with unchanged memory and zero pinned allocation
failures/fallbacks.

The complete 128-pair/256-image result was more nuanced:

- processing: 1,009.239 s versus 1,020.360 s unbound, a 1.09% improvement;
- pair interval: 7.8847 s versus 7.9716 s;
- overall/hot core means: 10.1252/10.0393 s versus 10.1903/10.1252 s;
- worker balance: 85/86/85 versus 83/90/83;
- packing mean: 1.8610 s versus 1.3430 s, a 38.6% regression;
- effective packing bandwidth: 6.120 versus 8.480 GB/s;
- mean GPU utilization: 71.04% versus 73.79%;
- peak memory: 87,960 MiB per tested GPU;
- all 256 pinned records had zero allocation failures/fallbacks, with exactly
  three cold key-cache requests and 253 hot requests.

The frozen-output oracle passed with relative-L2 max 0.028531 and max-absolute
error 0.005742. Security remains 128.318682 classical bits because no FHE
parameter, secret, arithmetic, model, or bootstrap change occurred. GPUs 1-3
returned to idle after shutdown.

This is retained as a positive topology-specific candidate, not the universal
default. The small aggregate gain coexists with CPU/memory-bandwidth contention
when all three workers share one NUMA node and with about 7.3% slower setup.
Persistent workers amortize setup. Disjoint node-0 CPU partitions were tested
next without forcing memory policy and were rejected, as recorded below. The
runnable shared-affinity record and exact metrics are in
`.codex/experiments/lattisense-numa-affinity`.

### Rejected disjoint node-0 worker partitions

The bounded follow-up retained the exact pinned-H2D runtime, first-touch memory
policy, GPUs, queue, model, parameters, and eight-bootstrap schedule. It changed
only worker affinity, assigning SMT-aware disjoint node-0 CPU sets to GPUs 1-3:
`0-10,64-74`, `11-21,75-85`, and `22-31,86-95`.

The one-pair and 16-pair gates passed accuracy and stability. The complete
128-pair/256-image gate also passed, but rejected the optimization:

- processing: 1,022.928 s versus 1,009.239 s shared-node0, +1.36%;
- pair interval: 7.9916 s versus 7.8847 s, +1.36%;
- overall/hot core means: 10.2997/10.2216 s versus 10.1252/10.0393 s;
- packing mean: 2.1226 s versus 1.8610 s, +14.1%;
- packing bandwidth: 5.485 GB/s versus 6.120 GB/s, -10.4%;
- mean GPU utilization: 70.11% versus 71.04%;
- worker balance: 85/87/84; peak memory: 87,962 MiB;
- exactly three cold cache misses and 253 hot requests, with zero allocation
  failures or fallbacks.

All 256 embeddings were finite; relative-L2 max was 0.028789 and maximum
absolute error was 0.005782. Security remains 128.318682 classical bits because
no cryptographic or arithmetic input changed. GPUs returned to idle after
shutdown. Hard partitioning reduced the CPU resources available to packing and
did not identify worker migration as the parent regression's cause. The shared
node-0 affinity candidate remains preferred. Complete evidence is in
`.codex/experiments/lattisense-numa-worker-partitions`.

### Rejected submission-thread-only affinity

The final CPU-placement follow-up left both task pools on the shared node-0
mask and narrowed only the calling GPU submission/dispatcher thread during
`run_tasks()`. Its inherited mask was restored after every request. Physical
GPUs 1, 2, and 3 mapped to CPUs 0, 1, and 2, respectively.

The idle one-pair point was favorable: processing was 27.398 s versus 31.028 s
for a same-binary disabled control, and hot core inference was 9.766 s versus
10.082 s. Packing moved the other way, from 0.991 to 1.047 s.

The accurate persistent 16-pair gate did not sustain the headline gain:

- processing: 366.192 s versus 367.516 s shared-node0, -0.36%;
- overall/hot core means: 9.865/9.726 s versus 9.924/9.781 s;
- candidate hot p50/p95: 9.674/9.999 s;
- hot packing: 1.045 s versus 1.051 s;
- mean/p50/p95 utilization: 74.19/94/100%; peak memory: 87,900 MiB;
- all 32 telemetry records confirmed CPU 0 and the 64-CPU inherited mask.

All outputs were finite; relative-L2 max was 0.024713 and maximum absolute
error was 0.005385. Parameters, eight-bootstrap arithmetic, and
128.318682-bit security were unchanged. The 1.324-second processing difference
across 32 images is within observed host/run variance, so the experiment was
rejected without a 128-pair run. Source and evidence are preserved at commit
`c2993d1` in `.codex/experiments/lattisense-submission-affinity`.

## Fresh profile of the fastest validated affinity deployment

On 2026-07-25, Nsight Systems profiled the exact pinned-H2D runtime
`27b2a872bf29...` under shared node-0 affinity. Tracing raised pair processing
to 29.876 s and cold/hot inference to 15.749/10.871 s, while correctness still
passed at relative L2 0.020809 and maximum absolute error 0.003781.

The device activity span was 26.602 s, but the union of kernel/copy activity
was only 16.650 s (62.59%), leaving 9.952 s of internal idle time. The pair
issued 679,626 kernels and 83,753 H2D copies. Those H2D copies moved 43.620 GB
but consumed only 1.768 s on the device, confirming that another buffer-only
optimization is unlikely to dominate.

Host API attribution was 9.675 s in 136,679 event synchronizes, 5.532 s in
88,921 async-copy submissions, 4.298 s in kernel launches, 2.381 s in 41,815
stream synchronizes, 1.252 s in 411,986 event records, 0.972 s in 176,524
stream waits, 0.843 s in 453,276 `cudaFree` calls, and 0.638 s in 95,018 event
creates. These times overlap and must not be added into a wall-clock estimate.

Exact event correlation found that 87,265 of 218,339 producer/consumer waits
(40.0%) targeted the producer's own stream. Three bounded experiments followed:

- One-time `MemoryPool` CUDA initialization removed repeated `cudaFree(0)`
  calls, but its accurate 16-pair gate regressed processing 0.76% and hot
  inference 1.43%; reject for latency.
- Native Blackwell `sm_120` cubins produced embeddings around `2.9e19` and
  failed relative L2 by nineteen orders of magnitude; reject before any
  sustained timing. The failure is in architecture-specific GPU arithmetic,
  not the CryptoFace model or parameters.
- Same-stream wait elision initially improved 16-pair processing/hot inference
  1.13%/1.33%, but its full run failed after 91 images with Go's fatal bad-heap
  pointer check. The ABI had stored C allocation addresses in slice headers of
  Go `ring.Poly` heap objects. Direct per-component modular transforms removed
  that invalid representation and passed pair/16-pair accuracy, but the safe
  combined candidate regressed processing/hot inference 1.08%/1.41% versus the
  affinity control. It was rejected without another full gate.

The crash must not be hidden with disabled garbage collection or cgo checks.
The failed full-run log and partial metrics remain in
`.codex/experiments/lattisense-skip-same-stream-waits`. The safe ABI repair is
worth retaining as a correctness candidate independently of the rejected
latency claim.

Timing-disabled dependency events were then tested in isolation. They passed
pair and 16-pair correctness. The persistent result was 370.361 s processing
and 9.873 s hot mean: 0.30%/0.46% better than the safe parent but 0.77%/0.94%
slower than the validated affinity control. This bounds the simple event-flag
opportunity below observed run variance. The candidate was rejected without a
full gate, and persistent event pooling is not justified as the next step.

## Earlier Nsight Systems attribution

Nsight Systems 2025.3 profiled two consecutive requests from the committed
task-cache control on idle GPU 1. Profiler processing time was 33.602 s; core
inference was 16.292 and 13.536 s. The device-work span was 29.956 s, while
the union of kernel/copy activity was 20.435 s, leaving about 9.521 s without
device work inside that span.

| Source | Calls / instances | Aggregate time or bytes |
| --- | ---: | ---: |
| GPU kernels | 681,168 | 18.40 s |
| Pageable H2D copies | 124,878 | 64.06 GB; 6.00 s device time |
| `cudaEventSynchronize` | 95,024 | 14.93 s host API time |
| `cudaMemcpyAsync` | 130,045 | 13.23 s host API time |
| `cudaStreamSynchronize` | 41,860 | 5.51 s host API time |
| `cudaLaunchKernel` | 681,168 | 4.56 s host API time |
| `cudaEventCreate` | 95,018 | 1.64 s host API time |

The largest between-request device-idle gap was 1.959 s. The remaining idle
time occurs inside requests and is dominated by fine-grained graph/event
scheduling plus pageable H2D loads. The GPU ABI bridge exports fresh malloc
backed C structs on each run, loads them into HEonGPU, and synchronizes the
stream after each backend load. GPU evaluation-key uploads are now removed on
hot requests by the validated cache, reducing mean core inference to 10.534 s.
Request-scoped plaintext/ciphertext transfers and fine-grained scheduling
remain future targets; convolution arithmetic is not the cause of this gap.

## Rejected cleanup-pool combination

- Path: `../latti-ai-8boot-q44p61-h640-task-cache-cleanup-pool-candidate`
- Branch: `codex/q44p61-h640-task-cache-cleanup-pool-candidate`
- Commit: `cffe43698a795418397d60379c0cfa1a8cc03bbd`

Moving event-completion cleanup waits off the two submission threads passed
correctness and held peak memory at 87,868 MiB. An immediate idle A/B was
slower, however: 32.446 s versus 31.105 s for the task-cache control, a 1.341
s (4.31%) regression. It is preserved as a diagnostic and not promoted.

## Rejected bounded ABI-export reuse

- Path: `.codex/experiments/lattisense-abi-export-cache`
- Branch: `codex/q44p61-h640-abi-export-cache`
- Commit: `a7ff96df56fb186d48c05676459cd6e6fffbb49f`
- Nested Lattigo commit: `19d1e4edc162d641ebb6b636beda08496a1a070e`

The experiment retained deep-copied host ABI exports under an explicit byte
cap and never admitted ciphertext inputs, intermediates, or outputs. The safe
exact-handle form passed correctness but recorded zero hits because the runner
creates new equivalent parameter handles and re-extracts key handles per
request. Its apparent timing gains were therefore scheduling variance, not
reuse.

Attempts to stabilize identity exposed an invalid lifetime contract before a
second request could complete. The variants failed request 1 with either
`free(): invalid size` or `runtime/cgo: misuse of an invalid Handle` in
`ExportCkksPlaintextMul`, including request-generation gating and a key-only
cache. The experiment is rejected. Pinned-memory testing was not started
because no persistent ABI buffer set passed the ownership/correctness gate.
See `EXPERIMENT.md` in the candidate for the retained measurements and failure
sequence.

## Rejected constrained Q31/P3/H640 parameter candidate

The final bounded parameter search started from pinned-H2D source `0226a75`
and changed the only limb-count reduction that could preserve the P3
decomposition path: one residual Q44 limb was removed, bootstrap stages shifted
down one level, and exactly eight redundant 15-to-14 drops were collapsed. The
main/ephemeral secrets remained H640/H32 and the physical bootstrap count
remained eight.

The complete pinned estimator suite passed before inference:

- Q/P limbs: 31/3; `log2(QP)=1693.999969`;
- limiting attack: sparse BDD MITM hybrid;
- classical security: **131.499984 bits**.

The matching compiled Lattigo preset and serialized graph passed structural and
Go compile gates. Encrypted correctness also passed: the one-pair relative-L2
and max-absolute errors were 0.020701/0.003806; the 16-pair maxima were
0.026216/0.005510 with all 32 embeddings finite.

Latency rejected the candidate:

- one pair: 31.935 s, cold/hot core 17.723/10.608 s;
- immediate 32Q control: 27.879 s, cold/hot 14.515/10.010 s;
- Q31 16-pair processing and overall/hot core: 376.209 s and
  10.097/9.957 s;
- 32Q ordinary-pinned control: 372.153 s and 9.939/9.805 s;
- Q31 hot packing: 1.213 s versus 1.167 s; peak memory 87,900 MiB.

Pinned staging stayed at exactly 20,827 objects and 11.389 GB per image.
Therefore the removed upper limb did not reduce the dominant request transfer
path, while processing/hot inference regressed 1.09%/1.55%. No 128-pair run was
performed. The clean bundle is `276fbe2`; matching LattiSense/Lattigo source
is `f7016fa`/`73452c0`.

## Ordered TODO

Benchmarking rule for every new candidate: run an idle one-GPU/single-pair gate
and a sustained 128-pair/three-GPU gate. Record single-pair wall time and
cold/hot runner-recorded encrypted inference separately from total throughput,
aggregate pair interval, images/second, and the sustained per-image distribution. A
16-pair run is an intermediate stability gate, not a substitute for either.
Missing historical measurements must be labelled rather than inferred.

- [x] Preserve and fingerprint the original eight-bootstrap control.
- [x] Pin the estimator and verify the full Q/P modulus and sparse-secret
  attack suite.
- [x] Promote H640 only after the >=128-bit security gate passed.
- [x] Fix the CKKS GPU ABI scale corruption and bootstrap level alignment.
- [x] Run one-pair, persistent 16-pair, and complete 128-pair correctness gates.
- [x] Freeze the secure H640 reproduction in a clean committed directory.
- [x] Profile persistent scheduling and remove repeated MegaAG parsing through
  thread-local task reuse.
- [x] Validate task reuse with idle one-pair, persistent, and full 128-pair
  multi-GPU gates; preserve the promoted commit.
- [x] Capture an operation-level Nsight Systems trace and attribute kernel,
  copy, event, stream, and host launch time.
- [x] Retest the event cleanup pool on top of task caching; reject it after the
  immediate idle control was faster.
- [x] Instrument ABI export/load nodes to measure invariant plaintext/key bytes,
  unique buffer lifetime, and per-type time. Do not retain dynamic ciphertext
  inputs or outputs.
- [x] Test bounded reuse of immutable plaintext/key ABI exports on top of commit
  `c1c8b0d...`, with an explicit host-memory cap and exact pointer/type/level
  cache key. Rejected: the safe form had zero hits and stable-identity forms
  violated ABI/Go handle lifetimes.
- [x] If ABI reuse helps, separately test persistent pinned registration for
  the reused buffers. Not applicable: reuse did not pass the ownership gate,
  so pinning was intentionally not attempted.
- [x] Implement backend-owned GPU evaluation-key reuse with explicit task,
  parameter, context-generation, and device scoping; retain the HEonGPU
  context/operator/upload stream required by key lifetimes.
- [x] Validate GPU key reuse with cache-disabled, one-pair, persistent
  16-pair, and complete idle 128-pair/3-GPU gates. Require explicit nonzero
  hit telemetry before attributing latency.
- [x] Separately test pinned memory only for request-scoped plaintext and
  ciphertext buffers with a valid ownership contract. Full one-pair,
  persistent 16-pair, and 128-pair/3-GPU gates passed at source `0226a75`.
  Persistent host ABI-export reuse remains rejected.
- [x] Profile the validated pinned path to separate CPU packing, completion
  event waits, and H2D gains. The paired trace attributes the gain to reduced
  pageable-host API blocking; write-combined allocation was rejected after a
  16-pair latency and packing regression at `e3cbaca`.
- [x] Test direct request export into a byte-capped pinned pool so fully pinned
  CKKS plaintext/ciphertext components bypass staging. A 2 GiB cap covered
  about 29% of bytes and cut packing 41.4%, but the accurate 16-pair gate
  regressed hot inference 1.15% and processing 2.25%; reject without a
  128-pair run. Evidence is in
  `.codex/experiments/lattisense-direct-pinned-export`.
- [x] Measure CPU/NUMA locality and sustained packing behavior before another
  buffer change. Node-0 affinity passed accuracy and improved full processing
  1.09%, but concurrent three-worker packing regressed 38.6%; preserve the
  result in `.codex/experiments/lattisense-numa-affinity`.
- [x] Test disjoint node-0 CPU partitions per persistent GPU worker while
  keeping the validated pinned runtime and first-touch memory policy fixed.
  Rejected after the full gate: processing regressed 1.36%, hot inference
  1.82%, and packing 14.1% versus shared-node0 despite passing all correctness,
  security, memory, and cleanup gates.
- [x] For the GPU-key-cache positive result, rerun correctness, repeatability,
  peak memory, persistent 16-pair, and complete 128-pair multi-GPU gates.
- [x] Preserve the fastest reliable secure result in standalone bundle
  `4cff63a` with LattiSense source `4233877`; never modify the immutable
  secure or promoted task-cache controls.
- [x] Test byte-capped GPU-resident reuse of immutable plaintexts with exact
  task/parameter/context-generation/device ownership and explicit hit/memory
  telemetry. Rejected: a 1 GiB hotset produced exact hits and passed accuracy,
  but the final 16-pair gate regressed hot inference 10.1% and processing 7.9%;
  a lock-free frozen lookup path did not recover the loss. See
  `.codex/experiments/lattisense-gpu-plaintext-hotset` at `7b027f5`.
- [x] Test bounded CUDA graph capture only after identifying a stable repeated
  launch region that does not include changing ciphertext pointers. Rejected
  at feasibility: simple capture/replay works, but legacy `cudaFree`, event
  sync, and stream sync invalidate capture and are pervasive in the current
  HEonGPU/request lifecycle. No latency was attributed to a nonexistent graph;
  persistent capture-safe workspaces are required first. See
  `.codex/experiments/lattisense-cuda-graph-capture` at `2d963c3`.
- [x] Test one isolated HEonGPU kernel-fusion or batched-operation candidate.
  Rejected: fused RingT `cmpac_sum` removed 32,984 arithmetic launches and
  34,048 result allocations per image, but the accurate 16-pair gate regressed
  hot inference 3.5% and processing 3.0%. See
  `.codex/experiments/lattisense-fused-cmpac` at `31f2d69` (nested HEonGPU
  `87b01f0`).
- [x] Test two-slot CPU packing/GPU execution overlap with bounded memory and
  completion-event ownership. Rejected at the one-pair screen: one-slot waits
  totaled only 40.002 ms/image; the second slot saved 1.253 ms, doubled pinned
  allocation bytes, increased hot packing 27.9%, and regressed hot inference
  2.1%. See `.codex/experiments/lattisense-pinned-two-slot` at `21f3a75`.
- [x] Test submission-thread-only affinity, leaving packing workers free to use
  the shared node-0 CPU pool. Rejected after the accurate 16-pair gate:
  processing and hot inference improved only 0.36% and 0.56%, respectively,
  so the favorable one-pair point did not justify a full gate.
- [x] Run a constrained FHE-parameter search only through the full sparse-secret
  security estimator and primitive/layer correctness gates before inference
  benchmarking. The clean Q31/P3/H640 candidate passed at 131.499984 bits and
  passed one-pair/16-pair numerical gates, but regressed sustained processing
  1.09% and hot inference 1.55%; reject without a full run. Historical P2,
  smaller-P, and smaller-bootstrap-prime families remain rejected at primitive
  or numerical gates.
- [x] Profile the fastest validated pinned-H2D plus shared-node0 deployment.
  The traced device was busy only 62.59% inside its activity span; event and
  stream synchronization/submission dominate the remaining gap, while H2D
  device time is only 1.768 s for 43.620 GB per pair.
- [x] Test one-time `MemoryPool` CUDA initialization. It removed 453,276
  repeated `cudaFree(0)` calls but regressed the accurate 16-pair gate; retain
  only as a graph-readiness cleanup.
- [x] Test native `sm_120` code generation. Reject at the one-pair correctness
  gate because encrypted outputs diverged catastrophically.
- [x] Test redundant same-stream wait elision. Preserve the initially positive
  16-pair result and the failed full gate. Replace the unsafe Go heap objects
  backed by C memory; the safe combined retest passed accuracy but lost the
  latency gain, so reject without a second full run.
- [x] Test timing-disabled CUDA dependency events as an isolated A/B. Pair and
  16-pair correctness passed, but the persistent gate remained 0.77%/0.94%
  slower than the validated affinity control in processing/hot inference;
  reject without a full run.
- [x] Decide whether a bounded persistent event pool is justified. Do not
  proceed: the simpler event-flag experiment bounded this opportunity below
  run variance. Higher-upside work must remove several lifecycle barriers
  together or repair native Blackwell arithmetic first.
