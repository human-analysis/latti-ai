# LattiAI

> **CryptoFace source snapshot.** This repository preserves the Latti-AI
> source used by the CryptoFace Latti GPU submission. LattiSense is pinned to
> the source-only `human-analysis/lattisense` repository. No prebuilt native
> runtime is tracked. See [`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md).

For an NVIDIA H200 source build, run `scripts/build_h200.sh` after cloning with
submodules.

[![Build & Test](https://github.com/cipherflow-fhe/latti-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/cipherflow-fhe/latti-ai/actions/workflows/ci.yml)
[![Format Check](https://github.com/cipherflow-fhe/latti-ai/actions/workflows/format.yml/badge.svg)](https://github.com/cipherflow-fhe/latti-ai/actions/workflows/format.yml)
[![Python Lint](https://github.com/cipherflow-fhe/latti-ai/actions/workflows/python-lint.yml/badge.svg)](https://github.com/cipherflow-fhe/latti-ai/actions/workflows/python-lint.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**LattiAI** is a development platform for privacy-preserving AI model inference, built on top of the [LattiSense](https://github.com/cipherflow-fhe/lattisense/tree/main) Fully Homomorphic Encryption (FHE) framework developed by [CipherFlow](https://cipherflow.ai/).

LattiAI covers the complete pipeline from plaintext models trained with frameworks like PyTorch to encrypted inference deployment. Through model adaptation, model compilation, and a high-performance HE operator library, it automatically converts standard AI models into encrypted inference services based on the CKKS fully homomorphic encryption scheme. Throughout the entire inference process, data remains encrypted — the server cannot access the user's raw data, and the user cannot access the model parameters, achieving bidirectional privacy protection for both data and model.

AI developers can complete end-to-end encrypted inference deployment without understanding the underlying cryptographic primitives.

## Main Features

- **Model Adaptation**: Provides plug-and-play polynomial approximation operators that replace non-polynomial activation functions (e.g., ReLU, SiLU) and MaxPooling with FHE-friendly polynomial activations and AvgPool. After fine-tuning or retraining, the adapted model achieves accuracy on par with the original. Validated on ResNet-18, ResNet-44, MobileNetV2, YOLOv5 and more, with additional conversion strategies under active development.

- **Model Compiler**: Takes adapted model files (`.pth`, `.onnx`) and, through operator mapping and computation graph optimization, automatically generates a CKKS-compatible directed acyclic computation graph (DAG) for encrypted inference, with automatic planning of bootstrapping insertion and data packing strategies.

- **HE Operator Library**: Implements encrypted versions of core neural network operators based on the CKKS scheme — convolution, deconvolution, fully connected, AvgPool, and BatchNorm — leveraging SIMD slot encoding for vectorized parallel computation and supporting arbitrary-depth ciphertext operations through bootstrapping.

- **Runtime**: Automatically schedules the complete inference pipeline based on the compiler-generated encrypted computation graph, with support for multi-threaded CPU parallelism and GPU acceleration. Encrypted inference results are nearly identical to plaintext inference.

## Build & Install

### CPU-Only Build (Without GPU Acceleration)

If GPU acceleration is not needed, use this simplified process:

```bash
git clone https://github.com/cipherflow-fhe/latti-ai.git
cd latti-ai
git submodule update --init
git -C inference/lattisense submodule update --init fhe_ops_lib/lattigo
cmake -B build
cmake --build build -j$(nproc)
```

### GPU Build (Recommended)

#### Step 1: Clone Repository

```bash
git clone --recursive https://github.com/cipherflow-fhe/latti-ai.git  # This may take ~6 minutes
cd latti-ai
```

#### Step 2: Build and install HEonGPU (GPU Acceleration Library)

```bash
cd inference/lattisense/HEonGPU
cmake -B build \
  -DCMAKE_CUDA_ARCHITECTURES=<arch> \
  -DCMAKE_CUDA_COMPILER=<path/to/cuda>/bin/nvcc \
  -DCMAKE_INSTALL_PREFIX=<path/to/HEonGPU>/install
cmake --build build --parallel $(nproc) --target install
```

#### Step 3: Build Project

```bash
cd ../../..  # Return to project root
cmake -B build -DINFERENCE_SDK_ENABLE_GPU=ON -DLATTISENSE_CUDA_ARCH=<arch>
cmake --build build -j$(nproc)
```

For detailed build prerequisites, troubleshooting, and build options, see the **[Build Guide](docs/en/build-guide.md)**.

---

## Quick Start

This guide demonstrates how to transform a standard PyTorch model into an inference service for encrypted queries using the **LattiAI** framework.

> **Want to try encrypted inference right away?** We provide pre-prepared task resources for several example models. If you would like to skip the model adaptation and compilation steps below, jump directly to [Running Examples](#running-examples).

We will use a **ResNet-20** model trained on the **CIFAR-10** dataset as an end-to-end example.

### Prerequisites

Before starting, ensure you have:

- Successfully built the project (see [Build & Install](#build--install) above).
- The standard CIFAR-10 dataset files (automatically downloaded on first run).

Install Python dependencies:

```bash
pip install -r training/requirements.txt
```

> **Note:** All commands in this guide are run from the **project root directory** unless otherwise specified.

### Phase 1: Model Adaptation & Compilation

In this phase, we convert a standard neural network into an **FHE-friendly** version and compile it into an encrypted computation graph.

```
Baseline Training  →  Operator Replacement & Fine-tuning  →  Model Compilation
     (Step 1)                   (Step 2)                        (Step 3)
```

#### Step 1: Baseline Training

Train a standard ResNet-20 on CIFAR-10 with ReLU activations:

```bash
python examples/test_cifar10/train.py --epochs 150 --batch-size 128 --lr 0.1 --output-dir ./runs/cifar10/model --input-shape 3 32 32
```

**Output:** `./runs/cifar10/model/train_baseline.pth`

#### Step 2: Operator Replacement & Fine-Tuning

FHE does not support non-linear activations like ReLU directly. Run the following command to replace ReLU layers with polynomial functions, swap max pooling for average pooling, and fine-tune the parameters to maintain accuracy. The script automatically exports the adapted model to ONNX format and saves model weights in an H5 file.

```bash
python examples/test_cifar10/train.py \
  --poly_model_convert \
  --pretrained ./runs/cifar10/model/train_baseline.pth \
  --epochs 10 \
  --batch-size 36 \
  --lr 0.001 \
  --input-dir ./runs/cifar10/model \
  --export-dir ./runs/cifar10/task/server \
  --input-shape 3 32 32 \
  --degree 4 \
  --upper-bound 3.0 \
  --poly-module RangeNormPoly2d
```

Workflow of `train.py`: when `--poly_model_convert` is enabled, the script replaces FHE-incompatible operators before training and exports the adapted model after training. Without this flag, it performs standard baseline training only.

```python
# 1. Replace FHE-incompatible operators (only when --poly_model_convert is set)
if args.poly_model_convert:
    replace_maxpool_with_avgpool(model)
    replace_activation_with_poly(
        model,
        old_cls=nn.ReLU,
        new_module_factory=RangeNormPoly2d,
        upper_bound=args.upper_bound,
        degree=args.degree,
    )

# 2. Train (or fine-tune) the model
for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    scheduler.step()

# 3. Export to ONNX & H5 (only when --poly_model_convert is set)
if args.poly_model_convert:
    export_to_onnx(model, save_path=onnx_path, ...)
    fuse_and_export_h5(model, h5_path=h5_path, ...)
```

- `--pretrained`: loads the baseline checkpoint.
- `--input-dir`: directory containing the baseline model (also used as output for `.pth` and `.onnx`).
- `--export-dir`: directory for the H5 weight file, corresponding to the server-side model weights.
- `--upper-bound`: normalization upper bound for RangeNormPoly2d (default: `3.0`). Controls the input range for polynomial approximation.
- `--degree`: degree of the polynomial activation (choices: `2`, `4`, `8`; default: `4`). Higher degree gives better approximation but increases FHE computational depth.
- `--poly-module`: type of polynomial activation to replace ReLU (choices: `RangeNormPoly2d`, `Simple_Polyrelu`).

**Output:**

| File | Description |
|------|-------------|
| `./runs/cifar10/model/train_poly.pth` | Adapted model checkpoint with polynomial activations |
| `./runs/cifar10/model/trained_poly.onnx` | Exported adapted model in ONNX format |
| `./runs/cifar10/task/server/model_parameters.h5` | Model weights (BatchNorm absorbed into Conv) |

#### Step 3: High-Level FHE Compilation

Next, compile the adapted model into an **FHE Model Graph**. This step performs the following optimizations:

- Selecting optimal FHE parameters.
- Determining bootstrapping positions.
- Assigning FHE levels and scales to each layer.

```bash
python training/run_compile.py \
  --input=./runs/cifar10/model/trained_poly.onnx \
  --output=./runs/cifar10/ \
  --style=multiplexed
```

- `--input`: the exported adapted model in ONNX format from the previous step.
- `--output`: root output directory; the compiler generates `task/server/` and `task/client/` subdirectories underneath.
- `--style`: packing style — `multiplexed` (channel-multiplexed packing for higher slot utilization) or `ordinary` (one channel per ciphertext). Default: `multiplexed`.

**Output:**

| File | Description |
|------|-------------|
| `./runs/cifar10/model/pt.json` | Intermediate computation graph (JSON) |
| `./runs/cifar10/task/server/task_config.json` | Server-side inference task configuration |
| `./runs/cifar10/task/server/ckks_parameter.json` | CKKS encryption parameter configuration |
| `./runs/cifar10/task/server/nn_layers_ct_0.json` | Compiled encrypted computation graph (DAG) |
| `./runs/cifar10/task/client/task_config.json` | Client-side inference task configuration |
| `./runs/cifar10/task/client/ckks_parameter.json` | CKKS encryption parameter configuration |

### Phase 2: Encrypted Inference

Once the high-level graph is ready, we lower it to hardware-specific instructions for actual execution.

#### Step 1: Generate Low-Level Instructions

Generate low-level instructions from the project root:

```bash
python inference/interface/gen_mega_ag.py --task-dir ./runs/cifar10/task
```

#### Step 2: Runtime Execution

Use the `InferenceClient` / `InferenceServer` interface to run encrypted inference (see `examples/inference.cpp` for the complete example):

```cpp
#include "interface/inference_client.h"
#include "interface/inference_server.h"

// Client: generate keys and encrypt input
InferenceClient client("./task/client");
client.setup();
auto eval_ctx = client.export_eval_context();           // serialize public keys (Bytes)
auto encrypted_input = client.encrypt("./task/client/img.csv");  // encrypt and serialize ciphertext (Bytes)

// Server: import public keys, load model, run inference on serialized ciphertext
InferenceServer server("./task/server", use_gpu);
server.import_eval_context(eval_ctx);           // deserialize public keys
server.load_model();
auto encrypted_output = server.evaluate(encrypted_input);  // deserialize input, inference, serialize output

// Client: decrypt result
auto result = client.decrypt(encrypted_output);  // decrypt and deserialize output ciphertext, with secret key
```

Run the built example:

```bash
./build/examples/inference --task-dir ./runs/cifar10/task --input ./examples/test_cifar10/task/client/img.csv --verify
./build/examples/inference --task-dir ./runs/cifar10/task --input ./examples/test_cifar10/task/client/img.csv --gpu --verify
```

---

## Running Examples

> For a complete end-to-end walkthrough (from model adaptation to encrypted inference), see [Quick Start](#quick-start). The commands below assume pre-built examples with pre-prepared `task/` folders.

### Prerequisites

Make sure the project has been built successfully. See [Build & Install](#build--install) above. Examples are built automatically along with the project.

### Run
First, generate instructions for each example (run from the **project root directory**):


```bash
python inference/interface/gen_mega_ag.py --task-dir examples/test_mnist/task
python inference/interface/gen_mega_ag.py --task-dir examples/test_cifar10/task
python inference/interface/gen_mega_ag.py --task-dir examples/test_imagenet/task
```

Then use CTest to run examples from the `build/` directory. Each test runs encrypted inference and verifies the result against plaintext output:

```bash
cd build
ctest -L example-cpu -V          # All CPU examples
ctest -L example-gpu -V          # All GPU examples
ctest -R mnist -V                # MNIST only (CPU + GPU)
ctest -R mnist-cpu -V            # MNIST CPU only
ctest -R mnist-gpu -V            # MNIST GPU only
ctest -R cifar10 -V              # CIFAR-10 only (CPU + GPU)
ctest -R cifar10-cpu -V          # CIFAR-10 CPU only
ctest -R cifar10-gpu -V          # CIFAR-10 GPU only
ctest -R imagenet -V             # ImageNet only (CPU + GPU)
ctest -R imagenet-cpu -V         # ImageNet CPU only
ctest -R imagenet-gpu -V         # ImageNet GPU only
```

---

#### Performance

> Testing environment — Server: Intel Xeon Gold 6226R (32 cores) + NVIDIA RTX 5880 Ada (48GB); 128-bit security level.

| Task | Model | Dataset | Baseline Accuracy | FHE Accuracy | 16-thread CPU Latency (s) | GPU Latency (s) |
|------|-------|---------|-------------------|-------------|-----------------|-----------------|
| Classification | MobileNetV2 | ImageNet | 71.8% | 70.1% | 1210.0 | 82.4 |

For detailed benchmarks and methodology, see the [Technical Whitepaper](docs/en/whitepaper.md#performance-evaluation).

---

## Future Work

The ordered 128-bit-security and GPU-scheduling checklist, including accepted
gates and rejected parameter candidates, is maintained in
[`docs/cryptoface_security_gpu_scheduling.md`](docs/cryptoface_security_gpu_scheduling.md).

### CryptoFace GPU bootstrap correctness and performance

The Q44/P61/H192 eight-bootstrap candidate passes the full numerical 128-pair gate but **fails the complete 128-bit security gate**: sparse BDD MITM hybrid costs only **121.724 bits**, despite the non-hybrid BDD estimate of 128.484 bits. Its measured performance remains useful: **14.264 seconds of core inference per image** (p50 14.259 s, p95 15.715 s) and one pair every **10.686 seconds** across three GPUs. All 256 embeddings are finite, with maximum relative L2 error 0.028499 and maximum absolute error 0.006135. The rejected candidate is preserved at `../latti-ai-8boot-q44p61-security-fail-repro`; a secure baseline has not yet been promoted.

The historical serialized/synchronized fallback completed single-image FHE inference in **25.1807 seconds** with relative L2 error **0.0016412**. Earlier **18–19 second** concurrent variants are faster but are not correct: layer-by-layer verification first diverges at `_nets_0_layer1_Add_output_0_refreshed`, whose output is approximately 4x the expected value (relative L2 error exactly **3.0**).

The SEAL-equivalent CryptoFace placement has been reproduced separately in Latti: one bootstrap follows each backbone's `layer3.Add`, each bootstrap receives level 1 and refreshes to level 16, L2Norm has no bootstrap, and the final output is at level 1. This reduces the generated task from 20 physical bootstrap operations to 4. SEAL's raw initial level 31 cannot be used directly for ordinary Latti arithmetic: the current runtime exposes only `Q[0..24]` to that path and aborts at level 25. The executable mapping therefore starts the four patch inputs at level 24, which still supplies the 23 levels required before the scheduled bootstrap.

The level-24 task passes structural and CKKS-chain validation, but its current GPU execution allocates approximately 87.9 GB and then stalls GPU-idle in both default and single-stream modes. Schedule reduction, this scheduling/memory issue, and the GPU bootstrap arithmetic regression are independent concerns; the reduced schedule must pass full-output verification before it can replace the known-correct fallback.

Recovery testing on 2026-07-18 identified two historical custom parameter states. The older state has 31 Q primes, bootstrap output level 15, and 55-bit Sine primes; an eight-bootstrap reconstruction ran in 20.436 seconds but produced an invalid embedding. The later backup state has 32 Q primes, bootstrap output level 16, and 60-bit Sine primes; an eight-bootstrap reconstruction ran in 16.829 seconds but was also numerically invalid. A preserved-runtime test of the four-bootstrap SEAL schedule with the statically embedded 32-Q/55-bit factory completed in 25.587 seconds and remained invalid. These timings are diagnostic only and must not be reported as restored performance.

The preserved correct runtime archive exposes an additional compatibility constraint: its archived `ckks_parameter.json` uses the 32-Q/60-bit chain, while its preserved `mega_ag.json` and statically linked CryptoFace factory use the 32-Q/55-bit chain. A newly generated 60-bit graph is therefore rejected by exact Q/P-chain validation even though the archived mixed metadata ran through its 55-bit graph. A reproducible restoration requires a matched task, graph, Go factory, and GPU arithmetic core; bootstrap count or parameter metadata alone is insufficient.

The symmetric eight-bootstrap reconstruction is now available at `runs/cryptoface/full_pcnn_64x64_seal_schedule_q55_sealnorm_l1_dp8_l13/task`. It uses two logical/physical bootstraps per backbone, after `layer1.Add` and `layer3.Add`. Every bootstrap follows `level 2 -> /120 -> level 1 -> bootstrap -> level 16 -> x120 -> level 15`; the first refreshed segment is dropped once to start the next segment at level 14. Inputs start at level 13 so ordinary model arithmetic stays inside the 46-bit residual chain. Static validation confirms exactly eight physical bootstraps, 32 Q / 3 P, Q55, MessageRatio 2048, output level 16, and synchronized level-13 client/server inputs.

This schedule restores the historical performance envelope but is not yet correct. A full diagnostic run completed encrypted inference in **16.781 seconds**, but the final relative L2 error was approximately `1.52e19`. Mutation-free pruned probes isolate the first real divergence before bootstrapping: a level-13 input round trip is correct (relative L2 `1.32e-11`), the same one-layer `conv0` is correct on CPU (relative L2 `1.10e-7`), but GPU `conv0` is explosive at level 13 and returns NaNs at level 25. The immediate blocker is therefore Q55 GPU convolution/plaintext arithmetic, likely in the `encode_ringt -> ringt_to_mul -> mult_plain_mul -> rescale` path, not bootstrap placement. Detailed artifacts and reproduction notes are in `docs/cryptoface_gpu_recovery_20260718.md`.

The `conv0` regression was subsequently fixed. The GPU's internal plaintext-preparation context used `CkksBtpParameter::create_parameter()` unconditionally and silently encoded weights with the default 60-bit `N16QP1546H192H32` chain, while HEonGPU executed the graph with the explicit Q55 CryptoFace chain. `init_empty_context` now honors `btp_param_name`, and GPU plaintext export converts Lattigo `PlaintextMul` values out of Montgomery form before HEonGPU pointwise multiplication. The scalar multiply probes now have relative L2 errors of `1.28e-11` without rescale and `2.06e-11` with rescale; pruned GPU `conv0` matches CPU at relative L2 `1.10e-7`. The corrected full eight-bootstrap graph produces a valid embedding (max absolute error `0.003067`, relative L2 `0.0156`) but takes **34.942 seconds**. The earlier 16.781-second result was invalid and is not a performance baseline.

Sequential tuning on 2026-07-19 found no safe large speedup. Upstream convolution reduced rotations from 20,280 to 18,420 and preserved correctness, but its 35.638-second full run was not faster than the 34.942-second custom-convolution control; the custom version is preserved at branch `codex/conv-current-snapshot-20260719`. Increasing the key-switch basis from P3 to P4 or P5 made isolated arithmetic and the first full run faster, but repeated full runs became nondeterministically explosive, so the active configuration is restored to P3. Serialized one-stream execution remained correct in two runs and averaged 34.214 seconds versus 35.173 seconds for the two-stream serialized control. Bypassing the bootstrap mutex corrupted the embedding (relative L2 approximately `3.08e19`) without improving latency, so serialized bootstrap execution remains mandatory. Full measurements and artifacts are recorded in `docs/cryptoface_gpu_recovery_20260718.md`.

Profiling on 2026-07-20 localized the remaining 33-second regression to plaintext transfer amplification rather than convolution or bootstrap placement. The correct slow path issued 234,034 host-to-device copies totaling approximately 122 GB, versus 64,583 copies and approximately 32 GB on the historical fast path; an individual GPU bootstrap interval was only approximately 109 ms. The compact RingT bridge transferred only one `N`-coefficient polynomial but constructed every GPU plaintext with full `N * (level + 1)` storage. Retaining those oversized plaintexts also explained the approximately 87.9 GB model-load stall. An isolated CKKS RingT allocation overload now allocates exactly `N` coefficients while leaving ordinary CKKS and BFV plaintext allocation unchanged.

The isolated compact-allocation candidate passed exact GPU `conv0` verification in 1.757 seconds and restored full eight-bootstrap inference to **18.649 seconds**. Independent comparison of all 256 decrypted values against PyTorch gave max absolute error `0.003757` and relative L2 error `0.015628`, consistent with the known-correct eight-bootstrap baseline. A separate full run took 20.285 seconds, so timing variance still needs profiling before this becomes the default runtime. The validated state is preserved at `.codex/snapshots/gpu-correct-18p649s-20260721`; the exact archived 18.120-second run remains arithmetically invalid and is not the recovered result.

The level-24 SEAL schedule is now also recovered with exactly four bootstraps. The high-prime arithmetic failure was isolated to HerPN ciphertext squares crossing the 55-bit bootstrap-special primes; scale compensation fixes the square without consuming a level. Pairing compact RingT allocation with compact RingT transfer removes the model-load stall. The full four-bootstrap graph passes all 256 PyTorch outputs in **21.846 seconds** with makespan-first scheduling and two GPU streams (relative L2 `0.05005`). Memory-first is a safe fallback at 33.807 seconds; four streams regressed to 23.842 seconds. Detailed gates are in `docs/cryptoface_gpu_recovery_20260718.md`.

Future GPU tuning should proceed in this order:

1. Add a permanent regression test that asserts the CPU preprocessing context, serialized task, and GPU context use identical Q/P values before a ciphertext-plaintext multiply.
2. Tighten the four-bootstrap output from relative L2 `0.0477`–`0.0500` and the eight-bootstrap output from `0.0156` toward the preserved approximately `0.0016` baseline.
3. Add an allocation-size regression test for compact RingT transfer and re-profile H2D traffic; the target is approximately 65,000 copies and 32 GB rather than 234,034 copies and 122 GB.
4. Produce an operation-level profile of the 21.846-second four-bootstrap run and explain the 18.649–20.285-second eight-bootstrap timing range without removing serialized bootstrap synchronization.
5. Reduce the four-bootstrap runtime toward 18–19 seconds while retaining the upstream convolution and the full-output oracle; revisit the historical level-cost-1 convolution only if profiling justifies its larger plaintext footprint.
6. Require repeated finite 256-element PyTorch comparisons before promoting any faster parameter, decomposition, concurrency, or bootstrap-serialization variant.

A 2026-07-22 operation-profile pass established a new clean eight-bootstrap
control of **16.220–16.289 seconds** with relative L2 error approximately
`0.0160`, without changing the preserved runtime. The trace recorded 340,587
kernel launches, 65,024 asynchronous copies, and 47,509 host event waits; the
approximately 109 ms bootstrap kernel is not the dominant cost. Makespan
scheduling (16.470 s), one CUDA connection (17.929 s), a separate two-worker
event-cleanup pool (16.387 s versus a paired 16.289 s control), and non-timing
dependency events (19.067 s) all preserved numerical correctness but did not
improve latency, so none is promoted. The next performance work should focus
on HEonGPU operation fusion or CUDA graph capture, then further resident
plaintext reuse or transfer batching. Full details are in
`docs/cryptoface_gpu_recovery_20260718.md`.

A source-reconstruction follow-up on 2026-07-22 found that the correct but
slow isolated build was executing the same 340,587 kernels per image as the
frozen runtime while GPU busy time fell from 65.5% to 27-30%. The regression
was scheduling rather than arithmetic: the reconstructed GPU task inherited
`MEMORY_FIRST`, whereas the frozen source anchor used `MAKESPAN_FIRST`.
Making the GPU policy explicit reduced reported encrypted inference from
32.604/23.754 seconds to **15.182/14.713 seconds** for two queued images.
Both 256-value embeddings remained finite with relative L2 errors
`0.019190/0.020056`, maximum absolute errors below `0.0038`, and all eight
physical bootstraps intact. The optimized source is preserved at
`../lattisense-8boot-opt` revision `f2293f910afb9f8b5242f918a206f0ddb7e7ab12`;
its detailed trace comparison is in
`doc/cryptoface_8boot_gpu_optimization.md` in that repository. The immutable
`latti-ai-8boot-multigpu-repro` archive remains unchanged.

The same optimized build then passed the complete three-GPU 128-pair gate.
Across 256 encrypted images it averaged **14.294 seconds** reported inference
and **16.024 seconds** queued wall time, compared with 14.448 and 16.178
seconds for the frozen reference run. Processing completed in 1372.047 seconds
versus 1383.503 seconds, with worker balance 86/85/85. Every embedding was
finite; image-wise relative L2 versus the frozen output had mean `0.020198`
and maximum `0.027118`, below the `0.03` threshold. P95/P99 inference latency
was slightly higher despite the better mean, and peak sampled memory increased
from 87,962 MiB to 89,607 MiB, so this result establishes performance parity
with a modest favorable mean rather than an across-the-board speedup.

---

## Documentation

- **Technical Whitepaper**: See [docs/en/whitepaper.md](docs/en/whitepaper.md)
- **Build Guide**: See [docs/en/build-guide.md](docs/en/build-guide.md)
- **API Reference**: See [docs/en/APIs_Reference.md](docs/en/APIs_Reference.md)

## Related Links

- **HEonGPU**: [GPU-accelerated Homomorphic Encryption Library](https://github.com/Alisah-Ozcan/HEonGPU)
- **Lattigo**: [Go Homomorphic Encryption Library](https://github.com/tuneinsight/lattigo)

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or feedback, please reach out:

- Email: lattisense.support@cipherflow.cn
