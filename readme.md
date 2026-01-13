# KLT (Kanade-Lucas-Tomasi) Feature Tracker with CUDA

This repository contains an implementation of the KLT feature tracker with both CPU and GPU (CUDA) versions.

## Building the Code

Navigate to the `src/v3` directory to build the code.

### CPU Version

# KLT (Kanade–Lucas–Tomasi) Feature Tracker — v3 (optimized GPU)

This repository implements the KLT feature tracker and includes multiple development versions. This README focuses on the v3 optimized GPU implementation and explains the KLT algorithm and the optimization process used across versions.

**About KLT**

- **Goal:** Track sparse image features across frames by estimating optical flow for small image patches.
- **Core idea (Lucas–Kanade):** Assume constant image motion within a small window. Solve for the 2D displacement that minimizes the sum of squared intensity differences using image gradients, yielding a small linear system per feature:

	- Compute image gradients (Ix, Iy) and the temporal difference (It).
	- Form the normal equations (A^T A)u = A^T b where A contains gradients and b contains temporal terms.
	- Solve the 2×2 system (often regularized) to get the flow vector for the feature.

- **Tomasi (feature selection):** Choose features with strong, well-conditioned gradient structure (eigenvalues of A^T A) so tracking is stable.
- **Pyramids:** To handle larger motions, operate on image pyramids (coarse-to-fine): estimate flow at coarse levels and refine at finer levels.

**Typical KLT pipeline**

- Detect good features in the first frame (Tomasi corner detector).
- Build image pyramids for both frames.
- For each pyramid level (coarse→fine):
	- Warp patches using current displacement estimate.
	- Compute gradients and temporal residuals.
	- Solve per-feature linear system, iterate (e.g., Gauss–Newton) for refinement.
	- Propagate refined displacement to next (finer) level.

**Optimization summary (project versions)**

- **v1 — Profiling & CPU hotspots:** Focused on profiling the original CPU implementation to identify hotspots: image convolution, pyramid construction, gradient computation, per-feature window ops, and the small linear-system solves. The profiling guided subsequent optimization choices.
- **v2 — Naive implementation:** A baseline implementation (minimal optimizations) used to measure performance and verify correctness against the CPU reference.
- **v3 — Optimized GPU implementation (this branch):** Targeted high-throughput GPU execution with these key optimizations:
	- Kernel design: fuse operations where practical (e.g., gradients + residuals) to reduce global memory traffic.
	- Memory: coalesce global accesses, use shared memory for sliding-window operations, and leverage texture/read-only caches for image sampling where beneficial.
	- Compute: implement warp- and block-level reductions for per-feature accumulation (A^T A and A^T b), use warp shuffles for small reductions, and minimize divergence in inner loops.
	- Parallelism: map features and pyramid levels carefully to threads/blocks to maximize occupancy and balance load.
	- Overlap & streaming: reduce host-device transfers, use asynchronous copies and CUDA streams for pipelining when processing sequences.
	- Numerical: stabilize 2×2 solves with small regularization and grouped solves to exploit vectorization/parallel reductions.

- **v4 — OpenACC implementation:** An alternative acceleration approach using OpenACC; available on the `v4` branch (see note below).

Note: v4 is available in the `v4` branch; this README does not include run instructions for v4 or other versions — only for v3.

## Build & run (v3)

All build and run instructions below assume you are working in the repository root and have the required toolchains installed (GCC/Make and, for GPU, the CUDA toolkit).

- Build GPU-accelerated v3 (requires CUDA):

```bash
cd src/v3
make gpu
```

This builds the GPU library and example programs in `src/v3`.

- (Optional) Build CPU v3 targets for comparison:

```bash
cd src/v3
make cpu
```

- Run an example (GPU):

```bash
cd src/v3
./example1_gpu
```

Replace `example1_gpu` with any of the provided GPU example binaries to run other test cases.

## Data

- Example input images are in the `data/` folder. Use these for quick tests.

## Notes and next steps

- For further investigation, consult the profiling outputs in `src/v1/Profile` to see the original hotspot analysis that informed the v3 optimizations.
- If you want, I can add a short walkthrough that maps specific profiling hotspots to the concrete optimizations implemented in v3.
