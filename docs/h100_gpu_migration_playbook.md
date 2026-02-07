# H100 GPU Migration Playbook for JAX Inference

## Purpose

This document explains why we should run inference on NVIDIA H100 GPUs, how to do it safely, and what to validate before calling results production-grade.

This codebase should continue to run on CPU by default on machines without CUDA support.

## Why H100 GPUs Are Worth It

For this project, runtime is dominated by repeated model evaluations during MAP + NUTS. H100-class GPUs help most when we execute many repeated, numerically similar JAX kernels.

Expected benefits:

- lower wall-clock for posterior sampling (especially NUTS warmup and long chains),
- faster throughput for case-study sweeps and synthetic recovery tests,
- better scaling for denser corner plots and higher ESS targets.

Important caveat:

- speedups depend on batching and parallel execution strategy; a purely sequential chain workflow will underuse the GPU.

## Physics and Precision Requirements

This branch uses `jax_enable_x64=True` for physical robustness. H100 supports float64 well, so GPU migration is compatible with current physics assumptions.

Rules:

- keep x64 enabled for production inference and final diagnostics,
- only use float32 for exploratory speed tests, never for final scientific outputs unless explicitly validated.

## Recommended Runtime Modes

### Mode A: CPU (this machine)

Use this when CUDA is unavailable or when debugging:

- `JAX_PLATFORMS=cpu`
- `JAX_ENABLE_X64=1`

### Mode B: H100 CUDA

Use this for production-scale inference:

- install CUDA-enabled JAX wheel matching cluster CUDA version,
- keep `JAX_ENABLE_X64=1`,
- run multiple chains in parallel workers (process-level) or multi-device strategy, not strictly sequential chains.

## Environment Setup Checklist (H100)

1. Verify CUDA driver/runtime and GPU visibility (`nvidia-smi`).
2. Install CUDA JAX build (version-pinned with `jaxlib`).
3. Confirm backend and devices in Python:
   - `jax.default_backend() == "gpu"`
   - `jax.devices()` includes expected H100s.
4. Confirm x64 mode:
   - `jax.config.read("jax_enable_x64")` is `True`.
5. Run a small smoke inference and ensure no backend fallback to CPU.

## Inference Strategy on H100

Preferred approach:

- keep MAP optimization single-process per fit,
- run NUTS chains as independent workers (or device-parallel chains) with deterministic seed offsets,
- aggregate draws for KDE/corner plots and diagnostics.

Why:

- improves hardware utilization,
- keeps code simple and robust,
- avoids over-coupling chain execution to one process.

## Benchmark and Acceptance Metrics

When switching CPU -> H100, compare:

- wall-clock time per fit,
- effective sample size per second (ESS/sec),
- divergences,
- `r_hat`,
- BFMI and energy diagnostics,
- posterior moment deltas (`M0`, `M1`, `M2`) versus CPU baseline.

Acceptance target:

- materially better ESS/sec and runtime with no degradation in diagnostics or physics consistency.

## Validation Protocol Before Production Use

1. Run identical inference case(s) on CPU and H100 with fixed seeds.
2. Compare posterior summaries (`theta` means/intervals) and predicted moments.
3. Confirm diagnostics are at least as good on GPU.
4. Confirm reproducibility across reruns on same hardware/software stack.
5. Record versions (`jax`, `jaxlib`, CUDA, driver) with results.

## Operational Guidance

- Keep CPU path first-class for portability and debugging.
- Keep GPU path version-pinned and documented.
- Never mix unvalidated precision changes with scientific result generation.

## Current Local Constraint

On this Apple/Metal machine we stay on CPU for reliable x64 inference and continue development/validation there. H100 migration should be exercised on a CUDA machine using this playbook.
