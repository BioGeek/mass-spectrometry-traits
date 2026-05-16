//! Equivalence tests comparing GPU kernel output against the in-crate CPU
//! similarity implementations.
//!
//! Gated on `feature = "burn-cuda"` or `feature = "burn-cpu"`. The CUDA path
//! is the primary dev-machine target (runs on a real GPU). The CPU path
//! uses `cubecl-cpu`'s MLIR-based CPU runtime and exists so the same
//! equivalence assertions can run in CI without GPU hardware. Per-file
//! gates further restrict CUDA-only tests (e.g. `autodiff`, `fusion`) to
//! the CUDA configuration.

#![cfg(any(feature = "burn-cuda", feature = "burn-cpu"))]

pub(super) mod fixtures;
#[cfg(feature = "burn-autodiff")]
mod autodiff;
mod cross;
mod edge_cases;
#[cfg(feature = "burn-fusion")]
mod fusion;
mod paired;
#[cfg(feature = "proptest")]
mod proptest;
mod ranking;
mod symmetry;
