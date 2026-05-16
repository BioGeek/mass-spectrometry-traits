//! Equivalence tests comparing GPU kernel output against the in-crate CPU
//! similarity implementations.
//!
//! Gated on `feature = "burn-cuda"` because the tests instantiate the CUDA
//! runtime explicitly. Tests are skipped (not compiled) on machines without
//! a CUDA toolchain.

#![cfg(feature = "burn-cuda")]

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
