//! Public Rust-level API for the Burn / CubeCL spectral similarity kernels.
//!
//! The module is split into two submodules for navigability:
//!
//! - [`configs`]: builder-style config types ([`ScoringParams`],
//!   [`RankingWindow`], [`PairedConfig`], [`CrossConfig`], [`RankingConfig`]).
//! - [`bundles`]: tensor-bundle structs ([`SpectrumBatch`],
//!   [`PairwiseParams`], [`SpectrumPrimitive`], [`PairwisePrimitive`]).
//!
//! Everything is re-exported here via `pub use {configs, bundles}::*;` so
//! external callers continue to import from `crate::burn::api::*` directly.
//! The submodule organization is internal navigation, not part of the public
//! path contract.

use burn::tensor::Int as TensorInt;
use burn::tensor::Tensor as BurnTensor;
use burn::tensor::TensorPrimitive;
use burn::tensor::backend::Backend;
use burn::tensor::ops::{FloatTensor, IntTensor};

use crate::burn::metrics::KernelMetric;

pub mod bundles;
pub mod configs;

pub use bundles::*;
pub use configs::*;

/// Hard upper bound on `max_peaks` for any kernel launch. Above this value
/// the modified-variant per-thread scratch arrays start to spill out of
/// register / local-memory budget on consumer NVIDIA hardware.
pub const MAX_PEAKS_LIMIT: usize = 256;

/// Backend trait for the spectral similarity GPU kernels.
///
/// Generic over the metric marker `M`. Implemented as a single blanket
/// `impl<…, M> SpectralKernelBackend<M> for …` on each backend type, so
/// new metrics get every shape on every backend with no extra wiring.
pub trait SpectralKernelBackend<M: KernelMetric>: Backend {
    /// Paired (1-to-1 batch) kernel: `score[i] = sim(left[i], right[i])`.
    ///
    /// All tensors in `left`, `right`, and `params` must live on the same
    /// device. Cosine metrics ignore the precursor tensors but still require
    /// them so the trait shape is uniform across metrics.
    fn paired_score(
        left: SpectrumPrimitive<Self>,
        right: SpectrumPrimitive<Self>,
        params: PairwisePrimitive<Self>,
        config: PairedConfig<M>,
    ) -> FloatTensor<Self>;

    /// Cross / all-pairs kernel: `output[i, j] = sim(left[i], right[j])`.
    ///
    /// `left` has shape `[M, P]`, `right` has shape `[N, P]`. Output is
    /// `[M, N]`. Scoring parameters live in [`CrossConfig`] as scalars, see
    /// its docs for the broadcasting rationale.
    fn cross_score(
        left: SpectrumPrimitive<Self>,
        right: SpectrumPrimitive<Self>,
        config: CrossConfig<M>,
    ) -> FloatTensor<Self>;

    /// Ranking kernel: deterministic LCG sampling of `k` non-self partners per
    /// anchor inside the teacher cache, plus top-2 reduce.
    ///
    /// `teacher` has shape `[N, P]`. Returns
    /// `(candidate_index[B, k], best_position[B], top2_gap[B])` with
    /// `B = config.batch_items()`.
    fn ranking_score(
        teacher: SpectrumPrimitive<Self>,
        config: RankingConfig<M>,
    ) -> (IntTensor<Self>, IntTensor<Self>, FloatTensor<Self>);
}

/// Public wrapper around [`SpectralKernelBackend::paired_score`].
///
/// `left` and `right` are `[batch, peak_width]` plus a `[batch]` precursor
/// each. `params` carries the three `[batch]` per-row scoring tensors.
/// Returns `[batch]`.
pub fn paired_kernel<B, M>(
    left: SpectrumBatch<B>,
    right: SpectrumBatch<B>,
    params: PairwiseParams<B>,
    config: PairedConfig<M>,
) -> BurnTensor<B, 1>
where
    B: SpectralKernelBackend<M>,
    M: KernelMetric,
{
    let out = B::paired_score(
        left.into_primitive(),
        right.into_primitive(),
        params.into_primitive(),
        config,
    );
    BurnTensor::from_primitive(TensorPrimitive::Float(out))
}

/// Public wrapper around [`SpectralKernelBackend::cross_score`].
///
/// `left` has shape `[M, P]`, `right` has shape `[N, P]`. Returns `[M, N]`.
pub fn cross_kernel<B, M>(
    left: SpectrumBatch<B>,
    right: SpectrumBatch<B>,
    config: CrossConfig<M>,
) -> BurnTensor<B, 2>
where
    B: SpectralKernelBackend<M>,
    M: KernelMetric,
{
    let out = B::cross_score(left.into_primitive(), right.into_primitive(), config);
    BurnTensor::from_primitive(TensorPrimitive::Float(out))
}

/// Named output of [`ranking_kernel`]. Mirrors the three-tensor result with
/// labelled fields so downstream callers don't repeat the tuple destructure
/// at every call site.
///
/// * `candidate_index`: `[batch_items, k]` int tensor, partner row indices
///   inside the teacher cache (still in the cache's index space, so add
///   `config.batch_start()` if you need absolute indices).
/// * `best_position`: `[batch_items]` int tensor, the column of
///   `candidate_index` that holds the top-1 partner per anchor.
/// * `top2_gap`: `[batch_items]` float tensor, `score(top-1) - score(top-2)`
///   per anchor, clamped to `[0, 1]`.
#[derive(Debug)]
pub struct RankingOutput<B: Backend> {
    pub candidate_index: BurnTensor<B, 2, TensorInt>,
    pub best_position: BurnTensor<B, 1, TensorInt>,
    pub top2_gap: BurnTensor<B, 1>,
}

/// Public wrapper around [`SpectralKernelBackend::ranking_score`].
///
/// `teacher` carries the `[N, P]` peak grids and `[N]` precursor masses for
/// the full cache. See [`RankingOutput`] for the shape of the returned
/// tensors.
pub fn ranking_kernel<B, M>(teacher: SpectrumBatch<B>, config: RankingConfig<M>) -> RankingOutput<B>
where
    B: SpectralKernelBackend<M>,
    M: KernelMetric,
{
    let (candidate_index, best_position, top2_gap) =
        B::ranking_score(teacher.into_primitive(), config);

    RankingOutput {
        candidate_index: BurnTensor::new(candidate_index),
        best_position: BurnTensor::new(best_position),
        top2_gap: BurnTensor::from_primitive(TensorPrimitive::Float(top2_gap)),
    }
}
