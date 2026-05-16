//! Ranking-kernel equivalence tests, swept across [`CANONICAL_PARAMETER_POINTS`]
//! (exponent + tolerance regimes). Each iteration replays the LCG sampling
//! schedule on the CPU and asserts identical candidate indices, best
//! position, and top-2 gap.

use crate::burn::{
    EntropyMetric, KernelMetric, LinearCosineMetric, LinearEntropyMetric,
    ModifiedLinearCosineMetric, ModifiedLinearEntropyMetric, RankingConfig, SpectralKernelBackend,
    ranking_kernel,
};

use super::fixtures::{
    CANONICAL_PARAMETER_POINTS, ParameterPoint, ReferenceSpectrum, TEST_EPSILON, TEST_MAX_PEAKS,
    cpu_linear_cosine, cpu_linear_entropy, cpu_modified_linear_cosine, cpu_modified_linear_entropy,
    reference_spectra_at, spectrum_batch, spectrum_rows,
};

#[cfg(feature = "burn-cuda")]
type TestBackend = burn::backend::Cuda<f32, i32>;
#[cfg(all(feature = "burn-cpu", not(feature = "burn-cuda")))]
type TestBackend = burn::backend::Cpu<f32, i32>;

type TestDevice = burn::tensor::Device<TestBackend>;

fn cosine_ranking_config<M: KernelMetric>(point: ParameterPoint) -> RankingConfig<M> {
    M::ranking_config()
        .with_batch_start(1)
        .with_batch_items(10)
        .with_candidates_per_anchor(7)
        .with_mz_power(point.mz_power)
        .with_intensity_power(point.intensity_power)
        .with_mz_tolerance(point.mz_tolerance)
        .with_max_peaks(TEST_MAX_PEAKS)
        .with_seed(12_345)
        .with_epsilon(TEST_EPSILON)
}

fn entropy_ranking_config<M: EntropyMetric>(
    point: ParameterPoint,
    weighted: bool,
) -> RankingConfig<M> {
    cosine_ranking_config::<M>(point).with_weighted(weighted)
}

fn run_ranking_test_with<M, F, MakeConfig>(
    cpu_score: F,
    tolerance: f32,
    make_config: MakeConfig,
) where
    M: KernelMetric,
    TestBackend: SpectralKernelBackend<M>,
    F: Fn(ParameterPoint, &ReferenceSpectrum, &ReferenceSpectrum) -> f32 + Copy,
    MakeConfig: Fn(ParameterPoint) -> RankingConfig<M>,
{
    let device = TestDevice::default();

    for &point in CANONICAL_PARAMETER_POINTS {
        let spectra = reference_spectra_at(point.mz_tolerance);
        let spectra = &spectra[..12];
        let rows = spectrum_rows(spectra);

        let config = make_config(point);

        let teacher = spectrum_batch::<TestBackend>(&rows, &device);
        let output = ranking_kernel::<TestBackend, M>(teacher, config);

        let candidate_index = output
            .candidate_index
            .into_data()
            .to_vec::<i32>()
            .expect("candidate indices should be i32");
        let best_position = output
            .best_position
            .into_data()
            .to_vec::<i32>()
            .expect("best positions should be i32");
        let top2_gap = output
            .top2_gap
            .into_data()
            .to_vec::<f32>()
            .expect("top-2 gaps should be f32");

        let candidate_count = config.effective_candidates_per_anchor();
        for anchor in 0..config.batch_items() {
            let expected = ranking_reference(spectra, anchor, &config, |left, right| {
                cpu_score(point, left, right)
            });
            let start = anchor * candidate_count;
            let actual_candidates = &candidate_index[start..start + candidate_count];
            assert_eq!(
                actual_candidates,
                expected.candidate_indices.as_slice(),
                "anchor {anchor} ({}) at {point:?}: candidate indices diverged",
                M::NAME,
            );
            assert_eq!(
                best_position[anchor] as usize, expected.best_candidate_position,
                "anchor {anchor} ({}) at {point:?}: best position diverged",
                M::NAME,
            );
            assert!(
                (top2_gap[anchor] - expected.top2_gap).abs() < tolerance,
                "anchor {anchor} ({}) at {point:?}: gpu={} cpu={}",
                M::NAME,
                top2_gap[anchor],
                expected.top2_gap,
            );
            assert!(
                !actual_candidates.contains(&(anchor as i32)),
                "anchor {anchor} ({}) at {point:?}: self-pairing in candidates",
                M::NAME,
            );
            for (left, left_value) in actual_candidates.iter().enumerate() {
                for right_value in actual_candidates.iter().skip(left + 1) {
                    assert_ne!(
                        left_value, right_value,
                        "anchor {anchor} ({}) at {point:?}: duplicate candidate {left_value}",
                        M::NAME,
                    );
                }
            }
        }
    }
}

struct RankingReference {
    candidate_indices: Vec<i32>,
    best_candidate_position: usize,
    top2_gap: f32,
}

/// CPU reference: replicates the LCG sampling schedule bit-for-bit and scores
/// each candidate using `cpu_score` (already bound to the current parameter
/// point by the caller).
fn ranking_reference<M: KernelMetric>(
    spectra: &[(&'static str, ReferenceSpectrum)],
    anchor: usize,
    config: &RankingConfig<M>,
    cpu_score: impl Fn(&ReferenceSpectrum, &ReferenceSpectrum) -> f32,
) -> RankingReference {
    let batch_start = config.batch_start();
    let batch_items = config.batch_items();
    assert!(batch_start + batch_items <= spectra.len());

    let mut state =
        config.seed() as u32 ^ (((anchor as u32) + 1) * 40503) ^ ((batch_start as u32) >> 16);
    if state == 0 {
        state = 0x6d2b_79f5;
    }
    let partner_slots = (batch_items - 1) as u32;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    let offset = state % partner_slots;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    let mut stride = (state % partner_slots) + 1;
    while gcd(stride, partner_slots) != 1 {
        stride += 1;
        if stride > partner_slots {
            stride = 1;
        }
    }

    let mut best_score = f32::NEG_INFINITY;
    let mut second_best_score = f32::NEG_INFINITY;
    let mut best_candidate_position = 0usize;
    let candidates = config.effective_candidates_per_anchor();
    let anchor_spectrum = &spectra[batch_start + anchor].1;
    let mut candidate_indices = Vec::with_capacity(candidates);

    for candidate_position in 0..candidates {
        let mut local_partner =
            ((offset + (candidate_position as u32) * stride) % partner_slots) as usize;
        if local_partner >= anchor {
            local_partner += 1;
        }
        candidate_indices.push(local_partner as i32);

        let partner_spectrum = &spectra[batch_start + local_partner].1;
        let score = cpu_score(anchor_spectrum, partner_spectrum);
        if score > best_score {
            second_best_score = best_score;
            best_score = score;
            best_candidate_position = candidate_position;
        } else if score > second_best_score {
            second_best_score = score;
        }
    }

    RankingReference {
        candidate_indices,
        best_candidate_position,
        top2_gap: (best_score - second_best_score).clamp(0.0, 1.0),
    }
}

fn gcd(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

#[test]
fn ranking_matches_cpu_linear_cosine() {
    run_ranking_test_with::<LinearCosineMetric, _, _>(
        cpu_linear_cosine,
        1.0e-4,
        cosine_ranking_config::<LinearCosineMetric>,
    );
}

#[test]
fn ranking_matches_cpu_modified_linear_cosine() {
    run_ranking_test_with::<ModifiedLinearCosineMetric, _, _>(
        cpu_modified_linear_cosine,
        2.0e-4,
        cosine_ranking_config::<ModifiedLinearCosineMetric>,
    );
}

#[test]
fn ranking_matches_cpu_linear_entropy_unweighted() {
    run_ranking_test_with::<LinearEntropyMetric, _, _>(
        |p, l, r| cpu_linear_entropy(p, false, l, r),
        1.0e-4,
        |p| entropy_ranking_config::<LinearEntropyMetric>(p, false),
    );
}

#[test]
fn ranking_matches_cpu_linear_entropy_weighted() {
    run_ranking_test_with::<LinearEntropyMetric, _, _>(
        |p, l, r| cpu_linear_entropy(p, true, l, r),
        2.0e-4,
        |p| entropy_ranking_config::<LinearEntropyMetric>(p, true),
    );
}

#[test]
fn ranking_matches_cpu_modified_linear_entropy_unweighted() {
    run_ranking_test_with::<ModifiedLinearEntropyMetric, _, _>(
        |p, l, r| cpu_modified_linear_entropy(p, false, l, r),
        2.0e-4,
        |p| entropy_ranking_config::<ModifiedLinearEntropyMetric>(p, false),
    );
}

#[test]
fn ranking_matches_cpu_modified_linear_entropy_weighted() {
    run_ranking_test_with::<ModifiedLinearEntropyMetric, _, _>(
        |p, l, r| cpu_modified_linear_entropy(p, true, l, r),
        2.0e-4,
        |p| entropy_ranking_config::<ModifiedLinearEntropyMetric>(p, true),
    );
}
