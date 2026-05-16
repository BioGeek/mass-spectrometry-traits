//! Equivalence tests for the `Fusion<Cuda<f32, i32>>` backend.
//!
//! The fusion runtime should produce identical forward-pass scores to the
//! raw `Cuda<f32, i32>` backend, it just batches and reorders surrounding
//! ops. These tests guard against regressions in the `CustomOpIr`-based
//! plumbing (one named op per metric x shape) since the autoencoder's fusion
//! code path has no upstream test coverage.

#![cfg(feature = "burn-fusion")]

use burn::backend::cuda::CudaDevice;
use burn_cubecl::CubeBackend;
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_fusion::Fusion;

use crate::burn::{
    KernelMetric, LinearCosineMetric, cross_kernel, paired_kernel, ranking_kernel,
};

use super::fixtures::{
    DEFAULT_TEST_POINT, PAIR_CHUNK_SIZE, TEST_EPSILON, TEST_INTENSITY_POWER, TEST_MAX_PEAKS,
    TEST_MZ_POWER, TEST_MZ_TOLERANCE, all_pair_indices, cpu_linear_cosine, pair_batches, pair_rows,
    pairwise_params_constant, reference_spectra, spectrum_batch, spectrum_rows,
};

type RawBackend = CubeBackend<CudaRuntime, f32, i32, u8>;
type FusionBackend = Fusion<RawBackend>;

#[test]
fn fusion_paired_matches_cpu_linear_cosine() {
    let device = CudaDevice::default();
    let spectra = reference_spectra();
    let spectra = &spectra[..16];
    let indices = all_pair_indices(spectra.len());

    let config = LinearCosineMetric::paired_config()
        .with_max_peaks(TEST_MAX_PEAKS)
        .with_epsilon(TEST_EPSILON);

    for chunk in indices.chunks(PAIR_CHUNK_SIZE) {
        let pairs = pair_rows(spectra, chunk);
        let row_count = pairs.indices.len();
        let (left, right) = pair_batches::<FusionBackend>(&pairs, &device);
        let params = pairwise_params_constant::<FusionBackend>(
            row_count,
            TEST_MZ_POWER,
            TEST_INTENSITY_POWER,
            TEST_MZ_TOLERANCE,
            &device,
        );

        let scores = paired_kernel::<FusionBackend, LinearCosineMetric>(left, right, params, config)
            .into_data()
            .to_vec::<f32>()
            .expect("kernel output should be f32");

        for (row, &(left_index, right_index)) in pairs.indices.iter().enumerate() {
            let (_, left) = &spectra[left_index];
            let (_, right) = &spectra[right_index];
            let cpu = cpu_linear_cosine(DEFAULT_TEST_POINT, left, right);
            assert!(
                (scores[row] - cpu).abs() < 1.0e-4,
                "fusion paired diverged at row {row}: gpu={} cpu={}",
                scores[row],
                cpu,
            );
        }
    }
}

#[test]
fn fusion_cross_matches_cpu_linear_cosine() {
    let device = CudaDevice::default();
    let spectra = reference_spectra();
    let left = &spectra[..6];
    let right = &spectra[6..12];

    let left_rows = spectrum_rows(left);
    let right_rows = spectrum_rows(right);
    let left_batch = spectrum_batch::<FusionBackend>(&left_rows, &device);
    let right_batch = spectrum_batch::<FusionBackend>(&right_rows, &device);

    let config = LinearCosineMetric::cross_config()
        .with_mz_power(TEST_MZ_POWER)
        .with_intensity_power(TEST_INTENSITY_POWER)
        .with_mz_tolerance(TEST_MZ_TOLERANCE)
        .with_max_peaks(TEST_MAX_PEAKS)
        .with_epsilon(TEST_EPSILON);

    let scores = cross_kernel::<FusionBackend, LinearCosineMetric>(left_batch, right_batch, config)
        .into_data()
        .to_vec::<f32>()
        .expect("cross kernel output should be f32");

    assert_eq!(scores.len(), left.len() * right.len());
    for (i, (_, left_spectrum)) in left.iter().enumerate() {
        for (j, (_, right_spectrum)) in right.iter().enumerate() {
            let gpu_score = scores[i * right.len() + j];
            let cpu = cpu_linear_cosine(DEFAULT_TEST_POINT, left_spectrum, right_spectrum);
            assert!(
                (gpu_score - cpu).abs() < 1.0e-4,
                "fusion cross diverged at ({i}, {j}): gpu={gpu_score} cpu={cpu}",
            );
        }
    }
}

#[test]
fn fusion_ranking_runs_and_returns_expected_shapes() {
    let device = CudaDevice::default();
    let spectra = reference_spectra();
    let spectra = &spectra[..12];
    let rows = spectrum_rows(spectra);

    let config = LinearCosineMetric::ranking_config()
        .with_batch_start(1)
        .with_batch_items(10)
        .with_candidates_per_anchor(7)
        .with_mz_power(TEST_MZ_POWER)
        .with_intensity_power(TEST_INTENSITY_POWER)
        .with_mz_tolerance(TEST_MZ_TOLERANCE)
        .with_max_peaks(TEST_MAX_PEAKS)
        .with_seed(12_345)
        .with_epsilon(TEST_EPSILON);

    let teacher = spectrum_batch::<FusionBackend>(&rows, &device);
    let output = ranking_kernel::<FusionBackend, LinearCosineMetric>(teacher, config);

    let candidate_count = config.effective_candidates_per_anchor();
    assert_eq!(
        output.candidate_index.dims(),
        [config.batch_items(), candidate_count]
    );
    assert_eq!(output.best_position.dims(), [config.batch_items()]);
    assert_eq!(output.top2_gap.dims(), [config.batch_items()]);

    // Round-trip the data so the fusion runtime materialises everything.
    let _ = output.candidate_index.into_data().to_vec::<i32>().expect("i32");
    let _ = output.best_position.into_data().to_vec::<i32>().expect("i32");
    let _ = output.top2_gap.into_data().to_vec::<f32>().expect("f32");
}
