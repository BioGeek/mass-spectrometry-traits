use burn_cubecl::cubecl::prelude::*;

use crate::burn::SpectralPairScorer;
use crate::burn::metrics::LinearCosineMetric;

#[cube]
impl SpectralPairScorer for LinearCosineMetric {
    fn score_rows<F: Float>(
        left_mz: &Tensor<F>,
        left_intensity: &Tensor<F>,
        _left_precursor: &Tensor<F>,
        left_row: usize,
        right_mz: &Tensor<F>,
        right_intensity: &Tensor<F>,
        _right_precursor: &Tensor<F>,
        right_row: usize,
        mz_p: F,
        intensity_p: F,
        tolerance: F,
        eps: F,
        #[comptime] max_peaks: u32,
        #[comptime] _weighted: bool,
    ) -> F {
        linear_cosine_score_rows::<F>(
            left_mz,
            left_intensity,
            left_row,
            right_mz,
            right_intensity,
            right_row,
            mz_p,
            intensity_p,
            tolerance,
            eps,
            max_peaks,
        )
    }
}

/// Two-pointer sweep over two preprocessed peak rows.
///
/// Mirrors `LinearCosine::similarity` (CPU) in `src/structs/linear_cosine.rs`:
/// computes per-row maxima for m/z, intensity, and product space, stores the
/// normalized peak products in per-thread scratch arrays
/// (`Array::<F>::new(max_peaks)`), then performs the matching sweep by
/// reading from the scratch arrays directly. This caching pattern avoids the
/// repeated `powf` calls that the naive call-`peak_product`-on-demand
/// formulation incurs, each peak's `intensity.powf(p)` and `mz.powf(p)` are
/// computed once per row pair, not once per pass.
#[cube]
pub fn linear_cosine_score_rows<F: Float>(
    left_mz: &Tensor<F>,
    left_intensity: &Tensor<F>,
    left_row: usize,
    right_mz: &Tensor<F>,
    right_intensity: &Tensor<F>,
    right_row: usize,
    mz_p: F,
    intensity_p: F,
    tolerance: F,
    eps: F,
    #[comptime] max_peaks: u32,
) -> F {
    let max_peaks_usize = comptime!(max_peaks as usize);
    let left_peaks = left_mz.shape(1);
    let right_peaks = right_mz.shape(1);
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);

    let mut left_products = Array::<F>::new(max_peaks_usize);
    let mut right_products = Array::<F>::new(max_peaks_usize);

    let left_norm_square = prepare_linear_cosine_row::<F>(
        left_mz,
        left_intensity,
        left_row,
        &mut left_products,
        left_peaks,
        max_peaks_usize,
        mz_p,
        intensity_p,
        eps,
    );
    let right_norm_square = prepare_linear_cosine_row::<F>(
        right_mz,
        right_intensity,
        right_row,
        &mut right_products,
        right_peaks,
        max_peaks_usize,
        mz_p,
        intensity_p,
        eps,
    );

    let mut left_cursor = 0usize;
    let mut right_cursor = 0usize;
    let mut score = zero;
    while left_cursor < left_peaks && right_cursor < right_peaks {
        let left_product = left_products[left_cursor];
        if left_product <= zero {
            left_cursor += 1;
        } else {
            let right_product = right_products[right_cursor];
            if right_product <= zero {
                right_cursor += 1;
            } else {
                let mz_left =
                    left_mz[left_mz.stride(0) * left_row + left_mz.stride(1) * left_cursor];
                let mz_right =
                    right_mz[right_mz.stride(0) * right_row + right_mz.stride(1) * right_cursor];
                let delta = mz_left - mz_right;

                if delta.abs() <= tolerance {
                    score += left_product * right_product;
                    left_cursor += 1;
                    right_cursor += 1;
                } else if mz_left + tolerance < mz_right {
                    left_cursor += 1;
                } else {
                    right_cursor += 1;
                }
            }
        }
    }

    let left_norm = (left_norm_square + eps).sqrt();
    let right_norm = (right_norm_square + eps).sqrt();
    let similarity = score / (left_norm * right_norm + eps);
    similarity.max(zero).min(one)
}

/// Three-pass prepass for one spectrum row:
///
/// 1. Find `intensity_max = max(intensity[i]^p)` and `mz_max = max(mz[i]^p)`
///    over non-padding peaks (`intensity > 0`).
/// 2. Compute `products[i] = (intensity[i]^p / intensity_max) * (mz[i]^p / mz_max)`
///    and store in `products`. Track `product_max`.
/// 3. Normalize `products[i] /= product_max` in place and accumulate
///    `norm_square = sum(products[i]^2)`.
///
/// Returns `norm_square`. Padding slots (`peak >= peak_count`) are zeroed so
/// the matching sweep can read them unconditionally without a guard.
#[cube]
pub fn prepare_linear_cosine_row<F: Float>(
    mz_tensor: &Tensor<F>,
    intensity_tensor: &Tensor<F>,
    row: usize,
    products: &mut Array<F>,
    peak_count: usize,
    #[comptime] max_peaks: usize,
    mz_p: F,
    intensity_p: F,
    eps: F,
) -> F {
    let zero = F::new(0.0_f32);

    let mut intensity_max = zero;
    let mut mz_max = zero;
    for peak in 0..peak_count {
        let intensity = intensity_tensor
            [row * intensity_tensor.stride(0) + peak * intensity_tensor.stride(1)];
        if intensity > zero {
            let mz = mz_tensor[row * mz_tensor.stride(0) + peak * mz_tensor.stride(1)];
            intensity_max = intensity_max.max(intensity.max(eps).powf(intensity_p));
            mz_max = mz_max.max(mz.max(eps).powf(mz_p));
        }
    }
    intensity_max += eps;
    mz_max += eps;

    let mut product_max = zero;
    for peak in 0..peak_count {
        let intensity = intensity_tensor
            [row * intensity_tensor.stride(0) + peak * intensity_tensor.stride(1)];
        if intensity > zero {
            let mz = mz_tensor[row * mz_tensor.stride(0) + peak * mz_tensor.stride(1)];
            let product = (intensity.max(eps).powf(intensity_p) / intensity_max)
                * (mz.max(eps).powf(mz_p) / mz_max);
            products[peak] = product;
            product_max = product_max.max(product);
        } else {
            products[peak] = zero;
        }
    }
    for peak in peak_count..max_peaks {
        products[peak] = zero;
    }
    product_max += eps;

    let mut norm_square = zero;
    for peak in 0..peak_count {
        let p = products[peak] / product_max;
        products[peak] = p;
        norm_square += p * p;
    }
    norm_square
}
