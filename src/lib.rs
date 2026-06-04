#![forbid(unsafe_code)]

//! Transform theory for ternary data on {-1, 0, +1}.
//!
//! Provides TernaryWavelet transform, ternary Fourier features, random features
//! approximation, kernel methods with ternary kernels, and RBF-like similarity.

/// A ternary value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ternary {
    Neg,
    Zero,
    Pos,
}

impl Ternary {
    pub fn to_f64(self) -> f64 {
        match self {
            Ternary::Neg => -1.0,
            Ternary::Zero => 0.0,
            Ternary::Pos => 1.0,
        }
    }

    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Ternary::Neg),
            0 => Some(Ternary::Zero),
            1 => Some(Ternary::Pos),
            _ => None,
        }
    }

    pub fn values() -> [Ternary; 3] {
        [Ternary::Neg, Ternary::Zero, Ternary::Pos]
    }
}

use std::f64::consts::{PI, E};

// ==================== Ternary Wavelet Transform ====================

/// Haar-like wavelet transform adapted for ternary data.
pub struct TernaryWavelet {
    pub levels: usize,
}

impl TernaryWavelet {
    pub fn new(levels: usize) -> Self {
        TernaryWavelet { levels }
    }

    /// Forward ternary wavelet transform.
    pub fn forward(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < 2 {
            return data.to_vec();
        }
        let mut result = data.to_vec();
        let mut current_n = n;

        for _ in 0..self.levels.min(log2_floor(n)) {
            let half = current_n / 2;
            let mut approx = vec![0.0; half];
            let mut detail = vec![0.0; half];

            for i in 0..half {
                // Ternary-aware: emphasize the sign structure
                let a = result[2 * i];
                let b = result[2 * i + 1];
                approx[i] = (a + b) / 2.0;
                detail[i] = (a - b) / 2.0;
            }

            // Ternary rounding on approximation coefficients
            for i in 0..half {
                result[i] = approx[i];
            }
            for i in 0..half {
                result[half + i] = detail[i];
            }
            current_n = half;
        }

        result
    }

    /// Inverse ternary wavelet transform.
    pub fn inverse(&self, coeffs: &[f64]) -> Vec<f64> {
        let n = coeffs.len();
        if n < 2 {
            return coeffs.to_vec();
        }

        let levels = self.levels.min(log2_floor(n));
        let mut result = coeffs.to_vec();

        for _ in 0..levels {
            let current_n = smallest_reconstruct(n, levels);
            let half = current_n / 2;
            let mut reconstructed = vec![0.0; current_n];

            for i in 0..half {
                let a = result[i];
                let d = result[half + i];
                reconstructed[2 * i] = a + d;
                reconstructed[2 * i + 1] = a - d;
            }

            for i in 0..current_n {
                result[i] = reconstructed[i];
            }
        }

        result
    }

    /// Get wavelet energy at each level.
    pub fn energy_per_level(&self, coeffs: &[f64]) -> Vec<f64> {
        let n = coeffs.len();
        let levels = self.levels.min(log2_floor(n));
        let mut energies = Vec::new();
        let mut current_n = n;

        for _level in 0..levels {
            let half = current_n / 2;
            let start = half;
            let energy: f64 = coeffs[start..current_n]
                .iter()
                .map(|&c| c * c)
                .sum();
            energies.push(energy);
            current_n = half;
        }

        energies
    }

    /// Denoise by thresholding detail coefficients.
    pub fn denoise(&self, data: &[f64], threshold: f64) -> Vec<f64> {
        let coeffs = self.forward(data);
        let n = coeffs.len();
        let levels = self.levels.min(log2_floor(n));

        let mut denoised = coeffs.clone();
        let mut current_n = n;

        for _ in 0..levels {
            let half = current_n / 2;
            for i in half..current_n {
                if denoised[i].abs() < threshold {
                    denoised[i] = 0.0;
                }
            }
            current_n = half;
        }

        self.inverse(&denoised)
    }
}

fn log2_floor(n: usize) -> usize {
    if n == 0 { 0 } else { (n as f64).log2() as usize }
}

fn smallest_reconstruct(n: usize, levels: usize) -> usize {
    // For reconstruction, start from the smallest level
    n / (1 << (levels - 1).max(0))
}

// ==================== Ternary Fourier Features ====================

/// Ternary Fourier feature extraction.
pub struct TernaryFourier {
    pub n_features: usize,
    pub frequencies: Vec<f64>,
}

impl TernaryFourier {
    pub fn new(n_features: usize) -> Self {
        let frequencies: Vec<f64> = (1..=n_features).map(|i| i as f64).collect();
        TernaryFourier {
            n_features,
            frequencies,
        }
    }

    /// Compute Fourier features for a ternary sequence.
    pub fn transform(&self, data: &[Ternary]) -> Vec<f64> {
        let n = data.len() as f64;
        let mut features = Vec::with_capacity(self.n_features * 2);

        for &freq in &self.frequencies {
            let mut cos_sum = 0.0;
            let mut sin_sum = 0.0;
            for (t, &val) in data.iter().enumerate() {
                let x = val.to_f64();
                let phase = 2.0 * PI * freq * t as f64 / n;
                cos_sum += x * phase.cos();
                sin_sum += x * phase.sin();
            }
            features.push(cos_sum / n);
            features.push(sin_sum / n);
        }

        features
    }

    /// Compute the power spectrum.
    pub fn power_spectrum(&self, data: &[Ternary]) -> Vec<f64> {
        let features = self.transform(data);
        features
            .chunks(2)
            .map(|chunk| {
                let cos_v = chunk[0];
                let sin_v = if chunk.len() > 1 { chunk[1] } else { 0.0 };
                cos_v * cos_v + sin_v * sin_v
            })
            .collect()
    }

    /// Reconstruct from Fourier features (approximate).
    pub fn reconstruct(&self, features: &[f64], length: usize) -> Vec<f64> {
        let mut result = vec![0.0; length];
        let n = length as f64;

        for (i, chunk) in features.chunks(2).enumerate() {
            let cos_v = chunk[0];
            let sin_v = if chunk.len() > 1 { chunk[1] } else { 0.0 };
            let freq = (i + 1) as f64;
            for t in 0..length {
                let phase = 2.0 * PI * freq * t as f64 / n;
                result[t] += cos_v * phase.cos() + sin_v * phase.sin();
            }
        }

        result
    }
}

// ==================== Random Features Approximation ====================

/// Random Fourier features for kernel approximation.
pub struct RandomFeatures {
    pub n_features: usize,
    pub dim: usize,
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
}

impl RandomFeatures {
    pub fn new(n_features: usize, dim: usize, seed: u64) -> Self {
        let weights: Vec<Vec<f64>> = (0..n_features)
            .map(|i| {
                (0..dim)
                    .map(|j| {
                        // Simple deterministic random from seed
                        let s = seed.wrapping_add((i * dim + j) as u64);
                        pseudo_random(s)
                    })
                    .collect()
            })
            .collect();

        let biases: Vec<f64> = (0..n_features)
            .map(|i| pseudo_random(seed.wrapping_add(n_features as u64 * dim as u64 + i as u64)) * 2.0 * PI)
            .collect();

        RandomFeatures {
            n_features,
            dim,
            weights,
            biases,
        }
    }

    /// Transform input using random features.
    pub fn transform(&self, input: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .map(|(w, &b)| {
                let dot: f64 = w.iter().zip(input.iter()).map(|(&a, &x)| a * x).sum();
                (dot + b).cos()
            })
            .collect()
    }

    /// Approximate kernel value between two inputs.
    pub fn kernel_approx(&self, a: &[f64], b: &[f64]) -> f64 {
        let fa = self.transform(a);
        let fb = self.transform(b);
        let dot: f64 = fa.iter().zip(fb.iter()).map(|(&x, &y)| x * y).sum();
        dot / self.n_features as f64
    }
}

fn pseudo_random(seed: u64) -> f64 {
    // Simple LCG-based pseudo-random in [-1, 1]
    let s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((s >> 33) as i64 as f64) / (1i64 << 31) as f64
}

// ==================== Ternary Kernels ====================

/// Kernel functions for ternary data.
pub struct TernaryKernel {
    pub sigma: f64,
}

impl TernaryKernel {
    pub fn new(sigma: f64) -> Self {
        TernaryKernel { sigma }
    }

    /// RBF-like kernel for ternary vectors.
    pub fn rbf(&self, a: &[Ternary], b: &[Ternary]) -> f64 {
        let dist_sq: f64 = a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = x.to_f64() - y.to_f64();
                d * d
            })
            .sum();
        (-dist_sq / (2.0 * self.sigma * self.sigma)).exp()
    }

    /// Ternary matching kernel: counts matching positions.
    pub fn matching(&self, a: &[Ternary], b: &[Ternary]) -> f64 {
        let matches = a.iter().zip(b.iter()).filter(|(&x, &y)| x == y).count();
        matches as f64 / a.len().max(1) as f64
    }

    /// Ternary agreement kernel: +1 for same, -1 for opposite, 0 otherwise.
    pub fn agreement(&self, a: &[Ternary], b: &[Ternary]) -> f64 {
        let score: f64 = a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let xv = x.to_f64();
                let yv = y.to_f64();
                xv * yv
            })
            .sum();
        score / a.len().max(1) as f64
    }

    /// Polynomial kernel for ternary vectors.
    pub fn polynomial(&self, a: &[Ternary], b: &[Ternary], degree: i32, offset: f64) -> f64 {
        let dot: f64 = a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| x.to_f64() * y.to_f64())
            .sum();
        (dot + offset).powi(degree)
    }

    /// Compute full kernel matrix for a set of ternary vectors.
    pub fn kernel_matrix(&self, data: &[Vec<Ternary>], kernel_type: KernelType) -> Vec<Vec<f64>> {
        let n = data.len();
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                matrix[i][j] = match kernel_type {
                    KernelType::Rbf => self.rbf(&data[i], &data[j]),
                    KernelType::Matching => self.matching(&data[i], &data[j]),
                    KernelType::Agreement => self.agreement(&data[i], &data[j]),
                    KernelType::Polynomial(d, o) => self.polynomial(&data[i], &data[j], d, o),
                };
            }
        }
        matrix
    }
}

/// Types of ternary kernels.
#[derive(Debug, Clone, Copy)]
pub enum KernelType {
    Rbf,
    Matching,
    Agreement,
    Polynomial(i32, f64),
}

// ==================== RBF-like Similarity ====================

/// RBF-like similarity measures for ternary data.
pub struct TernaryRBFSimilarity {
    pub gamma: f64,
}

impl TernaryRBFSimilarity {
    pub fn new(gamma: f64) -> Self {
        TernaryRBFSimilarity { gamma }
    }

    /// Compute similarity between two ternary sequences.
    pub fn similarity(&self, a: &[Ternary], b: &[Ternary]) -> f64 {
        let dist = self.hamming_distance(a, b) as f64;
        (-self.gamma * dist).exp()
    }

    /// Hamming distance between ternary sequences.
    pub fn hamming_distance(&self, a: &[Ternary], b: &[Ternary]) -> usize {
        a.iter().zip(b.iter()).filter(|(&x, &y)| x != y).count()
    }

    /// Weighted Hamming: opposite signs cost more than one being zero.
    pub fn weighted_distance(&self, a: &[Ternary], b: &[Ternary]) -> f64 {
        a.iter().zip(b.iter())
            .map(|(&x, &y)| {
                let xv = x.to_f64();
                let yv = y.to_f64();
                (xv - yv).abs()
            })
            .sum()
    }

    /// Similarity matrix for a set of sequences.
    pub fn similarity_matrix(&self, data: &[Vec<Ternary>]) -> Vec<Vec<f64>> {
        let n = data.len();
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            matrix[i][i] = 1.0;
            for j in (i + 1)..n {
                let s = self.similarity(&data[i], &data[j]);
                matrix[i][j] = s;
                matrix[j][i] = s;
            }
        }
        matrix
    }

    /// Find k nearest neighbors.
    pub fn knn(&self, query: &[Ternary], data: &[Vec<Ternary>], k: usize) -> Vec<(usize, f64)> {
        let mut scored: Vec<(usize, f64)> = data.iter()
            .enumerate()
            .map(|(i, d)| (i, self.similarity(query, d)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavelet_forward() {
        let w = TernaryWavelet::new(1);
        let data = vec![1.0, -1.0, 1.0, -1.0];
        let coeffs = w.forward(&data);
        assert_eq!(coeffs.len(), 4);
        // Approximation of alternating signal
        assert!(coeffs[0].abs() < 1e-10);
        assert!(coeffs[1].abs() < 1e-10);
    }

    #[test]
    fn test_wavelet_inverse() {
        let w = TernaryWavelet::new(1);
        let data = vec![3.0, 1.0, -1.0, 5.0];
        let coeffs = w.forward(&data);
        let reconstructed = w.inverse(&coeffs);
        // Should approximately reconstruct
        for i in 0..data.len() {
            assert!((reconstructed[i] - data[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_wavelet_energy() {
        let w = TernaryWavelet::new(2);
        let data = vec![1.0, -1.0, 1.0, -1.0];
        let coeffs = w.forward(&data);
        let energy = w.energy_per_level(&coeffs);
        assert_eq!(energy.len(), 2);
    }

    #[test]
    fn test_wavelet_denoise() {
        let w = TernaryWavelet::new(1);
        let data = vec![1.0, 0.01, -1.0, 0.01];
        let denoised = w.denoise(&data, 0.05);
        assert_eq!(denoised.len(), 4);
    }

    #[test]
    fn test_wavelet_single_element() {
        let w = TernaryWavelet::new(1);
        let data = vec![5.0];
        let coeffs = w.forward(&data);
        assert_eq!(coeffs, vec![5.0]);
    }

    #[test]
    fn test_fourier_transform() {
        let f = TernaryFourier::new(3);
        let data = vec![Ternary::Pos, Ternary::Neg, Ternary::Pos, Ternary::Neg];
        let features = f.transform(&data);
        assert_eq!(features.len(), 6); // 3 frequencies × 2 (cos + sin)
    }

    #[test]
    fn test_fourier_power_spectrum() {
        let f = TernaryFourier::new(4);
        let data = vec![Ternary::Pos, Ternary::Neg, Ternary::Pos, Ternary::Neg];
        let spectrum = f.power_spectrum(&data);
        assert_eq!(spectrum.len(), 4);
        // High frequency alternating should have strong frequency-2 component
        assert!(spectrum[1] > spectrum[0]);
    }

    #[test]
    fn test_fourier_reconstruct() {
        let f = TernaryFourier::new(2);
        let data = vec![Ternary::Pos, Ternary::Zero, Ternary::Neg, Ternary::Zero];
        let features = f.transform(&data);
        let reconstructed = f.reconstruct(&features, 4);
        assert_eq!(reconstructed.len(), 4);
    }

    #[test]
    fn test_fourier_constant_signal() {
        let f = TernaryFourier::new(2);
        let data = vec![Ternary::Zero, Ternary::Zero, Ternary::Zero, Ternary::Zero];
        let features = f.transform(&data);
        // All features should be ~0 for constant zero signal
        for &v in &features {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_random_features_transform() {
        let rf = RandomFeatures::new(10, 3, 42);
        let input = vec![1.0, 0.0, -1.0];
        let features = rf.transform(&input);
        assert_eq!(features.len(), 10);
        // Each feature should be in [-1, 1] since it's cos()
        for &v in &features {
            assert!(v >= -1.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_random_features_kernel() {
        let rf = RandomFeatures::new(100, 3, 42);
        let a = vec![1.0, 0.0, -1.0];
        let b = vec![1.0, 0.0, -1.0];
        let k = rf.kernel_approx(&a, &b);
        // Self-kernel should be close to 1
        assert!(k > 0.5);
    }

    #[test]
    fn test_ternary_kernel_rbf() {
        let k = TernaryKernel::new(1.0);
        let a = vec![Ternary::Pos, Ternary::Pos];
        let b = vec![Ternary::Pos, Ternary::Pos];
        let rbf = k.rbf(&a, &b);
        assert!((rbf - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ternary_kernel_matching() {
        let k = TernaryKernel::new(1.0);
        let a = vec![Ternary::Pos, Ternary::Neg];
        let b = vec![Ternary::Pos, Ternary::Zero];
        let m = k.matching(&a, &b);
        assert!((m - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_ternary_kernel_agreement() {
        let k = TernaryKernel::new(1.0);
        let a = vec![Ternary::Pos, Ternary::Neg];
        let b = vec![Ternary::Neg, Ternary::Pos];
        let ag = k.agreement(&a, &b);
        assert!((ag - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_ternary_kernel_polynomial() {
        let k = TernaryKernel::new(1.0);
        let a = vec![Ternary::Pos, Ternary::Pos];
        let b = vec![Ternary::Pos, Ternary::Pos];
        let p = k.polynomial(&a, &b, 2, 1.0);
        assert!((p - 9.0).abs() < 1e-10); // (2+1)^2 = 9
    }

    #[test]
    fn test_kernel_matrix() {
        let k = TernaryKernel::new(1.0);
        let data = vec![
            vec![Ternary::Pos, Ternary::Pos],
            vec![Ternary::Neg, Ternary::Neg],
        ];
        let m = k.kernel_matrix(&data, KernelType::Matching);
        assert!((m[0][0] - 1.0).abs() < 1e-10);
        assert!((m[1][1] - 1.0).abs() < 1e-10);
        assert!(m[0][1] < 1.0);
    }

    #[test]
    fn test_rbf_similarity() {
        let sim = TernaryRBFSimilarity::new(1.0);
        let a = vec![Ternary::Pos, Ternary::Pos];
        let b = vec![Ternary::Pos, Ternary::Pos];
        assert!((sim.similarity(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rbf_hamming() {
        let sim = TernaryRBFSimilarity::new(1.0);
        let a = vec![Ternary::Pos, Ternary::Neg, Ternary::Zero];
        let b = vec![Ternary::Pos, Ternary::Zero, Ternary::Zero];
        assert_eq!(sim.hamming_distance(&a, &b), 1);
    }

    #[test]
    fn test_rbf_weighted_distance() {
        let sim = TernaryRBFSimilarity::new(1.0);
        let a = vec![Ternary::Pos];
        let b = vec![Ternary::Neg];
        let d = sim.weighted_distance(&a, &b);
        assert!((d - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_rbf_similarity_matrix() {
        let sim = TernaryRBFSimilarity::new(1.0);
        let data = vec![
            vec![Ternary::Pos],
            vec![Ternary::Neg],
        ];
        let m = sim.similarity_matrix(&data);
        assert!((m[0][0] - 1.0).abs() < 1e-10);
        assert!((m[0][1] - m[1][0]).abs() < 1e-10);
    }

    #[test]
    fn test_rbf_knn() {
        let sim = TernaryRBFSimilarity::new(1.0);
        let query = vec![Ternary::Pos, Ternary::Pos];
        let data = vec![
            vec![Ternary::Pos, Ternary::Pos],
            vec![Ternary::Neg, Ternary::Neg],
            vec![Ternary::Zero, Ternary::Zero],
        ];
        let knn = sim.knn(&query, &data, 2);
        assert_eq!(knn.len(), 2);
        assert_eq!(knn[0].0, 0); // Most similar
    }
}
