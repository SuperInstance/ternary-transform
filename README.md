# ternary-transform

Transform theory for ternary data — Haar-like wavelet transforms, ternary Fourier features, random Fourier features for kernel approximation, and ternary kernel functions with RBF-like similarity.

## Why This Exists

Signal processing and kernel methods assume continuous, real-valued inputs. When your data is inherently ternary {-1, 0, +1} — multi-level signals, ternary logic outputs, tri-state encodings — standard transforms can waste capacity on values that never appear. This crate provides transforms adapted for ternary structure: wavelets that respect the sign pattern, Fourier features computed on ternary sequences, and kernel functions (RBF, matching, agreement, polynomial) designed for ternary vectors. Also includes k-nearest-neighbor search via RBF similarity. `forbid(unsafe_code)` throughout.

## Core Concepts

- **TernaryWavelet**: Haar-like wavelet transform with forward/inverse, energy-per-level analysis, and threshold-based denoising. Levels are capped at `log₂(n)`.
- **TernaryFourier**: Discrete Fourier-style feature extraction for ternary sequences. Computes cosine and sine sums at integer frequencies, power spectrum, and approximate reconstruction.
- **RandomFeatures**: Random Fourier features for scalable kernel approximation. Uses deterministic pseudo-random weights and biases seeded from a user-supplied seed.
- **TernaryKernel**: Kernel functions for ternary vectors — RBF, matching (fraction of identical positions), agreement (dot product / length), and polynomial. Computes full kernel matrices.
- **TernaryRBFSimilarity**: RBF-like similarity with Hamming and weighted distance, similarity matrices, and k-nearest-neighbor search.

## Quick Start

```toml
# Cargo.toml
[dependencies]
ternary-transform = "0.1"
```

```rust
use ternary_transform::{
    Ternary, TernaryWavelet, TernaryFourier, RandomFeatures,
    TernaryKernel, KernelType, TernaryRBFSimilarity,
};

fn main() {
    // Wavelet transform
    let wavelet = TernaryWavelet::new(2);
    let signal = vec![1.0, -1.0, 1.0, -1.0];
    let coeffs = wavelet.forward(&signal);
    let energy = wavelet.energy_per_level(&coeffs);
    let denoised = wavelet.denoise(&signal, 0.05);

    // Ternary Fourier features
    let fourier = TernaryFourier::new(4);
    let data = vec![Ternary::Pos, Ternary::Neg, Ternary::Pos, Ternary::Neg];
    let features = fourier.transform(&data); // 8 values: 4 freq × 2 (cos+sin)
    let spectrum = fourier.power_spectrum(&data);

    // Kernel functions
    let kernel = TernaryKernel::new(1.0);
    let a = vec![Ternary::Pos, Ternary::Neg];
    let b = vec![Ternary::Pos, Ternary::Zero];
    let rbf = kernel.rbf(&a, &b);
    let matching = kernel.matching(&a, &b); // 0.5

    // K-nearest neighbors
    let sim = TernaryRBFSimilarity::new(1.0);
    let data = vec![
        vec![Ternary::Pos, Ternary::Pos],
        vec![Ternary::Neg, Ternary::Neg],
        vec![Ternary::Zero, Ternary::Zero],
    ];
    let knn = sim.knn(&vec![Ternary::Pos, Ternary::Pos], &data, 2);
    println!("Nearest: {:?}", knn);
}
```

## API Overview

| Type | Description |
|---|---|
| `Ternary` | Value: `Neg`, `Zero`, `Pos` with `to_f64()` |
| `TernaryWavelet` | `forward()`, `inverse()`, `energy_per_level()`, `denoise()` |
| `TernaryFourier` | `transform()`, `power_spectrum()`, `reconstruct()` |
| `RandomFeatures` | `transform()`, `kernel_approx()` — deterministic from seed |
| `TernaryKernel` | `rbf()`, `matching()`, `agreement()`, `polynomial()`, `kernel_matrix()` |
| `KernelType` | Enum: `Rbf`, `Matching`, `Agreement`, `Polynomial(degree, offset)` |
| `TernaryRBFSimilarity` | `similarity()`, `hamming_distance()`, `weighted_distance()`, `knn()`, `similarity_matrix()` |

## How It Works

**TernaryWavelet** applies the standard Haar decomposition (average + difference) but operates on floating-point representations of ternary signals. Multiple levels decompose the approximation coefficients recursively. Denoising thresholds the detail coefficients at each level.

**TernaryFourier** computes discrete cosine and sine sums at integer frequencies `1..n_features`. For a ternary sequence of length `n`, the feature at frequency `k` is `Σ x[t]·cos(2πkt/n)` and `Σ x[t]·sin(2πkt/n)`. The power spectrum is `cos² + sin²` at each frequency.

**RandomFeatures** uses the random kitchen sinks trick: generates random weight vectors and biases, then maps inputs through `cos(w·x + b)` to approximate shift-invariant kernels in a low-dimensional space.

**TernaryKernel** implements four kernel functions directly on ternary vectors by projecting to `{-1.0, 0.0, 1.0}`. `TernaryRBFSimilarity` uses Hamming distance in the RBF exponent for a ternary-native similarity measure.

## Use Cases

- **Ternary signal denoising**: Clean up noisy ternary sensor data using wavelet thresholding.
- **Feature extraction for ternary ML**: Compute Fourier or random features from ternary sequences for downstream classifiers.
- **Ternary similarity search**: Find nearest neighbors in ternary-valued datasets using RBF or matching kernels.
- **Kernel methods on ternary data**: Build SVM-like classifiers with kernels designed for the ternary domain.

## Ecosystem

Part of the **SuperInstance** ternary computing suite:

- `ternary-lattice` — lattice structures for ternary values
- `ternary-codes` — error-correcting codes for ternary data
- `ternary-gradient` — gradient-free optimization on ternary landscapes
- `ternary-language` — ternary NLP and grammar processing
- `ternary-trees` — ternary decision trees and forests
- `ternary-transform` — this crate
- `ternary-planning` — planning and scheduling with ternary priorities
- `ternary-rl` — reinforcement learning with ternary actions
- `ternary-som` — self-organizing maps for ternary data
- `ternary-failure` — failure analysis with ternary classification

## License

MIT
