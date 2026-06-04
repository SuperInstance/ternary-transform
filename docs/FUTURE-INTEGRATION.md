# Future Integration: ternary-transform

## Current State
Provides `TernaryWavelet` (Haar-like forward/inverse with energy-per-level), ternary Fourier features, random feature approximation, kernel methods with ternary kernels, and RBF-like similarity functions.

## Integration Opportunities

### With ternary-cell
Wavelet decomposition becomes the **cell's frequency-domain perception**. `TernaryWavelet::forward()` applied to a cell's state history produces approximation coefficients (long-term trends) and detail coefficients (recent changes). During `tick()`, the cell can decide: respond to detail (reactive mode) or approximation (stable mode). The `energy_per_level()` method quantifies how much signal exists at each timescale — directly informing the cell's `surprise` computation.

### With ternary-attention
Transform preprocessing before attention: `TernaryWavelet::forward()` decomposes state sequences into multi-resolution representations. Multi-head attention then operates on different wavelet levels simultaneously, giving the system multi-scale awareness without increasing the attention dimension.

### With ternary-signal / ternary-entropy
The Fourier features and kernel methods provide frequency-domain tools that complement `ternary-sensor`'s time-domain processing. An `RbfSimilarity` kernel over room states enables smooth interpolation between discrete ternary configurations.

## Potential in Mature Systems
In construct-core's Layer 2 (`AsyncConstruct`), wavelet transforms compress long state histories into fixed-size representations. A construct that has been running for thousands of ticks can summarize its history via wavelet coefficients rather than storing raw state. The `random_features()` approximation maps to Johnson-Lindenstrauss-style dimensionality reduction for ternary vectors — critical for ESP32 deployment where memory is scarce.

## Cross-Pollination Ideas
**Music × Transforms:** Wavelet decomposition of ternary musical sequences directly produces multi-resolution rhythmic analysis. The approximation coefficients at level k represent the pulse at 2^k subdivision. This is how `agent-rhythm-rs` should process tempo hierarchically — each wavelet level is a metrical layer.

**Image × Transforms:** Ternary 2D wavelets (extend current 1D Haar to separable 2D) would process ternary images/room maps. The detail coefficients at each level detect edges at different scales — room boundaries, furniture clusters, door positions.

## Dependencies for Next Steps
- 2D wavelet extension needed for spatial room data
- `ternary-matrix` integration for batched transform operations
- Benchmark wavelet-based surprise vs. current entropy-based surprise in ternary-cell
