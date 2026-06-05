# ternary-transform

**Signal transformations that reshape ternary data. Shift, scale, invert, permute, convolve.**

Every processing chain is a sequence of transforms. Take a signal, shift it in time, scale its amplitude, invert its polarity, convolve it with a kernel, threshold the result. Each transform is simple. The magic is in the composition — chaining transforms into pipelines that sculpt raw data into meaningful structure.

This crate provides the fundamental signal transforms for ternary data: time-domain operations (delay, reverse, rotate), amplitude operations (invert, threshold, compress), spatial operations (convolve, correlate), and structural operations (interleave, decimate, permute). Each is a pure function — no state, no side effects, just input → output.

## What's Inside

- **`delay(signal, n)`** — shift signal forward by n samples, padding with zeros
- **`reverse(signal)`** — reverse the signal in time
- **`rotate(signal, n)`** — circular rotation (wrap around)
- **`invert(signal)`** — flip every value: +1↔-1, 0 stays
- **`threshold(signal, min, max)`** — zero out values outside the range
- **`convolve(signal, kernel)`** — convolution with a ternary kernel (edge detection, smoothing)
- **`correlate(a, b)`** — cross-correlation between two signals
- **`interleave(a, b)`** — alternate samples from two signals
- **`decimate(signal, factor)`** — keep every Nth sample
- **`permute(signal, order)`** — rearrange samples by index permutation
- **`amplify(signal, factor)`** — scale amplitude, clamp to ternary

## Quick Example

```rust
use ternary_transform::*;

let a = vec![1, 0, -1, 0, 1, 0, -1, 0];
let b = vec![0, 1, 0, -1, 0, 1, 0, -1];

// Invert: flip polarity
let inv = invert(&a);
// [-1, 0, 1, 0, -1, 0, 1, 0]

// Delay: shift forward 2 samples
let delayed = delay(&a, 2);
// [0, 0, 1, 0, -1, 0, 1, 0]

// Convolve with edge-detection kernel
let kernel = vec![1, 0, -1];
let edges = convolve(&a, &kernel);
// Detects transitions between +1 and -1

// Cross-correlation: how similar are a and b at each offset?
let corr = correlate(&a, &b);
// Peak at the offset where they align best

// Interleave: weave two signals together
let woven = interleave(&a, &b);
// [1, 0, 0, 1, -1, 0, 0, -1, ...]
```

## The Deeper Truth

**Every transform is a perspective shift.** Invert changes the sign of everything — what was positive becomes negative, what was negative becomes positive. It's hearing the world through a mirror. Convolution with a kernel is asking "does this shape appear here?" — it's pattern matching in the time domain. Correlation is asking "how are these two signals related?" — it's similarity measurement.

The power comes from composition. A pipeline like `invert → delay → convolve → threshold` transforms a raw signal into something that highlights specific features. This is how image processing works (edge detection is convolution with a specific kernel), how audio effects work (echo is delay + attenuate + add), and how neural networks work (learned convolution kernels). The ternary versions are the simplest possible implementations — the skeleton that all signal processing is built on.

**Use cases:**
- **Signal processing pipelines** — compose transforms into processing chains
- **Audio effects** — delay, invert, and convolve are the basis of every audio effect
- **Image processing** — ternary images convolved with edge-detection kernels
- **Data augmentation** — generate variations of ternary data for ML training
- **Pattern matching** — cross-correlation finds where patterns appear in signals

## See Also

- **ternary-bite** — destructive transforms (crush, wavefold, saturate)
- **ternary-warp** — simpler value transforms (clamp, quantize, fold, warp)
- **ternary-echo** — echo is delay + attenuate + add
- **ternary-mixer** — mixing is amplitude scaling + addition
- **ternary-filter** — (future) frequency-domain transforms

## Install

```bash
cargo add ternary-transform
```

## License

MIT
