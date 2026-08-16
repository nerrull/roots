// Small self-contained radix-2 Cooley-Tukey FFT, public-domain style (written
// from scratch for this project -- no external DSP library, matching how the
// rest of this repo vendors small headers instead of pulling a package
// manager). Only supports power-of-two sizes, which is all the spectrogram
// and spectral-envelope views need.
#pragma once

#include <vector>
#include <complex>
#include <cstdint>
#include <cmath>

namespace fft {

using Complex = std::complex<float>;

inline bool IsPowerOfTwo(uint32_t n) { return n != 0 && (n & (n - 1)) == 0; }

// In-place iterative radix-2 DIT FFT. `data.size()` must be a power of two.
// inverse=true computes the unnormalized inverse transform (divide by N
// yourself if you need it); the analysis views here only need magnitudes, so
// callers of the forward transform never do.
inline void Transform(std::vector<Complex>& data, bool inverse = false)
{
    const size_t n = data.size();
    if (n < 2)
        return;

    // Bit-reversal permutation.
    for (size_t i = 1, j = 0; i < n; ++i)
    {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(data[i], data[j]);
    }

    const float sign = inverse ? 1.0f : -1.0f;
    for (size_t len = 2; len <= n; len <<= 1)
    {
        const float theta = sign * 2.0f * 3.14159265358979323846f / float(len);
        const Complex wlen(std::cos(theta), std::sin(theta));
        for (size_t i = 0; i < n; i += len)
        {
            Complex w(1.0f, 0.0f);
            for (size_t k = 0; k < len / 2; ++k)
            {
                const Complex u = data[i + k];
                const Complex v = data[i + k + len / 2] * w;
                data[i + k] = u + v;
                data[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// Periodic Hann window, DFT-even (no 1/(N-1) tail), standard for STFT overlap use.
inline void HannWindow(std::vector<float>& w)
{
    const size_t n = w.size();
    for (size_t i = 0; i < n; ++i)
        w[i] = 0.5f - 0.5f * std::cos(2.0f * 3.14159265358979323846f * float(i) / float(n));
}

// Windows `samples` (length == window.size() == a power of two), FFTs it, and
// writes magnitudes for bins [0, N/2] into out (size N/2+1).
inline void MagnitudeSpectrum(const float* samples, const std::vector<float>& window, std::vector<float>& out)
{
    const size_t n = window.size();
    std::vector<Complex> buf(n);
    for (size_t i = 0; i < n; ++i)
        buf[i] = Complex(samples[i] * window[i], 0.0f);

    Transform(buf, false);

    out.resize(n / 2 + 1);
    for (size_t i = 0; i <= n / 2; ++i)
        out[i] = std::abs(buf[i]) * (2.0f / float(n));
}

} // namespace fft
