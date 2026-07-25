// Unit tests for the mic-array DSP. Synthetic signals give exact ground truth
// for things that are otherwise only checkable by ear with a live talker:
// whether the compressor curve is right, and whether the DOA estimator actually
// recovers a known source angle.
//
// Build target: kinect_v2_dsp_test. Exits non-zero on first failure.

#include "dsp.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
  std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what);
  if (!ok) ++g_failures;
}

void CheckNear(float got, float want, float tol, const char* what) {
  const bool ok = std::fabs(got - want) <= tol;
  std::printf("  [%s] %s (got %.4f, want %.4f +/- %.4f)\n",
              ok ? " ok " : "FAIL", what, got, want, tol);
  if (!ok) ++g_failures;
}

constexpr float kFs = 16000.f;

// --- compressor --------------------------------------------------------------

void TestCompressorCurve() {
  std::printf("compressor static curve\n");
  dsp::CompressorParams p;
  p.threshold_db = -20.f;
  p.ratio = 4.f;
  p.knee_db = 0.f;  // hard knee, so the maths is exact
  p.makeup_db = 0.f;

  CheckNear(dsp::Compressor::StaticCurveGainDb(-40.f, p), 0.f, 1e-4f,
            "no reduction below threshold");
  CheckNear(dsp::Compressor::StaticCurveGainDb(-20.f, p), 0.f, 1e-4f,
            "no reduction at threshold");
  // 20 dB over a 4:1 ratio -> output 5 dB over -> 15 dB of reduction.
  CheckNear(dsp::Compressor::StaticCurveGainDb(0.f, p), -15.f, 1e-3f,
            "20 dB over at 4:1 gives -15 dB");
  // 8 dB over -> output 2 dB over -> 6 dB reduction.
  CheckNear(dsp::Compressor::StaticCurveGainDb(-12.f, p), -6.f, 1e-3f,
            "8 dB over at 4:1 gives -6 dB");

  // Curve must be monotonically non-increasing in level.
  bool monotonic = true;
  float prev = 0.f;
  for (float l = -80.f; l <= 6.f; l += 0.25f) {
    const float g = dsp::Compressor::StaticCurveGainDb(l, p);
    if (g > prev + 1e-4f) monotonic = false;
    prev = g;
  }
  Check(monotonic, "gain is monotonically non-increasing");

  // Soft knee must start reducing before the threshold and stay continuous.
  dsp::CompressorParams k = p;
  k.knee_db = 12.f;
  const float below = dsp::Compressor::StaticCurveGainDb(-24.f, k);
  Check(below < -1e-4f, "soft knee engages below threshold");
  bool continuous = true;
  float last = dsp::Compressor::StaticCurveGainDb(-60.f, k);
  for (float l = -60.f; l <= 6.f; l += 0.1f) {
    const float g = dsp::Compressor::StaticCurveGainDb(l, k);
    if (std::fabs(g - last) > 0.5f) continuous = false;
    last = g;
  }
  Check(continuous, "soft knee curve is continuous");
}

void TestCompressorDynamics() {
  std::printf("compressor dynamics\n");
  dsp::Compressor comp;
  comp.prepare(kFs);
  dsp::CompressorParams p;
  p.threshold_db = -20.f;
  p.ratio = 4.f;
  p.attack_ms = 5.f;
  p.release_ms = 100.f;
  p.knee_db = 0.f;
  p.makeup_db = 0.f;

  // Feed a loud steady tone; reduction should approach the static curve value.
  const float amp = 1.0f;  // 0 dBFS peak -> 20 dB over threshold
  for (int i = 0; i < int(kFs * 0.5f); ++i) {
    const float x = amp * std::sin(2.f * float(M_PI) * 440.f * i / kFs);
    comp.process(x, p);
  }
  // At the tone's peak the target is -15 dB; the envelope tracks peaks, so the
  // settled reduction sits between the peak and zero. Just require real work.
  Check(comp.gainReductionDb() < -5.f, "compresses a loud steady tone");

  // Then go quiet: reduction must release back toward 0 dB.
  for (int i = 0; i < int(kFs * 2.0f); ++i) comp.process(0.f, p);
  CheckNear(comp.gainReductionDb(), 0.f, 0.5f, "releases to unity when quiet");

  // A quiet signal should pass essentially untouched (makeup aside).
  dsp::Compressor c2;
  c2.prepare(kFs);
  float max_out = 0.f;
  for (int i = 0; i < int(kFs * 0.2f); ++i) {
    const float x = 0.001f * std::sin(2.f * float(M_PI) * 300.f * i / kFs);
    max_out = std::max(max_out, std::fabs(c2.process(x, p)));
  }
  CheckNear(max_out, 0.001f, 2e-4f, "quiet signal passes through");
}

void TestLimiter() {
  std::printf("limiter\n");
  dsp::Limiter lim;
  lim.prepare(kFs);
  float worst = 0.f;
  for (int i = 0; i < int(kFs * 0.5f); ++i) {
    // Deliberately way too hot.
    const float x = 8.f * std::sin(2.f * float(M_PI) * 220.f * i / kFs);
    worst = std::max(worst, std::fabs(lim.process(x, -1.f)));
  }
  Check(worst <= dsp::DbToLin(-1.f) + 1e-5f, "never exceeds the -1 dB ceiling");
}

// --- biquad ------------------------------------------------------------------

// Magnitude response measured by driving a tone through the filter.
float MeasureGain(const dsp::BiquadCoeffs& c, float freq) {
  dsp::Biquad f;
  const int n = int(kFs * 0.5f);
  const int settle = int(kFs * 0.2f);
  float peak = 0.f;
  for (int i = 0; i < n; ++i) {
    const float x = std::sin(2.f * float(M_PI) * freq * i / kFs);
    const float y = f.process(x, c);
    if (i > settle) peak = std::max(peak, std::fabs(y));
  }
  return peak;
}

void TestBiquad() {
  std::printf("biquad highpass\n");
  const dsp::BiquadCoeffs hp = dsp::HighpassCoeffs(kFs, 200.f);
  CheckNear(MeasureGain(hp, 4000.f), 1.f, 0.05f, "passband is unity at 4 kHz");
  CheckNear(dsp::LinToDb(MeasureGain(hp, 200.f)), -3.f, 1.0f,
            "-3 dB at the corner");
  Check(MeasureGain(hp, 20.f) < 0.05f, "20 Hz rumble is strongly rejected");

  // DC must be blocked completely.
  dsp::Biquad f;
  float y = 0.f;
  for (int i = 0; i < 4000; ++i) y = f.process(1.f, hp);
  CheckNear(y, 0.f, 1e-3f, "DC is blocked");
}

// --- array / DOA -------------------------------------------------------------

// Renders a band-limited pulse train arriving from `azimuth_deg` onto each mic,
// applying the exact fractional propagation delay per mic.
void SynthesizePlaneWave(const dsp::MicArray& arr, float azimuth_deg,
                         int n, std::vector<std::vector<float>>& out,
                         float noise_amp, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> gauss(0.f, 1.f);

  // Source waveform: filtered noise in the speech band, so it is broadband
  // enough to localise but below the array's spatial-aliasing limit.
  const int pad = 64;
  std::vector<float> src(size_t(n + 2 * pad));
  dsp::Biquad hp, lp;
  const dsp::BiquadCoeffs hpc = dsp::HighpassCoeffs(kFs, 300.f);
  const dsp::BiquadCoeffs lpc = dsp::LowpassCoeffs(kFs, 3000.f);
  for (size_t i = 0; i < src.size(); ++i) {
    src[i] = lp.process(hp.process(gauss(rng), hpc), lpc);
  }

  const float s = std::sin(azimuth_deg * float(M_PI) / 180.f);
  out.assign(size_t(arr.count), std::vector<float>(size_t(n), 0.f));
  for (int c = 0; c < arr.count; ++c) {
    // Mic c observes src(t - TOA_c), with TOA_c = T0 - x_c*sin(theta)/c, so a
    // mic on the source's side hears it earlier. Folding T0 into `pad`:
    //     m_c[i] = src[i + pad + x_c*sin(theta)/c*fs]
    // i.e. reading the source *further ahead* is what "hears it earlier" means.
    const float advance = (arr.x_m[c] * s / arr.speed_of_sound) * kFs;
    for (int i = 0; i < n; ++i) {
      const float pos = float(i + pad) + advance;
      const int ip = int(std::floor(pos));
      const float frac = pos - float(ip);
      const float taps[4] = {src[size_t(ip - 1)], src[size_t(ip)],
                             src[size_t(ip + 1)], src[size_t(ip + 2)]};
      out[size_t(c)][size_t(i)] =
          dsp::Lagrange4(taps, frac) + noise_amp * gauss(rng);
    }
  }
}

void TestGeometry() {
  std::printf("array geometry\n");
  const dsp::MicArray a = dsp::KinectV2MicArray(0.15f);
  Check(a.count == 4, "4 mics");
  CheckNear(a.x_m[0], -0.075f, 1e-6f, "first mic at -aperture/2");
  CheckNear(a.x_m[3], 0.075f, 1e-6f, "last mic at +aperture/2");
  CheckNear(a.x_m[1] - a.x_m[0], 0.05f, 1e-6f, "uniform 50 mm spacing");
  // c / (2d) = 343 / 0.1 = 3430 Hz
  CheckNear(a.aliasingFreeHz(), 3430.f, 1.f, "aliasing-free up to 3.43 kHz");

  // Steering delays: broadside must be equal for all mics.
  float d[dsp::MicArray::kMaxMics];
  dsp::SteeringDelays(a, 0.f, kFs, d);
  Check(std::fabs(d[0] - d[3]) < 1e-5f, "broadside delays are equal");
  for (int c = 0; c < a.count; ++c) Check(d[c] >= 0.f, "delays are non-negative");

  // At +90 deg the geometric spread should equal aperture/c in samples.
  dsp::SteeringDelays(a, 90.f, kFs, d);
  CheckNear(d[3] - d[0], 0.15f / 343.f * kFs, 1e-3f,
            "endfire spread matches aperture/c");
}

void TestBeamformerRejection() {
  std::printf("delay-and-sum spatial response\n");
  const dsp::MicArray arr = dsp::KinectV2MicArray(0.15f);
  const int n = 8000;
  std::vector<std::vector<float>> mics;
  SynthesizePlaneWave(arr, 0.f, n, mics, 0.f, 1234);

  // Sum the mics steered on-axis (0 deg) vs steered far off-axis (60 deg).
  auto beam_power = [&](float steer) {
    float d[dsp::MicArray::kMaxMics];
    dsp::SteeringDelays(arr, steer, kFs, d);
    double power = 0.0;
    const int start = 32;
    for (int i = start; i < n - 4; ++i) {
      float sum = 0.f;
      for (int c = 0; c < arr.count; ++c) {
        const float pos = float(i) - d[c];
        const int ip = int(pos);
        const float frac = pos - float(ip);
        const float taps[4] = {mics[size_t(c)][size_t(ip - 1)],
                               mics[size_t(c)][size_t(ip)],
                               mics[size_t(c)][size_t(ip + 1)],
                               mics[size_t(c)][size_t(ip + 2)]};
        sum += dsp::Lagrange4(taps, frac);
      }
      power += double(sum) * sum;
    }
    return power;
  };

  const double on = beam_power(0.f);
  const double off = beam_power(60.f);
  const float rejection_db = float(10.0 * std::log10(on / std::max(off, 1e-12)));
  std::printf("        on-axis/off-axis = %.2f dB\n", rejection_db);
  Check(rejection_db > 3.f, "on-axis beam beats a 60 deg mis-steer by >3 dB");
}

void TestDoaAccuracy() {
  std::printf("SRP DOA accuracy (clean)\n");
  const dsp::MicArray arr = dsp::KinectV2MicArray(0.15f);
  dsp::SrpDoa doa;
  doa.configure(arr, kFs, -90.f, 90.f, 2.f);

  const float truths[] = {-60.f, -40.f, -20.f, 0.f, 20.f, 40.f, 60.f};
  for (float truth : truths) {
    std::vector<std::vector<float>> mics;
    SynthesizePlaneWave(arr, truth, 4096, mics, 0.f, unsigned(truth + 100));
    const float* ptrs[dsp::MicArray::kMaxMics];
    for (int c = 0; c < arr.count; ++c) ptrs[c] = mics[size_t(c)].data();

    const dsp::DoaResult r = doa.estimate(ptrs, arr.count, 4096);
    Check(r.valid, "estimate is valid");
    CheckNear(r.azimuth_deg, truth, 6.f, "recovers the true azimuth");
    Check(r.confidence > 0.f, "confidence is positive");
  }
}

void TestDoaNoisy() {
  std::printf("SRP DOA with noise\n");
  const dsp::MicArray arr = dsp::KinectV2MicArray(0.15f);
  dsp::SrpDoa doa;
  doa.configure(arr, kFs, -90.f, 90.f, 2.f);

  // Source RMS through the band filters lands around 0.1; add uncorrelated
  // noise at ~10% of that per mic.
  int within = 0;
  const int trials = 8;
  for (int t = 0; t < trials; ++t) {
    const float truth = -50.f + 100.f * float(t) / float(trials - 1);
    std::vector<std::vector<float>> mics;
    SynthesizePlaneWave(arr, truth, 4096, mics, 0.01f, unsigned(t + 7));
    const float* ptrs[dsp::MicArray::kMaxMics];
    for (int c = 0; c < arr.count; ++c) ptrs[c] = mics[size_t(c)].data();
    const dsp::DoaResult r = doa.estimate(ptrs, arr.count, 4096);
    if (r.valid && std::fabs(r.azimuth_deg - truth) <= 10.f) ++within;
  }
  std::printf("        %d/%d trials within 10 deg\n", within, trials);
  Check(within >= trials - 1, "noisy DOA is within 10 deg on nearly all trials");

  // Silence must not produce a confident direction.
  std::vector<std::vector<float>> quiet(
      size_t(arr.count), std::vector<float>(4096, 0.f));
  const float* ptrs[dsp::MicArray::kMaxMics];
  for (int c = 0; c < arr.count; ++c) ptrs[c] = quiet[size_t(c)].data();
  const dsp::DoaResult r = doa.estimate(ptrs, arr.count, 4096);
  Check(r.confidence < 0.05f, "silence yields near-zero confidence");
}

}  // namespace

int main() {
  std::printf("=== mic-array DSP tests ===\n\n");
  TestCompressorCurve();
  TestCompressorDynamics();
  TestLimiter();
  TestBiquad();
  TestGeometry();
  TestBeamformerRejection();
  TestDoaAccuracy();
  TestDoaNoisy();

  std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}
