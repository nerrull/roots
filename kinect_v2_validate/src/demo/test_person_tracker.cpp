// Unit tests for the depth person tracker, using synthetic depth frames so the
// expected azimuth is known exactly.
//
// Build target: kinect_v2_tracker_test. Exits non-zero on first failure.

#include "person_tracker.h"

#include <cmath>
#include <cstdio>
#include <random>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
  std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what);
  if (!ok) ++g_failures;
}

void CheckNear(float got, float want, float tol, const char* what) {
  const bool ok = std::fabs(got - want) <= tol;
  std::printf("  [%s] %s (got %.3f, want %.3f +/- %.3f)\n",
              ok ? " ok " : "FAIL", what, got, want, tol);
  if (!ok) ++g_failures;
}

constexpr int kW = 512, kH = 424;

// Nominal Kinect v2 depth-camera intrinsics.
KinectSource::DepthIntrinsics Intrinsics() {
  KinectSource::DepthIntrinsics i;
  i.fx = 365.5f;
  i.fy = 365.5f;
  i.cx = 256.f;
  i.cy = 212.f;
  i.valid = true;
  return i;
}

FrameSnapshot MakeDepthFrame() {
  FrameSnapshot f;
  f.width = kW;
  f.height = kH;
  f.bytes_per_pixel = 4;
  f.format = libfreenect2::Frame::Float;
  f.valid = true;
  f.data.assign(size_t(kW) * kH * 4, 0);  // all-zero == invalid depth
  return f;
}

float* Pixels(FrameSnapshot& f) {
  return reinterpret_cast<float*>(f.data.data());
}

// Paints an axis-aligned rectangle of constant depth.
void PaintRect(FrameSnapshot& f, int cu, int cv, int half_w, int half_h,
               float mm) {
  float* px = Pixels(f);
  for (int y = std::max(cv - half_h, 0); y < std::min(cv + half_h, kH); ++y) {
    for (int x = std::max(cu - half_w, 0); x < std::min(cu + half_w, kW); ++x) {
      px[size_t(y) * kW + size_t(x)] = mm;
    }
  }
}

// Azimuth a blob centred on column `u` at distance `mm` should produce.
float ExpectedAzimuth(float u, float mm) {
  const KinectSource::DepthIntrinsics i = Intrinsics();
  const float z = mm * 0.001f;
  const float x = (u - i.cx) * z / i.fx;
  return std::atan2(x, z) * 180.f / float(M_PI);
}

void TestCentredPerson() {
  std::printf("centred person\n");
  FrameSnapshot f = MakeDepthFrame();
  PaintRect(f, 256, 212, 40, 70, 2000.f);

  PersonTracker t;
  t.setIntrinsics(Intrinsics());
  PersonTracker::Params p;
  const PersonTrack& tr = t.update(f, p, 0.0);

  Check(tr.valid, "tracks a centred blob");
  CheckNear(tr.azimuth_deg, 0.f, 1.0f, "azimuth is ~0 dead ahead");
  CheckNear(tr.distance_mm, 2000.f, 20.f, "distance matches");
  Check(tr.support_px > 4000, "support is the full blob");
}

void TestOffAxisPerson() {
  std::printf("off-axis person\n");
  for (int u : {150, 360}) {
    FrameSnapshot f = MakeDepthFrame();
    PaintRect(f, u, 212, 40, 70, 2500.f);

    PersonTracker t;
    t.setIntrinsics(Intrinsics());
    PersonTracker::Params p;
    const PersonTrack& tr = t.update(f, p, 0.0);
    Check(tr.valid, "tracks an off-axis blob");
    CheckNear(tr.azimuth_deg, ExpectedAzimuth(float(u), 2500.f), 1.5f,
              u < 256 ? "negative azimuth to the left"
                      : "positive azimuth to the right");
  }

  // Sign convention must be invertible for whichever way the rig faces.
  FrameSnapshot f = MakeDepthFrame();
  PaintRect(f, 360, 212, 40, 70, 2500.f);
  PersonTracker t;
  t.setIntrinsics(Intrinsics());
  PersonTracker::Params p;
  p.mirror = true;
  const PersonTrack& tr = t.update(f, p, 0.0);
  CheckNear(tr.azimuth_deg, -ExpectedAzimuth(360.f, 2500.f), 1.5f,
            "mirror flips the azimuth sign");
}

void TestNearerPersonWinsOverWall() {
  std::printf("nearer person beats the back wall\n");
  FrameSnapshot f = MakeDepthFrame();
  // A full-frame wall at 4 m, plus a person at 1.5 m off to one side.
  PaintRect(f, 256, 212, kW, kH, 4000.f);
  PaintRect(f, 150, 212, 40, 70, 1500.f);

  PersonTracker t;
  t.setIntrinsics(Intrinsics());
  PersonTracker::Params p;
  const PersonTrack& tr = t.update(f, p, 0.0);

  Check(tr.valid, "tracks despite a large background surface");
  CheckNear(tr.distance_mm, 1500.f, 150.f, "locks onto the near person");
  CheckNear(tr.azimuth_deg, ExpectedAzimuth(150.f, 1500.f), 2.5f,
            "azimuth points at the person, not the wall");
}

void TestSpeckleDoesNotHijack() {
  std::printf("depth speckle rejection\n");
  FrameSnapshot f = MakeDepthFrame();
  PaintRect(f, 360, 212, 40, 70, 2500.f);  // the real person, to the right

  // Scatter a handful of spuriously-near single pixels far to the left. A
  // literal "closest point" tracker would jump to these every frame.
  float* px = Pixels(f);
  std::mt19937 rng(4);
  std::uniform_int_distribution<int> ux(5, 80), uy(5, kH - 5);
  for (int i = 0; i < 12; ++i) {
    px[size_t(uy(rng)) * kW + size_t(ux(rng))] = 600.f;
  }

  PersonTracker t;
  t.setIntrinsics(Intrinsics());
  PersonTracker::Params p;
  const PersonTrack& tr = t.update(f, p, 0.0);

  Check(tr.valid, "still tracks");
  CheckNear(tr.azimuth_deg, ExpectedAzimuth(360.f, 2500.f), 2.0f,
            "ignores isolated near-pixel speckle");
  Check(tr.distance_mm > 2000.f, "distance is the person, not the speckle");
}

void TestTooSmallRejected() {
  std::printf("undersized blob rejection\n");
  FrameSnapshot f = MakeDepthFrame();
  PaintRect(f, 256, 212, 5, 5, 1200.f);  // 100 px, below min_support_px

  PersonTracker t;
  t.setIntrinsics(Intrinsics());
  PersonTracker::Params p;
  p.min_support_px = 300;
  const PersonTrack& tr = t.update(f, p, 0.0);
  Check(!tr.valid, "does not track a blob below the support threshold");
}

void TestHoldAndRelease() {
  std::printf("hold then release\n");
  PersonTracker t;
  t.setIntrinsics(Intrinsics());
  PersonTracker::Params p;
  p.hold_seconds = 1.0f;
  p.smoothing = 1.0f;  // no smoothing, so the azimuth is exact

  FrameSnapshot person = MakeDepthFrame();
  PaintRect(person, 360, 212, 40, 70, 2500.f);
  const float expect = ExpectedAzimuth(360.f, 2500.f);
  CheckNear(t.update(person, p, 0.0).azimuth_deg, expect, 1.5f,
            "acquires the person");

  FrameSnapshot empty = MakeDepthFrame();
  const PersonTrack& held = t.update(empty, p, 0.5);
  Check(held.valid && held.holding, "coasts within the hold window");
  CheckNear(held.azimuth_deg, expect, 1.5f, "holds the last azimuth");

  const PersonTrack& lost = t.update(empty, p, 2.0);
  Check(!lost.valid, "drops the track after the hold window");
}

void TestSmoothing() {
  std::printf("azimuth smoothing\n");
  PersonTracker t;
  t.setIntrinsics(Intrinsics());
  PersonTracker::Params p;
  p.smoothing = 0.25f;

  FrameSnapshot left = MakeDepthFrame();
  PaintRect(left, 150, 212, 40, 70, 2500.f);
  FrameSnapshot right = MakeDepthFrame();
  PaintRect(right, 360, 212, 40, 70, 2500.f);

  const float a_left = t.update(left, p, 0.0).azimuth_deg;
  // First acquisition snaps; a subsequent jump must not.
  const float after_jump = t.update(right, p, 0.1).azimuth_deg;
  const float a_right = ExpectedAzimuth(360.f, 2500.f);
  Check(std::fabs(after_jump - a_right) > std::fabs(a_right - a_left) * 0.5f,
        "a sudden jump is smoothed, not followed instantly");

  // Repeated frames must converge on the new position.
  float last = after_jump;
  for (int i = 0; i < 40; ++i) last = t.update(right, p, 0.2 + 0.03 * i).azimuth_deg;
  CheckNear(last, a_right, 1.5f, "converges on the new azimuth");
}

void TestInvalidFrame() {
  std::printf("invalid input\n");
  PersonTracker t;
  t.setIntrinsics(Intrinsics());
  PersonTracker::Params p;

  FrameSnapshot bad;
  Check(!t.update(bad, p, 0.0).valid, "an empty snapshot yields no track");

  FrameSnapshot wrong = MakeDepthFrame();
  wrong.format = libfreenect2::Frame::BGRX;
  Check(!t.update(wrong, p, 0.0).valid, "a non-float frame is rejected");

  FrameSnapshot blank = MakeDepthFrame();
  Check(!t.update(blank, p, 0.0).valid, "an all-invalid depth frame is rejected");
}

}  // namespace

int main() {
  std::printf("=== depth person tracker tests ===\n\n");
  TestCentredPerson();
  TestOffAxisPerson();
  TestNearerPersonWinsOverWall();
  TestSpeckleDoesNotHijack();
  TestTooSmallRejected();
  TestHoldAndRelease();
  TestSmoothing();
  TestInvalidFrame();

  std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}
