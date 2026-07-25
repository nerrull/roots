// Locates a person in the depth image and reports the azimuth to steer the
// audio beam at.
//
// "Closest point" taken literally is a terrible tracker: a single hot pixel of
// depth noise wins over an actual person. So instead we find the nearest depth
// *surface with real support* -- a coarse block grid, each block scored by a low
// percentile of its valid depths and required to have enough valid pixels --
// then take the centroid of everything within a slab behind that surface. That
// is stable enough to steer a beam with, and cheap enough to run per frame.

#pragma once

#include <cstdint>
#include <vector>

#include "kinect_source.h"

struct PersonTrack {
  bool valid = false;      // a person-like blob is currently visible
  bool holding = false;    // no blob right now, coasting on the last one
  float distance_mm = 0.f;
  float x_m = 0.f, y_m = 0.f, z_m = 0.f;  // camera frame, +x right, +y down
  float azimuth_deg = 0.f;                // smoothed, for beam steering
  float raw_azimuth_deg = 0.f;            // this frame, unsmoothed
  int support_px = 0;
  int centroid_u = 0, centroid_v = 0;  // for drawing on the depth image
};

class PersonTracker {
 public:
  struct Params {
    float min_mm = 500.f;    // ignore closer than this (sensor floor / lens)
    float max_mm = 4500.f;   // ignore further than this
    float slab_mm = 450.f;   // depth slab behind the nearest surface
    int min_support_px = 300;       // reject blobs smaller than a torso
    int min_block_px = 24;          // valid pixels a block needs to count
    float smoothing = 0.35f;        // per-frame blend toward the new azimuth
    float hold_seconds = 1.0f;      // coast this long after losing the blob
    bool mirror = false;            // flip azimuth sign
  };

  void setIntrinsics(const KinectSource::DepthIntrinsics& intr) {
    intrinsics_ = intr;
  }

  // `depth` must be a Frame::Depth snapshot (float mm). `now_s` drives the hold
  // timer. Returns the current track (also available via current()).
  const PersonTrack& update(const FrameSnapshot& depth, const Params& p,
                            double now_s);

  const PersonTrack& current() const { return track_; }
  void reset();

 private:
  KinectSource::DepthIntrinsics intrinsics_;
  PersonTrack track_;
  bool have_smoothed_ = false;
  float smoothed_azimuth_ = 0.f;
  double last_seen_s_ = -1.0;

  // Scratch reused across frames so update() does not allocate steady-state.
  std::vector<float> block_depth_;
  std::vector<int> block_count_;
  std::vector<float> scratch_;
};
