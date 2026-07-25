#include "person_tracker.h"

#include <algorithm>
#include <cmath>

namespace {

// Coarse grid cell size in pixels. 16 gives 32x27 blocks on a 512x424 frame --
// fine enough to isolate a person, coarse enough that speckle noise cannot
// carry a block on its own.
constexpr int kBlock = 16;

// Depth percentile used as a block's "near surface". A low percentile follows
// the nearest real surface while ignoring the handful of spuriously-near pixels
// that a hard minimum would latch onto.
constexpr float kNearPercentile = 0.15f;

}  // namespace

void PersonTracker::reset() {
  track_ = PersonTrack{};
  have_smoothed_ = false;
  smoothed_azimuth_ = 0.f;
  last_seen_s_ = -1.0;
}

const PersonTrack& PersonTracker::update(const FrameSnapshot& depth,
                                         const Params& p, double now_s) {
  const int w = depth.width, h = depth.height;
  if (!depth.valid || w <= 0 || h <= 0 ||
      depth.format != libfreenect2::Frame::Float) {
    track_.valid = false;
    return track_;
  }

  const float* src = reinterpret_cast<const float*>(depth.data.data());
  const int bw = (w + kBlock - 1) / kBlock;
  const int bh = (h + kBlock - 1) / kBlock;
  block_depth_.assign(size_t(bw * bh), 0.f);
  block_count_.assign(size_t(bw * bh), 0);

  // Pass 1: per-block near-surface depth from a low percentile of valid pixels.
  for (int by = 0; by < bh; ++by) {
    for (int bx = 0; bx < bw; ++bx) {
      scratch_.clear();
      const int y1 = std::min((by + 1) * kBlock, h);
      const int x1 = std::min((bx + 1) * kBlock, w);
      for (int y = by * kBlock; y < y1; ++y) {
        const float* row = src + size_t(y) * size_t(w);
        for (int x = bx * kBlock; x < x1; ++x) {
          const float mm = row[x];
          if (mm >= p.min_mm && mm <= p.max_mm && std::isfinite(mm)) {
            scratch_.push_back(mm);
          }
        }
      }
      const size_t idx = size_t(by * bw + bx);
      block_count_[idx] = int(scratch_.size());
      if (int(scratch_.size()) >= p.min_block_px) {
        const size_t k =
            size_t(float(scratch_.size() - 1) * kNearPercentile);
        std::nth_element(scratch_.begin(), scratch_.begin() + k,
                         scratch_.end());
        block_depth_[idx] = scratch_[k];
      }
    }
  }

  // Pass 2: the nearest qualifying block defines the slab of interest.
  float near_mm = 0.f;
  bool found_block = false;
  for (size_t i = 0; i < block_depth_.size(); ++i) {
    if (block_count_[i] < p.min_block_px) continue;
    const float d = block_depth_[i];
    if (!found_block || d < near_mm) {
      near_mm = d;
      found_block = true;
    }
  }

  bool found = false;
  if (found_block) {
    // Pass 3: centroid of every pixel inside the slab, in 3D. Averaging in
    // metres (not pixels) keeps the azimuth right for off-centre blobs.
    const float slab_hi = near_mm + p.slab_mm;
    double sx = 0, sy = 0, sz = 0, su = 0, sv = 0;
    int count = 0;
    for (int y = 0; y < h; ++y) {
      const float* row = src + size_t(y) * size_t(w);
      for (int x = 0; x < w; ++x) {
        const float mm = row[x];
        if (mm < near_mm || mm > slab_hi || !std::isfinite(mm)) continue;
        const float z = mm * 0.001f;
        float px, py;
        if (intrinsics_.valid) {
          px = (float(x) - intrinsics_.cx) * z / intrinsics_.fx;
          py = (float(y) - intrinsics_.cy) * z / intrinsics_.fy;
        } else {
          // No intrinsics yet: fall back to a nominal 70.6 deg horizontal FOV
          // (the Kinect v2 depth camera) so azimuth is still roughly right.
          const float fx = float(w) / (2.f * std::tan(70.6f * 0.5f *
                                                      float(M_PI) / 180.f));
          px = (float(x) - float(w) * 0.5f) * z / fx;
          py = (float(y) - float(h) * 0.5f) * z / fx;
        }
        sx += px;
        sy += py;
        sz += z;
        su += x;
        sv += y;
        ++count;
      }
    }

    if (count >= p.min_support_px) {
      const double inv = 1.0 / double(count);
      track_.x_m = float(sx * inv);
      track_.y_m = float(sy * inv);
      track_.z_m = float(sz * inv);
      track_.distance_mm = track_.z_m * 1000.f;
      track_.support_px = count;
      track_.centroid_u = int(su * inv);
      track_.centroid_v = int(sv * inv);
      // Azimuth off the sensor's optical axis; +x is to the sensor's right.
      float az = std::atan2(track_.x_m, std::max(track_.z_m, 1e-3f)) * 180.f /
                 float(M_PI);
      if (p.mirror) az = -az;
      track_.raw_azimuth_deg = az;
      found = true;
    }
  }

  if (found) {
    // One-pole smoothing: the beam should follow a person, not chase per-frame
    // centroid jitter.
    const float a = std::min(std::max(p.smoothing, 0.f), 1.f);
    smoothed_azimuth_ = have_smoothed_
                            ? smoothed_azimuth_ +
                                  a * (track_.raw_azimuth_deg - smoothed_azimuth_)
                            : track_.raw_azimuth_deg;
    have_smoothed_ = true;
    last_seen_s_ = now_s;
    track_.azimuth_deg = smoothed_azimuth_;
    track_.valid = true;
    track_.holding = false;
  } else {
    // Coast briefly so a dropped frame or a turned head does not swing the beam
    // back to centre.
    const bool within_hold =
        have_smoothed_ && last_seen_s_ >= 0.0 &&
        (now_s - last_seen_s_) <= double(p.hold_seconds);
    track_.azimuth_deg = smoothed_azimuth_;
    track_.valid = within_hold;
    track_.holding = within_hold;
    if (!within_hold) {
      track_.support_px = 0;
      track_.distance_mm = 0.f;
    }
  }
  return track_;
}
