#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include "rtmpose_utils.h"

struct Vec3 { float x, y, z; };
struct Quat { float w, x, y, z; };

struct Joint3D {
    Vec3  pos{};
    Quat  orientation{1, 0, 0, 0};
    float confidence{0};
    bool  valid{false};
};

// COCO-17 bone pairs (parent → child).  Used for orientation and drawing.
static const std::pair<int,int> COCO_BONES[] = {
    {5, 7}, {7, 9},     // left arm
    {6, 8}, {8, 10},    // right arm
    {11, 13}, {13, 15}, // left leg
    {12, 14}, {14, 16}, // right leg
    {5, 6},             // shoulders
    {11, 12},           // hips
    {5, 11},            // left torso
    {6, 12},            // right torso
    {0, 5}, {0, 6},     // head to shoulders
};
static const int NUM_COCO_BONES = (int)(sizeof(COCO_BONES) / sizeof(COCO_BONES[0]));

// ── minimal 3-D math ─────────────────────────────────────────────────────────

static inline float  v3dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline Vec3   v3cross(Vec3 a, Vec3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
static inline Vec3   v3norm(Vec3 v) {
    float l = sqrtf(v3dot(v, v));
    return l < 1e-6f ? Vec3{0,1,0} : Vec3{v.x/l, v.y/l, v.z/l};
}
static inline Quat   qnorm(Quat q) {
    float l = sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    return l < 1e-6f ? Quat{1,0,0,0} : Quat{q.w/l, q.x/l, q.y/l, q.z/l};
}

// Shortest-arc quaternion that rotates unit vector `from` onto unit vector `to`.
static inline Quat quat_between(Vec3 from, Vec3 to) {
    from = v3norm(from); to = v3norm(to);
    float d = v3dot(from, to);
    if (d > 0.9999f)  return {1, 0, 0, 0};
    if (d < -0.9999f) {
        Vec3 perp = (fabsf(from.x) < 0.9f) ? Vec3{1,0,0} : Vec3{0,1,0};
        Vec3 ax   = v3norm(v3cross(from, perp));
        return {0, ax.x, ax.y, ax.z};
    }
    Vec3 ax = v3cross(from, to);
    return qnorm({1.f + d, ax.x, ax.y, ax.z});
}

// ── Kinect v1 depth conversion ───────────────────────────────────────────────

// Convert 11-bit raw disparity to depth in mm.
static inline float kinect_raw_to_mm(uint16_t raw) {
    if (raw == 0 || raw == 2047) return 0.f;
    float d = 1000.f / (raw * -0.0030711016f + 3.3309495161f);
    return (d > 200.f && d < 7000.f) ? d : 0.f;
}

// Median depth (mm) over a (2r+1)² patch; returns 0 if no valid samples.
static float sample_depth_mm(const uint16_t* raw_depth, int cx, int cy, int r = 3) {
    float vals[49];  // worst case (2*3+1)^2 = 49
    int   n = 0;
    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            int x = cx + dx, y = cy + dy;
            if (x < 0 || x >= 640 || y < 0 || y >= 480) continue;
            float d = kinect_raw_to_mm(raw_depth[y * 640 + x] & 0x7FFu);
            if (d > 0.f) vals[n++] = d;
        }
    }
    if (n == 0) return 0.f;
    std::sort(vals, vals + n);
    return vals[n / 2];
}

// ── main entry point ─────────────────────────────────────────────────────────

// Kinect v1 depth-camera intrinsics (OpenNI / ROS calibration defaults).
static const float K_FX = 594.21f, K_FY = 591.04f, K_CX = 339.5f, K_CY = 242.7f;

// All bones share the same rest direction (+Y = upward in camera space).
static const Vec3 BONE_REST = {0, 1, 0};

inline std::vector<Joint3D> lift_to_3d(const std::vector<PosePoint>& kps,
                                        const uint16_t*               raw_depth)
{
    constexpr int N = 17;
    std::vector<Joint3D> joints(N);

    if ((int)kps.size() < N) return joints;

    // Project each visible keypoint to camera-space 3-D.
    for (int i = 0; i < N; i++) {
        joints[i].confidence = kps[i].score;
        if (kps[i].score < 0.3f) continue;

        float d = sample_depth_mm(raw_depth, kps[i].x, kps[i].y);
        if (d <= 0.f) continue;

        joints[i].pos   = { (kps[i].x - K_CX) * d / K_FX,
                             (kps[i].y - K_CY) * d / K_FY,
                             d };
        joints[i].valid = true;
    }

    // Compute bone orientation: shortest arc from BONE_REST to bone direction.
    for (auto& [pi, ci] : COCO_BONES) {
        if (!joints[pi].valid || !joints[ci].valid) continue;
        Vec3 dir = { joints[ci].pos.x - joints[pi].pos.x,
                     joints[ci].pos.y - joints[pi].pos.y,
                     joints[ci].pos.z - joints[pi].pos.z };
        joints[pi].orientation = quat_between(BONE_REST, dir);
    }

    return joints;
}
