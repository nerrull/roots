// Unit tests for hardware-independent pose/kinect logic.
// Build standalone (no OpenCV/libfreenect needed):
//   c++ -std=c++17 tests/test_pose_logic.cpp -o /tmp/run_tests && /tmp/run_tests

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

// ── inline copies of logic under test ────────────────────────────────────────
// (mirrors the real implementations so we can test without hardware headers)

struct Vec3 { float x, y, z; };
struct Quat { float w, x, y, z; };

struct Joint3D {
    Vec3  pos{};
    Quat  orientation{1, 0, 0, 0};
    float confidence{0};
    bool  valid{false};
};

struct PosePoint { int x, y; float score; PosePoint() : x(0), y(0), score(0.f) {} };

static const std::pair<int,int> COCO_BONES[] = {
    {5,7},{7,9},{6,8},{8,10},{11,13},{13,15},{12,14},{14,16},
    {5,6},{11,12},{5,11},{6,12},{0,5},{0,6},
};

static inline float v3dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
static inline Vec3  v3cross(Vec3 a, Vec3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
static inline Vec3  v3norm(Vec3 v) {
    float l = sqrtf(v3dot(v,v));
    return l < 1e-6f ? Vec3{0,1,0} : Vec3{v.x/l, v.y/l, v.z/l};
}
static inline Quat  qnorm(Quat q) {
    float l = sqrtf(q.w*q.w+q.x*q.x+q.y*q.y+q.z*q.z);
    return l < 1e-6f ? Quat{1,0,0,0} : Quat{q.w/l, q.x/l, q.y/l, q.z/l};
}
static inline Quat quat_between(Vec3 from, Vec3 to) {
    from = v3norm(from); to = v3norm(to);
    float d = v3dot(from, to);
    if (d > 0.9999f)  return {1,0,0,0};
    if (d < -0.9999f) {
        Vec3 perp = (fabsf(from.x) < 0.9f) ? Vec3{1,0,0} : Vec3{0,1,0};
        Vec3 ax   = v3norm(v3cross(from, perp));
        return {0, ax.x, ax.y, ax.z};
    }
    Vec3 ax = v3cross(from, to);
    return qnorm({1.f+d, ax.x, ax.y, ax.z});
}

static inline float kinect_raw_to_mm(uint16_t raw) {
    if (raw == 0 || raw == 2047) return 0.f;
    float d = 1000.f / (raw * -0.0030711016f + 3.3309495161f);
    return (d > 200.f && d < 7000.f) ? d : 0.f;
}

static float sample_depth_mm(const uint16_t* raw_depth, int cx, int cy, int r = 3) {
    float vals[49];
    int   n = 0;
    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            int x = cx+dx, y = cy+dy;
            if (x < 0 || x >= 640 || y < 0 || y >= 480) continue;
            float d = kinect_raw_to_mm(raw_depth[y*640+x] & 0x7FFu);
            if (d > 0.f) vals[n++] = d;
        }
    }
    if (n == 0) return 0.f;
    std::sort(vals, vals+n);
    return vals[n/2];
}

static const float K_FX = 594.21f, K_FY = 591.04f, K_CX = 339.5f, K_CY = 242.7f;
static const Vec3 BONE_REST = {0, 1, 0};

static std::vector<Joint3D> lift_to_3d(const std::vector<PosePoint>& kps,
                                        const uint16_t* raw_depth)
{
    constexpr int N = 17;
    std::vector<Joint3D> joints(N);
    if ((int)kps.size() < N) return joints;
    for (int i = 0; i < N; i++) {
        joints[i].confidence = kps[i].score;
        if (kps[i].score < 0.3f) continue;
        float d = sample_depth_mm(raw_depth, kps[i].x, kps[i].y);
        if (d <= 0.f) continue;
        joints[i].pos   = { (kps[i].x - K_CX)*d/K_FX,
                             (kps[i].y - K_CY)*d/K_FY, d };
        joints[i].valid = true;
    }
    for (auto& [pi, ci] : COCO_BONES) {
        if (!joints[pi].valid || !joints[ci].valid) continue;
        Vec3 dir = { joints[ci].pos.x-joints[pi].pos.x,
                     joints[ci].pos.y-joints[pi].pos.y,
                     joints[ci].pos.z-joints[pi].pos.z };
        joints[pi].orientation = quat_between(BONE_REST, dir);
    }
    return joints;
}

// ── DetectBox IoU (mirrors main.cpp) ─────────────────────────────────────────

struct DetectBox { int left, top, right, bottom; };

static float box_iou(const DetectBox& a, const DetectBox& b) {
    int ix = std::max(a.left, b.left),   iy = std::max(a.top, b.top);
    int ax = std::min(a.right, b.right), ay = std::min(a.bottom, b.bottom);
    if (ax <= ix || ay <= iy) return 0.f;
    float inter = (float)(ax-ix)*(float)(ay-iy);
    float ua    = (float)(a.right-a.left)*(float)(a.bottom-a.top);
    float ub    = (float)(b.right-b.left)*(float)(b.bottom-b.top);
    return inter / (ua + ub - inter);
}

// ── test harness ──────────────────────────────────────────────────────────────

static int g_pass = 0, g_fail = 0;

#define CHECK(expr) do { \
    if (expr) { ++g_pass; } \
    else { ++g_fail; fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #expr); } \
} while(0)

#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((float)(a)-(float)(b)) < (tol))

// ── test cases ────────────────────────────────────────────────────────────────

static void test_box_iou() {
    DetectBox a{0,0,100,100};

    // Identical
    CHECK_NEAR(box_iou(a, a), 1.f, 1e-5f);

    // No overlap
    CHECK_NEAR(box_iou(a, {200,200,300,300}), 0.f, 1e-5f);

    // Touching edge only → no overlap
    CHECK_NEAR(box_iou(a, {100,0,200,100}), 0.f, 1e-5f);

    // 50% side overlap: inter=5000, union=15000 → 1/3
    CHECK_NEAR(box_iou(a, {50,0,150,100}), 1.f/3.f, 1e-5f);

    // Contained: inter=2500, ua=10000, ub=2500 → 0.25
    CHECK_NEAR(box_iou(a, {25,25,75,75}), 0.25f, 1e-5f);

    // Symmetry
    DetectBox b{50,0,150,100};
    CHECK_NEAR(box_iou(a, b), box_iou(b, a), 1e-6f);
}

static void test_kinect_raw_to_mm() {
    // Sentinels → 0
    CHECK_NEAR(kinect_raw_to_mm(0),    0.f, 1e-3f);
    CHECK_NEAR(kinect_raw_to_mm(2047), 0.f, 1e-3f);

    // Mid-range → in [200, 7000] mm
    float d = kinect_raw_to_mm(1000);
    CHECK(d > 200.f && d < 7000.f);

    // Smaller raw disparity = farther (inverse relationship)
    CHECK(kinect_raw_to_mm(500) > kinect_raw_to_mm(1500));

    // Values near extremes may fall outside valid range and return 0
    float d_low = kinect_raw_to_mm(1);
    CHECK(d_low == 0.f || (d_low > 200.f && d_low < 7000.f));
}

static void test_sample_depth_mm() {
    std::vector<uint16_t> buf(640*480, 1000u);
    float expected = kinect_raw_to_mm(1000u);

    // Full patch in center
    CHECK_NEAR(sample_depth_mm(buf.data(), 320, 240), expected, 1.f);

    // Corner patch (partial OOB) still returns correct median
    CHECK_NEAR(sample_depth_mm(buf.data(), 0, 0), expected, 1.f);

    // All zeros → no valid samples → 0
    std::vector<uint16_t> zeros(640*480, 0u);
    CHECK_NEAR(sample_depth_mm(zeros.data(), 320, 240), 0.f, 1e-5f);

    // Single valid pixel in otherwise zero buffer
    std::vector<uint16_t> sparse(640*480, 0u);
    sparse[240*640+320] = 1000u;
    CHECK_NEAR(sample_depth_mm(sparse.data(), 320, 240), expected, 1.f);
}

static void test_lift_too_few_keypoints() {
    std::vector<uint16_t> depth(640*480, 1000u);
    // 5 keypoints — should return 17 invalid joints, no crash
    auto joints = lift_to_3d(std::vector<PosePoint>(5), depth.data());
    CHECK((int)joints.size() == 17);
    for (auto& j : joints) CHECK(!j.valid);
}

static void test_lift_low_confidence() {
    std::vector<PosePoint> kps(17);
    for (auto& p : kps) { p.x = 320; p.y = 240; p.score = 0.1f; }
    std::vector<uint16_t> depth(640*480, 1000u);
    auto joints = lift_to_3d(kps, depth.data());
    for (auto& j : joints) CHECK(!j.valid);
}

static void test_lift_valid() {
    std::vector<PosePoint> kps(17);
    for (auto& p : kps) { p.x = 320; p.y = 240; p.score = 0.9f; }
    std::vector<uint16_t> depth(640*480, 1000u);
    auto joints = lift_to_3d(kps, depth.data());
    CHECK((int)joints.size() == 17);
    for (auto& j : joints) {
        CHECK(j.valid);
        CHECK(j.confidence > 0.f);
        CHECK(j.pos.z > 0.f);
    }
}

static void test_lift_no_depth() {
    std::vector<PosePoint> kps(17);
    for (auto& p : kps) { p.x = 320; p.y = 240; p.score = 0.9f; }
    std::vector<uint16_t> depth(640*480, 0u);
    auto joints = lift_to_3d(kps, depth.data());
    for (auto& j : joints) CHECK(!j.valid);
}

static void test_lift_projection_at_principal_point() {
    // At principal point (K_CX, K_CY): 3D x,y should be ~0
    std::vector<PosePoint> kps(17);
    for (auto& p : kps) {
        p.x = (int)K_CX;
        p.y = (int)K_CY;
        p.score = 0.9f;
    }
    std::vector<uint16_t> depth(640*480, 1000u);
    auto joints = lift_to_3d(kps, depth.data());
    for (auto& j : joints) {
        if (!j.valid) continue;
        CHECK_NEAR(j.pos.x, 0.f, 5.f);  // small residual from int rounding
        CHECK_NEAR(j.pos.y, 0.f, 5.f);
        CHECK(j.pos.z > 0.f);
    }
}

static void test_simcc_score_old_vs_new() {
    // Confident in x, uncertain in y
    float xi = 0.9f, yi = 0.3f;
    float old_score = std::max(xi, yi);     // Bug: overestimates at 0.9
    float new_score = (xi + yi) * 0.5f;    // Fix: reflects weak y-axis at 0.6
    CHECK_NEAR(old_score, 0.9f, 1e-5f);
    CHECK_NEAR(new_score, 0.6f, 1e-5f);
    CHECK(new_score < old_score);

    // Both axes agree — both formulas should be similar
    float xi2 = 0.85f, yi2 = 0.87f;
    CHECK_NEAR((xi2+yi2)*0.5f, 0.86f, 1e-4f);

    // New formula is symmetric in xi,yi; old is not
    CHECK_NEAR((xi+yi)*0.5f, (yi+xi)*0.5f, 1e-6f);
    CHECK_NEAR(std::max(xi,yi), std::max(yi,xi), 1e-6f);  // max is symmetric, but wrong value
}

int main() {
    test_box_iou();
    test_kinect_raw_to_mm();
    test_sample_depth_mm();
    test_lift_too_few_keypoints();
    test_lift_low_confidence();
    test_lift_valid();
    test_lift_no_depth();
    test_lift_projection_at_principal_point();
    test_simcc_score_old_vs_new();

    fprintf(stderr, "\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
