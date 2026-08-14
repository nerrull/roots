#include "face_fit.h"

#include <algorithm>
#include <cmath>

namespace mirror {

const std::vector<int>& MP68Indices() {
    // The standard dlib-68 -> MediaPipe FaceMesh mapping, in dlib order:
    // 0-16 jaw, 17-26 brows, 27-35 nose, 36-47 eyes, 48-67 mouth.
    static const std::vector<int> kMp68 = {
        162, 234, 93,  58,  172, 136, 149, 148, 152, 377, 378, 365, 397, 288,
        323, 454, 389,
        71,  63,  105, 66,  107, 336, 296, 334, 293, 301,
        168, 197, 5,   4,   75,  97,  2,   326, 305,
        33,  160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
        61,  39,  37,  0,   267, 269, 291, 405, 314, 17,  84,  181,
        78,  82,  13,  312, 308, 317, 14,  87,
    };
    return kMp68;
}

FacePose Similarity2D(const std::vector<float>& src, const std::vector<float>& dst) {
    FacePose p;
    const size_t n = std::min(src.size(), dst.size()) / 2;
    if (n == 0) return p;

    double msx = 0, msy = 0, mdx = 0, mdy = 0;
    for (size_t i = 0; i < n; ++i) {
        msx += src[i * 2]; msy += src[i * 2 + 1];
        mdx += dst[i * 2]; mdy += dst[i * 2 + 1];
    }
    msx /= double(n); msy /= double(n); mdx /= double(n); mdy /= double(n);

    // A 2x2 rotation is a single angle, so the SVD the general Umeyama needs
    // collapses to an atan2 of the summed dot and cross products.
    double sdot = 0, scross = 0, svar = 0;
    for (size_t i = 0; i < n; ++i) {
        const double ax = src[i * 2] - msx, ay = src[i * 2 + 1] - msy;
        const double bx = dst[i * 2] - mdx, by = dst[i * 2 + 1] - mdy;
        sdot   += ax * bx + ay * by;
        scross += ax * by - ay * bx;
        svar   += ax * ax + ay * ay;
    }
    const double theta = std::atan2(scross, sdot);
    const double c = std::cos(theta), s = std::sin(theta);
    const double scale = svar > 1e-20 ? std::sqrt(sdot * sdot + scross * scross) / svar : 1.0;

    p.s = float(scale);
    p.R[0] = float(c);  p.R[1] = float(-s);
    p.R[2] = float(s);  p.R[3] = float(c);
    p.t[0] = float(mdx - scale * (c * msx - s * msy));
    p.t[1] = float(mdy - scale * (s * msx + c * msy));
    return p;
}

namespace {

// Cholesky solve of a small symmetric positive-definite system, in place.
// The ridge term guarantees positive-definiteness, so there is no pivoting and
// no fallback path: without it the normal equations are singular whenever the
// landmarks under-determine a mode, which for the tail modes they always do.
bool SolveSPD(std::vector<double>& A, std::vector<double>& b, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A[size_t(i) * n + j];
            for (int k = 0; k < j; ++k) sum -= A[size_t(i) * n + k] * A[size_t(j) * n + k];
            if (i == j) {
                if (sum <= 0.0) return false;
                A[size_t(i) * n + i] = std::sqrt(sum);
            } else {
                A[size_t(i) * n + j] = sum / A[size_t(j) * n + j];
            }
        }
    }
    for (int i = 0; i < n; ++i) {          // forward
        double sum = b[i];
        for (int k = 0; k < i; ++k) sum -= A[size_t(i) * n + k] * b[k];
        b[i] = sum / A[size_t(i) * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {     // back
        double sum = b[i];
        for (int k = i + 1; k < n; ++k) sum -= A[size_t(k) * n + i] * b[k];
        b[i] = sum / A[size_t(i) * n + i];
    }
    return true;
}

std::string NormaliseName(const std::string& in) {
    // MediaPipe says "mouthSmileLeft"; the basis says "mouthSmile_L". Lowercase,
    // drop underscores, and collapse left/right to l/r and the two agree.
    std::string s;
    s.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        const char c = in[i];
        if (c == '_') continue;
        s.push_back(char(std::tolower(static_cast<unsigned char>(c))));
    }
    for (const char* word : {"left", "right"}) {
        const std::string w = word;
        size_t p;
        while ((p = s.find(w)) != std::string::npos) s.replace(p, w.size(), 1, w[0]);
    }
    return s;
}

}  // namespace

// ---------------------------------------------------------------------------

bool FaceFitter::load(const std::string& basis_path, std::string& err) {
    if (!basis_.load(basis_path, err)) return false;
    alpha_.assign(size_t(basis_.identityModes()), 0.0f);
    expr_.assign(size_t(basis_.expressionModes()), 0.0f);
    basis_.reconstruct(alpha_, expr_, verts_);
    return true;
}

float FaceFitter::Frontality(const std::vector<FaceLandmark>& lm) {
    // Nose tip against the midpoint of the two cheek extremes, in units of half
    // the face width. A turned head slides the nose off that midpoint long
    // before anything else is obviously wrong.
    constexpr int kNose = 1, kLeft = 234, kRight = 454;
    if (int(lm.size()) <= kRight) return 0.0f;
    const float mid = 0.5f * (lm[kLeft].x + lm[kRight].x);
    const float half = 0.5f * std::fabs(lm[kRight].x - lm[kLeft].x) + 1e-6f;
    return std::exp(-3.0f * std::fabs(lm[kNose].x - mid) / half);
}

float FaceFitter::Neutrality(const std::vector<float>& bs,
                             const std::vector<std::string>& names) {
    if (bs.empty()) return 1.0f;
    float act = 0.0f;
    for (size_t i = 0; i < bs.size(); ++i) {
        // Blinks and eye direction are involuntary and constant; scoring them
        // as expression would reject every frame roughly a third of the time.
        if (i < names.size()) {
            const std::string& n = names[i];
            if (n.find("eyeBlink") != std::string::npos ||
                n.find("eyeLook") != std::string::npos ||
                n.find("Pupil") != std::string::npos ||
                n == "_neutral")
                continue;
        }
        act += bs[i];
    }
    return std::exp(-0.6f * act);
}

void FaceFitter::buildNameMap(const std::vector<std::string>& names) const {
    if (mapped_names_ == names) return;
    mapped_names_ = names;
    name_map_.assign(names.size(), -1);

    std::vector<std::string> basis_norm;
    basis_norm.reserve(basis_.expressionNames().size());
    for (const std::string& n : basis_.expressionNames())
        basis_norm.push_back(NormaliseName(n));

    for (size_t i = 0; i < names.size(); ++i) {
        const std::string key = NormaliseName(names[i]);
        for (size_t j = 0; j < basis_norm.size(); ++j) {
            if (basis_norm[j] == key) { name_map_[i] = int(j); break; }
        }
    }
}

void FaceFitter::mapExpression(const std::vector<float>& bs,
                               const std::vector<std::string>& names,
                               std::vector<float>& out) const {
    out.assign(size_t(basis_.expressionModes()), 0.0f);
    if (bs.empty()) return;
    buildNameMap(names);
    for (size_t i = 0; i < bs.size() && i < name_map_.size(); ++i) {
        const int j = name_map_[i];
        if (j >= 0) out[size_t(j)] = bs[i];
    }
}

// ---------------------------------------------------------------------------

void FaceFitter::clearIdentity() {
    frames_.clear();
    alpha_.assign(size_t(basis_.identityModes()), 0.0f);
    has_identity_ = false;
}

void FaceFitter::identityScores(float& best, float& worst) const {
    best = worst = 0.0f;
    if (frames_.empty()) return;
    best = worst = frames_[0].score;
    for (const Sample& s : frames_) {
        best = std::max(best, s.score);
        worst = std::min(worst, s.score);
    }
}

bool FaceFitter::offerIdentityFrame(const FaceResult& r, int w, int h) {
    if (!basis_.valid() || !r.valid) return false;
    const float front = Frontality(r.landmarks);
    if (front < cfg_.min_frontality) return false;
    const float score = front * Neutrality(r.blendshapes, r.blendshape_names);

    // Full: keep this frame only if it beats the worst one held.
    int worst_at = -1;
    if (int(frames_.size()) >= cfg_.max_frames) {
        float worst = 0.0f;
        for (size_t i = 0; i < frames_.size(); ++i) {
            if (worst_at < 0 || frames_[i].score < worst) {
                worst = frames_[i].score;
                worst_at = int(i);
            }
        }
        if (worst_at < 0 || score <= frames_[size_t(worst_at)].score) return false;
    }

    const std::vector<int>& mp = MP68Indices();
    Sample s;
    s.score = score;
    s.target.resize(mp.size() * 2);
    for (size_t i = 0; i < mp.size(); ++i) {
        if (mp[i] >= int(r.landmarks.size())) return false;
        const FaceLandmark& l = r.landmarks[size_t(mp[i])];
        s.target[i * 2]     = l.x * float(w);
        s.target[i * 2 + 1] = -l.y * float(h);   // pixels are y-down; the basis is y-up
    }
    mapExpression(r.blendshapes, r.blendshape_names, s.expr);
    if (worst_at >= 0) frames_[size_t(worst_at)] = std::move(s);
    else frames_.push_back(std::move(s));
    return true;
}

bool FaceFitter::fitIdentity(float* residual_px) {
    if (!basis_.valid() || frames_.empty()) return false;

    const int n_id = std::min(cfg_.n_identity, basis_.identityModes());
    const int n_lm = FaceBasis::kLandmarks;
    const size_t n_frames = frames_.size();

    // Landmark basis, xy only. The z rows are dead weight here: the fit is
    // against a 2D image, and a weak-perspective similarity has no depth.
    const std::vector<float>& lm_neutral = basis_.lmNeutral();
    const std::vector<float>& lm_id = basis_.lmIdentity();
    const std::vector<float>& lm_ex = basis_.lmExpression();

    // Per frame: the expression's landmark contribution, subtracted up front so
    // identity is fitted to the face rather than to the face's expression.
    std::vector<std::vector<double>> base(n_frames);   // neutral + expression, xy
    for (size_t k = 0; k < n_frames; ++k) {
        base[k].assign(size_t(n_lm) * 2, 0.0);
        for (int p = 0; p < n_lm; ++p) {
            base[k][size_t(p) * 2]     = lm_neutral[size_t(p) * 3];
            base[k][size_t(p) * 2 + 1] = lm_neutral[size_t(p) * 3 + 1];
        }
        const std::vector<float>& w = frames_[k].expr;
        for (size_t m = 0; m < w.size() && m < size_t(basis_.expressionModes()); ++m) {
            if (w[m] == 0.0f) continue;
            const float* src = &lm_ex[m * size_t(n_lm) * 3];
            for (int p = 0; p < n_lm; ++p) {
                base[k][size_t(p) * 2]     += double(w[m]) * src[size_t(p) * 3];
                base[k][size_t(p) * 2 + 1] += double(w[m]) * src[size_t(p) * 3 + 1];
            }
        }
    }

    std::vector<double> a(size_t(n_id), 0.0);
    std::vector<FacePose> poses(n_frames);
    std::vector<float> cur(size_t(n_lm) * 2);
    std::vector<double> AtA(size_t(n_id) * n_id, 0.0);
    std::vector<double> Atb(size_t(n_id), 0.0);
    std::vector<double> J(size_t(n_lm) * 2 * n_id);   // d(projected landmark)/d(alpha)

    for (int it = 0; it < cfg_.iterations; ++it) {
        // --- pose given shape ------------------------------------------------
        for (size_t k = 0; k < n_frames; ++k) {
            for (int p = 0; p < n_lm; ++p) {
                double x = base[k][size_t(p) * 2], y = base[k][size_t(p) * 2 + 1];
                for (int m = 0; m < n_id; ++m) {
                    if (a[size_t(m)] == 0.0) continue;
                    const float* src = &lm_id[size_t(m) * n_lm * 3 + size_t(p) * 3];
                    x += a[size_t(m)] * src[0];
                    y += a[size_t(m)] * src[1];
                }
                cur[size_t(p) * 2]     = float(x);
                cur[size_t(p) * 2 + 1] = float(y);
            }
            poses[k] = Similarity2D(cur, frames_[k].target);
        }

        // --- shape given poses -----------------------------------------------
        std::fill(AtA.begin(), AtA.end(), 0.0);
        std::fill(Atb.begin(), Atb.end(), 0.0);
        for (int m = 0; m < n_id; ++m) AtA[size_t(m) * n_id + m] = cfg_.ridge;

        for (size_t k = 0; k < n_frames; ++k) {
            const FacePose& P = poses[k];
            const double s = P.s;
            const double r00 = P.R[0], r01 = P.R[1], r10 = P.R[2], r11 = P.R[3];

            // J: each identity mode's landmark offset, rotated and scaled into
            // image space -- i.e. how the projection moves per unit of alpha.
            for (int m = 0; m < n_id; ++m) {
                const float* src = &lm_id[size_t(m) * n_lm * 3];
                for (int p = 0; p < n_lm; ++p) {
                    const double vx = src[size_t(p) * 3], vy = src[size_t(p) * 3 + 1];
                    J[(size_t(p) * 2) * n_id + m]     = s * (r00 * vx + r01 * vy);
                    J[(size_t(p) * 2 + 1) * n_id + m] = s * (r10 * vx + r11 * vy);
                }
            }

            // Residual of the current base (neutral + expression) projection.
            std::vector<double> res(size_t(n_lm) * 2);
            for (int p = 0; p < n_lm; ++p) {
                const double bx = base[k][size_t(p) * 2], by = base[k][size_t(p) * 2 + 1];
                const double px = s * (r00 * bx + r01 * by) + P.t[0];
                const double py = s * (r10 * bx + r11 * by) + P.t[1];
                res[size_t(p) * 2]     = frames_[k].target[size_t(p) * 2]     - px;
                res[size_t(p) * 2 + 1] = frames_[k].target[size_t(p) * 2 + 1] - py;
            }

            for (size_t row = 0; row < size_t(n_lm) * 2; ++row) {
                const double* Jr = &J[row * n_id];
                const double rv = res[row];
                for (int m = 0; m < n_id; ++m) {
                    Atb[size_t(m)] += Jr[m] * rv;
                    double* out = &AtA[size_t(m) * n_id];
                    const double jm = Jr[m];
                    for (int m2 = m; m2 < n_id; ++m2) out[m2] += jm * Jr[m2];
                }
            }
        }
        for (int m = 0; m < n_id; ++m)                     // mirror the upper half
            for (int m2 = 0; m2 < m; ++m2)
                AtA[size_t(m) * n_id + m2] = AtA[size_t(m2) * n_id + m];

        std::vector<double> A = AtA, b = Atb;
        if (!SolveSPD(A, b, n_id)) return false;
        a = b;
    }

    alpha_.assign(size_t(basis_.identityModes()), 0.0f);
    for (int m = 0; m < n_id; ++m) alpha_[size_t(m)] = float(a[size_t(m)]);
    has_identity_ = true;

    if (residual_px) {
        double err = 0.0;
        size_t count = 0;
        std::vector<float> lm;
        for (size_t k = 0; k < n_frames; ++k) {
            basis_.reconstructLandmarks(alpha_, frames_[k].expr, lm);
            for (int p = 0; p < n_lm; ++p) {
                cur[size_t(p) * 2]     = lm[size_t(p) * 3];
                cur[size_t(p) * 2 + 1] = lm[size_t(p) * 3 + 1];
            }
            const FacePose P = Similarity2D(cur, frames_[k].target);
            for (int p = 0; p < n_lm; ++p) {
                const double x = P.s * (P.R[0] * cur[size_t(p) * 2] +
                                        P.R[1] * cur[size_t(p) * 2 + 1]) + P.t[0];
                const double y = P.s * (P.R[2] * cur[size_t(p) * 2] +
                                        P.R[3] * cur[size_t(p) * 2 + 1]) + P.t[1];
                const double dx = frames_[k].target[size_t(p) * 2] - x;
                const double dy = frames_[k].target[size_t(p) * 2 + 1] - y;
                err += std::sqrt(dx * dx + dy * dy);
                ++count;
            }
        }
        *residual_px = count ? float(err / double(count)) : 0.0f;
    }
    return true;
}

// ---------------------------------------------------------------------------

bool FaceFitter::update(const FaceResult& r, int w, int h) {
    if (!basis_.valid() || !r.valid) return false;

    mapExpression(r.blendshapes, r.blendshape_names, expr_);
    basis_.reconstruct(alpha_, expr_, verts_);
    basis_.reconstructLandmarks(alpha_, expr_, lm_model_);

    // Head rotation from MediaPipe's 4x4. Its canonical face space is y-up with
    // z toward the viewer, the same convention as the basis, so the rotation
    // block transfers without a change of basis. Translation and scale are not
    // taken from it: they are in MediaPipe's own metric space, whereas what is
    // wanted here is placement in *this* image, which the 2D similarity below
    // solves directly against the observed landmarks.
    if (use_tracker_pose_) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) rot_[i * 3 + j] = r.transform[i * 4 + j];
        // Guard against a degenerate matrix (identity is what the tracker
        // leaves when it has no pose to report, which is harmless).
        const float det =
            rot_[0] * (rot_[4] * rot_[8] - rot_[5] * rot_[7]) -
            rot_[1] * (rot_[3] * rot_[8] - rot_[5] * rot_[6]) +
            rot_[2] * (rot_[3] * rot_[7] - rot_[4] * rot_[6]);
        if (!(det > 0.5f && det < 1.5f)) {
            for (int i = 0; i < 9; ++i) rot_[i] = (i % 4 == 0) ? 1.0f : 0.0f;
        }
        // Rotate about the mesh centroid, so the head turns in place rather
        // than swinging around the model origin.
        float cx = 0, cy = 0, cz = 0;
        const size_t n = verts_.size() / 3;
        for (size_t i = 0; i < n; ++i) {
            cx += verts_[i * 3]; cy += verts_[i * 3 + 1]; cz += verts_[i * 3 + 2];
        }
        cx /= float(n); cy /= float(n); cz /= float(n);
        auto rotate = [&](std::vector<float>& v) {
            for (size_t i = 0; i < v.size() / 3; ++i) {
                const float x = v[i * 3] - cx, y = v[i * 3 + 1] - cy, z = v[i * 3 + 2] - cz;
                v[i * 3]     = rot_[0] * x + rot_[1] * y + rot_[2] * z + cx;
                v[i * 3 + 1] = rot_[3] * x + rot_[4] * y + rot_[5] * z + cy;
                v[i * 3 + 2] = rot_[6] * x + rot_[7] * y + rot_[8] * z + cz;
            }
        };
        rotate(verts_);
        rotate(lm_model_);   // the same rotation, so the similarity below agrees
    } else {
        for (int i = 0; i < 9; ++i) rot_[i] = (i % 4 == 0) ? 1.0f : 0.0f;
    }

    // Observed landmarks, and the similarity that places the model on them.
    const std::vector<int>& mp = MP68Indices();
    obs_px_.resize(mp.size() * 2);
    std::vector<float> model_xy(mp.size() * 2);
    for (size_t i = 0; i < mp.size(); ++i) {
        if (mp[i] >= int(r.landmarks.size())) return false;
        const FaceLandmark& l = r.landmarks[size_t(mp[i])];
        obs_px_[i * 2]     = l.x * float(w);
        obs_px_[i * 2 + 1] = -l.y * float(h);
        model_xy[i * 2]     = lm_model_[i * 3];
        model_xy[i * 2 + 1] = lm_model_[i * 3 + 1];
    }
    pose_ = Similarity2D(model_xy, obs_px_);
    return true;
}

void FaceFitter::headAngles(float& yaw, float& pitch, float& roll) const {
    // ZYX Euler from the rotation block.
    pitch = std::asin(std::max(-1.0f, std::min(1.0f, -rot_[6])));
    if (std::fabs(rot_[6]) < 0.9999f) {
        yaw  = std::atan2(rot_[3], rot_[0]);
        roll = std::atan2(rot_[7], rot_[8]);
    } else {
        yaw  = std::atan2(-rot_[1], rot_[4]);
        roll = 0.0f;
    }
}

void FaceFitter::projectVertices(std::vector<float>& out) const {
    const size_t n = verts_.size() / 3;
    out.resize(n * 2);
    for (size_t i = 0; i < n; ++i) {
        const float x = verts_[i * 3], y = verts_[i * 3 + 1];
        // Back to image space: the similarity lands in y-up, pixels are y-down.
        out[i * 2]     = pose_.s * (pose_.R[0] * x + pose_.R[1] * y) + pose_.t[0];
        out[i * 2 + 1] = -(pose_.s * (pose_.R[2] * x + pose_.R[3] * y) + pose_.t[1]);
    }
}

void FaceFitter::projectNormalised(int src_w, int src_h, float scale, float u_off,
                                   float v_off, std::vector<float>& out) const {
    std::vector<float> px;
    projectVertices(px);   // pixels in the fit's coordinate space, y-down
    const size_t n = px.size() / 2;
    out.resize(n * 2);
    if (src_w <= 0 || src_h <= 0) return;
    for (size_t i = 0; i < n; ++i) {
        out[i * 2]     = px[i * 2]     / float(src_w) * scale + u_off;
        out[i * 2 + 1] = px[i * 2 + 1] / float(src_h) * scale + v_off;
    }
}

void FaceFitter::sampleTexture(const std::vector<float>& image, int img_w, int img_h,
                               int src_w, int src_h, std::vector<float>& out,
                               float scale, float u_off, float v_off) const {
    const size_t n = verts_.size() / 3;
    out.assign(n * 3, 0.5f);
    if (image.size() < size_t(img_w) * img_h * 3 || img_w < 2 || img_h < 2 ||
        src_w <= 0 || src_h <= 0 || n == 0)
        return;

    std::vector<float> uv;
    projectNormalised(src_w, src_h, scale, u_off, v_off, uv);

    for (size_t i = 0; i < n; ++i) {
        // Normalised -> image pixels. Going through normalised coordinates is
        // what lets the mirror render at a different resolution from the one
        // the fit was solved at, which it always does.
        const float u = uv[i * 2]     * float(img_w) - 0.5f;
        const float v = uv[i * 2 + 1] * float(img_h) - 0.5f;

        const float uc = std::min(std::max(u, 0.0f), float(img_w - 1));
        const float vc = std::min(std::max(v, 0.0f), float(img_h - 1));
        const int x0 = int(uc), y0 = int(vc);
        const int x1 = std::min(x0 + 1, img_w - 1), y1 = std::min(y0 + 1, img_h - 1);
        const float fx = uc - float(x0), fy = vc - float(y0);

        for (int c = 0; c < 3; ++c) {
            const float a = image[(size_t(y0) * img_w + x0) * 3 + c];
            const float b = image[(size_t(y0) * img_w + x1) * 3 + c];
            const float d = image[(size_t(y1) * img_w + x0) * 3 + c];
            const float e = image[(size_t(y1) * img_w + x1) * 3 + c];
            out[i * 3 + c] = (a * (1 - fx) + b * fx) * (1 - fy) +
                             (d * (1 - fx) + e * fx) * fy;
        }
    }
}

void FaceFitter::projectLandmarks(std::vector<float>& out) const {
    const size_t n = lm_model_.size() / 3;
    out.resize(n * 2);
    for (size_t i = 0; i < n; ++i) {
        const float x = lm_model_[i * 3], y = lm_model_[i * 3 + 1];
        out[i * 2]     = pose_.s * (pose_.R[0] * x + pose_.R[1] * y) + pose_.t[0];
        out[i * 2 + 1] = -(pose_.s * (pose_.R[2] * x + pose_.R[3] * y) + pose_.t[1]);
    }
}

}  // namespace mirror
