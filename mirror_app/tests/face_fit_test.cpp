// face_fit_test — the morphable-model fit, against synthetic landmarks.
//
// Synthetic because the property worth testing is exact: if the landmarks were
// *generated* by a known identity under a known similarity, the fitter should
// recover that identity, and the residual should go to roughly zero. Against a
// real face there is no ground truth and the only check available is "the mesh
// looks like it is on the face", which a test cannot make.
//
// Needs external/face_basis.bin (tools/export_face_basis.py). Skips cleanly
// when it is absent rather than failing, matching how the MediaPipe-dependent
// paths behave when the dylib has not been built.

#include "face_basis.h"
#include "face_fit.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace mirror;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("  %-56s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

// A deterministic pseudo-random identity: no <random> engine differences, and
// the same face every run so a regression is reproducible.
std::vector<float> makeAlpha(int n, float amp) {
    std::vector<float> a(size_t(n), 0.0f);
    uint32_t s = 12345;
    for (int i = 0; i < n && i < 40; ++i) {     // only the modes the fit solves
        s = s * 1664525u + 1013904223u;
        a[size_t(i)] = amp * ((float(s >> 8) / 8388608.0f) - 1.0f);
    }
    return a;
}

// Build a FaceResult whose MP68 landmarks are `lm_model` placed by (s, angle,
// tx, ty) into a w*h image, normalised the way MediaPipe reports them.
FaceResult synthesise(const std::vector<float>& lm_model, int w, int h,
                      float scale, float angle, float tx, float ty,
                      const std::vector<float>& blendshapes,
                      const std::vector<std::string>& names) {
    FaceResult r;
    r.valid = true;
    r.landmarks.assign(478, FaceLandmark{});
    const std::vector<int>& mp = MP68Indices();
    const float c = std::cos(angle), s = std::sin(angle);
    for (size_t i = 0; i < mp.size(); ++i) {
        const float x = lm_model[i * 3], y = lm_model[i * 3 + 1];
        const float px = scale * (c * x - s * y) + tx;
        const float py = scale * (s * x + c * y) + ty;      // y-up
        FaceLandmark& l = r.landmarks[size_t(mp[i])];
        l.x = px / float(w);
        l.y = -py / float(h);                               // back to y-down
    }
    // Frontality reads landmarks 1 (nose tip), 234 and 454 (cheeks). 234 and
    // 454 are themselves MP68 targets (dlib 1 and 15, on the jawline) and are
    // already set by the projection above -- writing to them would corrupt two
    // of the 68 points the fit solves against. Landmark 1 is *not* in MP68, so
    // placing it midway between the cheeks is free and makes the synthetic face
    // score as frontal, which is what it is.
    r.landmarks[1].x = 0.5f * (r.landmarks[234].x + r.landmarks[454].x);
    r.blendshapes = blendshapes;
    r.blendshape_names = names;
    return r;
}

// MediaPipe-style names, so the ARKit name matching is exercised rather than
// bypassed. Deliberately in a different order and spelling from the basis's.
std::vector<std::string> mpNames() {
    return {"_neutral", "jawOpen", "mouthSmileLeft", "mouthSmileRight",
            "eyeBlinkLeft", "browDownLeft", "mouthPucker", "noseSneerRight"};
}

}  // namespace

// The synthetic image size, used before W/H come into scope.
static const int W_TEST = 640, H_TEST = 480;

int main() {
    std::printf("face_fit_test: morphable face fit\n\n");

    const std::string path = std::string(MIRROR_APP_EXTERNAL_DIR) + "/face_basis.bin";
    FaceFitter fitter;
    std::string err;
    if (!fitter.load(path, err)) {
        std::printf("SKIP: %s\n", err.c_str());
        std::printf("      generate it with:\n"
                    "      ../../neuromirror/.venv/bin/python "
                    "tools/export_face_basis.py\n");
        return 0;
    }

    const FaceBasis& B = fitter.basis();
    std::printf("basis: %d verts  %d tris  %d identity  %d expression  (%s topology)\n",
                B.vertexCount(), B.triangleCount(), B.identityModes(),
                B.expressionModes(), B.nvfTopology() ? "NVF/Maxine" : "ICT");

    // --- the 2D similarity, on its own ------------------------------------
    std::printf("\nSimilarity2D\n");
    {
        std::vector<float> src = {0, 0, 1, 0, 1, 1, 0, 1};
        const float k = 2.5f, ang = 0.4f;
        const float c = std::cos(ang), s = std::sin(ang);
        std::vector<float> dst(src.size());
        for (size_t i = 0; i < 4; ++i) {
            dst[i * 2]     = k * (c * src[i * 2] - s * src[i * 2 + 1]) + 7.0f;
            dst[i * 2 + 1] = k * (s * src[i * 2] + c * src[i * 2 + 1]) - 3.0f;
        }
        const FacePose p = Similarity2D(src, dst);
        check(std::fabs(p.s - k) < 1e-4f, "recovers scale");
        check(std::fabs(p.R[0] - c) < 1e-4f && std::fabs(p.R[2] - s) < 1e-4f,
              "recovers rotation");
        check(std::fabs(p.t[0] - 7.0f) < 1e-3f && std::fabs(p.t[1] + 3.0f) < 1e-3f,
              "recovers translation");
    }

    // --- the frame-quality gates -------------------------------------------
    // These decide which frames reach the identity solve at all, so they are
    // worth testing separately from it -- a gate that accepts everything and a
    // gate that accepts nothing both look like "the fit is bad" from outside.
    std::printf("\nidentity-frame gating\n");
    {
        const std::vector<std::string> names = mpNames();
        std::vector<FaceLandmark> lm(478);
        lm[234].x = 0.3f; lm[454].x = 0.7f;
        lm[1].x = 0.5f;
        check(FaceFitter::Frontality(lm) > 0.99f, "centred nose scores frontal");
        lm[1].x = 0.62f;                       // nose well off centre: turned head
        check(FaceFitter::Frontality(lm) < 0.55f, "turned head scores non-frontal");

        std::vector<float> neutral(names.size(), 0.0f);
        check(FaceFitter::Neutrality(neutral, names) > 0.99f, "neutral face scores neutral");
        std::vector<float> blinking(names.size(), 0.0f);
        blinking[4] = 1.0f;                    // eyeBlinkLeft
        check(FaceFitter::Neutrality(blinking, names) > 0.99f,
              "a blink does not count as expression");
        std::vector<float> smiling(names.size(), 0.0f);
        smiling[2] = 0.8f; smiling[3] = 0.8f;
        check(FaceFitter::Neutrality(smiling, names) < 0.5f, "a smile does");
    }

    // --- the retained set is a ranking, not a threshold --------------------
    std::printf("\nidentity sample ranking\n");
    {
        FaceFitter f2;
        std::string e2;
        f2.load(path, e2);
        f2.config().max_frames = 2;
        f2.config().min_frontality = 0.0f;
        const std::vector<std::string> names = mpNames();
        std::vector<float> lm0;
        f2.basis().reconstructLandmarks(std::vector<float>(), std::vector<float>(), lm0);

        // Three frames, increasingly expressive -> decreasing score. With only
        // two slots, the two calmest must be the ones held.
        for (int k = 0; k < 3; ++k) {
            std::vector<float> b(names.size(), 0.0f);
            b[2] = 0.3f * float(k);            // mouthSmileLeft
            b[6] = 0.3f * float(k);            // mouthPucker
            FaceResult r = synthesise(lm0, W_TEST, H_TEST, 12.f, 0.f, 320.f, -240.f,
                                      b, names);
            f2.offerIdentityFrame(r, W_TEST, H_TEST);
        }
        check(f2.identityFrames() == 2, "retained set stays at max_frames");
        float best = 0, worst = 0;
        f2.identityScores(best, worst);
        std::printf("  retained scores: best %.3f  worst %.3f\n", best, worst);
        check(worst > 0.5f, "the expressive frame was evicted, not the calm one");
    }

    // --- identity recovery -------------------------------------------------
    // The frontality floor is off here: the point is to test the solver, and
    // specifically that it subtracts a *known* expression rather than baking it
    // into identity -- which needs a deliberately expressive frame.
    std::printf("\nidentity fit (synthetic ground truth)\n");
    fitter.config().min_frontality = 0.0f;
    const int W = 640, H = 480;
    const std::vector<float> truth = makeAlpha(B.identityModes(), 1.2f);
    const std::vector<std::string> names = mpNames();
    // A smile and an open jaw, so the fit has to subtract a known expression
    // rather than absorbing it into identity.
    std::vector<float> bs(names.size(), 0.0f);
    bs[1] = 0.45f;  bs[2] = 0.6f;  bs[3] = 0.55f;

    std::vector<float> expr;
    {
        // The expression the fitter will map to, so ground truth uses the same.
        FaceResult probe = synthesise(std::vector<float>(68 * 3, 0.f), W, H,
                                      1, 0, 0, 0, bs, names);
        fitter.update(probe, W, H);
        expr = fitter.expression();
    }
    int mapped = 0;
    for (float v : expr) if (v != 0.0f) ++mapped;
    check(mapped == 3, "MediaPipe blendshapes map onto basis modes by ARKit name");

    std::vector<float> lm_truth;
    B.reconstructLandmarks(truth, expr, lm_truth);

    // Three frames at slightly different placements, as a real collection would
    // be -- the fit shares one identity across them and solves pose per frame.
    fitter.useTrackerPose(false);   // synthetic landmarks carry no head rotation
    const float scales[3] = {11.0f, 12.5f, 10.2f};
    const float angles[3] = {0.0f, 0.05f, -0.04f};
    const float txs[3]    = {320.0f, 310.0f, 330.0f};
    const float tys[3]    = {-240.0f, -235.0f, -250.0f};
    for (int k = 0; k < 3; ++k) {
        FaceResult r = synthesise(lm_truth, W, H, scales[k], angles[k],
                                  txs[k], tys[k], bs, names);
        check(fitter.offerIdentityFrame(r, W, H), "identity frame accepted");
    }
    check(fitter.identityFrames() == 3, "three frames collected");

    float residual = -1.0f;
    check(fitter.fitIdentity(&residual), "fitIdentity solves");
    std::printf("  mean landmark residual: %.4f px\n", residual);
    check(residual >= 0.0f && residual < 1.0f, "residual under 1 px");

    // The coefficients themselves. This is a correlation, not an equality, and
    // it is not close to 1 by design: ridge=6 deliberately shrinks the solution
    // toward the mean face, and the identity modes are far from orthogonal once
    // projected to 2D, so many coefficient vectors explain the same landmarks
    // about equally well. What the number rules out is a fit that found a
    // *different* face.
    //
    // The reference is neuromirror's fit_identity_frames on identical synthetic
    // input, which scores 0.7609 / 0.8061 / 0.8443 at ridge 6.0 / 1.0 / 0.1.
    // This port reproduces those to four decimals -- so a drift here is a
    // regression against the Python original, not a tuning question.
    {
        const std::vector<float>& got = fitter.alpha();
        double dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < truth.size() && i < got.size(); ++i) {
            dot += double(truth[i]) * got[i];
            na  += double(truth[i]) * truth[i];
            nb  += double(got[i]) * got[i];
        }
        const double corr = dot / (std::sqrt(na * nb) + 1e-12);
        std::printf("  alpha correlation with ground truth: %.4f "
                    "(python reference at ridge 6.0: 0.7609)\n", corr);
        check(corr > 0.70, "recovered identity correlates with ground truth");
    }

    // --- per-frame update --------------------------------------------------
    std::printf("\nper-frame update\n");
    {
        FaceResult r = synthesise(lm_truth, W, H, 12.0f, 0.02f, 315.f, -238.f,
                                  bs, names);
        check(fitter.update(r, W, H), "update() succeeds");
        check(int(fitter.vertices().size()) == B.vertexCount() * 3,
              "mesh has the render topology's vertex count");

        // The fitted landmarks should land on the observed ones.
        std::vector<float> proj;
        fitter.projectLandmarks(proj);
        const std::vector<float>& obs = fitter.observedLandmarks();
        double worst = 0.0, mean = 0.0;
        for (size_t i = 0; i < proj.size() / 2; ++i) {
            // observed is y-up; projected is flipped back to y-down pixels
            const double dx = proj[i * 2] - obs[i * 2];
            const double dy = proj[i * 2 + 1] + obs[i * 2 + 1];
            const double d = std::sqrt(dx * dx + dy * dy);
            worst = std::max(worst, d);
            mean += d;
        }
        mean /= double(proj.size() / 2);
        std::printf("  landmark reprojection: mean %.4f px  worst %.4f px\n", mean, worst);
        check(mean < 1.0, "fitted landmarks reproject onto the observed ones");

        // The projected mesh should sit inside the frame, roughly where the
        // face is. A sign error in the y-flip puts it above the image instead,
        // which is exactly the mistake the header warns about.
        std::vector<float> vpx;
        fitter.projectVertices(vpx);
        float minx = vpx[0], maxx = vpx[0], miny = vpx[1], maxy = vpx[1];
        for (size_t i = 0; i < vpx.size() / 2; ++i) {
            minx = std::min(minx, vpx[i * 2]);   maxx = std::max(maxx, vpx[i * 2]);
            miny = std::min(miny, vpx[i * 2 + 1]); maxy = std::max(maxy, vpx[i * 2 + 1]);
        }
        std::printf("  projected mesh bounds: x %.1f..%.1f  y %.1f..%.1f  (image %dx%d)\n",
                    minx, maxx, miny, maxy, W, H);
        check(miny > 0.0f && maxy < float(H) && minx > 0.0f && maxx < float(W),
              "projected mesh lands inside the image");
    }

    // --- expression drives the mesh ---------------------------------------
    std::printf("\nexpression animation\n");
    {
        std::vector<float> neutral_bs(names.size(), 0.0f);
        FaceResult a = synthesise(lm_truth, W, H, 12.f, 0.f, 320.f, -240.f,
                                  neutral_bs, names);
        fitter.update(a, W, H);
        const std::vector<float> rest = fitter.vertices();

        std::vector<float> open_bs(names.size(), 0.0f);
        open_bs[1] = 1.0f;   // jawOpen
        FaceResult b = synthesise(lm_truth, W, H, 12.f, 0.f, 320.f, -240.f,
                                  open_bs, names);
        fitter.update(b, W, H);
        const std::vector<float>& open = fitter.vertices();

        double moved = 0.0;
        for (size_t i = 0; i < rest.size(); ++i)
            moved = std::max(moved, double(std::fabs(rest[i] - open[i])));
        std::printf("  jawOpen 0 -> 1 moves the mesh by up to %.3f model units\n", moved);
        check(moved > 0.1, "jawOpen visibly deforms the mesh");
    }

    // --- texture sampling --------------------------------------------------
    std::printf("\ntexture sampling\n");
    {
        FaceResult r = synthesise(lm_truth, W, H, 12.f, 0.f, 320.f, -240.f, bs, names);
        fitter.update(r, W, H);

        // A flat image: every vertex must come back with exactly that colour,
        // whatever the projection does.
        const int IW = 64, IH = 48;
        std::vector<float> flat(size_t(IW) * IH * 3);
        for (size_t i = 0; i < flat.size(); i += 3) {
            flat[i] = 0.25f; flat[i + 1] = 0.5f; flat[i + 2] = 0.75f;
        }
        std::vector<float> col;
        fitter.sampleTexture(flat, IW, IH, W, H, col);
        check(int(col.size()) == B.vertexCount() * 3, "one colour per vertex");
        bool ok = true;
        for (size_t i = 0; i + 2 < col.size(); i += 3)
            ok &= std::fabs(col[i] - 0.25f) < 1e-4f &&
                  std::fabs(col[i + 1] - 0.5f) < 1e-4f &&
                  std::fabs(col[i + 2] - 0.75f) < 1e-4f;
        check(ok, "a flat image samples to a flat mesh colour");

        // Left half red, right half blue. The mesh spans the middle of the
        // image, so both must appear -- and, critically, the left of the *mesh*
        // must take the left of the *image*. A mirrored or transposed sampling
        // passes the flat test above and fails this one.
        std::vector<float> split(size_t(IW) * IH * 3, 0.f);
        for (int y = 0; y < IH; ++y)
            for (int x = 0; x < IW; ++x) {
                float* p = &split[(size_t(y) * IW + x) * 3];
                if (x < IW / 2) p[0] = 1.f; else p[2] = 1.f;
            }
        fitter.sampleTexture(split, IW, IH, W, H, col);

        std::vector<float> px;
        fitter.projectVertices(px);
        int leftmost = 0, rightmost = 0;
        for (size_t i = 0; i < px.size() / 2; ++i) {
            if (px[i * 2] < px[size_t(leftmost) * 2])  leftmost = int(i);
            if (px[i * 2] > px[size_t(rightmost) * 2]) rightmost = int(i);
        }
        std::printf("  leftmost vertex rgb  %.2f %.2f %.2f\n",
                    col[size_t(leftmost) * 3], col[size_t(leftmost) * 3 + 1],
                    col[size_t(leftmost) * 3 + 2]);
        std::printf("  rightmost vertex rgb %.2f %.2f %.2f\n",
                    col[size_t(rightmost) * 3], col[size_t(rightmost) * 3 + 1],
                    col[size_t(rightmost) * 3 + 2]);
        check(col[size_t(leftmost) * 3] > 0.5f, "mesh left samples the image left (red)");
        check(col[size_t(rightmost) * 3 + 2] > 0.5f,
              "mesh right samples the image right (blue)");
    }

    std::printf("\n%s\n", g_failures ? "FAILURES" : "all checks passed");
    return g_failures ? 1 : 0;
}
