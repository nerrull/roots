// face_fit — fit a morphable face model to MediaPipe landmarks, then animate it.
//
// A C++ port of neuromirror/emotion/ict_fit.py, split the way the cost splits:
//
//   identity  expensive, occasional. Alternates a 2D similarity (Umeyama) for
//             pose given shape with a ridge-regularised linear solve for the
//             identity coefficients given pose. Runs over several collected
//             frames at once, which averages out landmark noise. Milliseconds,
//             once per person.
//
//   per frame cheap. Expression comes straight from MediaPipe's blendshapes
//             (matched to the basis by ARKit name), head rotation straight from
//             MediaPipe's 4x4, and the mesh is one weighted sum of modes. This
//             is what runs at frame rate.
//
// Expression is *known* during the identity fit rather than solved for, and its
// landmark contribution is subtracted first -- otherwise a person who happened
// to be smiling in the collected frames gets a smile baked into their identity.
//
// Two things the Python version needs that this does not:
//
//   * OpenCV. `solve_pose` used cv2.solvePnP for head rotation; MediaPipe's
//     Tasks API already hands back a facial transformation matrix, so the
//     rotation is read off it directly and there is no PnP and no new
//     dependency. `useTrackerPose(false)` falls back to the flat 2D similarity,
//     which is enough for placement if the pose matrix ever misbehaves.
//   * NumPy. The solves are small and dense -- a 2x2 similarity over 68 points
//     and an 80x80 symmetric system -- so they are written out (Cholesky).
//
// Units and axes are the basis's (see face_basis.h): ICT's y-up centimetres.
// MediaPipe's pixels are y-down, so targets are flipped into y-up on the way
// in and results are flipped back on the way out. That flip is easy to lose in
// translation and produces an upside-down face when lost.

#pragma once

#include <string>
#include <vector>

#include "face_basis.h"
#include "face_tracker.h"

namespace mirror {

// dlib-68 landmark points as MediaPipe FaceMesh vertex indices. Needed no
// matter which morphable model is used -- it is the bridge between what
// MediaPipe outputs (478 mesh points) and what the basis is indexed by (68).
const std::vector<int>& MP68Indices();

// A weak-perspective placement: p_image_yup = s * R * p_model_xy + t.
struct FacePose {
    float s = 1.0f;
    float R[4] = {1, 0, 0, 1};   // row-major 2x2
    float t[2] = {0, 0};
};

// Best similarity (rotation + uniform scale + translation) mapping src -> dst,
// both n interleaved xy pairs. The 2D case of Umeyama, in closed form: no SVD
// is needed because a 2x2 rotation is one angle.
FacePose Similarity2D(const std::vector<float>& src, const std::vector<float>& dst);

class FaceFitter {
public:
    struct Config {
        // Identity modes actually solved for, of the basis's 100. The tail
        // modes are increasingly fine detail that 68 landmarks cannot resolve,
        // and fitting them is how a fit starts chasing landmark noise.
        int   n_identity = 80;
        float ridge      = 6.0f;   // pulls the solve toward the mean face
        int   iterations = 12;     // pose/shape alternations
        int   max_frames = 8;      // identity samples retained
        // A floor, not the selection mechanism -- see offerIdentityFrame().
        // Only rejects frames with no useful information in them at all.
        float min_frontality = 0.25f;
    };

    FaceFitter() = default;

    bool load(const std::string& basis_path, std::string& err);
    bool valid() const { return basis_.valid(); }
    const FaceBasis& basis() const { return basis_; }
    Config& config() { return cfg_; }

    // --- scoring (exposed so the app can show why a frame was rejected) -----
    // 1 for a face looking straight at the camera, falling off with yaw.
    static float Frontality(const std::vector<FaceLandmark>& landmarks);
    // 1 for a neutral face, falling off with total expression. Blinks and eye
    // darts are ignored -- they say nothing about identity and never stop.
    static float Neutrality(const std::vector<float>& blendshapes,
                            const std::vector<std::string>& names);

    // --- identity ----------------------------------------------------------
    // Offer a frame as an identity sample. Returns true if it was kept.
    //
    // The retained set is the best `max_frames` seen so far, ranked by
    // frontality * neutrality -- a *ranking*, not a threshold. A fixed
    // threshold cannot work: whether any frame clears it depends on the person,
    // the lighting and the sensor, and MediaPipe reports substantial baseline
    // activation on an ordinary face (a frame of someone mid-sentence scores
    // 0.04 neutrality, and that is correct rather than a bug). With a threshold
    // the collection either fills instantly or never fills at all. Ranking over
    // a fixed window always terminates and adapts to whoever is in front of it.
    //
    // So the caller collects for a few seconds and then fits, rather than
    // waiting for a count.
    bool offerIdentityFrame(const FaceResult& r, int w, int h);
    int  identityFrames() const { return int(frames_.size()); }
    // Score range of the retained set, for showing the user whether holding
    // still is helping.
    void identityScores(float& best, float& worst) const;
    void clearIdentity();

    // Solve identity over the collected frames. Returns false if there are
    // none. `residual_px` receives the mean landmark error afterwards, which is
    // the number worth watching: a good fit lands within a couple of pixels.
    bool fitIdentity(float* residual_px = nullptr);
    bool hasIdentity() const { return has_identity_; }
    const std::vector<float>& alpha() const { return alpha_; }

    // --- per frame ---------------------------------------------------------
    // Expression + pose + mesh for the current frame. Returns false if the
    // result is invalid or no basis is loaded. Works before fitIdentity() --
    // the mesh is then the mean face, animated, which is a reasonable mask
    // while identity samples are still being collected.
    bool update(const FaceResult& r, int w, int h);

    // Fitted mesh, 3 floats/vertex, in model units, rotated by head pose about
    // its own centroid when the tracker pose is in use.
    const std::vector<float>& vertices() const { return verts_; }
    const std::vector<float>& expression() const { return expr_; }
    const FacePose& pose() const { return pose_; }
    // Head rotation in radians, for display. Zero without a tracker pose.
    void headAngles(float& yaw, float& pitch, float& roll) const;

    void useTrackerPose(bool on) { use_tracker_pose_ = on; }
    bool trackerPose() const { return use_tracker_pose_; }

    // Project the fitted mesh into pixel space via the current pose, as
    // interleaved xy. This is what makes the fit checkable against the video:
    // if these do not land on the face, nothing downstream will either.
    void projectVertices(std::vector<float>& px_out) const;
    // The same for the 68 landmarks, alongside the observed targets -- the
    // pair is the fit residual made visible.
    void projectLandmarks(std::vector<float>& px_out) const;
    const std::vector<float>& observedLandmarks() const { return obs_px_; }

    // --- texturing ---------------------------------------------------------
    // Colour every vertex of the fitted mesh by projecting it into `image`
    // (h*w*3 floats in [0,1], row-major RGB) and sampling. Output is
    // vertexCount()*3 floats, ready to hand to the root scene's face pass.
    //
    // `src_w`/`src_h` are the pixel dimensions the *fit* was solved in, which
    // is what projectVertices() lands in; `image` may be a different size (the
    // neural mirror renders at its own resolution), and is addressed in
    // normalised coordinates, so the two need not match.
    //
    // Bilinear, and clamped at the edges: a vertex whose projection falls
    // outside the image takes the nearest edge pixel rather than black, so a
    // partly out-of-frame face degrades at the boundary instead of growing a
    // hard dark band across the mask.
    //
    // `u_off`/`v_off` shift the sample point in normalised coordinates, and are
    // what pin the texture to the fit rather than to the camera. The tracker
    // says where the face was *seen*; the mirror draws it where the fit *put*
    // it, and those are only the same place in some of the head-movement modes.
    // Centring the subject moves the rendered face to the middle of the frame
    // while the landmarks stay out at the edge -- sampling at the landmark
    // position then lifts colour from whatever generative field happens to be
    // there, which is the texture mapping visibly failing.
    // The pin is a similarity, not just a shift: with a face-size control in
    // play the subject is scaled about the frame centre as well as moved.
    // uv_sampled = uv_projected * scale + (u_off, v_off).
    void sampleTexture(const std::vector<float>& image, int img_w, int img_h,
                       int src_w, int src_h, std::vector<float>& rgb_out,
                       float scale = 1.f, float u_off = 0.f,
                       float v_off = 0.f) const;

    // The fitted mesh projected into normalised frame coordinates (0..1, y-down,
    // the same space the landmarks are in), with the same pinning offset. This
    // is what a view of the fit draws, and sharing the offset with
    // sampleTexture is what keeps the drawn mask and its colours in register.
    void projectNormalised(int src_w, int src_h, float scale, float u_off,
                           float v_off, std::vector<float>& uv_out) const;

private:
    struct Sample {
        std::vector<float> target;   // 68 xy, pixels, y-up
        std::vector<float> expr;     // expression weights
        float score = 0.0f;          // frontality * neutrality
    };

    // MediaPipe blendshape scores -> basis expression weights, by ARKit name.
    void mapExpression(const std::vector<float>& blendshapes,
                       const std::vector<std::string>& names,
                       std::vector<float>& out) const;
    void buildNameMap(const std::vector<std::string>& names) const;

    FaceBasis basis_;
    Config cfg_;
    std::vector<Sample> frames_;
    std::vector<float> alpha_, expr_, verts_, lm_model_, obs_px_;
    FacePose pose_;
    float rot_[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    bool has_identity_ = false;
    bool use_tracker_pose_ = true;

    // Cached MediaPipe-name -> expression-mode mapping, rebuilt when the
    // tracker's name list changes (i.e. once).
    mutable std::vector<int> name_map_;
    mutable std::vector<std::string> mapped_names_;
};

}  // namespace mirror
