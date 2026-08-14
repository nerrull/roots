// face_tracker — MediaPipe face landmarks, and the mask they produce.
//
// Two things come out of this, per PLAN.md: a training mask (fit the face,
// leave the background generative) and placement for the root scene's face
// mask. Both need the same thing — 478 landmarks in normalised image space —
// so they share one tracker.
//
// The backend is MediaPipe's *official* C Tasks API
// (mediapipe/tasks/c/vision/face_landmarker), built as a shared library. That
// is deliberately not cpvrlab/libmediapipe, which pins MediaPipe v0.8.11:
// v0.8.11 predates the Tasks API entirely, so it offers the legacy 468-point
// face_mesh with no blendshapes and no transformation matrix, and it is
// GPL-3.0. The upstream C API gives 478 landmarks, 52 blendshapes and a 4x4
// pose matrix under Apache-2.0.
//
// Compiled only when that library is present (MIRROR_HAVE_MEDIAPIPE); without
// it the app builds and runs with face tracking simply absent.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mirror {

// A landmark in normalised image space: x, y in [0,1] from the top-left, z
// roughly in the same units as x with the head centre near zero (negative is
// toward the camera). Not metric -- MediaPipe's z is a relative depth.
struct FaceLandmark {
    float x = 0, y = 0, z = 0;
};

struct FaceResult {
    bool valid = false;
    std::vector<FaceLandmark> landmarks;   // 478 when valid
    std::vector<float> blendshapes;        // 52 when requested and valid
    // Category names for `blendshapes`, in the same order ("jawOpen",
    // "mouthSmileLeft", ...). Filled once and then left alone -- the tracker
    // only refills it if it no longer matches the score count.
    std::vector<std::string> blendshape_names;
    // Row-major 4x4 head pose, when requested. Identity if unavailable.
    float transform[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

    // Axis-aligned bounds of the landmarks, normalised. Cheap placement info
    // that does not require touching the full landmark list.
    float min_x = 0, min_y = 0, max_x = 0, max_y = 0;
    float centre_x = 0, centre_y = 0;
};

// The 468-point face-mesh silhouette, in order, as MediaPipe indexes it. This
// is the outline used to build a face mask; the extra 10 points (468..477) are
// the iris refinements and are deliberately not part of it.
const std::vector<int>& FaceOvalIndices();

// Rasterise a filled polygon over the landmark outline into an h*w mask
// (non-zero = inside). `dilate_px` grows the region, which matters for
// training: a mask tight to the silhouette gives the network no "background"
// pixels near the edge, and with only positive supervision it has no reason to
// form a boundary there (measured on the square: a tight mask converges to
// loss 0 while producing a soft blob rather than an edge).
//
// `indices` selects which landmarks form the outline; FaceOvalIndices() is the
// usual choice.
void RasteriseFaceMask(const std::vector<FaceLandmark>& landmarks,
                       const std::vector<int>& indices, int w, int h,
                       int dilate_px, std::vector<unsigned char>& mask);

// The landmarks' bounding box as a filled rectangle -- the "face crop" rather
// than the face outline. Coarser than the hull on purpose: the hull cuts along
// the silhouette, so the network is supervised on skin and never on the edge
// between a person and the room, while a box hands it the whole crop including
// hair, jawline and the background immediately around them.
//
// `pad_frac` grows the box by a fraction of its own size, so the margin scales
// with how close the person is; `pad_px` adds a fixed margin on top of that.
void RasteriseFaceBox(const std::vector<FaceLandmark>& landmarks, int w, int h,
                      float pad_frac, int pad_px,
                      std::vector<unsigned char>& mask);

// Euclidean distance, in pixels, from each pixel to the nearest set pixel of
// `mask`. Zero inside the mask, growing outward.
//
// This is what lets the soft edge follow a face instead of a rectangle: the
// contours of a distance field are offset curves of whatever shape produced it,
// evenly spaced in every direction. The analytic box it replaces normalised the
// distance per axis, which made the fade band a different width horizontally
// than vertically and flared it at the corners -- the gradient visibly pooling
// along the box edges.
//
// Exact (Felzenszwalb & Huttenlocher's separable transform), O(w*h), and cheap
// enough to redo every frame at fit-grid size.
void DistanceOutside(const std::vector<unsigned char>& mask, int w, int h,
                     std::vector<float>& dist_px);

// The same fill from an explicit normalised box, for the head-centred fit mode:
// there the crop is pinned to the middle of the frame while the landmarks are
// off wherever the person is, so the box cannot be derived from them.
void RasteriseBox(float cx, float cy, float hx, float hy, int w, int h,
                  int pad_px, std::vector<unsigned char>& mask);

class FaceTracker {
public:
    FaceTracker();
    ~FaceTracker();

    // `model_path` is a face_landmarker .task bundle. Returns false with `err`
    // set if the model is missing or the backend is not compiled in.
    bool open(const std::string& model_path, std::string& err);
    void close();
    bool isOpen() const;

    // Detect on an RGB8 frame. `timestamp_ms` must increase monotonically --
    // the video-mode graph rejects a repeated or decreasing timestamp, which
    // shows up as a hard error rather than a dropped frame.
    bool detect(const unsigned char* rgb, int w, int h, int64_t timestamp_ms,
                FaceResult& out);

    std::string error() const;
    static bool available();   // compiled in?

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mirror
