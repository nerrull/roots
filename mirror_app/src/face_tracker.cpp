// FaceTracker over MediaPipe's C Tasks API.
//
// Compiled unconditionally; the MediaPipe calls are behind MIRROR_HAVE_MEDIAPIPE
// so the app links whether or not libface_landmarker.dylib has been built. That
// keeps FaceTracker::available() answerable rather than a link error, and lets
// the mask geometry be tested with no backend present.

#include "face_tracker.h"

#include <algorithm>

#if MIRROR_HAVE_MEDIAPIPE
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker.h"
#endif

namespace mirror {

bool FaceTracker::available() {
#if MIRROR_HAVE_MEDIAPIPE
    return true;
#else
    return false;
#endif
}

#if MIRROR_HAVE_MEDIAPIPE

struct FaceTracker::Impl {
    MpFaceLandmarkerPtr landmarker = nullptr;
    std::string err;
    int64_t last_ts = -1;
};

FaceTracker::FaceTracker() : impl_(new Impl()) {}
FaceTracker::~FaceTracker() { close(); }

bool FaceTracker::isOpen() const { return impl_->landmarker != nullptr; }
std::string FaceTracker::error() const { return impl_->err; }

bool FaceTracker::open(const std::string& model_path, std::string& err) {
    if (impl_->landmarker) return true;

    MpFaceLandmarkerOptions opts{};
    opts.base_options.model_asset_path = model_path.c_str();
    // VIDEO rather than IMAGE: the video graph keeps tracking state between
    // frames, so it only runs the (expensive) face detector when it loses the
    // face and otherwise refines from the previous result. On a live feed that
    // is both faster and much steadier frame to frame.
    opts.running_mode = MpRunningMode::MP_RUNNING_MODE_VIDEO;
    opts.num_faces = 1;
    opts.min_face_detection_confidence = 0.5f;
    opts.min_face_presence_confidence = 0.5f;
    opts.min_tracking_confidence = 0.5f;
    // Both requested: blendshapes and the pose matrix are what PLAN.md wants
    // fed into the root scene's mask placement, and asking for them at create
    // time is the only way to get them.
    opts.output_face_blendshapes = true;
    opts.output_facial_transformation_matrixes = true;
    opts.result_callback = nullptr;

    char* msg = nullptr;
    const MpStatus st = MpFaceLandmarkerCreate(&opts, &impl_->landmarker, &msg);
    if (st != kMpOk) {
        err = msg ? msg : "MpFaceLandmarkerCreate failed";
        err += " (model: " + model_path + ")";
        if (msg) MpErrorFree(msg);
        impl_->landmarker = nullptr;
        impl_->err = err;
        return false;
    }
    impl_->err.clear();
    impl_->last_ts = -1;
    return true;
}

void FaceTracker::close() {
    if (impl_ && impl_->landmarker) {
        char* msg = nullptr;
        MpFaceLandmarkerClose(impl_->landmarker, &msg);
        if (msg) MpErrorFree(msg);
        impl_->landmarker = nullptr;
    }
}

bool FaceTracker::detect(const unsigned char* rgb, int w, int h,
                         int64_t timestamp_ms, FaceResult& out) {
    out = FaceResult{};
    if (!impl_->landmarker || !rgb || w <= 0 || h <= 0) return false;

    // The video graph rejects a non-increasing timestamp with a hard error
    // rather than dropping the frame, and a render loop faster than the camera
    // will happily present the same millisecond twice. Nudge instead of failing.
    if (timestamp_ms <= impl_->last_ts) timestamp_ms = impl_->last_ts + 1;
    impl_->last_ts = timestamp_ms;

    char* msg = nullptr;
    MpImagePtr image = nullptr;
    if (MpImageCreateFromUint8Data(kMpImageFormatSrgb, w, h, rgb, w * h * 3,
                                   &image, &msg) != kMpOk) {
        impl_->err = msg ? msg : "MpImageCreateFromUint8Data failed";
        if (msg) MpErrorFree(msg);
        return false;
    }

    MpFaceLandmarkerResult res{};
    const MpStatus st = MpFaceLandmarkerDetectForVideo(
        impl_->landmarker, image, /*options=*/nullptr, timestamp_ms, &res, &msg);
    MpImageFree(image);
    if (st != kMpOk) {
        impl_->err = msg ? msg : "MpFaceLandmarkerDetectForVideo failed";
        if (msg) MpErrorFree(msg);
        return false;
    }
    impl_->err.clear();

    // No face is a normal outcome, not an error: nobody is in front of the
    // sensor most of the time.
    if (res.face_landmarks_count == 0 || !res.face_landmarks) {
        MpFaceLandmarkerCloseResult(&res);
        return false;
    }

    const MpNormalizedLandmarks& face = res.face_landmarks[0];
    out.landmarks.resize(face.landmarks_count);
    for (uint32_t i = 0; i < face.landmarks_count; ++i) {
        out.landmarks[i].x = face.landmarks[i].x;
        out.landmarks[i].y = face.landmarks[i].y;
        out.landmarks[i].z = face.landmarks[i].z;
    }

    if (res.face_blendshapes && res.face_blendshapes_count > 0) {
        const MpCategories& bs = res.face_blendshapes[0];
        out.blendshapes.resize(bs.categories_count);
        for (uint32_t i = 0; i < bs.categories_count; ++i) {
            out.blendshapes[i] = bs.categories[i].score;
        }
        // Names, once. The fitter matches blendshapes to the face basis's
        // expression modes by ARKit name, so it needs to know which score is
        // which -- and taking that from the model rather than from a
        // hardcoded 52-item list means a different .task bundle cannot
        // silently shift every expression by one slot.
        if (out.blendshape_names.size() != out.blendshapes.size()) {
            out.blendshape_names.clear();
            out.blendshape_names.reserve(bs.categories_count);
            for (uint32_t i = 0; i < bs.categories_count; ++i) {
                const char* n = bs.categories[i].category_name;
                out.blendshape_names.emplace_back(n ? n : "");
            }
        }
    }

    if (res.facial_transformation_matrixes &&
        res.facial_transformation_matrixes_count > 0) {
        const MpMatrix& m = res.facial_transformation_matrixes[0];
        // MpMatrix is column-major; FaceResult::transform is row-major.
        if (m.rows == 4 && m.cols == 4 && m.data) {
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    out.transform[r * 4 + c] = m.data[c * 4 + r];
        }
    }

    MpFaceLandmarkerCloseResult(&res);

    out.min_x = out.max_x = out.landmarks[0].x;
    out.min_y = out.max_y = out.landmarks[0].y;
    for (const FaceLandmark& l : out.landmarks) {
        out.min_x = std::min(out.min_x, l.x);
        out.max_x = std::max(out.max_x, l.x);
        out.min_y = std::min(out.min_y, l.y);
        out.max_y = std::max(out.max_y, l.y);
    }
    out.centre_x = 0.5f * (out.min_x + out.max_x);
    out.centre_y = 0.5f * (out.min_y + out.max_y);
    out.valid = true;
    return true;
}

#else   // no MediaPipe

struct FaceTracker::Impl {
    std::string err = "MediaPipe face tracking is not compiled in "
                      "(build libface_landmarker.dylib -- see CMakeLists.txt)";
};

FaceTracker::FaceTracker() : impl_(new Impl()) {}
FaceTracker::~FaceTracker() = default;
bool FaceTracker::isOpen() const { return false; }
std::string FaceTracker::error() const { return impl_->err; }
void FaceTracker::close() {}

bool FaceTracker::open(const std::string&, std::string& err) {
    err = impl_->err;
    return false;
}

bool FaceTracker::detect(const unsigned char*, int, int, int64_t,
                         FaceResult& out) {
    out = FaceResult{};
    return false;
}

#endif  // MIRROR_HAVE_MEDIAPIPE

}  // namespace mirror
