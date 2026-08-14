// mirror_app — unified Metal app hosting the neural mirror and the 3D root scene.
//
// This file is the shell: GLFW window + CAMetalLayer + Dear ImGui (Metal backend)
// + the main loop, adapted from neuromirror/reactor_cpp/src/main.mm. Scenes
// (MirrorScene, MetalRootRenderer) plug into MetalContext and render into offscreen
// textures that get composited here. For now it is a runnable empty-window
// checkpoint that verifies the Metal + GLFW + ImGui + CMake toolchain end to end.
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <Cocoa/Cocoa.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_metal.h"

#include "metal_context.h"
#include "mirror_scene.h"
#include "fit_target.h"
#include "face_tracker.h"
#include "face_fit.h"
#if MIRROR_HAVE_KINECT
#include "kinect_target.h"
#endif
#include "root_scene.h"
#include "transition_scene.h"
#include "fit_view_scene.h"
#include "ui_params.h"
#include "midi_in.h"
#include "fullscreen_present.h"
#include "text_overlay.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

// Headless check of the MLX→Metal texture path (no window): render a few mirror
// frames and read back a pixel. Used to smoke-test without a GUI.
static int selftest() {
    MetalContext ctx;
    if (!ctx.device()) { fprintf(stderr, "selftest: no Metal device\n"); return 1; }
    MirrorScene mirror(ctx);
    if (!mirror.valid()) { fprintf(stderr, "selftest: mirror invalid\n"); return 1; }
    // Exercise the transition path too, so the whole pipeline is covered.
    mirror.params().transition = 0.5f;
    id<MTLTexture> tex = nil;
    for (int i = 0; i < 5; ++i) { mirror.advance(1.0 / 60.0); tex = mirror.render(); }
    // Read back one RGBA16F texel (center) to confirm real data landed.
    uint16_t px[4] = {0, 0, 0, 0};
    NSUInteger cx = tex.width / 2, cy = tex.height / 2;
    [tex getBytes:px bytesPerRow:sizeof(px)
       fromRegion:MTLRegionMake2D(cx, cy, 1, 1) mipmapLevel:0];
    auto h2f = [](uint16_t h) {
        uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, bits;
        if (e == 0) bits = (s << 31) | 0; else bits = (s << 31) | ((e + 112) << 23) | (m << 13);
        float f; __builtin_memcpy(&f, &bits, 4); return f;
    };
    printf("selftest: %dx%d center rgba = %.3f %.3f %.3f %.3f  OK\n",
           (int)tex.width, (int)tex.height, h2f(px[0]), h2f(px[1]), h2f(px[2]), h2f(px[3]));
    return 0;
}

// Headless throughput benchmark of the mirror's MLX compute (features + fused
// MLP), matching demo_pond's method: eval each frame, synchronize once at the
// end. Reports ms/frame at 1920x1080 / downscale.  Usage: --bench [downscale] [frames]
static int bench(int downscale, int frames) {
    MetalContext ctx;
    if (!ctx.device()) { fprintf(stderr, "bench: no Metal device\n"); return 1; }
    const int W = 1920, H = 1080;
    const int lw = std::max(2, W / downscale), lh = std::max(2, H / downscale);
    mirror::Pond pond(11);
    mirror::PondParams p;

    auto now = [] { return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count(); };

    { auto a = pond.render(lh, lw, 0.0, p); mirror::mx::eval(a); }   // warmup
    mirror::mx::synchronize();
    double t0 = now();
    for (int f = 0; f < frames; ++f) {
        auto a = pond.render(lh, lw, f / 60.0, p);
        mirror::mx::eval(a);
    }
    mirror::mx::synchronize();
    double dt = (now() - t0) / frames;
    printf("bench: pond compute %dx%d (ds=%d, %d frames): %.3f ms/frame (%.0f fps)\n",
           lw, lh, downscale, frames, dt * 1e3, 1.0 / dt);
    return 0;
}

// Headless check of the root scene's Metal pipeline (no window): compiles the
// MSL passes, renders a few frames of the synthetic root structure into the
// offscreen fog texture, and reads back the centre texel to confirm real data.
static int roottest() {
    MetalContext ctx;
    if (!ctx.device()) { fprintf(stderr, "roottest: no Metal device\n"); return 1; }
    RootScene roots(ctx, 640, 360);
    if (!roots.valid()) { fprintf(stderr, "roottest: root scene invalid (shader compile?)\n"); return 1; }
    id<MTLTexture> tex = nil;
    for (int i = 0; i < 3; ++i) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [ctx.queue() commandBuffer];
            roots.advance(1.0 / 60.0);
            tex = roots.render(cb);
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
    if (!tex) { fprintf(stderr, "roottest: no texture\n"); return 1; }
    uint16_t px[4] = {0, 0, 0, 0};
    NSUInteger cx = tex.width / 2, cy = tex.height / 2;
    [tex getBytes:px bytesPerRow:sizeof(px)
       fromRegion:MTLRegionMake2D(cx, cy, 1, 1) mipmapLevel:0];
    auto h2f = [](uint16_t h) {
        uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, bits;
        if (e == 0) bits = (s << 31) | 0; else bits = (s << 31) | ((e + 112) << 23) | (m << 13);
        float f; __builtin_memcpy(&f, &bits, 4); return f;
    };
    printf("roottest: %dx%d center rgba = %.3f %.3f %.3f %.3f  OK\n",
           (int)tex.width, (int)tex.height, h2f(px[0]), h2f(px[1]), h2f(px[2]), h2f(px[3]));
    return 0;
}

// Headless render of the root scene to a PPM file for visual validation.
// Usage: --rootshot <out.ppm> [az] [el] [radius] [mode] [overlays]
// mode: 0 Phong (default), 1 PBR, 2 Invert.  overlays: 1 = axes + grid.
static int rootshot(const char* path, float az, float el, float rad, int mode, bool overlays) {
    MetalContext ctx;
    if (!ctx.device()) { fprintf(stderr, "rootshot: no Metal device\n"); return 1; }
    const int W = 960, H = 540;
    RootScene roots(ctx, W, H);
    if (!roots.valid()) { fprintf(stderr, "rootshot: root scene invalid\n"); return 1; }
    roots.autoOrbit = false;
    roots.azimuth = az; roots.elevation = el; roots.radius = rad;
    roots.renderer().shaderMode = (MetalRootRenderer::ShaderMode)mode;
    if (overlays) {
        roots.renderer().overlay.showAxes = true;
        roots.renderer().overlay.showGrid = true;
    }
    id<MTLTexture> tex = nil;
    for (int i = 0; i < 3; ++i) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [ctx.queue() commandBuffer];
            roots.advance(1.0 / 60.0);
            tex = roots.render(cb);
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
    std::vector<uint16_t> px((size_t)W * H * 4);
    [tex getBytes:px.data() bytesPerRow:W * 4 * sizeof(uint16_t)
       fromRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0];
    auto h2f = [](uint16_t h) {
        uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, bits;
        if (e == 0) bits = (s << 31) | 0; else bits = (s << 31) | ((e + 112) << 23) | (m << 13);
        float f; __builtin_memcpy(&f, &bits, 4); return f;
    };
    FILE* fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "rootshot: cannot open %s\n", path); return 1; }
    fprintf(fp, "P6\n%d %d\n255\n", W, H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const uint16_t* p = &px[((size_t)y * W + x) * 4];
            for (int c = 0; c < 3; ++c) {
                float v = h2f(p[c]);
                v = v <= 0.f ? 0.f : (v >= 1.f ? 1.f : v);
                unsigned char b = (unsigned char)(powf(v, 1.0f / 2.2f) * 255.0f + 0.5f);
                fputc(b, fp);
            }
        }
    fclose(fp);
    printf("rootshot: wrote %s (%dx%d)\n", path, W, H);
    return 0;
}

// Headless live-growth check: steps the CPlantBox sim `steps` frames, then
// renders to a PPM. Usage: --growshot <out.ppm> [steps] [az] [el] [radius]
static int growshot(const char* path, int steps, float az, float el, float rad,
                    float faceScale = -1.f, float targetY = -1e9f) {
    MetalContext ctx;
    if (!ctx.device()) { fprintf(stderr, "growshot: no Metal device\n"); return 1; }
    const int W = 960, H = 540;
    RootScene roots(ctx, W, H);
    if (!roots.valid()) { fprintf(stderr, "growshot: root scene invalid\n"); return 1; }
    printf("growshot: sim active = %d\n", roots.simActive() ? 1 : 0);
    roots.autoOrbit = false;
    if (rad > 0) roots.radius = rad;
    // Framing overrides, so a single mask can be filled the frame with. The
    // masks are a couple of centimetres on a fifty-centimetre cone, and their
    // orientation is not decidable at the scale the whole system is shot at.
    if (faceScale > 0) roots.faceScale = faceScale;
    if (targetY > -1e8f) roots.target[1] = targetY;
    roots.azimuth = az; roots.elevation = el;
    id<MTLTexture> tex = nil;
    for (int i = 0; i < steps; ++i) roots.advance(1.0 / 60.0);   // grow (no GPU work)
    for (int i = 0; i < 2; ++i) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [ctx.queue() commandBuffer];
            tex = roots.render(cb);
            [cb commit]; [cb waitUntilCompleted];
        }
    }
    std::vector<uint16_t> px((size_t)W * H * 4);
    [tex getBytes:px.data() bytesPerRow:W * 4 * sizeof(uint16_t)
       fromRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0];
    auto h2f = [](uint16_t h) {
        uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, bits;
        if (e == 0) bits = (s << 31) | 0; else bits = (s << 31) | ((e + 112) << 23) | (m << 13);
        float f; __builtin_memcpy(&f, &bits, 4); return f;
    };
    FILE* fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "growshot: cannot open %s\n", path); return 1; }
    fprintf(fp, "P6\n%d %d\n255\n", W, H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const uint16_t* p = &px[((size_t)y * W + x) * 4];
            for (int c = 0; c < 3; ++c) {
                float v = h2f(p[c]); v = v <= 0.f ? 0.f : (v >= 1.f ? 1.f : v);
                fputc((unsigned char)(powf(v, 1.0f / 2.2f) * 255.0f + 0.5f), fp);
            }
        }
    fclose(fp);
    printf("growshot: wrote %s (%dx%d, %d steps, done=%d)\n",
           path, W, H, steps, roots.simDone() ? 1 : 0);
    return 0;
}

// Headless GPU benchmark of the root render: grow the sim to completion, then
// time `frames` full render()s (geometry + face + fog). Usage:
//   --rootbench [downscale] [frames]
static int rootbench(int downscale, int frames, int baseW, int baseH) {
    MetalContext ctx;
    if (!ctx.device()) { fprintf(stderr, "rootbench: no Metal device\n"); return 1; }
    const int W = baseW / std::max(1, downscale), H = baseH / std::max(1, downscale);
    RootScene roots(ctx, W, H);
    if (!roots.valid()) { fprintf(stderr, "rootbench: invalid\n"); return 1; }
    roots.autoOrbit = false;
    if (const char* m = getenv("ROOTBENCH_MODE"))
        roots.renderer().shaderMode = (MetalRootRenderer::ShaderMode)atoi(m);
    if (const char* r = getenv("ROOTBENCH_RADIUS")) roots.radius = atof(r);

    auto now = [] { return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count(); };

    // Grow to done, timing the CPU cost of advance() (sim step + geometry rebuild
    // + buffer uploads) — this is what runs every frame while roots are growing.
    double advAccum = 0.0; int advN = 0;
    for (int i = 0; i < 4000 && !roots.simDone(); ++i) {
        double a0 = now(); roots.advance(1.0 / 60.0); advAccum += now() - a0; advN++;
    }
    if (advN > 0)
        printf("rootbench: advance() CPU during growth: %.3f ms/frame (%d frames)\n",
               advAccum / advN * 1e3, advN);

    @autoreleasepool {   // warmup
        id<MTLCommandBuffer> cb = [ctx.queue() commandBuffer];
        roots.render(cb); [cb commit]; [cb waitUntilCompleted];
    }
    double t0 = now();
    for (int f = 0; f < frames; ++f) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [ctx.queue() commandBuffer];
            roots.render(cb); [cb commit]; [cb waitUntilCompleted];
        }
    }
    double dt = (now() - t0) / frames;
    printf("rootbench: %dx%d (ds=%d, sim done=%d): %.3f ms/frame (%.0f fps)\n",
           W, H, downscale, roots.simDone() ? 1 : 0, dt * 1e3, 1.0 / dt);
    return 0;
}

// Write an RGBA16F texture to a gamma-corrected PPM.
static void writePPM(const char* path, id<MTLTexture> tex, int W, int H) {
    std::vector<uint16_t> px((size_t)W * H * 4);
    [tex getBytes:px.data() bytesPerRow:W * 4 * sizeof(uint16_t)
       fromRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0];
    auto h2f = [](uint16_t h) {
        uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, bits;
        if (e == 0) bits = (s << 31) | 0; else bits = (s << 31) | ((e + 112) << 23) | (m << 13);
        float f; __builtin_memcpy(&f, &bits, 4); return f;
    };
    FILE* fp = fopen(path, "wb");
    if (!fp) return;
    fprintf(fp, "P6\n%d %d\n255\n", W, H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const uint16_t* p = &px[((size_t)y * W + x) * 4];
            for (int c = 0; c < 3; ++c) {
                float v = h2f(p[c]); v = v <= 0.f ? 0.f : (v >= 1.f ? 1.f : v);
                fputc((unsigned char)(powf(v, 1.0f / 2.2f) * 255.0f + 0.5f), fp);
            }
        }
    fclose(fp);
}

// Build a field of cached root systems and render it. Usage:
//   --fieldshot <out.ppm> [grid] [az] [el]
static int fieldshot(const char* path, int grid, float az, float el) {
    MetalContext ctx;
    if (!ctx.device()) return 1;
    const int W = 1280, H = 720;
    RootScene roots(ctx, W, H);
    if (!roots.valid()) { fprintf(stderr, "fieldshot: invalid\n"); return 1; }
    const float spacing = 30.f;
    roots.buildField(grid, spacing);
    roots.autoOrbit = false;
    roots.azimuth = az; roots.elevation = el;
    roots.target[0] = 0; roots.target[1] = 10; roots.target[2] = 0;
    roots.radius = grid * spacing * 0.85f;
    id<MTLTexture> tex = nil;
    for (int i = 0; i < 2; ++i) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [ctx.queue() commandBuffer];
            tex = roots.render(cb); [cb commit]; [cb waitUntilCompleted];
        }
    }
    writePPM(path, tex, W, H);
    MetalRootRenderer& R = roots.renderer();
    printf("fieldshot: wrote %s  instances=%d visible=%d culled=%d drawnSegs=%ld\n",
           path, R.instanceCount(), R.lastVisibleInstances, R.lastCulledInstances,
           R.lastDrawnSegments);
    return 0;
}

// Benchmark a field with/without culling+LOD. Usage: --fieldbench [grid] [frames]

// Load an image as h*w*3 floats in [0,1] for fitting. NSImage handles whatever
// the user drops in (png/jpeg/heic/tiff), and the explicit bitmap context
// normalises colour space and alpha so the fit target does not silently depend
// on the file's encoding.
// Fit loop settings, shared between the UI and the frame loop.
//
// Two sets, because fitting a face crop and fitting a whole frame are not the
// same problem. A crop is a few percent of the pixels, so a step costs a few
// percent as much and many more of them fit in a frame -- and with that much
// less data behind each gradient, a smaller step keeps it from chasing the
// landmark box's own jitter. The whole feed is the opposite: one expensive step
// per frame, over enough pixels to average out. Sharing one pair of numbers
// meant every crop/no-crop transition silently changed what the numbers meant.
// The grid divisor belongs to the pair for the same reason: a crop of the frame
// at divisor 2 is a few thousand pixels, so it can afford a finer grid than the
// whole feed ever could, and a face is where the detail has to go.
struct FitTune {
    int   steps     = 1;
    float lr        = 3e-3f;
    int   downscale = 2;        // fit grid = display size / this
};
FitTune g_tune_crop{4, 2e-3f, 1};
FitTune g_tune_full{1, 3e-3f, 3};
bool  g_fit_live = false;       // retarget from the camera every frame
#if MIRROR_HAVE_KINECT
mirror::KinectFitTarget g_kinect;
#endif

// --- face tracking ----------------------------------------------------------
//
// One tracker feeds two consumers, which is why it lives here rather than
// inside either scene:
//
//   the mirror  the landmark hull becomes the training mask, so the network
//               fits the person and leaves the background generative.
//   the roots   the landmarks + blendshapes drive a morphable-model fit, and
//               the fitted mesh replaces the static canonical face on the root
//               scene's masks.
//
// Tracking runs at its own resolution, well above the fit grid: the fit is a
// couple of hundred pixels wide, where a face is too few pixels for the
// landmarks to be worth anything. MediaPipe crops to the detected face ROI, so
// the cost is roughly resolution-independent (measured 4.5 ms/frame).
// MIDI stays open across the session; the registry holds the bindings.
midi::Input g_midi;
std::string g_midi_err;

mirror::FaceTracker g_tracker;
mirror::FaceResult  g_face;
mirror::FaceFitter  g_fitter;
bool  g_track_on     = false;   // run the tracker at all
bool  g_mask_fit     = true;    // crop the live fit to the face when there is one
bool  g_drive_roots  = true;    // fitted mesh -> the root scene's face masks
int   g_track_w      = 480;     // tracker input size
int   g_track_h      = 360;
int   g_mask_dilate  = 6;       // px, at fit-grid scale
// What the crop is: the landmarks' bounding box, or the silhouette they trace.
// The box is the default because it is what "fit the face" usually means in
// practice -- the hull supervises skin only and never the boundary between a
// person and the room, so the network has no reason to draw an edge there.
enum class MaskShape { Box = 0, Hull = 1 };
int   g_mask_shape   = (int)MaskShape::Box;
float g_crop_pad     = 0.30f;   // box padding, as a fraction of the box's size
bool  g_collect_id   = false;   // gathering identity samples
double g_last_id_sample = 0.0;
double g_id_started = 0.0;
float g_id_collect_secs = 5.0f;
float g_id_residual  = -1.f;
std::string g_track_err;
std::vector<unsigned char> g_track_rgb;
std::vector<unsigned char> g_fit_mask;
int64_t g_track_ts = 0;         // must increase monotonically for video mode

// The neural texture the mask wears: per-vertex RGB sampled from the mirror's
// own output at the fitted mesh's projected positions. Lives here rather than
// in either scene because it is produced by one and consumed by the other.
std::vector<float> g_face_colors;
bool  g_texture_mask = true;
bool  g_face_colors_fresh = false;

// --- the frame source -------------------------------------------------------
//
// Tracking and fitting want the same picture at different sizes and depths
// (RGB8 for MediaPipe, floats for the fit), so both go through here rather than
// reaching for the sensor themselves.
//
// A still photo can stand in for the camera. That is not only a convenience for
// working without a person in front of the Kinect: it makes the whole pipeline
// reproducible, which is what lets a known face be fitted and compared run to
// run. The photo is simply substituted into the stream -- nothing downstream
// knows the difference.
enum class Source { Kinect = 0, Photo = 1 };
int g_source = (int)Source::Kinect;
std::vector<unsigned char> g_photo;      // full-res RGB8
int  g_photo_w = 0, g_photo_h = 0;
char g_photo_path[512] =
    "/Users/erichan/Documents/Development/neuromirror/emotion/_track_frames/f0016.png";

// A corner picture-in-picture of whatever the source is currently handing out.
// Worth having because every symptom of "the fit is not following me" looks the
// same on the mirror itself -- a closed sensor, a stale snapshot, the photo
// still selected, a mirrored image -- and they are all immediately obvious on
// the raw frame.
bool  g_show_source   = false;
int   g_source_pip_w  = 320;    // overlay width in points
int   g_source_corner = 1;      // 0 TL, 1 TR, 2 BL, 3 BR
bool  g_pip_landmarks = true;   // draw the tracker's landmarks over it

static bool LoadImageRGB(const char* path, int w, int h,
                         std::vector<float>& out, std::string& err);

// Load a still as the stand-in camera frame. Decoded once at a working size --
// large enough that MediaPipe has a real face to land landmarks on, which the
// fit grid alone would not provide.
static bool LoadPhotoSource(const char* path, std::string& err) {
    const int W = 640, H = 480;
    std::vector<float> f;
    if (!LoadImageRGB(path, W, H, f, err)) return false;
    g_photo.resize(f.size());
    for (size_t i = 0; i < f.size(); ++i)
        g_photo[i] = (unsigned char)std::min(255.f, std::max(0.f, f[i] * 255.f + 0.5f));
    g_photo_w = W; g_photo_h = H;
    return true;
}

// --- the camera mask --------------------------------------------------------
//
// A rectangle of the sensor's view that is kept; everything outside it goes
// black. The mirror only sells if the frame contains the person and nothing
// that says "room" -- a doorway, a window, the edge of the rig.
//
// Applied in *camera* space, before any head-mode placement, because that is
// the space it is authored in: the rectangle covers a fixed part of the room,
// and a mask that moved with the subject would be a vignette rather than a
// piece of set dressing. It is applied to the tracker's frame as well as the
// fit's, so nothing outside it can be detected either.
bool  g_cam_mask_on = false;
float g_cam_x0 = 0.15f, g_cam_y0 = 0.05f, g_cam_x1 = 0.85f, g_cam_y1 = 0.95f;
float g_cam_feather = 0.03f;   // soft edge, as a fraction of the frame

// Coverage at a normalised point: 1 inside, 0 outside, smooth across the edge.
static float CamMaskAt(float u, float v) {
    if (!g_cam_mask_on) return 1.f;
    const float f = std::max(g_cam_feather, 1e-4f);
    // Distance inside the rectangle on each axis, in normalised units; the
    // nearest edge wins, which rounds the corners slightly at wide feathers.
    const float dx = std::min(u - std::min(g_cam_x0, g_cam_x1),
                              std::max(g_cam_x0, g_cam_x1) - u);
    const float dy = std::min(v - std::min(g_cam_y0, g_cam_y1),
                              std::max(g_cam_y0, g_cam_y1) - v);
    const float d = std::min(dx, dy);
    const float t = std::min(1.f, std::max(0.f, d / f));
    return t * t * (3.f - 2.f * t);
}

static void ApplyCamMaskF(std::vector<float>& rgb, int w, int h) {
    if (!g_cam_mask_on || w <= 0 || h <= 0 || rgb.size() != size_t(w) * h * 3) return;
    for (int y = 0; y < h; ++y) {
        const float v = (float(y) + 0.5f) / float(h);
        for (int x = 0; x < w; ++x) {
            const float m = CamMaskAt((float(x) + 0.5f) / float(w), v);
            if (m >= 1.f) continue;
            float* p = &rgb[(size_t(y) * w + x) * 3];
            p[0] *= m; p[1] *= m; p[2] *= m;
        }
    }
}

static void ApplyCamMask8(std::vector<unsigned char>& rgb, int w, int h) {
    if (!g_cam_mask_on || w <= 0 || h <= 0 || rgb.size() != size_t(w) * h * 3) return;
    for (int y = 0; y < h; ++y) {
        const float v = (float(y) + 0.5f) / float(h);
        for (int x = 0; x < w; ++x) {
            const float m = CamMaskAt((float(x) + 0.5f) / float(w), v);
            if (m >= 1.f) continue;
            unsigned char* p = &rgb[(size_t(y) * w + x) * 3];
            for (int c = 0; c < 3; ++c) p[c] = (unsigned char)(p[c] * m + 0.5f);
        }
    }
}

static bool SourceRGB8(int w, int h, std::vector<unsigned char>& out) {
    if (g_source == (int)Source::Photo) {
        if (g_photo.empty()) return false;
        mirror::DownsampleToRGB8(g_photo.data(), g_photo_w, g_photo_h, 3, 0, 2, w, h, out);
        ApplyCamMask8(out, w, h);
        return true;
    }
#if MIRROR_HAVE_KINECT
    if (!g_kinect.lastFrameRGB8(w, h, out)) return false;
    ApplyCamMask8(out, w, h);
    return true;
#else
    return false;
#endif
}

static bool SourceRGBF(int w, int h, std::vector<float>& out) {
    if (g_source == (int)Source::Photo) {
        if (g_photo.empty()) return false;
        mirror::DownsampleRGB8(g_photo.data(), g_photo_w, g_photo_h, 3, 0, 2, w, h, out);
        ApplyCamMaskF(out, w, h);
        return true;
    }
#if MIRROR_HAVE_KINECT
    if (!g_kinect.poll(w, h, out)) return false;
    ApplyCamMaskF(out, w, h);
    return true;
#else
    return false;
#endif
}

// --- head movement ----------------------------------------------------------
//
// A live fit has to answer a question a still image never poses: the subject
// moves, and the network is a function of position. Three answers, none of them
// strictly better than the others:
//
//   centred     Move the subject to the middle of the frame and fit it there.
//               The network is always shown the same problem, so the weights
//               are as stable as they can be -- at the cost of the mirror no
//               longer showing where in the room anyone is standing.
//   track       Fit the subject where it is. Honest to the camera, and the
//               least stable: a face that walks across the frame is a
//               different function at every step, and the weights spend their
//               capacity re-learning the same face at a new address.
//   stabilised  Fit the subject where it is, but shift the network's *input*
//               coordinates by the head's displacement. The subject stays put
//               in the network's own frame while staying put on screen too --
//               the picture of "track" with the weights of "centred".
//
// The cost sits in different places: centred resamples the image once a frame,
// stabilised rebuilds the fit features once a frame, track does neither.
enum class HeadMode { Centred = 0, Track = 1, Stabilised = 2 };
int   g_head_mode = (int)HeadMode::Track;
float g_head_smooth = 0.25f;    // EMA per frame; 1 = no smoothing

// The tracked head, smoothed, in normalised frame coords. Smoothed because the
// landmark box jitters by a pixel or two on a perfectly still head, and every
// consumer here is something that must not jitter: an input offset that shakes
// makes the fit chase its own coordinate system, and a region edge that shakes
// is visible on screen directly.
bool  g_head_valid = false;
float g_head_cx = 0.5f, g_head_cy = 0.5f;
float g_head_hx = 0.15f, g_head_hy = 0.2f;   // half-extent, padding included

// Tracking is not all-or-nothing: MediaPipe drops a frame on a blink, a turn,
// a hand across the face. Treated as "no face" those gaps are expensive --
// the fit target flips from a crop to the whole frame and back, which resizes
// the trained pixel set, rebuilds the feature gather, and throws the region's
// soft edge on and off. So a detection has to be missing for a while before it
// counts as gone, and present for a moment before it counts as arrived.
float  g_face_hold_secs = 0.6f;   // keep the last box this long after the last hit
int    g_face_acquire   = 2;      // consecutive hits before a face is believed
double g_face_last_seen = -1e9;   // when the tracker last returned a face
int    g_face_streak    = 0;      // consecutive detections so far
bool   g_face_held      = false;  // showing a held box rather than a fresh one

// The soft edge around the fit, and what rides it.
bool  g_region_on   = true;
bool  g_region_hull = true;      // follow the mask's outline, not its bounding box
float g_fade_start  = 0.02f;     // where the fade begins, in coord units
float g_fade_width  = 0.35f;     // and how far it runs
bool  g_z_free      = false;     // animate the latent outside the crop
float g_grey_out    = 0.0f;      // drain colour outside the crop
std::vector<float> g_region_dist;   // scratch: distance field at fit-grid size

static void UpdateHeadBox() {
    if (!g_track_on || !g_face.valid) { g_head_valid = false; return; }
    const float cx = g_face.centre_x, cy = g_face.centre_y;
    const float pf = 1.f + std::max(g_crop_pad, 0.f);
    const float hx = 0.5f * (g_face.max_x - g_face.min_x) * pf;
    const float hy = 0.5f * (g_face.max_y - g_face.min_y) * pf;
    if (!g_head_valid) {   // first detection: snap, do not ease in from nowhere
        g_head_cx = cx; g_head_cy = cy; g_head_hx = hx; g_head_hy = hy;
        g_head_valid = true;
        return;
    }
    const float a = std::min(1.f, std::max(0.01f, g_head_smooth));
    g_head_cx += a * (cx - g_head_cx);
    g_head_cy += a * (cy - g_head_cy);
    g_head_hx += a * (hx - g_head_hx);
    g_head_hy += a * (hy - g_head_hy);
}

// Defined below, next to the placement it describes: the mask and the region
// both need it, and both are built before it.
static void PinTransform(float& scale, float& u, float& v);

// Is there a face to narrow the fit to at all?
static bool HaveCrop() {
    return g_mask_fit && g_track_on && g_face.valid && g_head_valid;
}

// Where the crop lands in the frame, normalised. Everything but the centred
// mode leaves the subject where the camera found it.
static float CropCX() {
    return g_head_mode == (int)HeadMode::Centred ? 0.5f : g_head_cx;
}
static float CropCY() {
    return g_head_mode == (int)HeadMode::Centred ? 0.5f : g_head_cy;
}

// Which pixels of the fit grid the live target supervises.
//
// The rule the app is built around: fit the video feed, and narrow to the face
// only when there actually is one. So an empty mask (false) is not a failure
// state -- it is the normal case of "no face tracked, fit the whole frame", and
// it is what the fit falls back to the moment someone leaves the room.
static bool BuildFitMask(int fw, int fh, std::vector<unsigned char>& mask) {
    if (!HaveCrop()) return false;
    // The same similarity the pixels were placed by -- the mask marks which of
    // them are trained, so it has to land on top of them exactly.
    float s = 1.f, ou = 0.f, ov = 0.f;
    PinTransform(s, ou, ov);
    if (g_mask_shape == (int)MaskShape::Hull) {
        // The hull moves with the pixels: in the centred mode the image was
        // resampled under it, so landmarks left where they were detected would
        // cut a face-shaped hole out of the background.
        std::vector<mirror::FaceLandmark> lm = g_face.landmarks;
        if (s != 1.f || ou != 0.f || ov != 0.f)
            for (mirror::FaceLandmark& L : lm) {
                L.x = L.x * s + ou;
                L.y = L.y * s + ov;
            }
        mirror::RasteriseFaceMask(lm, mirror::FaceOvalIndices(), fw, fh,
                                  g_mask_dilate, mask);
    } else {
        // Built from the *smoothed* box rather than the raw landmarks, so the
        // trained pixel set stops flickering between frames -- a mask whose
        // area changes forces the optimiser's feature gather to rebuild.
        mirror::RasteriseBox(g_head_cx * s + ou, g_head_cy * s + ov,
                             g_head_hx * s, g_head_hy * s,
                             fw, fh, g_mask_dilate, mask);
    }
    return true;
}

// The trained mask, the input shift and the soft region, all derived from the
// head box. Set once per frame, before anything reads them: the fit features
// and the render must agree on the offset, or the network is trained as one
// function and drawn as another.
bool g_have_mask = false;

static void ApplyHeadMode(mirror::PondParams& P, int fw, int fh) {
    P.coord_off_x = P.coord_off_y = 0.f;
    P.region.on = false;
    P.region.use_field = false;
    P.z_free_outside = g_z_free;
    P.grey_outside = g_grey_out;
    g_have_mask = false;
    if (!HaveCrop() || fw <= 0 || fh <= 0) return;

    const float asp = float(fw) / float(fh);
    if (g_head_mode == (int)HeadMode::Stabilised) {
        // Coord space spans (-asp, asp) x (-1, 1) over the frame, so a
        // normalised displacement doubles going in.
        P.coord_off_x = (g_head_cx - 0.5f) * 2.f * asp;
        P.coord_off_y = (g_head_cy - 0.5f) * 2.f;
    }

    // Built here rather than at the point of use so the render's soft edge and
    // the training mask are the same shape by construction -- the edge is
    // supposed to mark where supervision stops, and deriving it separately
    // would let the two drift.
    g_have_mask = BuildFitMask(fw, fh, g_fit_mask);
    if (!g_region_on) return;

    P.region.on = true;
    if (g_region_hull && g_have_mask) {
        // Distance outward from the mask itself, so the fade follows a face
        // outline when the mask is a hull. One coord unit is fh/2 pixels
        // (y spans -1..1 over the grid), and x uses the same scale because the
        // grid's x range is the aspect ratio -- so a single conversion is right
        // for both axes, which is exactly why the fade band comes out the same
        // width in every direction.
        mirror::DistanceOutside(g_fit_mask, fw, fh, g_region_dist);
        const float per_px = 2.f / float(fh);
        P.region_field.resize(g_region_dist.size());
        for (size_t i = 0; i < g_region_dist.size(); ++i)
            P.region_field[i] = g_region_dist[i] * per_px;
        P.region.use_field = true;
        P.region.fw = fw;
        P.region.fh = fh;
        P.region.ax = asp;
    } else {
        float s = 1.f, ou = 0.f, ov = 0.f;
        PinTransform(s, ou, ov);
        P.region.cx = ((g_head_cx * s + ou) - 0.5f) * 2.f * asp;
        P.region.cy = ((g_head_cy * s + ov) - 0.5f) * 2.f;
        P.region.hx = g_head_hx * s * 2.f * asp;
        P.region.hy = g_head_hy * s * 2.f;
    }
    P.region.fade_start = g_fade_start;
    P.region.fade_width = std::max(0.01f, g_fade_width);
}

// How big the subject should be on screen, in the centred mode: the half-height
// the head box is resampled to, as a fraction of the frame. Off by default,
// where the size is simply whatever distance the person is standing at.
bool  g_face_size_on = false;
float g_face_size    = 0.25f;

// The scale the centred mode is currently applying.
static float PlaceScale() {
    if (g_head_mode != (int)HeadMode::Centred || !HaveCrop() || !g_face_size_on)
        return 1.f;
    return std::min(6.f, std::max(0.1f, g_face_size / std::max(g_head_hy, 1e-3f)));
}

// Where the fit drew the face, relative to where the tracker saw it, in
// normalised units: uv_drawn = uv_tracked * scale + (u, v).
//
// This is the pin. The landmarks are in camera space, the rendered face is
// wherever the head mode put it, and anything that reads the render at a
// landmark position -- the mask's texture, a drawn overlay of the mesh -- has
// to cross that gap or it samples the wrong pixels entirely.
static void PinTransform(float& scale, float& u, float& v) {
    scale = 1.f;
    u = v = 0.f;
    // Only the centred mode moves the subject on screen. The input-shift mode
    // moves the network's coordinates and the render undoes it again, so the
    // face lands back where the camera found it.
    if (g_head_mode != (int)HeadMode::Centred || !HaveCrop()) return;
    scale = PlaceScale();
    u = 0.5f - g_head_cx * scale;
    v = 0.5f - g_head_cy * scale;
}

// Place the subject for the centred mode. At scale 1 this is a whole-pixel
// shift, which is what the mode wants -- resampling the face every frame is
// precisely the noise it exists to remove. Asking for a specific size makes
// interpolation unavoidable, so that path costs a bilinear resample and says so.
static void PlaceLiveFrame(std::vector<float>& rgb, int fw, int fh) {
    if (g_head_mode != (int)HeadMode::Centred || !HaveCrop()) return;
    const float s = PlaceScale();
    if (s == 1.f) {
        mirror::ShiftRGBF(fw, fh, (int)std::lround((0.5f - g_head_cx) * fw),
                          (int)std::lround((0.5f - g_head_cy) * fh), rgb);
    } else {
        mirror::PlaceRGBF(fw, fh, g_head_cx, g_head_cy, s, rgb);
    }
}

static bool SourceReady() {
    if (g_source == (int)Source::Photo) return !g_photo.empty();
#if MIRROR_HAVE_KINECT
    return g_kinect.isOpen();
#else
    return false;
#endif
}

static bool LoadImageRGB(const char* path, int w, int h,
                         std::vector<float>& out, std::string& err) {
    @autoreleasepool {
        NSString* p = [NSString stringWithUTF8String:path];
        NSImage* img = [[NSImage alloc] initWithContentsOfFile:p];
        if (!img) { err = std::string("could not open ") + path; return false; }
        CGImageRef cg = [img CGImageForProposedRect:nil context:nil hints:nil];
        if (!cg) { err = "could not decode image"; return false; }

        std::vector<uint8_t> rgba(size_t(w) * h * 4, 0);
        CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
        CGContextRef ctx = CGBitmapContextCreate(
            rgba.data(), w, h, 8, size_t(w) * 4, cs,
            kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
        CGColorSpaceRelease(cs);
        if (!ctx) { err = "could not create bitmap context"; return false; }
        // Aspect-preserving, centred. Stretching to fill would deform the face,
        // and the fit would then dutifully recover a squashed identity --
        // FHIBE's crops are square while the working canvas is 4:3, so this is
        // not a hypothetical.
        const double iw = double(CGImageGetWidth(cg)), ih = double(CGImageGetHeight(cg));
        double dw = w, dh = h;
        if (iw > 0 && ih > 0) {
            const double s = std::min(double(w) / iw, double(h) / ih);
            dw = iw * s; dh = ih * s;
        }
        CGContextDrawImage(ctx, CGRectMake((w - dw) * 0.5, (h - dh) * 0.5, dw, dh), cg);
        CGContextRelease(ctx);

        out.resize(size_t(w) * h * 3);
        for (size_t i = 0; i < size_t(w) * h; ++i) {
            for (int c = 0; c < 3; ++c) out[i * 3 + c] = rgba[i * 4 + c] / 255.0f;
        }
        return true;
    }
}

static int fieldbench(int grid, int frames) {
    MetalContext ctx;
    if (!ctx.device()) return 1;
    const int W = 1920, H = 1080;
    RootScene roots(ctx, W, H);
    if (!roots.valid()) { fprintf(stderr, "fieldbench: invalid\n"); return 1; }
    const float spacing = 30.f;
    roots.buildField(grid, spacing);
    roots.autoOrbit = false;
    // Immersive viewpoint: camera low and near the field edge looking across it,
    // so a good share of systems fall off-screen (culling) and the rest recede
    // into the distance (LOD) — the target end-goal viewing condition.
    roots.target[0] = grid * spacing * 0.15f; roots.target[1] = 8; roots.target[2] = 0;
    roots.radius = spacing * 1.6f;
    roots.elevation = 0.12f; roots.azimuth = 0.9f;
    MetalRootRenderer& R = roots.renderer();

    auto now = [] { return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count(); };
    auto timeIt = [&](const char* label) {
        @autoreleasepool { id<MTLCommandBuffer> cb = [ctx.queue() commandBuffer];
            roots.render(cb); [cb commit]; [cb waitUntilCompleted]; }   // warmup
        double t0 = now();
        for (int f = 0; f < frames; ++f) @autoreleasepool {
            id<MTLCommandBuffer> cb = [ctx.queue() commandBuffer];
            roots.render(cb); [cb commit]; [cb waitUntilCompleted];
        }
        double dt = (now() - t0) / frames;
        printf("fieldbench %-18s %.3f ms/frame (%.0f fps)  visible=%d culled=%d drawnSegs=%ld\n",
               label, dt*1e3, 1.0/dt, R.lastVisibleInstances, R.lastCulledInstances, R.lastDrawnSegments);
    };
    printf("fieldbench: %dx%d field = %d instances @ %dx%d\n", grid, grid, R.instanceCount(), W, H);
    R.cullInstances = false; R.subpixelCull = false; R.lodBias = 0.0001f;  timeIt("naive(all,full)");
    R.cullInstances = false; R.subpixelCull = false; R.lodBias = 1.0f;     timeIt("LOD-only");
    R.cullInstances = true;  R.subpixelCull = true;  R.lodBias = 1.0f;     timeIt("cull+subpx+LOD");
    return 0;
}

// Headless end-to-end face path on a still image: MediaPipe -> landmarks ->
// training mask, and -> morphable fit -> mesh. face_fit_test covers the solver
// against synthetic ground truth, but everything upstream of it -- the tracker,
// the MP68 mapping, the blendshape name matching, the y-flip -- only runs when
// a real detector looks at a real face. Usage: --facetest [image.png]
static int facetest(const char* path) {
    if (!mirror::FaceTracker::available()) {
        printf("facetest: MediaPipe not compiled in -- run ./setup-mediapipe.sh\n");
        return 0;
    }
    const int W = 640, H = 480;
    std::vector<float> rgbf;
    std::string err;
    if (!LoadImageRGB(path, W, H, rgbf, err)) {
        fprintf(stderr, "facetest: %s: %s\n", path, err.c_str());
        return 1;
    }
    std::vector<unsigned char> rgb(rgbf.size());
    for (size_t i = 0; i < rgbf.size(); ++i)
        rgb[i] = (unsigned char)std::min(255.f, std::max(0.f, rgbf[i] * 255.f + 0.5f));

    mirror::FaceTracker tracker;
    if (!tracker.open(std::string(MIRROR_APP_EXTERNAL_DIR) + "/face_landmarker.task", err)) {
        fprintf(stderr, "facetest: %s\n", err.c_str());
        return 1;
    }
    mirror::FaceResult face;
    if (!tracker.detect(rgb.data(), W, H, 1000, face)) {
        fprintf(stderr, "facetest: no face detected in %s\n", path);
        return 1;
    }
    printf("facetest: %s  %dx%d\n", path, W, H);
    printf("  landmarks %zu  blendshapes %zu  named %zu\n",
           face.landmarks.size(), face.blendshapes.size(),
           face.blendshape_names.size());
    printf("  bounds x %.3f..%.3f  y %.3f..%.3f\n",
           face.min_x, face.max_x, face.min_y, face.max_y);

    // --- the mirror's consumer: the training mask -------------------------
    std::vector<unsigned char> mask;
    mirror::RasteriseFaceMask(face.landmarks, mirror::FaceOvalIndices(),
                              W, H, 6, mask);
    size_t on = 0;
    for (unsigned char m : mask) on += m ? 1 : 0;
    printf("  training mask: %.2f%% of frame (%zu px)\n",
           100.0 * double(on) / double(mask.size()), on);
    if (on == 0) { fprintf(stderr, "facetest: empty mask\n"); return 1; }

    // --- the roots' consumer: the fitted mesh -----------------------------
    mirror::FaceFitter fitter;
    if (!fitter.load(std::string(MIRROR_APP_EXTERNAL_DIR) + "/face_basis.bin", err)) {
        printf("  %s\n  (skipping the fit)\n", err.c_str());
        return 0;
    }
    printf("  frontality %.3f  neutrality %.3f\n",
           mirror::FaceFitter::Frontality(face.landmarks),
           mirror::FaceFitter::Neutrality(face.blendshapes, face.blendshape_names));

    fitter.offerIdentityFrame(face, W, H);
    float residual = -1.f;
    if (fitter.identityFrames() > 0 && fitter.fitIdentity(&residual))
        printf("  identity fitted from 1 frame, residual %.3f px\n", residual);
    else
        printf("  frame rejected for identity (not frontal/neutral enough)\n");

    if (!fitter.update(face, W, H)) { fprintf(stderr, "facetest: update failed\n"); return 1; }
    float yaw, pitch, roll;
    fitter.headAngles(yaw, pitch, roll);
    printf("  head pose: yaw %+.1f  pitch %+.1f  roll %+.1f (deg)\n",
           yaw * 57.2958f, pitch * 57.2958f, roll * 57.2958f);

    int nz = 0;
    for (float v : fitter.expression()) if (v > 0.01f) ++nz;
    printf("  expression: %d of %d modes active\n", nz, fitter.basis().expressionModes());

    // The check that matters: does the fitted mesh land on the face? Compare
    // the projected mesh's bounds to the tracker's own landmark bounds. A lost
    // y-flip or a bad pose puts it somewhere else entirely.
    std::vector<float> vpx;
    fitter.projectVertices(vpx);
    float x0 = vpx[0], x1 = vpx[0], y0 = vpx[1], y1 = vpx[1];
    for (size_t i = 0; i < vpx.size() / 2; ++i) {
        x0 = std::min(x0, vpx[i * 2]);     x1 = std::max(x1, vpx[i * 2]);
        y0 = std::min(y0, vpx[i * 2 + 1]); y1 = std::max(y1, vpx[i * 2 + 1]);
    }
    printf("  projected mesh px: x %.0f..%.0f  y %.0f..%.0f\n", x0, x1, y0, y1);
    printf("  tracker face  px: x %.0f..%.0f  y %.0f..%.0f\n",
           face.min_x * W, face.max_x * W, face.min_y * H, face.max_y * H);

    // Centres should agree to well within a face width; the mesh is a mask, so
    // its extent is legitimately smaller than the full landmark set's.
    const float mcx = 0.5f * (x0 + x1), mcy = 0.5f * (y0 + y1);
    const float fcx = face.centre_x * W, fcy = face.centre_y * H;
    const float fw = (face.max_x - face.min_x) * W;
    const float off = std::sqrt((mcx - fcx) * (mcx - fcx) + (mcy - fcy) * (mcy - fcy));
    printf("  mesh centre is %.1f px from the face centre (%.0f%% of face width)\n",
           off, 100.0 * off / fw);
    if (off > 0.35f * fw) {
        fprintf(stderr, "facetest: FAIL -- fitted mesh is not on the face\n");
        return 1;
    }

    // --- and into the root scene ------------------------------------------
    // The fitted mesh replacing the canonical model is the other half of the
    // wiring, and it is the half that can fail silently: a bad vertex count or
    // a stale triangle list produces an empty face pass rather than an error.
    {
        MetalContext ctx;
        if (!ctx.device()) { fprintf(stderr, "facetest: no Metal device\n"); return 1; }
        RootScene roots(ctx, 640, 360);
        if (!roots.valid()) { fprintf(stderr, "facetest: root scene invalid\n"); return 1; }
        roots.setFittedFace(fitter.vertices(), fitter.basis().triangles());
        if (!roots.usingFittedFace()) {
            fprintf(stderr, "facetest: FAIL -- root scene rejected the fitted mesh\n");
            return 1;
        }
        // Animate it: a second mesh from a different expression must actually
        // reach the renderer, not just the first one.
        for (int i = 0; i < 3; ++i) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [ctx.queue() commandBuffer];
                roots.advance(1.0 / 60.0);
                roots.setFittedFace(fitter.vertices(), std::vector<int>());
                (void)roots.render(cb);
                [cb commit];
                [cb waitUntilCompleted];
            }
        }
        printf("  root scene: fitted face uploaded (%d verts, %d tris) and rendered\n",
               fitter.basis().vertexCount(), fitter.basis().triangleCount());
        roots.clearFittedFace();
        if (roots.usingFittedFace()) {
            fprintf(stderr, "facetest: FAIL -- clearFittedFace did not revert\n");
            return 1;
        }
    }
    printf("facetest: OK\n");
    return 0;
}

// --- demo renders -----------------------------------------------------------

static void writePPM_rgb(const std::string& path, const std::vector<float>& rgb,
                         int w, int h) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    std::vector<unsigned char> row(size_t(w) * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w * 3; ++x) {
            const float v = rgb[size_t(y) * w * 3 + x];
            row[x] = (unsigned char)std::min(255.f, std::max(0.f, v * 255.f + 0.5f));
        }
        fwrite(row.data(), 1, row.size(), f);
    }
    fclose(f);
}

// Rasterise the fitted mesh over `dst` with per-vertex colour, back-face culled
// and depth-sorted (painter's). A z-buffer would be more correct, but the mesh
// is a convex-ish mask with no self-occlusion once back faces are gone, and
// sorting 4048 triangles is simpler than carrying depth through the raster.
static void rasteriseMesh(std::vector<float>& dst, int W, int H,
                          const std::vector<float>& px,      // projected xy
                          const std::vector<float>& verts,   // model xyz (for depth)
                          const std::vector<int>& tris,
                          const std::vector<float>& vcol,
                          float alpha) {
    struct Tri { int i; float z; };
    std::vector<Tri> order;
    order.reserve(tris.size() / 3);
    for (size_t t = 0; t + 2 < tris.size(); t += 3) {
        const int a = tris[t], b = tris[t + 1], c = tris[t + 2];
        // Back-face cull by 2D winding in image (y-down) space.
        const float e1x = px[b * 2] - px[a * 2], e1y = px[b * 2 + 1] - px[a * 2 + 1];
        const float e2x = px[c * 2] - px[a * 2], e2y = px[c * 2 + 1] - px[a * 2 + 1];
        if (e1x * e2y - e1y * e2x >= 0.f) continue;
        const float z = (verts[a * 3 + 2] + verts[b * 3 + 2] + verts[c * 3 + 2]) / 3.f;
        order.push_back({int(t), z});
    }
    std::sort(order.begin(), order.end(),
              [](const Tri& a, const Tri& b) { return a.z < b.z; });

    for (const Tri& tr : order) {
        const int a = tris[tr.i], b = tris[tr.i + 1], c = tris[tr.i + 2];
        const float ax = px[a * 2], ay = px[a * 2 + 1];
        const float bx = px[b * 2], by = px[b * 2 + 1];
        const float cx = px[c * 2], cy = px[c * 2 + 1];
        int x0 = int(std::floor(std::min({ax, bx, cx})));
        int x1 = int(std::ceil (std::max({ax, bx, cx})));
        int y0 = int(std::floor(std::min({ay, by, cy})));
        int y1 = int(std::ceil (std::max({ay, by, cy})));
        x0 = std::max(0, x0); y0 = std::max(0, y0);
        x1 = std::min(W - 1, x1); y1 = std::min(H - 1, y1);
        const float den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
        if (std::fabs(den) < 1e-9f) continue;
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const float fx = float(x) + 0.5f, fy = float(y) + 0.5f;
                float l0 = ((by - cy) * (fx - cx) + (cx - bx) * (fy - cy)) / den;
                float l1 = ((cy - ay) * (fx - cx) + (ax - cx) * (fy - cy)) / den;
                float l2 = 1.f - l0 - l1;
                if (l0 < 0 || l1 < 0 || l2 < 0) continue;
                float* o = &dst[(size_t(y) * W + x) * 3];
                for (int k = 0; k < 3; ++k) {
                    const float col = vcol.empty()
                        ? 0.8f
                        : l0 * vcol[a * 3 + k] + l1 * vcol[b * 3 + k] + l2 * vcol[c * 3 + k];
                    o[k] = o[k] * (1.f - alpha) + col * alpha;
                }
            }
        }
    }
}

// Wireframe over an image: front-facing edges only, so the mesh reads as a
// surface on the face rather than a ball of lines.
static void drawMeshWire(std::vector<float>& dst, int W, int H,
                         const std::vector<float>& px, const std::vector<int>& tris,
                         const float col[3], float a) {
    auto line = [&](float x0, float y0, float x1, float y1) {
        const int steps = int(std::max(std::fabs(x1 - x0), std::fabs(y1 - y0))) + 1;
        for (int s = 0; s <= steps; ++s) {
            const float t = float(s) / float(steps);
            const int x = int(x0 + (x1 - x0) * t + 0.5f);
            const int y = int(y0 + (y1 - y0) * t + 0.5f);
            if (x < 0 || y < 0 || x >= W || y >= H) continue;
            float* o = &dst[(size_t(y) * W + x) * 3];
            for (int k = 0; k < 3; ++k) o[k] = o[k] * (1.f - a) + col[k] * a;
        }
    };
    for (size_t t = 0; t + 2 < tris.size(); t += 3) {
        const int i0 = tris[t], i1 = tris[t + 1], i2 = tris[t + 2];
        const float e1x = px[i1 * 2] - px[i0 * 2], e1y = px[i1 * 2 + 1] - px[i0 * 2 + 1];
        const float e2x = px[i2 * 2] - px[i0 * 2], e2y = px[i2 * 2 + 1] - px[i0 * 2 + 1];
        if (e1x * e2y - e1y * e2x >= 0.f) continue;   // back-facing
        line(px[i0 * 2], px[i0 * 2 + 1], px[i1 * 2], px[i1 * 2 + 1]);
        line(px[i1 * 2], px[i1 * 2 + 1], px[i2 * 2], px[i2 * 2 + 1]);
        line(px[i2 * 2], px[i2 * 2 + 1], px[i0 * 2], px[i0 * 2 + 1]);
    }
}

// --maskshot <photo> <out.ppm>
//
// The whole face path on one still, as four panels: the source, the fitted
// Maxine/NVF mask over it, what the neural mirror made of that face, and the
// mask wearing that reconstruction. This is the picture that shows the fit and
// the texturing are actually doing what they claim.
static int maskshot(const char* photo, const char* out, int fit_steps, int W, int H) {
    if (!mirror::FaceTracker::available()) {
        fprintf(stderr, "maskshot: MediaPipe not compiled in\n"); return 1;
    }
    std::string err;
    std::vector<float> src;
    if (!LoadImageRGB(photo, W, H, src, err)) {
        fprintf(stderr, "maskshot: %s: %s\n", photo, err.c_str()); return 1;
    }
    std::vector<unsigned char> rgb8(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        rgb8[i] = (unsigned char)std::min(255.f, std::max(0.f, src[i] * 255.f + 0.5f));

    mirror::FaceTracker tracker;
    if (!tracker.open(std::string(MIRROR_APP_EXTERNAL_DIR) + "/face_landmarker.task", err)) {
        fprintf(stderr, "maskshot: %s\n", err.c_str()); return 1;
    }
    mirror::FaceResult face;
    if (!tracker.detect(rgb8.data(), W, H, 1000, face)) {
        fprintf(stderr, "maskshot: no face in %s\n", photo); return 1;
    }
    mirror::FaceFitter fitter;
    if (!fitter.load(std::string(MIRROR_APP_EXTERNAL_DIR) + "/face_basis.bin", err)) {
        fprintf(stderr, "maskshot: %s\n", err.c_str()); return 1;
    }
    fitter.config().min_frontality = 0.f;
    fitter.offerIdentityFrame(face, W, H);
    float residual = -1.f;
    fitter.fitIdentity(&residual);
    fitter.update(face, W, H);
    printf("maskshot: identity residual %.2f px\n", residual);

    // --- the neural mirror, fitted to the face ---------------------------
    MetalContext ctx;
    if (!ctx.device()) { fprintf(stderr, "maskshot: no Metal device\n"); return 1; }
    MirrorScene mirror(ctx, 11, W / 2, H / 2);
    if (!mirror.valid()) { fprintf(stderr, "maskshot: mirror invalid\n"); return 1; }

    const int fw = W / 2, fh = H / 2;
    std::vector<float> target;
    mirror::DownsampleRGB8(rgb8.data(), W, H, 3, 0, 2, fw, fh, target);
    std::vector<unsigned char> mask;
    mirror::RasteriseFaceMask(face.landmarks, mirror::FaceOvalIndices(), fw, fh, 6, mask);
    mirror.pond().beginFit(target, fh, fw, mirror.params(), mask);
    for (int i = 0; i < fit_steps; ++i) mirror.fitSteps(1, 3e-3f);
    mirror.render();
    printf("maskshot: %d fit steps, loss %.5f\n", fit_steps, mirror.lastLoss());

    std::vector<float> colors;
    fitter.sampleTexture(mirror.lastImageRGB(), mirror.lowW(), mirror.lowH(),
                         W, H, colors);

    // --- compose ----------------------------------------------------------
    std::vector<float> px;
    fitter.projectVertices(px);
    const std::vector<int>& tris = fitter.basis().triangles();

    std::vector<float> p1 = src;
    // The fitted mask as a wireframe: what is being shown here is that the
    // *geometry* landed on the face, so the topology has to be visible. A
    // filled mask hides exactly the thing the panel exists to demonstrate.
    std::vector<float> p2 = src;
    const float wire[3] = {0.25f, 1.0f, 0.55f};
    drawMeshWire(p2, W, H, px, tris, wire, 0.55f);

    // The mirror's own output, upscaled to panel size.
    std::vector<float> p3(size_t(W) * H * 3);
    {
        const std::vector<float>& m = mirror.lastImageRGB();
        const int mw = mirror.lowW(), mh = mirror.lowH();
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const int sx = std::min(mw - 1, x * mw / W), sy = std::min(mh - 1, y * mh / H);
                for (int k = 0; k < 3; ++k)
                    p3[(size_t(y) * W + x) * 3 + k] = m[(size_t(sy) * mw + sx) * 3 + k];
            }
    }
    std::vector<float> p4(size_t(W) * H * 3, 0.06f);
    rasteriseMesh(p4, W, H, px, fitter.vertices(), tris, colors, 1.0f);

    const int PW = W * 2, PH = H * 2;
    std::vector<float> sheet(size_t(PW) * PH * 3, 0.f);
    auto blit = [&](const std::vector<float>& p, int ox, int oy) {
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W * 3; ++x)
                sheet[(size_t(y + oy) * PW) * 3 + size_t(ox) * 3 + x] =
                    p[size_t(y) * W * 3 + x];
    };
    blit(p1, 0, 0);   blit(p2, W, 0);
    blit(p3, 0, H);   blit(p4, W, H);

    writePPM_rgb(out, sheet, PW, PH);
    printf("maskshot: wrote %s (%dx%d)\n"
           "  [source | fitted NVF mask]  [neural mirror fit | mask textured from it]\n",
           out, PW, PH);
    return 0;
}

// --mirrorclip <out_prefix> <secs> [photo]
//
// A clip of the neural mirror converging onto a face: frames written as PPMs
// for an external encoder. The interesting thing about the mirror is temporal
// -- the fit never finishes, it tracks -- so a still cannot show it.
static int mirrorclip(const char* prefix, float secs, const char* photo, int fps,
                      int steps_per_frame) {
    const int W = 960, H = 540;
    MetalContext ctx;
    if (!ctx.device()) { fprintf(stderr, "mirrorclip: no Metal device\n"); return 1; }
    MirrorScene mirror(ctx, 11, W, H);
    if (!mirror.valid()) { fprintf(stderr, "mirrorclip: mirror invalid\n"); return 1; }

    if (photo && *photo) {
        std::string err;
        std::vector<float> src;
        if (!LoadImageRGB(photo, W, H, src, err)) {
            fprintf(stderr, "mirrorclip: %s: %s\n", photo, err.c_str()); return 1;
        }
        std::vector<unsigned char> rgb8(src.size());
        for (size_t i = 0; i < src.size(); ++i)
            rgb8[i] = (unsigned char)std::min(255.f, std::max(0.f, src[i] * 255.f + 0.5f));

        const int fw = W / 2, fh = H / 2;
        std::vector<float> target;
        mirror::DownsampleRGB8(rgb8.data(), W, H, 3, 0, 2, fw, fh, target);

        std::vector<unsigned char> mask;
        if (mirror::FaceTracker::available()) {
            mirror::FaceTracker tracker;
            std::string terr;
            mirror::FaceResult face;
            if (tracker.open(std::string(MIRROR_APP_EXTERNAL_DIR) + "/face_landmarker.task",
                             terr) &&
                tracker.detect(rgb8.data(), W, H, 1000, face)) {
                mirror::RasteriseFaceMask(face.landmarks, mirror::FaceOvalIndices(),
                                          fw, fh, 8, mask);
                printf("mirrorclip: masked to the face (%zu of %d px)\n",
                       std::count_if(mask.begin(), mask.end(),
                                     [](unsigned char c) { return c != 0; }),
                       fw * fh);
            }
        }
        mirror.pond().beginFit(target, fh, fw, mirror.params(), mask);
    }

    const int frames = std::max(1, int(secs * float(fps)));
    const double dt = 1.0 / double(fps);
    for (int i = 0; i < frames; ++i) {
        @autoreleasepool {
            mirror.advance(dt);
            if (mirror.pond().fitting()) mirror.fitSteps(steps_per_frame, 3e-3f);
            mirror.render();
            char path[512];
            snprintf(path, sizeof(path), "%s%04d.ppm", prefix, i);
            writePPM_rgb(path, mirror.lastImageRGB(), mirror.lowW(), mirror.lowH());
        }
    }
    printf("mirrorclip: wrote %d frames %s0000.ppm.. (%dx%d, %d fps, loss %.5f)\n",
           frames, prefix, W, H, fps, mirror.lastLoss());
    return 0;
}

// --transhot <prefix> <frames> [photo]
//
// The full 4-phase transition, offscreen. Mirrors cloth_cpp's `--shots`, but
// against live assets: the pond is a real MirrorScene fitted to the photo, and
// the face is the real fitted NVF mesh rather than a baked heightmap.
static int transhot(const char* prefix, int frames, const char* photo, float fps) {
    MetalContext ctx;
    if (!ctx.device()) { fprintf(stderr, "transhot: no Metal device\n"); return 1; }

    const int W = 960, H = 960;   // square: the front-on frustum is square
    MirrorScene mirror(ctx, 11, W / 2, H / 2);
    if (!mirror.valid()) { fprintf(stderr, "transhot: mirror invalid\n"); return 1; }
    TransitionScene trans(ctx, W, H);
    FitViewScene fitview(ctx, std::string(MIRROR_APP_SHADER_DIR) + "/fit_view.metal", W, H);
    if (!trans.valid()) { fprintf(stderr, "transhot: transition invalid (shader?)\n"); return 1; }

    // Fit the mirror to the photo and fit the face mesh to it, so both assets
    // are the live ones. This is the whole point of the port: cloth_cpp could
    // only ever run this against two frozen files.
    if (photo && *photo) {
        std::string err;
        std::vector<float> src;
        if (!LoadImageRGB(photo, W, H, src, err)) {
            fprintf(stderr, "transhot: %s: %s\n", photo, err.c_str()); return 1;
        }
        std::vector<unsigned char> rgb8(src.size());
        for (size_t i = 0; i < src.size(); ++i)
            rgb8[i] = (unsigned char)std::min(255.f, std::max(0.f, src[i] * 255.f + 0.5f));

        const int fw = W / 2, fh = H / 2;
        std::vector<float> target;
        mirror::DownsampleRGB8(rgb8.data(), W, H, 3, 0, 2, fw, fh, target);

        if (mirror::FaceTracker::available()) {
            mirror::FaceTracker tracker;
            std::string terr;
            mirror::FaceResult face;
            if (tracker.open(std::string(MIRROR_APP_EXTERNAL_DIR) + "/face_landmarker.task", terr) &&
                tracker.detect(rgb8.data(), W, H, 1000, face)) {
                std::vector<unsigned char> mask;
                mirror::RasteriseFaceMask(face.landmarks, mirror::FaceOvalIndices(),
                                          fw, fh, 8, mask);
                mirror.pond().beginFit(target, fh, fw, mirror.params(), mask);

                mirror::FaceFitter fitter;
                if (fitter.load(std::string(MIRROR_APP_EXTERNAL_DIR) + "/face_basis.bin", terr)) {
                    fitter.config().min_frontality = 0.f;
                    fitter.offerIdentityFrame(face, W, H);
                    float res = -1.f;
                    fitter.fitIdentity(&res);
                    fitter.update(face, W, H);
                    trans.setFaceMesh(fitter.vertices(), fitter.basis().triangles());
                    printf("transhot: face fitted (residual %.2f px), %d verts\n",
                           res, fitter.basis().vertexCount());
                } else {
                    printf("transhot: no face basis (%s)\n", terr.c_str());
                }
            }
        }
        // Converge the mirror before the clip starts, so the film is a face and
        // not noise on frame 0.
        for (int i = 0; i < 1200; ++i) mirror.fitSteps(1, 3e-3f);
        printf("transhot: mirror fitted, loss %.5f\n", mirror.lastLoss());
    }

    const double dt = 1.0 / double(fps);
    for (int f = 0; f < frames; ++f) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [ctx.queue() commandBuffer];
            mirror.advance(dt);
            if (mirror.pond().fitting()) mirror.fitSteps(1, 3e-3f);
            id<MTLTexture> pond = mirror.render();
            trans.setPondTexture(pond);
            trans.advance(dt);
            id<MTLTexture> tex = trans.render(cb);
            [cb commit];
            [cb waitUntilCompleted];
            char path[512];
            snprintf(path, sizeof(path), "%s%04d.ppm", prefix, f);
            writePPM(path, tex, W, H);
        }
    }
    printf("transhot: wrote %d frames %s0000.ppm.. (%dx%d @ %.0f fps)\n",
           frames, prefix, W, H, fps);
    return 0;
}

// --textshot <out.ppm> [text] [warp]
//
// The text overlay over a live pond, through the real present pass. Headless
// because the two things most likely to be wrong about this effect -- whether
// present.metal still compiles with the overlay in it, and whether the glyphs
// land where the placement says they do -- are both visible in one still frame
// and neither needs a person at the window.
//
// Ripples are turned on here even though they default off: the warp is the half
// of the effect that has anything to go wrong in it, and with no sources the
// shader's refraction branch never runs.
static int textshot(const char* path, const char* str, float warp,
                    float reveal, float softness) {
    MetalContext ctx;
    if (!ctx.device()) { fprintf(stderr, "textshot: no Metal device\n"); return 1; }

    const int W = 1280, H = 720;
    MirrorScene mirror(ctx, 11, W / 2, H / 2);
    if (!mirror.valid()) { fprintf(stderr, "textshot: mirror invalid\n"); return 1; }
    mirror.params().drops = 5;
    mirror.params().orbit_on = true;
    mirror.params().warp = 0.3f;

    FullscreenPresent present(ctx, std::string(MIRROR_APP_SHADER_DIR) + "/present.metal",
                              MTLPixelFormatRGBA16Float);
    if (!present.valid()) { fprintf(stderr, "textshot: present shader failed\n"); return 1; }

    mirror::TextOverlay text(ctx);
    mirror::TextParams tp;
    tp.on = true;
    if (str && *str) tp.text = str;
    tp.warp = warp;
    tp.reveal = reveal;
    tp.softness = softness;
    text.update(tp);
    if (!text.valid()) { fprintf(stderr, "textshot: no field built\n"); return 1; }
    printf("textshot: field %dx%d for \"%s\"\n", text.fieldW(), text.fieldH(),
           tp.text.c_str());

    MTLTextureDescriptor* d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                     width:W height:H mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModeShared;
    id<MTLTexture> out = [ctx.device() newTextureWithDescriptor:d];

    @autoreleasepool {
        mirror.advance(1.0);          // off zero, so the ripples have a phase
        id<MTLTexture> pond = mirror.render();

        mirror::TextRipple tr;
        const mirror::PondParams& P = mirror.params();
        tr.k = P.ring_freq;
        tr.decay = P.decay;
        tr.core_r2 = P.core_rolloff ? P.core_radius * P.core_radius : 0.f;
        const auto& srcs = mirror.pond().lastSources();
        tr.n = int(std::min(srcs.size(), size_t(16)));
        for (int i = 0; i < tr.n; ++i)
            for (int j = 0; j < 4; ++j) tr.src[i][j] = srcs[i][j];
        printf("textshot: %d ripple sources, warp %.2f, reveal %.2f\n",
               tr.n, warp, reveal);

        MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
        rpd.colorAttachments[0].texture = out;
        rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
        rpd.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);

        id<MTLCommandBuffer> cb = [ctx.queue() commandBuffer];
        id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rpd];
        present.encode(re, pond, text.texture(),
                       text.uniforms(tp, float(W) / float(H), tr, 1.0));
        [re endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        writePPM(path, out, W, H);
    }
    printf("textshot: wrote %s (%dx%d)\n", path, W, H);
    return 0;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--selftest") return selftest();
        if (a == "--roottest") return roottest();
        if (a == "--transhot") {
            const char* prefix = (i + 1 < argc) ? argv[i + 1] : "trans_";
            int n = (i + 2 < argc) ? atoi(argv[i + 2]) : 150;
            const char* photo = (i + 3 < argc) ? argv[i + 3] : "";
            float fps = (i + 4 < argc) ? (float)atof(argv[i + 4]) : 30.f;
            return transhot(prefix, n, photo, fps);
        }
        if (a == "--textshot") {
            const char* out = (i + 1 < argc) ? argv[i + 1] : "text.ppm";
            const char* str = (i + 2 < argc) ? argv[i + 2] : "";
            float warp = (i + 3 < argc) ? (float)atof(argv[i + 3]) : 0.08f;
            float reveal = (i + 4 < argc) ? (float)atof(argv[i + 4]) : 1.f;
            float soft = (i + 5 < argc) ? (float)atof(argv[i + 5]) : 1.f;
            return textshot(out, str, warp, reveal, soft);
        }
        if (a == "--fitviewtest") {
            // The fit view's shaders are loaded from disk at run time, so a
            // clean compile of the app says nothing about whether they work.
            // This builds the scene, feeds it a mask and a mesh, renders, and
            // reads a pixel back -- enough to catch a shader that does not
            // compile, a pipeline that fails to build, and a pass that draws
            // nothing at all.
            MetalContext ctx;
            if (!ctx.device()) { fprintf(stderr, "fitviewtest: no Metal device\n"); return 1; }
            const int W = 320, H = 240;
            FitViewScene fv(ctx, std::string(MIRROR_APP_SHADER_DIR) + "/fit_view.metal", W, H);
            if (!fv.valid()) { fprintf(stderr, "fitviewtest: pipelines failed\n"); return 1; }

            // A background that is not black, so "drew nothing" and "drew the
            // background" are distinguishable in the readback.
            MTLTextureDescriptor* td =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                   width:W height:H mipmapped:NO];
            td.usage = MTLTextureUsageShaderRead;
            td.storageMode = MTLStorageModeShared;
            id<MTLTexture> bg = [ctx.device() newTextureWithDescriptor:td];
            std::vector<uint16_t> px(size_t(W) * H * 4, 0x3400);   // ~0.25 in fp16
            [bg replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0
                    withBytes:px.data() bytesPerRow:size_t(W) * 8];
            fv.setBackground(bg);

            std::vector<unsigned char> mask(size_t(64) * 48, 0);
            for (int y = 12; y < 36; ++y)
                for (int x = 16; x < 48; ++x) mask[size_t(y) * 64 + x] = 1;
            fv.setMask(mask, 64, 48);

            // One big red triangle across the middle of the frame.
            const std::vector<float> uv = {0.3f, 0.3f, 0.7f, 0.3f, 0.5f, 0.7f};
            const std::vector<float> rgb = {1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f};
            const std::vector<int> tris = {0, 1, 2};
            fv.setMesh(uv, rgb, tris);

            id<MTLTexture> out = nil;
            {
                id<MTLCommandBuffer> cb = [ctx.queue() commandBuffer];
                out = fv.render(cb);
                [cb commit];
                [cb waitUntilCompleted];
            }
            if (!out) { fprintf(stderr, "fitviewtest: render returned nil\n"); return 1; }
            writePPM("fitview.ppm", out, W, H);

            // Read back three pixels that must differ from each other: bare
            // background, background under the mask tint, and the mesh. Only
            // checking that *something* rendered would miss the failure this
            // was written for -- a vertex-layout mismatch drops the mesh while
            // leaving every other layer perfect.
            auto probe = [&](int x, int y) {
                uint16_t p[4] = {0, 0, 0, 0};
                [out getBytes:p bytesPerRow:sizeof(p)
                   fromRegion:MTLRegionMake2D(x, y, 1, 1) mipmapLevel:0];
                auto h2f = [](uint16_t h) {
                    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, bits;
                    if (e == 0) bits = (s << 31); else bits = (s << 31) | ((e + 112) << 23) | (m << 13);
                    float f; __builtin_memcpy(&f, &bits, 4); return f;
                };
                return simd_make_float3(h2f(p[0]), h2f(p[1]), h2f(p[2]));
            };
            const simd_float3 bare = probe(5, 5);            // outside everything
            const simd_float3 masked = probe(100, 150);      // mask, below the mesh
            const simd_float3 mesh = probe(160, 120);        // inside the triangle
            printf("fitviewtest: bg %.3f %.3f %.3f | mask %.3f %.3f %.3f | "
                   "mesh %.3f %.3f %.3f\n",
                   bare.x, bare.y, bare.z, masked.x, masked.y, masked.z,
                   mesh.x, mesh.y, mesh.z);

            int fail = 0;
            if (!(bare.x > 0.2f && bare.x < 0.3f)) {
                fprintf(stderr, "fitviewtest: background did not render\n"); ++fail;
            }
            if (!(masked.y > bare.y + 0.05f)) {
                fprintf(stderr, "fitviewtest: mask overlay missing\n"); ++fail;
            }
            if (!(mesh.x > 0.8f && mesh.y < 0.2f && mesh.z < 0.2f)) {
                fprintf(stderr, "fitviewtest: mesh did not draw (expected red)\n"); ++fail;
            }
            printf("fitviewtest: %s\n", fail ? "FAIL" : "OK");
            return fail ? 1 : 0;
        }
        if (a == "--maskshot") {
            const char* photo = (i + 1 < argc) ? argv[i + 1]
                                : NEUROMIRROR_DIR "/emotion/_track_frames/f0016.png";
            const char* out = (i + 2 < argc) ? argv[i + 2] : "maskshot.ppm";
            int steps = (i + 3 < argc) ? atoi(argv[i + 3]) : 800;
            int mw = (i + 4 < argc) ? atoi(argv[i + 4]) : 640;
            int mh = (i + 5 < argc) ? atoi(argv[i + 5]) : 480;
            return maskshot(photo, out, steps, mw, mh);
        }
        if (a == "--mirrorclip") {
            const char* prefix = (i + 1 < argc) ? argv[i + 1] : "mirror_";
            float secs = (i + 2 < argc) ? (float)atof(argv[i + 2]) : 5.f;
            const char* photo = (i + 3 < argc) ? argv[i + 3] : "";
            int fps = (i + 4 < argc) ? atoi(argv[i + 4]) : 30;
            int spf = (i + 5 < argc) ? atoi(argv[i + 5]) : 12;
            return mirrorclip(prefix, secs, photo, fps, spf);
        }
        if (a == "--facetest") {
            return facetest(i + 1 < argc
                                ? argv[i + 1]
                                : NEUROMIRROR_DIR "/emotion/_track_frames/f0016.png");
        }
        if (a == "--fieldshot") {
            const char* path = (i + 1 < argc) ? argv[i + 1] : "field.ppm";
            int grid = (i + 2 < argc) ? atoi(argv[i + 2]) : 6;
            float az = (i + 3 < argc) ? atof(argv[i + 3]) : 0.5f;
            float el = (i + 4 < argc) ? atof(argv[i + 4]) : 0.35f;
            return fieldshot(path, grid, az, el);
        }
        if (a == "--fieldbench") {
            int grid = (i + 1 < argc) ? atoi(argv[i + 1]) : 8;
            int fr   = (i + 2 < argc) ? atoi(argv[i + 2]) : 100;
            return fieldbench(grid, fr);
        }
        if (a == "--rootbench") {
            int ds = (i + 1 < argc) ? atoi(argv[i + 1]) : 1;
            int fr = (i + 2 < argc) ? atoi(argv[i + 2]) : 200;
            int bw = (i + 3 < argc) ? atoi(argv[i + 3]) : 1920;
            int bh = (i + 4 < argc) ? atoi(argv[i + 4]) : 1080;
            return rootbench(ds, fr, bw, bh);
        }
        if (a == "--growshot") {
            const char* path = (i + 1 < argc) ? argv[i + 1] : "grow.ppm";
            int steps = (i + 2 < argc) ? atoi(argv[i + 2]) : 400;
            float az  = (i + 3 < argc) ? atof(argv[i + 3]) : 0.6f;
            float el  = (i + 4 < argc) ? atof(argv[i + 4]) : 0.2f;
            float rad = (i + 5 < argc) ? atof(argv[i + 5]) : -1.f;
            float fs  = (i + 6 < argc) ? atof(argv[i + 6]) : -1.f;
            float ty  = (i + 7 < argc) ? atof(argv[i + 7]) : -1e9f;
            return growshot(path, steps, az, el, rad, fs, ty);
        }
        if (a == "--uitest") {
            // The parameter registry, headless. ImGui runs without a backend
            // as long as it has a display size and a built font atlas, which
            // is enough to exercise declaration, MIDI routing and preset I/O --
            // all three of which fail *silently* when they fail.
            IMGUI_CHECKVERSION();
            ImGui::CreateContext();
            ImGuiIO& io = ImGui::GetIO();
            io.DisplaySize = ImVec2(400, 400);
            io.Fonts->Build();
            unsigned char* tex_px; int tex_w, tex_h;
            io.Fonts->GetTexDataAsRGBA32(&tex_px, &tex_w, &tex_h);
            io.Fonts->SetTexID((ImTextureID)1);

            float amp = 0.5f, gain = 2.0f;
            int   steps = 4;
            bool  flag = false;

            // One frame of the panel, declaring four controls in two sections.
            auto frame = [&]() {
                ImGui::NewFrame();
                ui::BeginFrame();
                ImGui::Begin("t");
                {
                    ui::Section a("mirror");
                    ui::SliderFloat("amp", &amp, 0.f, 1.f);
                    ui::SliderInt("steps", &steps, 0, 10);
                    ui::Checkbox("flag", &flag);
                }
                {
                    ui::Section b("roots");
                    ui::SliderFloat("amp", &gain, 0.f, 4.f);   // same label, other section
                }
                ImGui::End();
                ImGui::Render();
            };
            frame();
            int bad = 0;
            if (ui::DeclaredCount() != 4) {
                printf("  declared %d, want 4\n", ui::DeclaredCount()); ++bad;
            }

            // Bind mirror/amp to cc 7 and sweep it. The two "amp" controls
            // share a label and must not share a binding -- that collision is
            // the whole reason parameters are named by section.
            ui::SetLearnTarget("mirror/amp");
            ui::ApplyCC(0, 7, 64);              // consumed by learn, not applied
            if (ui::BindingCount() != 1) { printf("  learn did not bind\n"); ++bad; }
            if (std::fabs(amp - 0.5f) > 1e-6f) {
                printf("  learn moved the value (%.3f); it should only bind\n", amp); ++bad;
            }
            ui::ApplyCC(0, 7, 127);
            frame();
            if (std::fabs(amp - 1.0f) > 1e-3f) { printf("  cc did not apply: %.3f\n", amp); ++bad; }
            if (std::fabs(gain - 2.0f) > 1e-6f) {
                printf("  cc leaked to the same label in another section\n"); ++bad;
            }
            ui::ApplyCC(0, 7, 0);
            frame();
            if (std::fabs(amp) > 1e-3f) { printf("  cc floor wrong: %.3f\n", amp); ++bad; }
            // A different cc must do nothing.
            ui::ApplyCC(0, 9, 127);
            frame();
            if (std::fabs(amp) > 1e-3f) { printf("  unbound cc moved a value\n"); ++bad; }

            // Preset round-trip, including the binding.
            amp = 0.25f; gain = 3.5f; steps = 9; flag = true;
            frame();
            const std::string path = ui::PresetDir() + "/__uitest.set";
            std::string err;
            if (!ui::SavePreset(path, err)) { printf("  save: %s\n", err.c_str()); ++bad; }
            amp = 0.f; gain = 0.f; steps = 0; flag = false;
            ui::ClearAllBindings();
            frame();
            if (!ui::LoadPreset(path, err)) { printf("  load: %s\n", err.c_str()); ++bad; }
            frame();                            // values land as controls declare
            if (std::fabs(amp - 0.25f) > 1e-4f) { printf("  amp %.4f != 0.25\n", amp); ++bad; }
            if (std::fabs(gain - 3.5f) > 1e-4f) { printf("  gain %.4f != 3.5\n", gain); ++bad; }
            if (steps != 9) { printf("  steps %d != 9\n", steps); ++bad; }
            if (!flag) { printf("  flag did not restore\n"); ++bad; }
            if (ui::BindingCount() != 1) { printf("  binding not restored\n"); ++bad; }
            if (!ui::UnclaimedKeys().empty()) {
                printf("  %zu unclaimed keys after a self-written preset\n",
                       ui::UnclaimedKeys().size());
                ++bad;
            }
            // An unknown key is reported, not fatal, and must not disturb
            // anything real.
            {
                std::ofstream f(path);
                f << "p mirror/amp = 0.75\n";
                f << "p mirror/gone = 1\n";
            }
            gain = 1.25f;
            if (!ui::LoadPreset(path, err)) { printf("  partial load failed\n"); ++bad; }
            frame();
            if (std::fabs(amp - 0.75f) > 1e-4f) { printf("  partial: amp not read\n"); ++bad; }
            if (std::fabs(gain - 1.25f) > 1e-4f) { printf("  partial: absent key clobbered\n"); ++bad; }
            if (ui::UnclaimedKeys().size() != 1) { printf("  unknown key not reported\n"); ++bad; }
            remove(path.c_str());
            ImGui::DestroyContext();
            printf("uitest: %s\n", bad ? "FAIL" : "OK");
            return bad ? 1 : 0;
        }
        if (a == "--presettest") {
            // Round-trip the growth parameters through a file. The failure a
            // preset system has is silent: a field that is written but never
            // read (or the reverse) comes back as its default and nobody
            // notices until a saved look cannot be reproduced.
            MetalContext ctx;
            if (!ctx.device()) { fprintf(stderr, "presettest: no Metal device\n"); return 1; }
            RootScene roots(ctx, 320, 240);
            if (!roots.valid()) { fprintf(stderr, "presettest: invalid\n"); return 1; }

            rootsim::SimParams& SP = roots.simParams();
            // Deliberately non-default values, all distinct, so a field that
            // reads back another field's value is caught too.
            SP.speciesXml = "Glycine_max.xml";
            SP.N = 11;  SP.R0 = 17.5f; SP.Hh = 61.25f;
            SP.startFrac = 0.21f; SP.endFrac = 0.87f; SP.taperPower = 1.37f;
            SP.angleStepGoldenMult = 0.93f; SP.distStepFrac = 0.11f;
            SP.dwellDays = 23.5f; SP.weight = 0.77f; SP.mainTravelTrials = 19.f;
            SP.lateralWeight = 0.31f; SP.dwellWeight = 0.83f;
            SP.dwellLateralWeight = 0.71f; SP.sigma = 0.47f; SP.viewCylLen = 9.5f;
            SP.maxHopDays = 71.f; SP.reachMult = 1.9f; SP.travelPullReach = 1.45f;
            SP.coneSurfaceTravel = true; SP.coneShellThickness = 5.5f;
            SP.growthDt = 0.35f; SP.targetLift = 1.25f; SP.spawnBehind = 0.75f;
            SP.seed = 4242u;
            const rootsim::SimParams want = SP;

            const std::string path = std::string(RootScene::presetDir()) + "/__roundtrip.root";
            if (!roots.saveConfig(path)) {
                fprintf(stderr, "presettest: could not write %s\n", path.c_str());
                return 1;
            }
            SP = rootsim::SimParams{};          // wipe to defaults
            if (!roots.loadConfig(path)) {
                fprintf(stderr, "presettest: could not read back\n"); return 1;
            }
            const rootsim::SimParams& got = roots.simParams();

            int bad = 0;
            auto cf = [&](const char* n, float a, float b) {
                if (std::fabs(a - b) > 1e-4f) { printf("  MISMATCH %-22s %g != %g\n", n, a, b); ++bad; }
            };
            auto ci = [&](const char* n, long a, long b) {
                if (a != b) { printf("  MISMATCH %-22s %ld != %ld\n", n, a, b); ++bad; }
            };
            if (got.speciesXml != want.speciesXml) {
                printf("  MISMATCH %-22s %s != %s\n", "speciesXml",
                       got.speciesXml.c_str(), want.speciesXml.c_str());
                ++bad;
            }
            ci("N", got.N, want.N);
            cf("R0", got.R0, want.R0);                cf("Hh", got.Hh, want.Hh);
            cf("startFrac", got.startFrac, want.startFrac);
            cf("endFrac", got.endFrac, want.endFrac);
            cf("taperPower", got.taperPower, want.taperPower);
            cf("angleStepGoldenMult", got.angleStepGoldenMult, want.angleStepGoldenMult);
            cf("distStepFrac", got.distStepFrac, want.distStepFrac);
            cf("dwellDays", got.dwellDays, want.dwellDays);
            cf("weight", got.weight, want.weight);
            cf("mainTravelTrials", got.mainTravelTrials, want.mainTravelTrials);
            cf("lateralWeight", got.lateralWeight, want.lateralWeight);
            cf("dwellWeight", got.dwellWeight, want.dwellWeight);
            cf("dwellLateralWeight", got.dwellLateralWeight, want.dwellLateralWeight);
            cf("sigma", got.sigma, want.sigma);
            cf("viewCylLen", got.viewCylLen, want.viewCylLen);
            cf("maxHopDays", got.maxHopDays, want.maxHopDays);
            cf("reachMult", got.reachMult, want.reachMult);
            cf("travelPullReach", got.travelPullReach, want.travelPullReach);
            ci("coneSurfaceTravel", got.coneSurfaceTravel, want.coneSurfaceTravel);
            cf("coneShellThickness", got.coneShellThickness, want.coneShellThickness);
            cf("growthDt", got.growthDt, want.growthDt);
            cf("targetLift", got.targetLift, want.targetLift);
            cf("spawnBehind", got.spawnBehind, want.spawnBehind);
            ci("seed", got.seed, want.seed);

            // An unknown key must be skipped and a missing one must keep its
            // default, or a preset written before a parameter existed stops
            // loading the day one is added.
            {
                std::ofstream partial(path);
                partial << "N = 7\nsomethingUnknown = 3\n";
            }
            SP = rootsim::SimParams{};
            SP.sigma = 0.99f;
            if (!roots.loadConfig(path)) { printf("  partial load failed\n"); ++bad; }
            if (roots.simParams().N != 7) { printf("  partial: N not read\n"); ++bad; }
            if (std::fabs(roots.simParams().sigma - 0.99f) > 1e-6f) {
                printf("  partial: absent key clobbered an existing value\n"); ++bad;
            }
            remove(path.c_str());
            printf("presettest: %s\n", bad ? "FAIL" : "OK");
            return bad ? 1 : 0;
        }
        if (a == "--maskframes") {
            // The mask frames as numbers. A face is placed as
            // p + tangent*x + bitangent*y + normal*z, so the bitangent is the
            // direction the top of the head points: if its world Y is negative
            // the face is upside down, and that is a fact about three floats,
            // not something to squint at a render for.
            MetalContext ctx;
            if (!ctx.device()) { fprintf(stderr, "maskframes: no Metal device\n"); return 1; }
            RootScene roots(ctx, 320, 240);
            if (!roots.valid()) { fprintf(stderr, "maskframes: invalid\n"); return 1; }
            for (int k = 0; k < 4000 && !roots.simDone(); ++k) roots.advance(1.0 / 60.0);
            const auto& ms = roots.revealedMasks();
            printf("maskframes: %zu revealed\n", ms.size());
            int bad = 0;
            for (size_t k = 0; k < ms.size(); ++k) {
                const auto& m = ms[k];
                const float dotUp = m.bitangent[1];
                // The normal must also point away from the cone axis, or the
                // face is buried facing inward.
                const float outward = m.normal[0] * m.pos[0] + m.normal[2] * m.pos[2];
                printf("  [%zu] pos %6.2f %6.2f %6.2f | up %+.3f | outward %+.2f%s\n",
                       k, m.pos[0], m.pos[1], m.pos[2], dotUp, outward,
                       (dotUp > 0.f && outward > 0.f) ? "" : "   <-- WRONG");
                if (!(dotUp > 0.f && outward > 0.f)) ++bad;
            }
            printf("maskframes: %s\n", (ms.size() && !bad) ? "OK" : "FAIL");
            return (ms.size() && !bad) ? 0 : 1;
        }
        if (a == "--rootshot") {
            const char* path = (i + 1 < argc) ? argv[i + 1] : "root.ppm";
            float az  = (i + 2 < argc) ? atof(argv[i + 2]) : 0.6f;
            float el  = (i + 3 < argc) ? atof(argv[i + 3]) : 0.35f;
            float rad = (i + 4 < argc) ? atof(argv[i + 4]) : 42.0f;
            int   md  = (i + 5 < argc) ? atoi(argv[i + 5]) : 0;
            bool  ov  = (i + 6 < argc) ? atoi(argv[i + 6]) != 0 : false;
            return rootshot(path, az, el, rad, md, ov);
        }
        if (a == "--bench") {
            int ds = (i + 1 < argc) ? atoi(argv[i + 1]) : 4;
            int fr = (i + 2 < argc) ? atoi(argv[i + 2]) : 200;
            return bench(ds, fr);
        }
    }

    if (!glfwInit()) { fprintf(stderr, "glfw init failed\n"); return 1; }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);   // Metal owns the surface
    int W = 1280, H = 720;
    GLFWwindow* win = glfwCreateWindow(W, H, "neuromirror ⇄ roots", nullptr, nullptr);
    if (!win) { fprintf(stderr, "window failed\n"); glfwTerminate(); return 1; }

    MetalContext ctx;
    if (!ctx.device()) { fprintf(stderr, "no Metal device\n"); return 1; }
    id<MTLDevice> device = ctx.device();
    id<MTLCommandQueue> queue = ctx.queue();

    // Attach a CAMetalLayer to the GLFW window.
    NSWindow* nswin = glfwGetCocoaWindow(win);
    CAMetalLayer* layer = [CAMetalLayer layer];
    layer.device = device;
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer.framebufferOnly = YES;
    nswin.contentView.layer = layer;
    nswin.contentView.wantsLayer = YES;

    // ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOther(win, true);
    ImGui_ImplMetal_Init(device);

    printf("mirror_app — Metal shell up (device: %s)\n", device.name.UTF8String);

    // Scenes / presenter.
    MirrorScene mirror(ctx);
    RootScene roots(ctx, W, H);
    TransitionScene trans(ctx, W, H);
    FitViewScene fitview(ctx, std::string(MIRROR_APP_SHADER_DIR) + "/fit_view.metal", W, H);
    FullscreenPresent present(ctx, std::string(MIRROR_APP_SHADER_DIR) + "/present.metal",
                              layer.pixelFormat);
    // Composited in the present pass, so it sits over whichever scene is up
    // rather than belonging to any one of them.
    mirror::TextOverlay text(ctx);
    mirror::TextParams textp;

    enum class Scene { Mirror = 0, Roots = 1, Transition = 2, FitView = 3,
                       CamMask = 4 };
    int scene = (int)Scene::Mirror;
    int downscale = 4;       // mirror render-resolution divisor (low-res + upsample)
    int rootDownscale = 1;   // roots render-resolution divisor (manual, when auto off)
    bool rootAutoScale = true;   // cap the roots' internal resolution (see below)
    int  rootTargetDim = 1920;   // target max internal dimension when auto-scaling
    int rootSeed = 1;
    int fieldGrid = 6;           // NxN cached-system field for the LOD/cull demo

    double lastTime = glfwGetTime();
    double fpsAccum = 0.0; int fpsFrames = 0; double fpsShown = 0.0;

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        int fbw, fbh;
        glfwGetFramebufferSize(win, &fbw, &fbh);
        layer.drawableSize = CGSizeMake(std::max(1, fbw), std::max(1, fbh));

        @autoreleasepool {
            id<CAMetalDrawable> drawable = [layer nextDrawable];
            if (!drawable) { continue; }

            id<MTLCommandBuffer> cb = [queue commandBuffer];

            // Update the active scene's texture (MLX compute happens here).
            static double prevT = glfwGetTime();
            double nowT = glfwGetTime();
            double dt = nowT - prevT; prevT = nowT;
            // --- the frame, pulled once ---------------------------------
            //
            // Everything downstream reads the retained snapshot, so the pull
            // happens here, before any of it: pollColor is latest-wins and
            // consuming, and two pulls in a frame would hand the tracker and
            // the fit different moments -- a face outline from one instant
            // over the pixels of another, which shows up as the fit smearing.
            //
            // It runs whenever anything wants a frame, not only while the fit
            // is training: with the live feed on but the fit stopped, the
            // preview and the tracker still have to be live.
            static std::vector<float> live_rgb;
            int  fit_w = 0, fit_h = 0;
            bool live_fresh = false;
            bool source_polled = false;   // the overlay must not pull a second time
            if ((g_fit_live || g_track_on || g_show_source) && SourceReady()) {
                // Sized from the framebuffer rather than mirror.lowW(), which
                // this frame's ensureSize() has not set yet -- the two agree by
                // construction, and reading it after the fact would be a frame
                // behind on a resize.
                const int lw = std::max(1, fbw / std::max(1, downscale));
                const int lh = std::max(1, fbh / std::max(1, downscale));
                // Which tuning applies is decided by the *previous* frame's
                // crop state: the grid has to be chosen before the frame is
                // pulled, and whether there is a crop is not known until the
                // tracker has run on it. One frame of lag on a grid change is
                // nothing next to the hysteresis already in front of it -- and
                // everything within this frame stays consistent, because the
                // mask, the field and the target are all built at whatever
                // fit_w/fit_h this picks.
                const FitTune& grid = g_have_mask ? g_tune_crop : g_tune_full;
                fit_w = std::max(8, lw / std::max(1, grid.downscale));
                fit_h = std::max(8, lh / std::max(1, grid.downscale));
                live_fresh = SourceRGBF(fit_w, fit_h, live_rgb);
                source_polled = true;
            }

            // --- face tracking ------------------------------------------
            // Once per frame, before either scene uses it, so the mirror's
            // mask and the roots' mesh are built from the same detection
            // rather than from two frames a scene apart.
            if (g_track_on && g_tracker.isOpen() && SourceReady()) {
                if (SourceRGB8(g_track_w, g_track_h, g_track_rgb)) {
                    // Video mode rejects a repeated or decreasing timestamp
                    // with a hard error rather than dropping the frame, and
                    // the render loop can outrun the sensor, so the clock here
                    // is a counter rather than wall time.
                    g_track_ts += 33;
                    mirror::FaceResult r;
                    r.blendshape_names = g_face.blendshape_names;   // filled once
                    const bool hit = g_tracker.detect(g_track_rgb.data(), g_track_w,
                                                      g_track_h, g_track_ts, r);
                    // Hysteresis both ways. A detection is not believed until
                    // it has repeated, and a gap is not believed until it has
                    // lasted: MediaPipe drops frames on a blink or a turn, and
                    // treating those as "nobody there" costs a target resize, a
                    // feature-gather rebuild and a visible snap of the soft
                    // edge -- several frames of upheaval to report something
                    // that was over before it began.
                    if (hit) {
                        ++g_face_streak;
                        g_face_last_seen = nowT;
                        g_face_held = false;
                    } else {
                        g_face_streak = 0;
                        if (nowT - g_face_last_seen > g_face_hold_secs) {
                            g_face.valid = false;
                            g_face_held = false;
                        } else {
                            // Inside the hold: keep the last good landmarks,
                            // untouched. Everything downstream carries on
                            // against a frozen face rather than losing one.
                            g_face_held = g_face.valid;
                        }
                    }
                    if (hit && g_face_streak >= std::max(1, g_face_acquire)) {
                        g_face = std::move(r);
                        if (g_fitter.valid()) {
                            // Space the identity samples out in time. Taking
                            // them on consecutive render frames would collect
                            // eight views of the same 130 ms -- the multi-frame
                            // fit exists to average out landmark noise, and
                            // near-identical frames have the same noise in them.
                            if (g_collect_id && nowT - g_last_id_sample > 0.1) {
                                g_fitter.offerIdentityFrame(g_face, g_track_w,
                                                            g_track_h);
                                g_last_id_sample = nowT;
                                // Collection ends on the clock, not on a count:
                                // the retained set is a ranking, so it keeps
                                // improving for as long as it runs and would
                                // never "fill".
                                if (nowT - g_id_started > g_id_collect_secs) {
                                    if (g_fitter.identityFrames() > 0)
                                        g_fitter.fitIdentity(&g_id_residual);
                                    g_collect_id = false;
                                }
                            }
                            g_fitter.update(g_face, g_track_w, g_track_h);
                        }
                    }
                    // No `else` clearing the face: a miss is handled by the
                    // hold above, and a hit that has not yet met the acquire
                    // threshold is simply not adopted. Clearing here is what
                    // used to make a single dropped frame a whole event.
                }
            }

            // --- head movement ------------------------------------------
            // After the detection, before anything that consumes it. The
            // input shift and the region have to be settled here: the fit
            // features and the render both read them later in this frame and
            // must read the same values.
            UpdateHeadBox();
            ApplyHeadMode(mirror.params(), fit_w, fit_h);
            if (live_fresh) PlaceLiveFrame(live_rgb, fit_w, fit_h);

            // --- source overlay: upload the raw frame ---------------------
            //
            // Deliberately the *same* call the tracker makes (SourceRGB8), not a
            // second path to the sensor: the point of the overlay is to show the
            // picture the rest of the pipeline is looking at, mirroring and all.
            // A prettier but independently-fetched preview would be exactly the
            // kind of thing that agrees with the sensor while disagreeing with
            // the fit.
            static id<MTLTexture> srcTex = nil;
            static int srcTexW = 0, srcTexH = 0;
            static std::vector<unsigned char> srcRGB, srcRGBA;
            bool srcFresh = false;
            int pipW = 0, pipH = 0;
            const bool wantSource = g_show_source || scene == (int)Scene::CamMask;
            if (wantSource && SourceReady()) {
                // Preview at the source's own aspect. 320 wide is enough to see
                // a face in the corner and cheap to box-filter down to; the mask
                // editor draws it full-frame, so it gets a real resolution --
                // edges are what is being placed there.
                const int base = scene == (int)Scene::CamMask ? 960 : 320;
                if (g_source == (int)Source::Photo && g_photo_w > 0 && g_photo_h > 0) {
                    pipW = base;
                    pipH = std::max(1, base * g_photo_h / g_photo_w);
                } else {
                    pipW = base;                 // the colour camera is 1920x1080
                    pipH = std::max(1, base * 9 / 16);
                }
#if MIRROR_HAVE_KINECT
                // Advance the retained snapshot when nothing else did. Without
                // this the overlay (and the tracker, which reads the same
                // snapshot) would sit on one frozen frame whenever the live fit
                // is off -- and a frozen preview is worse than none, because it
                // looks like a working camera.
                if (!source_polled && g_source == (int)Source::Kinect &&
                    g_kinect.isOpen()) {
                    static std::vector<float> pip_scratch;
                    g_kinect.poll(pipW, pipH, pip_scratch);
                }
#endif
                if (SourceRGB8(pipW, pipH, srcRGB) &&
                    srcRGB.size() == size_t(pipW) * pipH * 3) {
                    if (!srcTex || srcTexW != pipW || srcTexH != pipH) {
                        MTLTextureDescriptor* td = [MTLTextureDescriptor
                            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                         width:pipW
                                                        height:pipH
                                                     mipmapped:NO];
                        td.usage = MTLTextureUsageShaderRead;
                        td.storageMode = MTLStorageModeManaged;
                        srcTex = [ctx.device() newTextureWithDescriptor:td];
                        srcTexW = pipW; srcTexH = pipH;
                    }
                    srcRGBA.resize(size_t(pipW) * pipH * 4);
                    for (size_t i = 0, n = size_t(pipW) * pipH; i < n; ++i) {
                        srcRGBA[i * 4 + 0] = srcRGB[i * 3 + 0];
                        srcRGBA[i * 4 + 1] = srcRGB[i * 3 + 1];
                        srcRGBA[i * 4 + 2] = srcRGB[i * 3 + 2];
                        srcRGBA[i * 4 + 3] = 255;
                    }
                    [srcTex replaceRegion:MTLRegionMake2D(0, 0, pipW, pipH)
                              mipmapLevel:0
                                withBytes:srcRGBA.data()
                              bytesPerRow:size_t(pipW) * 4];
                    srcFresh = true;
                }
            }

            id<MTLTexture> sceneTex = nil;

            // The mirror's frame, trained and rendered. A lambda because two
            // scenes need exactly this and a copy in each would be two places
            // for the training schedule to drift apart.
            auto renderMirror = [&]() -> id<MTLTexture> {
                mirror.ensureSize(fbw / std::max(1, downscale), fbh / std::max(1, downscale));
                mirror.advance(dt);
                // Training runs here, not inside render(): one place, once per
                // frame, so the cost is attributable and the displayed frame is
                // always the post-step state.
                if (mirror.pond().fitting()) {
                    // Live feed: swap the target, keeping weights and Adam
                    // state. beginFit() here would reset the optimiser every
                    // frame and the fit would never build enough momentum to
                    // follow motion (measured 677x worse tracking error).
                    if (g_fit_live && live_fresh) {
                        // The mask was built at the fit grid's size in
                        // ApplyHeadMode, from normalised landmarks -- a rescale
                        // of the outline rather than a resample of any image.
                        // With a mask, the training pass gathers only those
                        // pixels: a face is ~4% of the frame, so a step costs
                        // ~4% of the unmasked one.
                        if (g_have_mask) {
                            mirror.pond().updateFitTarget(live_rgb, fit_h, fit_w,
                                                          g_fit_mask);
                        } else {
                            mirror.pond().updateFitTarget(live_rgb, fit_h, fit_w);
                        }
                    }
                    const FitTune& tune = g_have_mask ? g_tune_crop : g_tune_full;
                    mirror.fitSteps(tune.steps, tune.lr);
                }
                id<MTLTexture> out = mirror.render();

                // Sample the neural texture onto the fitted mesh. This runs on
                // the mirror's own output, so the mask ends up wearing the
                // network's reconstruction of the face rather than the raw
                // camera pixels -- which is the point: what the mask carries
                // into the root scene is what the mirror made of the person.
                //
                // Sampled at the *pinned* position, not the landmark one: the
                // render puts the face wherever the head mode put it.
                //
                // Cheap enough to do every frame: 2056 bilinear samples.
                if (g_texture_mask && g_track_on && g_face.valid &&
                    g_fitter.valid() && mirror.pond().fitted()) {
                    float ps = 1.f, uo = 0.f, vo = 0.f;
                    PinTransform(ps, uo, vo);
                    g_fitter.sampleTexture(mirror.lastImageRGB(),
                                           mirror.lowW(), mirror.lowH(),
                                           g_track_w, g_track_h, g_face_colors,
                                           ps, uo, vo);
                    g_face_colors_fresh = true;
                }
                return out;
            };

            if (scene == (int)Scene::Mirror && mirror.valid()) {
                sceneTex = renderMirror();
            } else if (scene == (int)Scene::FitView && fitview.valid()) {
                // Same mirror frame, drawn flat with the mask and the mesh on
                // top of it. Deliberately runs the training too: a diagnostic
                // that froze the thing it is diagnosing would only ever show
                // the moment the scene was entered.
                fitview.setBackground(renderMirror());
                if (g_have_mask && fit_w > 0) {
                    fitview.setMask(g_fit_mask, fit_w, fit_h);
                } else {
                    fitview.clearMask();
                }
                if (g_track_on && g_face.valid && g_fitter.valid()) {
                    float ps = 1.f, uo = 0.f, vo = 0.f;
                    PinTransform(ps, uo, vo);
                    static std::vector<float> mesh_uv;
                    g_fitter.projectNormalised(g_track_w, g_track_h, ps, uo, vo, mesh_uv);
                    static bool tris_sent = false;
                    fitview.setMesh(mesh_uv, g_face_colors,
                                    tris_sent ? std::vector<int>()
                                              : g_fitter.basis().triangles());
                    tris_sent = true;
                } else {
                    fitview.clearMesh();
                }
                fitview.ensureSize(fbw, fbh);
                sceneTex = fitview.render(cb);
            } else if (scene == (int)Scene::CamMask && fitview.valid()) {
                // The raw camera frame, full-frame, with the mask already
                // applied to it -- so what is being edited is the result, not a
                // rectangle floating over an unmasked preview. The handles are
                // drawn as an ImGui overlay further down, where the mouse is.
                fitview.setBackground(srcTex);
                fitview.clearMask();
                fitview.clearMesh();
                fitview.ensureSize(fbw, fbh);
                sceneTex = fitview.render(cb);
            } else if (scene == (int)Scene::Transition && trans.valid()) {
                // The transition is driven by the mirror, so the mirror keeps
                // rendering underneath it -- that texture *is* the film. Its
                // training is left alone: the effect is a handoff, and a fit
                // that kept moving during it would change the sheet's skin
                // mid-fall.
                mirror.ensureSize(fbw / std::max(1, downscale), fbh / std::max(1, downscale));
                mirror.advance(dt);
                trans.setPondTexture(mirror.render());
                if (g_track_on && g_fitter.valid() && g_face.valid && !trans.hasFace())
                    trans.setFaceMesh(g_fitter.vertices(), g_fitter.basis().triangles());
                trans.ensureSize(fbw, fbh);
                trans.advance(dt);
                sceneTex = trans.render(cb);
            } else if (scene == (int)Scene::Roots && roots.valid()) {
                // The roots pass is overdraw-bound (per-fragment ray-capsule
                // intersection, multiplied by how many capsules stack per pixel),
                // so cost scales with pixels x overdraw. Rendering below the window
                // resolution and bilinear-upsampling (present.metal) — with the fog
                // pass's FXAA-lite smoothing the low-res image first — is the main
                // lever. Auto-scale caps the internal max dimension so a 4K/Retina
                // window stays fast instead of collapsing on a dense/zoomed nest.
                int effDs = std::max(1, rootDownscale);
                if (rootAutoScale) {
                    int maxdim = std::max(fbw, fbh);
                    effDs = std::max(1, (maxdim + rootTargetDim - 1) / rootTargetDim);
                }
                // A fitted face on the root scene's masks, re-uploaded only
                // when the tracker actually produced a new detection -- the
                // rebuild walks every placed mask, so doing it on a frame where
                // nothing changed is pure cost.
                if (g_drive_roots && g_track_on && g_fitter.valid() && g_face.valid) {
                    static bool uploaded_tris = false;
                    roots.setFittedFace(g_fitter.vertices(),
                                        uploaded_tris ? std::vector<int>()
                                                      : g_fitter.basis().triangles());
                    uploaded_tris = true;
                } else if (!g_drive_roots && roots.usingFittedFace()) {
                    roots.clearFittedFace();
                }

                // Hand the captured neural texture over. The mirror is not
                // running now -- only one sim runs at a time -- so there is no
                // live texture left to sample; what the mask wears is whatever
                // was captured while the mirror still had the person. Uploaded
                // once, on the frame after capture, because it does not change
                // again until the mirror runs again.
                if (g_face_colors_fresh && !g_face_colors.empty()) {
                    roots.setFaceColors(g_face_colors);
                    g_face_colors_fresh = false;
                }
                roots.ensureSize(fbw / effDs, fbh / effDs);
                roots.advance(dt);
                sceneTex = roots.render(cb);   // encodes geometry + fog passes into cb
            }

            MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
            rpd.colorAttachments[0].texture = drawable.texture;
            rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
            rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.05, 0.05, 0.06, 1.0);
            rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

            // MIDI, then the panel. Draining first means a knob moved this
            // frame is already in the parameter when the control draws it, so
            // the slider and the hardware never disagree on screen.
            if (g_midi.isOpen()) {
                static std::vector<midi::CC> ccs;
                g_midi.drain(ccs);
                for (const midi::CC& c : ccs) ui::ApplyCC(c.channel, c.cc, c.value);
                // Devices plugged in mid-session are the normal case.
                static double last_scan = 0.0;
                if (nowT - last_scan > 2.0) { g_midi.rescan(); last_scan = nowT; }
            }
            ui::BeginFrame();

            ImGui_ImplMetal_NewFrame(rpd);
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(340, 0), ImGuiCond_FirstUseEver);
            ImGui::Begin("neuromirror — controls");
            ImGui::Text("%.0f fps   t=%5.1fs", fpsShown, mirror.clock());
            ImGui::TextUnformatted("scene:"); ImGui::SameLine();
            ImGui::RadioButton("mirror", &scene, (int)Scene::Mirror); ImGui::SameLine();
            ImGui::RadioButton("roots",  &scene, (int)Scene::Roots); ImGui::SameLine();
            ImGui::RadioButton("transition", &scene, (int)Scene::Transition);
            ImGui::SameLine();
            ImGui::RadioButton("fit view", &scene, (int)Scene::FitView);
            ImGui::SameLine();
            ImGui::RadioButton("cam mask", &scene, (int)Scene::CamMask);
            ImGui::Separator();

            // --- camera mask ---------------------------------------------
            ui::PushSection("camera mask");
            if (ImGui::CollapsingHeader("camera mask",
                                        scene == (int)Scene::CamMask
                                            ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
                ui::Checkbox("mask the camera", &g_cam_mask_on);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "Keep a rectangle of the sensor's view and black out\n"
                        "the rest. The mirror only sells if the frame holds the\n"
                        "person and nothing that says 'room' -- a doorway, a\n"
                        "window, the edge of the rig.\n\n"
                        "Applied in camera space, ahead of everything: the fit\n"
                        "never sees the masked pixels and the tracker cannot\n"
                        "find a face in them.");
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("reset")) {
                    g_cam_x0 = 0.15f; g_cam_y0 = 0.05f;
                    g_cam_x1 = 0.85f; g_cam_y1 = 0.95f;
                }
                ImGui::BeginDisabled(!g_cam_mask_on);
                ImGui::PushItemWidth(-90);
                ImGui::DragFloatRange2("x", &g_cam_x0, &g_cam_x1, 0.002f, 0.f, 1.f,
                                       "%.3f", "%.3f", ImGuiSliderFlags_AlwaysClamp);
                ImGui::DragFloatRange2("y", &g_cam_y0, &g_cam_y1, 0.002f, 0.f, 1.f,
                                       "%.3f", "%.3f", ImGuiSliderFlags_AlwaysClamp);
                ui::SliderFloat("soft edge", &g_cam_feather, 0.f, 0.2f, "%.3f");
                ImGui::PopItemWidth();
                ImGui::EndDisabled();
                if (scene != (int)Scene::CamMask) {
                    ImGui::TextDisabled("(the 'cam mask' scene has drag handles)");
                }
            }
            ui::PopSection();

            // --- face tracking (feeds both scenes) ------------------------
            ui::PushSection("face tracking");
            if (ImGui::CollapsingHeader("face tracking")) {
                if (!mirror::FaceTracker::available()) {
                    ImGui::TextDisabled("MediaPipe not compiled in");
                    ImGui::TextDisabled("run ./setup-mediapipe.sh, then re-cmake");
                } else {
                    if (ui::Checkbox("track faces", &g_track_on) && g_track_on) {
                        if (!g_tracker.isOpen()) {
                            const std::string model =
                                std::string(MIRROR_APP_EXTERNAL_DIR) + "/face_landmarker.task";
                            if (!g_tracker.open(model, g_track_err)) g_track_on = false;
                        }
                        if (g_track_on && !g_fitter.valid()) {
                            const std::string basis =
                                std::string(MIRROR_APP_EXTERNAL_DIR) + "/face_basis.bin";
                            std::string ferr;
                            if (!g_fitter.load(basis, ferr)) g_track_err = ferr;
                        }
                    }
                    ImGui::SameLine();
                    if (g_face_held) {
                        ImGui::TextColored(ImVec4(1.f, 0.85f, 0.4f, 1.f), "held %.2fs",
                                           nowT - g_face_last_seen);
                    } else if (g_face.valid) {
                        ImGui::TextColored(ImVec4(0.6f, 1.f, 0.7f, 1.f), "face");
                    } else {
                        ImGui::TextDisabled("no face");
                    }
                    ImGui::PushItemWidth(90);
                    ui::SliderFloat("hold on loss", &g_face_hold_secs, 0.f, 3.f,
                                       "%.2fs");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "How long a detection survives after the tracker\n"
                            "stops returning one. A blink or a turn drops\n"
                            "frames, and without this the fit target flips from\n"
                            "a crop to the whole frame and back -- resizing the\n"
                            "trained pixel set and rebuilding the feature\n"
                            "gather to report something already over.");
                    }
                    ImGui::SameLine();
                    ui::SliderInt("acquire", &g_face_acquire, 1, 10);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Consecutive detections before a face is believed.\n"
                            "The other half of the same idea: keeps a single\n"
                            "spurious hit from starting everything up.");
                    }
                    ImGui::PopItemWidth();
#if !MIRROR_HAVE_KINECT
                    ImGui::TextDisabled("(no camera: tracking needs the Kinect target)");
#endif
                    if (!g_track_err.empty())
                        ImGui::TextColored(ImVec4(1.f, 0.5f, 0.5f, 1.f), "%s",
                                           g_track_err.c_str());

                    ui::Checkbox("crop the fit to the face", &g_mask_fit);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "With a face tracked, train only on its crop; with\n"
                            "none, fit the whole video feed. The network stays\n"
                            "global -- nothing constrains it outside the crop --\n"
                            "so the person is fitted and the rest of the frame\n"
                            "stays generative. A face is a few percent of the\n"
                            "frame, and a cropped step costs about that fraction\n"
                            "of a full one.");
                    }
                    ImGui::BeginDisabled(!g_mask_fit);
                    ImGui::Indent();
                    ImGui::TextUnformatted("crop:"); ImGui::SameLine();
                    ImGui::RadioButton("landmark box", &g_mask_shape,
                                       (int)MaskShape::Box);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "The landmarks' bounding box: the whole face crop,\n"
                            "hair and jawline and the background just around\n"
                            "them included.");
                    }
                    ImGui::SameLine();
                    ImGui::RadioButton("hull", &g_mask_shape, (int)MaskShape::Hull);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "The silhouette the landmarks trace. Tighter, but it\n"
                            "supervises skin only -- the network never sees the\n"
                            "boundary between a person and the room, so it has\n"
                            "no reason to draw one.");
                    }
                    ImGui::PushItemWidth(90);
                    if (g_mask_shape == (int)MaskShape::Box) {
                        ui::SliderFloat("pad", &g_crop_pad, 0.f, 0.6f, "%.2f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip(
                                "Grow the box by a fraction of its own size, so\n"
                                "the margin scales with how close the person is.");
                        }
                        ImGui::SameLine();
                    }
                    ui::SliderInt("dilate", &g_mask_dilate, 0, 24);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "A fixed margin in fit-grid pixels, on top of the\n"
                            "crop. A region tight to the outline gives the\n"
                            "network no background pixels near the edge, and\n"
                            "with only positive supervision it has no reason to\n"
                            "form a boundary there -- it converges to a soft\n"
                            "blob instead of an edge.");
                    }
                    ImGui::PopItemWidth();

                    // --- head movement ---------------------------------
                    ImGui::TextUnformatted("head moves:");
                    ImGui::RadioButton("centre it", &g_head_mode,
                                       (int)HeadMode::Centred);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Shift the frame so the head sits in the middle and\n"
                            "fit it there. The network is shown the same problem\n"
                            "every frame, which is the most stable thing to ask\n"
                            "of it -- but the mirror stops showing where in the\n"
                            "room the person is.");
                    }
                    ImGui::SameLine();
                    ImGui::RadioButton("follow it", &g_head_mode,
                                       (int)HeadMode::Track);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Fit the head where the camera found it. Honest, and\n"
                            "the least stable: a face crossing the frame is a\n"
                            "different function at every step, so the weights\n"
                            "spend themselves re-learning one face at a hundred\n"
                            "addresses.");
                    }
                    ImGui::SameLine();
                    ImGui::RadioButton("shift the inputs", &g_head_mode,
                                       (int)HeadMode::Stabilised);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Fit the head where it is, but offset the network's\n"
                            "input coordinates by its displacement -- so the\n"
                            "subject holds still in the network's own frame\n"
                            "while still moving on screen. The picture of\n"
                            "'follow', the weights of 'centre'.\n\n"
                            "The whole field shifts with the head, background\n"
                            "included: the offset is on the coordinates, not on\n"
                            "the subject.");
                    }
                    // Size is only meaningful where the app owns the placement.
                    // In the other two modes the subject is where the camera
                    // found it, and rescaling would be fighting that.
                    ImGui::BeginDisabled(g_head_mode != (int)HeadMode::Centred);
                    ui::Checkbox("set face size", &g_face_size_on);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Resample the crop so the head is a chosen size on\n"
                            "screen, instead of whatever distance the person\n"
                            "happens to be standing at.\n\n"
                            "Costs a bilinear resample every frame: at size 1:1\n"
                            "the centred mode only shifts by whole pixels, which\n"
                            "is deliberate -- refiltering the face every frame is\n"
                            "the noise that mode exists to remove.");
                    }
                    ImGui::SameLine();
                    ImGui::BeginDisabled(!g_face_size_on);
                    ImGui::SetNextItemWidth(110);
                    ui::SliderFloat("size", &g_face_size, 0.05f, 0.5f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Half the head's height as a fraction of the frame,\n"
                            "so 0.25 fills half the screen top to bottom.");
                    }
                    ImGui::EndDisabled();
                    if (g_face_size_on && HaveCrop()) {
                        ImGui::SameLine();
                        ImGui::TextDisabled("x%.2f", PlaceScale());
                    }
                    ImGui::EndDisabled();

                    ImGui::SetNextItemWidth(110);
                    ui::SliderFloat("head smoothing", &g_head_smooth, 0.02f, 1.f,
                                       "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "How fast the tracked box follows the landmarks.\n"
                            "1 is raw. The box jitters a pixel or two on a still\n"
                            "head, and both the input shift and the soft edge\n"
                            "show that jitter directly.");
                    }
                    ImGui::Unindent();
                    ImGui::EndDisabled();

                    ui::Checkbox("fitted mesh drives the root masks", &g_drive_roots);
                    ui::Checkbox("texture the mask from the neural fit",
                                    &g_texture_mask);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Colour the mask's vertices by projecting them into\n"
                            "the mirror's own output and sampling it -- so the\n"
                            "mask wears the network's reconstruction of the\n"
                            "face, not the camera's pixels.\n\n"
                            "The colour is captured, not looked up live: the\n"
                            "mirror and the roots never run at the same time,\n"
                            "so by the time the roots draw there is no neural\n"
                            "texture left to sample. Capturing at the handoff\n"
                            "is what lets the mask keep the face.");
                    }
                    if (!g_face_colors.empty()) {
                        ImGui::SameLine();
                        ImGui::TextColored(ImVec4(0.6f, 1.f, 0.7f, 1.f), "captured");
                    }

                    // --- frame source --------------------------------------
                    ImGui::Separator();
                    ImGui::TextUnformatted("source:"); ImGui::SameLine();
                    ImGui::RadioButton("sensor", &g_source, (int)Source::Kinect);
                    ImGui::SameLine();
                    ImGui::RadioButton("photo", &g_source, (int)Source::Photo);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Substitute a still photo for the camera. Nothing\n"
                            "downstream knows the difference, so the whole\n"
                            "path -- tracking, fit, mask, texture -- runs\n"
                            "without a person in front of the sensor, and runs\n"
                            "reproducibly on a known face.");
                    }
                    if (g_source == (int)Source::Photo) {
                        ImGui::PushItemWidth(-70);
                        ImGui::InputText("##photo", g_photo_path, sizeof(g_photo_path));
                        ImGui::PopItemWidth();
                        ImGui::SameLine();
                        if (ImGui::Button("load")) {
                            std::string perr;
                            if (!LoadPhotoSource(g_photo_path, perr)) g_track_err = perr;
                            else g_track_err.clear();
                        }
                        if (g_photo.empty()) ImGui::TextDisabled("no photo loaded");
                        else ImGui::TextDisabled("photo %dx%d", g_photo_w, g_photo_h);
                    }

                    // The overlay lives here rather than under the mirror's
                    // fit controls because the radio above it is the usual
                    // reason the mirror is not following the camera, and the
                    // two want to be read together.
                    ui::PushSection("overlay");
                    ui::Checkbox("camera overlay", &g_show_source);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Show the raw source frame in a corner -- the same\n"
                            "image the tracker sees, mirroring included. The\n"
                            "mirror's own output cannot tell a closed sensor\n"
                            "from a stale frame from the photo still being\n"
                            "selected; this can.");
                    }
                    if (g_show_source) {
                        ImGui::SameLine();
                        ui::Checkbox("landmarks", &g_pip_landmarks);
                        ImGui::PushItemWidth(110);
                        ui::SliderInt("size", &g_source_pip_w, 160, 640);
                        ImGui::SameLine();
                        const char* corners[] = {"top-left", "top-right",
                                                 "bottom-left", "bottom-right"};
                        ImGui::Combo("corner", &g_source_corner, corners, 4);
                        ui::DeclareInt("corner", &g_source_corner, 0, 3);
                        ImGui::PopItemWidth();
                    }
                    ui::PopSection();

                    // --- identity ------------------------------------------
                    ImGui::Separator();
                    ui::PushSection("identity");
                    ImGui::Text("IDENTITY");
                    ImGui::SameLine();
                    if (g_fitter.hasIdentity()) {
                        ImGui::TextColored(ImVec4(0.6f, 1.f, 0.7f, 1.f),
                                           "fitted (%.2f px residual)", g_id_residual);
                    } else if (g_collect_id) {
                        float best = 0, worst = 0;
                        g_fitter.identityScores(best, worst);
                        ImGui::TextDisabled("collecting %.1fs  best %.2f",
                                            g_id_collect_secs - (nowT - g_id_started),
                                            best);
                    } else {
                        ImGui::TextDisabled("mean face");
                    }
                    ImGui::BeginDisabled(!g_fitter.valid());
                    if (ImGui::Button(g_collect_id ? "cancel" : "fit identity")) {
                        if (g_collect_id) {
                            g_collect_id = false;
                        } else {
                            g_fitter.clearIdentity();
                            g_id_residual = -1.f;
                            g_collect_id = true;
                            g_id_started = nowT;
                        }
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Look at the camera with a still face for a few\n"
                            "seconds, then the 100 identity coefficients are\n"
                            "solved over the best frames at once.\n\n"
                            "Frames are ranked by frontality x neutrality and\n"
                            "the best few kept -- not thresholded. MediaPipe\n"
                            "reports substantial baseline activation on an\n"
                            "ordinary face, so any fixed threshold either\n"
                            "accepts everything or nothing depending on the\n"
                            "person and the lighting.\n\n"
                            "Expression is known per frame (from the\n"
                            "blendshapes) and subtracted first, so a smile does\n"
                            "not get baked into the face.");
                    }
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(90);
                    ui::SliderFloat("secs", &g_id_collect_secs, 1.f, 15.f, "%.0fs");
                    ImGui::EndDisabled();

                    if (g_fitter.valid()) {
                        ImGui::PushItemWidth(90);
                        ui::SliderInt("modes", &g_fitter.config().n_identity, 10, 100);
                        ImGui::SameLine();
                        ui::SliderFloat("ridge", &g_fitter.config().ridge, 0.01f, 20.f,
                                           "%.2f", ImGuiSliderFlags_Logarithmic);
                        ImGui::SameLine();
                        ui::SliderInt("frames", &g_fitter.config().max_frames, 1, 24);
                        ImGui::PopItemWidth();
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip(
                                "modes: of the basis's 100. The tail modes are\n"
                                "detail 68 landmarks cannot resolve, and fitting\n"
                                "them is how the fit starts chasing noise.\n"
                                "ridge: pulls the solve toward the mean face.\n"
                                "frames: more samples average out landmark jitter.");
                        }

                        bool tp = g_fitter.trackerPose();
                        if (ui::Checkbox("head pose from tracker", &tp))
                            g_fitter.useTrackerPose(tp);
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip(
                                "Rotate the mesh by MediaPipe's 4x4 facial\n"
                                "transformation matrix. The Python original ran\n"
                                "solvePnP for this; the Tasks API hands back the\n"
                                "pose directly, so there is no PnP and no OpenCV.\n"
                                "Off falls back to the flat 2D similarity, which\n"
                                "is enough for placement but does not turn.");
                        }
                        if (g_face.valid && tp) {
                            float yaw, pitch, roll;
                            g_fitter.headAngles(yaw, pitch, roll);
                            ImGui::TextDisabled("yaw %+.0f°  pitch %+.0f°  roll %+.0f°",
                                                yaw * 57.2958f, pitch * 57.2958f,
                                                roll * 57.2958f);
                        }
                        ImGui::TextDisabled("basis: %d verts, %d tris (%s)",
                                            g_fitter.basis().vertexCount(),
                                            g_fitter.basis().triangleCount(),
                                            g_fitter.basis().nvfTopology() ? "Maxine/NVF"
                                                                           : "ICT");
                    } else {
                        ImGui::TextDisabled("no face_basis.bin -- run");
                        ImGui::TextDisabled("tools/export_face_basis.py");
                    }
                    ui::PopSection();       // "identity"
                }
            }
            ui::PopSection();               // "face tracking"
            ImGui::Separator();

            ui::PushSection("transition");
            if (scene == (int)Scene::Transition) {
                ImGui::Text("%s   t=%.2fs   emergence %.0f%%",
                            trans.phaseName(), trans.clock(), trans.emergence() * 100.f);
                ImGui::SameLine();
                if (ImGui::Button("replay")) trans.restart();
                if (!trans.hasFace()) {
                    ImGui::TextDisabled("no fitted face -- enable face tracking");
                    ImGui::TextDisabled("(the pond alone still plays)");
                }
                ImGui::SeparatorText("timing (seconds)");
                ImGui::PushItemWidth(110);
                ui::SliderFloat("hold",   &trans.timing.hold,   0.f, 3.f);
                ImGui::SameLine();
                ui::SliderFloat("emerge", &trans.timing.emerge, 0.2f, 6.f);
                ui::SliderFloat("settle", &trans.timing.settle, 0.f, 2.f);
                ImGui::SameLine();
                ui::SliderFloat("fade",   &trans.timing.fade,   0.f, 1.f);
                ImGui::PopItemWidth();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "Crossfade across the swap. The sheet is held flat for\n"
                        "its whole duration, so the blend is between two\n"
                        "pixel-identical images -- letting gravity start on the\n"
                        "swap frame would make it a blend of two different ones\n"
                        "and the seam would show through regardless.");
                }
                ImGui::SeparatorText("look");
                ui::SliderFloat("refraction", &trans.refract, 0.f, 0.25f);
                ui::SliderFloat("velocity refraction", &trans.refractVel, 0.f, 3.f);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "Scales refraction by how fast the face is moving\n"
                        "through the film. The prototype used a fixed constant,\n"
                        "so the distortion peaked at the midpoint of the\n"
                        "timeline no matter how the emergence was paced.");
                }
                ImGui::SeparatorText("cloth");
                ui::Checkbox("show cloth", &trans.showCloth);
                ImGui::SameLine();
                ui::Checkbox("wireframe", &trans.wireframe);
                ui::SliderFloat("gravity back (-z)", &trans.gravityBack, 0.f, 20.f);
                ui::SliderFloat("gravity down (-y)", &trans.gravityDown, 0.f, 5.f);
                ImGui::PushItemWidth(110);
                ui::SliderInt("substeps", &trans.substeps, 1, 8);
                ImGui::SameLine();
                ui::SliderInt("iterations", &trans.iterations, 4, 64);
                ImGui::PopItemWidth();
                ImGui::TextDisabled("%d verts, %zu tris, minZ %.2f",
                                    (int)trans.cloth().pos.size(),
                                    trans.cloth().tris.size() / 3, trans.cloth().minZ());
            }
            ui::PopSection();               // "transition"

            // --- text overlay -------------------------------------------
            // Outside the per-scene blocks: it composites in the present pass,
            // so it is available over every scene.
            ui::PushSection("text");
            if (ImGui::CollapsingHeader("text overlay")) {
                ui::Checkbox("show text", &textp.on);
                static char buf[256] = {};
                static bool buf_init = false;
                if (!buf_init) {
                    std::snprintf(buf, sizeof(buf), "%s", textp.text.c_str());
                    buf_init = true;
                }
                if (ImGui::InputTextMultiline("##text", buf, sizeof(buf),
                                              ImVec2(-1, 46))) {
                    textp.text = buf;
                }
                ImGui::TextDisabled("newlines split lines");

                ui::SliderFloat("size", &textp.size, 0.02f, 0.6f);
                ui::SliderFloat("x", &textp.cx, -2.f, 2.f);
                ui::SliderFloat("y", &textp.cy, -1.f, 1.f);
                ui::SliderFloat("inversion", &textp.strength, 0.f, 1.f);
                ui::SliderFloat("text refraction", &textp.warp, 0.f, 2.f);
                // Antialiasing, not a glow -- the field only carries distance
                // out to its spread, and the shader clamps the ramp there.
                ui::SliderFloat("edge softness", &textp.softness, 0.2f, 3.f);
                ui::SliderFloat("stroke weight", &textp.dilate, -0.02f, 0.02f,
                                "%.4f");

                ImGui::Separator();
                // The one to bind to a fader: it is the whole emerge/dissolve
                // timeline, and the three under it only shape what it looks
                // like on the way through.
                ui::SliderFloat("reveal", &textp.reveal, 0.f, 1.f);
                ui::SliderFloat("turbulence", &textp.turbulence, 0.f, 1.f);
                ui::SliderFloat("turb scale", &textp.turb_scale, 0.5f, 30.f);
                ui::SliderFloat("turb drift", &textp.turb_speed, 0.f, 2.f);
                ImGui::Separator();
                // These rebuild the field rather than moving a uniform, which is
                // why they sit apart from the live knobs above: dragging one
                // re-rasterises the glyphs and re-runs the distance transform.
                ui::SliderFloat("tracking", &textp.tracking, -0.1f, 0.5f);
                static char fontbuf[128] = {};
                static bool font_init = false;
                if (!font_init) {
                    std::snprintf(fontbuf, sizeof(fontbuf), "%s",
                                  textp.font.c_str());
                    font_init = true;
                }
                if (ImGui::InputText("font", fontbuf, sizeof(fontbuf))) {
                    textp.font = fontbuf;
                }
                ui::SliderInt("raster px", &textp.raster_px, 64, 1024);
                ImGui::TextDisabled("field %dx%d", text.fieldW(), text.fieldH());
                if (textp.on && scene != (int)Scene::Mirror && textp.warp > 0.f) {
                    ImGui::TextDisabled("(no ripples in this scene: unwarped)");
                }
            }
            ui::PopSection();               // "text"

            if (scene == (int)Scene::Mirror) {
                ui::PushSection("mirror");
                mirror::PondParams& P = mirror.params();
                // ripples
                ui::SliderFloat("ring freq", &P.ring_freq, 0.3f, 10.0f);
                ui::SliderFloat("ripple decay", &P.decay, 0.0f, 5.0f);
                ui::SliderFloat("ripple speed", &P.speed, 0.0f, 6.0f);
                ui::SliderFloat("ripple phase", &P.ripple_offset, 0.0f, 2.0f * (float)M_PI);
                ui::SliderFloat("refraction (warp)", &P.warp, 0.0f, 1.0f);
                ui::SliderInt("raindrops", &P.drops, 0, 12);
                ui::Checkbox("moving ripple", &P.orbit_on);
                ui::Checkbox("soft centers (anti-alias)", &P.core_rolloff);
                if (P.core_rolloff) {
                    ImGui::SameLine(); ImGui::SetNextItemWidth(120);
                    ui::SliderFloat("radius", &P.core_radius, 0.02f, 0.5f);
                }
                ImGui::Separator();
                // --- live fitting -----------------------------------------
                {
                    static char fit_path[512] =
                        "/Users/erichan/Documents/Development/neuromirror/"
                        "emotion/_track_frames/f0016.png";
                    static std::string fit_err;

                    ImGui::Text("FIT  %s", mirror.pond().fitted()
                                    ? (mirror.pond().fitting() ? "training" : "held")
                                    : "not fitted");
                    ImGui::SameLine();
                    ImGui::TextDisabled("| %d steps | loss %.5f",
                                        mirror.pond().fitSteps(), mirror.lastLoss());

                    ImGui::PushItemWidth(-1);
                    ImGui::InputText("##fitpath", fit_path, sizeof(fit_path));
                    ImGui::PopItemWidth();

#if MIRROR_HAVE_KINECT
                    {
                        ImGui::Separator();
                        const bool open = g_kinect.isOpen();
                        ImGui::Text("LIVE");
                        ImGui::SameLine();
                        if (open) {
                            ImGui::TextColored(ImVec4(0.6f, 1.f, 0.7f, 1.f), "%s",
                                               g_kinect.deviceInfo().c_str());
                        } else {
                            ImGui::TextDisabled("sensor closed");
                        }
                        ImGui::SameLine();
                        ImGui::TextDisabled("| %llu frames",
                                            (unsigned long long)g_kinect.frames());

                        if (ImGui::Button(open ? "close sensor" : "open sensor")) {
                            if (open) {
                                g_fit_live = false;
                                g_kinect.close();
                            } else {
                                std::string kerr;
                                if (!g_kinect.open(kerr)) fit_err = kerr;
                                else fit_err.clear();
                            }
                        }
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip(
                                "Only one process can hold the sensor -- close\n"
                                "kinect_v2_demo first, or opening fails with\n"
                                "LIBUSB_ERROR_NO_DEVICE.");
                        }
                        ImGui::SameLine();
                        ImGui::BeginDisabled(!open);
                        // Arming the feed does not start training: the frame
                        // pull, the tracker and the preview all come alive, and
                        // "fit" below is what begins (or restarts) the fit on
                        // whatever the camera is showing at that moment.
                        ui::Checkbox("track live feed", &g_fit_live);
                        ImGui::EndDisabled();
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip(
                                "Retarget from the camera every frame. The fit\n"
                                "never finishes -- it tracks, running a few\n"
                                "hundred ms behind whoever is in front of the\n"
                                "sensor. That lag is the effect.\n\n"
                                "This only arms the feed. Press fit to start\n"
                                "training on it.");
                        }
                        // --- outside the crop ---------------------------
                        //
                        // Inside, the network is reproducing a person and
                        // every input must hold still. Outside, nothing is
                        // constrained. These are what that difference is
                        // allowed to look like.
                        ui::Checkbox("soft edge", &g_region_on);
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip(
                                "Fade the effects below in across a band around\n"
                                "the crop instead of switching at its border.");
                        }
                        ImGui::BeginDisabled(!g_region_on);
                        ImGui::SameLine();
                        ui::Checkbox("follow the outline", &g_region_hull);
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip(
                                "Fade outward from the mask's actual shape --\n"
                                "the face outline when the crop is a hull --\n"
                                "rather than from its bounding box.\n\n"
                                "Also what makes the band an even width all the\n"
                                "way round: the box form measures distance as a\n"
                                "fraction of each half-extent, so a tall crop\n"
                                "fades over a longer distance vertically than\n"
                                "horizontally and flares at the corners, which\n"
                                "is the gradient pooling along the edges.");
                        }
                        ImGui::PushItemWidth(90);
                        ui::SliderFloat("fade starts", &g_fade_start, 0.f, 0.8f,
                                           "%.3f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip(
                                "How far out from the crop the fade begins, in\n"
                                "coord units (the frame is 2 tall). Push it out\n"
                                "to keep a clean margin of untouched pixels\n"
                                "around the subject before anything happens.");
                        }
                        ImGui::SameLine();
                        ui::SliderFloat("fade width", &g_fade_width, 0.01f, 1.5f,
                                           "%.3f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip(
                                "And how far it runs. Start and width are\n"
                                "separate because 'where the gradient sits' and\n"
                                "'how long it takes' are separate complaints --\n"
                                "a fade pinned to the edge pools against it\n"
                                "however wide you make it.");
                        }
                        ImGui::PopItemWidth();

                        ui::Checkbox("animate z outside", &g_z_free);
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip(
                                "Let the latent move everywhere except on the\n"
                                "subject, which stays pinned to the z the fit\n"
                                "was begun at.\n\n"
                                "z is an MLP input, not a colour knob: moving it\n"
                                "under a fitted network asks the same weights a\n"
                                "different question, and the face comes apart.\n"
                                "Pinning it inside the crop is what lets the\n"
                                "rest of the frame keep breathing.");
                        }
                        ImGui::BeginDisabled(!g_z_free);
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(90);
                        ui::SliderFloat("z rate /s", &P.z_rate, -2.f, 2.f);
                        ImGui::EndDisabled();

                        ImGui::SetNextItemWidth(110);
                        ui::SliderFloat("grey outside", &g_grey_out, 0.f, 1.f,
                                           "%.2f");
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip(
                                "Drain colour outside the crop: 1 leaves the\n"
                                "subject in colour on a greyscale field. Rides\n"
                                "the same falloff as the latent, so there is one\n"
                                "edge in the picture rather than two that nearly\n"
                                "agree.");
                        }
                        ImGui::EndDisabled();
                        if (!mirror.pond().fitted()) {
                            ImGui::TextDisabled("(these need a fit: there is no "
                                                "inside without one)");
                        } else if (!HaveCrop()) {
                            ImGui::TextDisabled("(these need a tracked face)");
                        }

                        if (open) {
                            bool mir = g_kinect.mirrored();
                            if (ui::Checkbox("mirror image", &mir)) g_kinect.setMirrored(mir);
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip(
                                    "A mirror should put your left hand on your\n"
                                    "left. The sensor does not.");
                            }
                        }
                        ImGui::Separator();
                    }
#endif
                    // Two tunings, switched by whether a crop is active. The
                    // live one is marked, because otherwise sliders that do
                    // nothing right now look broken rather than inactive.
                    const bool crop_live = g_have_mask;
                    for (int which = 0; which < 2; ++which) {
                        FitTune& T = which == 0 ? g_tune_crop : g_tune_full;
                        const bool active = (which == 0) == crop_live;
                        ImGui::PushID(which);
                        // Own section per row: the two rows carry the same
                        // labels, and without this they would be one parameter
                        // in the registry rather than two.
                        ui::PushSection(which == 0 ? "crop" : "feed");
                        if (active) {
                            ImGui::TextColored(ImVec4(0.6f, 1.f, 0.7f, 1.f), "%s",
                                               which == 0 ? "face crop  >" : "whole feed >");
                        } else {
                            ImGui::TextDisabled("%s",
                                which == 0 ? "face crop   " : "whole feed  ");
                        }
                        ImGui::SameLine();
                        ImGui::PushItemWidth(70);
                        ui::SliderInt("grid", &T.downscale, 1, 8);
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip(
                                "Fit resolution as a divisor of the display\n"
                                "size, so 1 is the full render grid and 4 is a\n"
                                "sixteenth of the area.\n\n"
                                "A step costs roughly 3x a render over the same\n"
                                "points -- but a crop is only a few percent of\n"
                                "those points, so it can afford a far finer grid\n"
                                "than the whole feed ever could, and a face is\n"
                                "where the detail has to go. The network is\n"
                                "continuous either way: whatever grid the\n"
                                "gradient came from, the result renders at full\n"
                                "size.");
                        }
                        ImGui::SameLine();
                        ui::SliderInt("steps", &T.steps, 1, 32);
                        ImGui::SameLine();
                        ui::SliderFloat("lr", &T.lr, 1e-4f, 2e-2f, "%.4f",
                                           ImGuiSliderFlags_Logarithmic);
                        ImGui::PopItemWidth();
                        ui::PopSection();
                        ImGui::PopID();
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "A crop is a few percent of the pixels, so a step\n"
                            "costs a few percent as much and many more fit in a\n"
                            "frame -- but each gradient sees far less data, so a\n"
                            "smaller step keeps it from chasing the crop box's\n"
                            "own jitter. The whole feed is the opposite. One\n"
                            "shared set of numbers meant every crop/no-crop\n"
                            "transition quietly changed what they meant.");
                    }
                    if (fit_w > 0) {
                        ImGui::TextDisabled("fit grid %d x %d  (%d px%s)", fit_w, fit_h,
                                            mirror.pond().fitPixels(),
                                            crop_live ? ", cropped" : "");
                    }

                    if (ImGui::Button(mirror.pond().fitting() ? "stop" : "fit")) {
                        if (mirror.pond().fitting()) {
                            mirror.pond().stopFit();
                        } else if (g_fit_live &&
                                   live_rgb.size() != size_t(fit_w) * fit_h * 3) {
                            // Armed but nothing has arrived yet. Falling back to
                            // the still here would silently fit the photo and
                            // then look like the camera fit had failed.
                            fit_err = "no camera frame yet";
                        } else if (g_fit_live) {
                            // Start on the frame the camera is showing right
                            // now, cropped the same way the per-frame retarget
                            // will crop it -- beginFit sizes the optimiser to
                            // the pixel set, so starting unmasked and narrowing
                            // a frame later would rebuild it immediately.
                            fit_err.clear();
                            if (g_have_mask) {
                                mirror.pond().beginFit(live_rgb, fit_h, fit_w, P,
                                                       g_fit_mask);
                            } else {
                                mirror.pond().beginFit(live_rgb, fit_h, fit_w, P);
                            }
                        } else {
                            // The still-image path has no crop, so it is the
                            // whole-feed grid that applies.
                            const int ds = std::max(1, g_tune_full.downscale);
                            const int fw = std::max(8, mirror.lowW() / ds);
                            const int fh = std::max(8, mirror.lowH() / ds);
                            std::vector<float> rgb;
                            fit_err.clear();
                            if (LoadImageRGB(fit_path, fw, fh, rgb, fit_err)) {
                                mirror.pond().beginFit(rgb, fh, fw, P);
                            }
                        }
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            g_fit_live
                                ? "Fit the live feed -- the face crop when one\n"
                                  "is tracked, the whole frame otherwise."
                                : "Fit the still image above. Arm 'track live\n"
                                  "feed' to fit the camera instead.");
                    }
                    ImGui::SameLine();
                    ImGui::BeginDisabled(!mirror.pond().fitted());
                    if (ImGui::Button("clear fit")) mirror.pond().clearFit();
                    ImGui::EndDisabled();
                    ImGui::SameLine();
                    ImGui::TextDisabled("(clear returns to the generated field)");

                    if (!fit_err.empty()) {
                        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 140, 120, 255));
                        ImGui::TextWrapped("%s", fit_err.c_str());
                        ImGui::PopStyleColor();
                    }
                    if (mirror.pond().fitted()) {
                        ImGui::TextDisabled(
                            "weights are learned: detail / contrast / tilt no "
                            "longer apply");
                    }
                }

                ImGui::Separator();
                // --- the network itself -----------------------------------
                ui::PushSection("network");
                if (ImGui::CollapsingHeader("network")) {
                ui::SliderInt("sine layers (0 = tanh only)", &P.sine_layers,
                                 0, 5);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "SIREN sine activations on the leading hidden layers,\n"
                        "tanh behind them. One layer is enough: measured on a\n"
                        "face fit, 1 sine layer scores 0.00250 against 0.00273\n"
                        "for all-sine and 0.00562 for all-tanh.\n\n"
                        "Changing this rebuilds the weights (sine layers are\n"
                        "SIREN-initialised) and recompiles the kernel.");
                }
                ImGui::BeginDisabled(P.sine_layers == 0);
                ui::SliderFloat("sine w0 (composition)", &P.sine_w0, 1.0f, 60.0f,
                                   "%.1f", ImGuiSliderFlags_Logarithmic);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "How many regions the field breaks into. Low (2-10)\n"
                        "gives large open areas with detail only at the\n"
                        "boundaries; past ~40 the frame is uniform texture with\n"
                        "no background left.\n\n"
                        "Pairs with 'detail' below, which sets how hard those\n"
                        "boundaries are without changing the layout.");
                }
                ImGui::EndDisabled();
                ImGui::Separator();
                // weight shaping
                ui::SliderFloat("detail (w hidden)", &P.detail, 0.5f, 10.0f);
                ui::SliderFloat("gain tilt (front<->back)", &P.gain_tilt, -3.0f, 3.0f);
                ui::SliderFloat("w shape (gauss<->uniform)", &P.uniform_mix, 0.0f, 1.0f);
                ui::SliderFloat("contrast (w out)", &P.contrast, 1.0f, 12.0f);
                if (ImGui::Button("reseed network")) mirror.reseed();
                }
                ui::PopSection();

                // --- colour & tone ----------------------------------------
                ui::PushSection("colour");
                if (ImGui::CollapsingHeader("colour & tone")) {
                ui::Checkbox("sRGB fix", &P.srgb_fix); ImGui::SameLine();
                if (ImGui::Button("reset color")) { P.srgb_fix = false; P.gamma = 1.0f; }
                ui::SliderFloat("gamma (>1 darkens)", &P.gamma, 0.3f, 2.0f);
                ui::SliderFloat("color mix (0 grey -> 1 RGB)", &P.color_mix, 0.0f, 1.0f);
                ImGui::SameLine(); ImGui::SetNextItemWidth(90);
                const char* greyItems[] = {"R", "G", "B"};
                ImGui::Combo("grey ch", &P.grey_channel, greyItems, 3);
                ui::Checkbox("ripple amp -> color", &P.amp_drives_color);
                if (P.amp_drives_color) {
                    ImGui::SameLine(); ImGui::SetNextItemWidth(120);
                    ui::SliderFloat("amp gain", &P.amp_gain, 0.2f, 6.0f);
                }
                ui::Checkbox("swap R/B", &P.swap_rb);
                ui::Checkbox("color travel (palette follows orbit)", &P.color_travel);
                }
                ui::PopSection();

                // --- the z latent -----------------------------------------
                ui::PushSection("z");
                if (ImGui::CollapsingHeader("z latent")) {
                // z latent
                ImGui::Text("z phase = %6.2f  (circular morph)", P.z);
                ImGui::DragFloat("z", &P.z, 0.02f);
                ui::SliderFloat("z amplitude", &P.z_amp, 0.0f, 3.0f);
                ui::SliderFloat("z auto-rate /s", &P.z_rate, -2.0f, 2.0f);
                ui::SliderFloat("z step size", &P.z_step, 0.01f, 1.0f);
                if (ImGui::Button("z - step")) P.z -= P.z_step; ImGui::SameLine();
                if (ImGui::Button("z + step")) P.z += P.z_step; ImGui::SameLine();
                if (ImGui::Button("z = 0")) P.z = 0.0f;
                }
                ui::PopSection();

                // --- clock & render ---------------------------------------
                ui::PushSection("render");
                if (ImGui::CollapsingHeader("clock & render")) {
                // time
                ui::SliderFloat("ripple time scale", &P.time_scale, 0.0f, 4.0f);
                ui::Checkbox("pause", &P.paused);
                ui::SliderInt("downscale", &downscale, 1, 10);
                ImGui::Text("render %d x %d -> %d x %d", mirror.lowW(), mirror.lowH(), fbw, fbh);
                }
                ui::PopSection();
                // mask emergence transition
                ui::PushSection("mask emergence (transition)");
                if (ImGui::CollapsingHeader("mask emergence (transition)")) {
                    ui::SliderFloat("transition (0 pond -> 1 mask)", &P.transition, 0.0f, 1.0f);
                    ui::Checkbox("auto-play", &P.trans_auto); ImGui::SameLine();
                    if (ImGui::Button("reset t")) { P.transition = 0.0f; P.trans_auto = false; }
                    ui::SliderFloat("play rate /s", &P.trans_rate, 0.05f, 1.0f);
                    ui::SliderFloat("relief height", &P.relief_h, 0.0f, 1.5f);
                    ui::SliderFloat("mask width", &P.mask_ax, 0.2f, 1.0f);
                    ui::SliderFloat("mask height", &P.mask_ay, 0.2f, 1.2f);
                    ui::SliderFloat("light azimuth", &P.light_az, -(float)M_PI, (float)M_PI);
                    ui::SliderFloat("light elevation", &P.light_elev, 0.1f, (float)M_PI / 2.0f);
                    ui::SliderFloat("wet sheen (spec)", &P.spec_amt, 0.0f, 1.5f);
                    ui::SliderFloat("sheen tightness", &P.shininess, 4.0f, 96.0f);
                    ui::SliderFloat("background dim", &P.bg_dim, 0.0f, 1.0f);
                }
                ui::PopSection();
                ui::PopSection();          // "mirror"
            } else {
                ui::PushSection("roots");
                MetalRootRenderer& R = roots.renderer();
                ImGui::Text("%.0f fps   t=%5.1fs", fpsShown, roots.clock());
                ImGui::Text("render %d x %d -> %d x %d  (overdraw-bound)",
                            roots.width(), roots.height(), fbw, fbh);
                ui::Checkbox("auto render-scale", &rootAutoScale);
                if (rootAutoScale) {
                    ImGui::SameLine(); ImGui::SetNextItemWidth(120);
                    ui::SliderInt("target px", &rootTargetDim, 720, 3840);
                } else {
                    ui::SliderInt("root downscale", &rootDownscale, 1, 6);
                }
                ImGui::Separator();

                // --- growth ------------------------------------------------
                ui::PushSection("growth");
                if (ImGui::CollapsingHeader("growth", ImGuiTreeNodeFlags_DefaultOpen)) {
                    rootsim::SimParams& SP = roots.simParams();
                    ImGui::Text("%s", roots.simActive()
                                    ? (roots.simDone() ? "grown" : "growing")
                                    : "stand-in (no CPlantBox parameters)");

                    const auto& sp = RootScene::species();
                    int si = roots.speciesIndex();
                    ImGui::PushItemWidth(-90);
                    if (ImGui::BeginCombo("species",
                                          si >= 0 ? sp[size_t(si)].first.c_str()
                                                  : SP.speciesXml.c_str())) {
                        for (int i = 0; i < (int)sp.size(); ++i) {
                            const bool selected = (i == si);
                            if (ImGui::Selectable(sp[size_t(i)].first.c_str(), selected))
                                roots.setSpeciesIndex(i);
                            if (selected) ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::PopItemWidth();

                    ImGui::PushItemWidth(110);
                    ui::SliderInt("masks", &SP.N, 1, 24);
                    ImGui::SameLine();
                    ui::SliderFloat("cone radius", &SP.R0, 6.f, 24.f, "%.1f cm");
                    ui::SliderFloat("cone height", &SP.Hh, 24.f, 96.f, "%.1f cm");
                    ImGui::SameLine();
                    ui::SliderFloat("taper", &SP.taperPower, 0.4f, 2.5f);
                    ui::SliderFloat("spiral x golden", &SP.angleStepGoldenMult,
                                       0.2f, 2.0f);
                    ImGui::SameLine();
                    ui::SliderFloat("jitter", &SP.sigma, 0.f, 1.2f);
                    ui::SliderFloat("travel pull", &SP.weight, 0.f, 1.f);
                    ImGui::SameLine();
                    ui::SliderFloat("pull reach", &SP.travelPullReach, 0.4f, 3.f);
                    ui::SliderFloat("lateral", &SP.lateralWeight, 0.f, 1.f);
                    ImGui::SameLine();
                    ui::SliderFloat("dwell", &SP.dwellWeight, 0.f, 1.f);
                    ui::SliderFloat("dwell days", &SP.dwellDays, 2.f, 60.f);
                    ImGui::SameLine();
                    ui::SliderFloat("hop days", &SP.maxHopDays, 10.f, 160.f);
                    ImGui::PopItemWidth();

                    ui::Checkbox("crawl the cone surface", &SP.coneSurfaceTravel);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Confine the travelling root to a thin shell around\n"
                            "the cone the masks sit on, so it crawls over the\n"
                            "surface between them instead of cutting through the\n"
                            "interior.\n\n"
                            "Travel only: the dwell wrapping stays free, or the\n"
                            "nests around each mask would be flattened onto the\n"
                            "surface instead of bulging into 3D.");
                    }
                    if (SP.coneSurfaceTravel) {
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(90);
                        ui::SliderFloat("shell", &SP.coneShellThickness, 1.f, 20.f,
                                           "%.1f cm");
                    }

                    ImGui::SetNextItemWidth(110);
                    ui::SliderFloat("days / step", &SP.growthDt, 0.05f, 3.f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Read live, so it takes effect mid-grow. Not a\n"
                            "structural knob -- it does not need a regrow.");
                    }
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(90);
                    ui::SliderInt("steps/frame", &roots.simStepsPerFrame, 1, 30);

                    if (ImGui::Button("regrow")) roots.regrow();
                    ImGui::SameLine();
                    if (ImGui::Button("reseed")) roots.reseed((uint32_t)(++rootSeed));
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "A new random seed for the same parameters. Reseeds\n"
                            "the growth itself -- it used to drop a synthetic\n"
                            "stand-in structure over a running grow, which the\n"
                            "next frame then overwrote.");
                    }
                    ImGui::SameLine();
                    ImGui::TextDisabled("seed %u", SP.seed);
                }
                ui::PopSection();

                // --- presets -----------------------------------------------
                ui::PushSection("presets");
                if (ImGui::CollapsingHeader("presets")) {
                    static std::vector<std::string> presets = RootScene::listPresets();
                    static char preset_name[128] = "untitled";
                    static std::string preset_msg;
                    ImGui::PushItemWidth(-90);
                    if (ImGui::BeginCombo("load", "choose...")) {
                        for (const std::string& nm : presets) {
                            if (ImGui::Selectable(nm.c_str())) {
                                if (roots.loadConfig(RootScene::presetDir() + "/" +
                                                     nm + ".root")) {
                                    snprintf(preset_name, sizeof(preset_name), "%s",
                                             nm.c_str());
                                    roots.regrow();
                                    preset_msg = "loaded " + nm;
                                } else {
                                    preset_msg = "could not read " + nm;
                                }
                            }
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::InputText("name", preset_name, sizeof(preset_name));
                    ImGui::PopItemWidth();
                    if (ImGui::Button("save")) {
                        const std::string path = RootScene::presetDir() + "/" +
                                                 preset_name + ".root";
                        preset_msg = roots.saveConfig(path) ? ("saved " + path)
                                                            : ("could not write " + path);
                        presets = RootScene::listPresets();
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("rescan")) presets = RootScene::listPresets();
                    if (!preset_msg.empty()) ImGui::TextDisabled("%s", preset_msg.c_str());
                }
                ui::PopSection();

                // --- camera ------------------------------------------------
                ui::PushSection("camera");
                if (ImGui::CollapsingHeader("camera", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ui::Checkbox("frame automatically", &roots.autoFrame);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Derive the target and distance from the scene's own\n"
                            "bounds. The constants this replaced were tuned to\n"
                            "one cone size and pointed at the wrong part of any\n"
                            "other.");
                    }
                    if (roots.autoFrame) {
                        const int nm = roots.maskCount();
                        std::string label = roots.focusMask >= 0 && roots.focusMask < nm
                                                ? ("mask " + std::to_string(roots.focusMask))
                                                : std::string("whole scene");
                        ImGui::PushItemWidth(-90);
                        if (ImGui::BeginCombo("focus", label.c_str())) {
                            if (ImGui::Selectable("whole scene", roots.focusMask < 0))
                                roots.focusMask = -1;
                            for (int i = 0; i < nm; ++i) {
                                const std::string it = "mask " + std::to_string(i);
                                if (ImGui::Selectable(it.c_str(), roots.focusMask == i))
                                    roots.focusMask = i;
                            }
                            ImGui::EndCombo();
                        }
                        ImGui::PopItemWidth();
                        ImGui::SetNextItemWidth(110);
                        ui::SliderFloat("zoom", &roots.zoom, 0.15f, 5.f, "%.2fx");
                        ImGui::SameLine();
                        if (ImGui::SmallButton("reset zoom")) roots.zoom = 1.f;
                    }
                    ui::Checkbox("auto-orbit", &roots.autoOrbit); ImGui::SameLine();
                    ImGui::SetNextItemWidth(120);
                    ui::SliderFloat("orbit rate", &roots.orbitRate, -1.0f, 1.0f);
                    ImGui::BeginDisabled(roots.autoFrame);
                    ui::SliderFloat("radius", &roots.radius, 5.0f, 120.0f);
                    ImGui::EndDisabled();
                    ui::SliderFloat("azimuth", &roots.azimuth, -(float)M_PI, (float)M_PI);
                    ui::SliderFloat("elevation", &roots.elevation, -1.5f, 1.5f);
                    ui::SliderFloat("fov", &roots.fov, 0.2f, 1.2f);
                }
                ui::PopSection();
                ImGui::Separator();
                ImGui::Separator();
                // shading
                ui::PushSection("material");
                const char* modes[] = {"Phong", "PBR", "Invert (approx)"};
                int sm = (int)R.shaderMode;
                if (ImGui::Combo("shader", &sm, modes, 3)) R.shaderMode = (MetalRootRenderer::ShaderMode)sm;
                ui::ColorEdit3("base color", R.mat.baseColor);
                ui::ColorEdit3("base color 2", R.mat.baseColor2);
                ui::SliderFloat("color noise", &R.mat.colorNoiseStrength, 0.0f, 1.0f);
                ui::SliderFloat("ambient", &R.mat.ambient, 0.0f, 0.5f);
                ui::SliderFloat("diffuse", &R.mat.diffuse, 0.0f, 1.5f);
                ui::SliderFloat("shininess", &R.mat.shininess, 4.0f, 300.0f);
                if (sm == 1) {
                    ui::SliderFloat("metallic", &R.pbr.metallic, 0.0f, 1.0f);
                    ui::SliderFloat("roughness", &R.pbr.roughness, 0.05f, 1.0f);
                }
                ui::SliderFloat("radius scale", &R.radiusScale, 0.2f, 4.0f);
                ui::PopSection();           // "material"
                ImGui::Separator();
                // fog
                ui::PushSection("fog & atmosphere");
                if (ImGui::CollapsingHeader("fog & atmosphere")) {
                    ui::ColorEdit3("fog color", R.fog.color);
                    ui::SliderFloat("fog density", &R.fog.density, 0.0f, 0.06f);
                    ui::SliderFloat("fog falloff", &R.fog.falloff, 0.0f, 0.3f);
                    ui::SliderFloat("fog noise", &R.fog.noiseStrength, 0.0f, 1.0f);
                    ui::SliderFloat("fog refDist", &R.fog.refDist, 0.0f, 120.0f);
                    ui::SliderFloat("wisp glow", &R.wispGlowStrength, 0.0f, 3.0f);
                    ui::SliderInt("wisps", &R.wispCount, 0, 8);
                }
                ui::PopSection();
                // pulses
                ui::PushSection("travelling pulses");
                if (ImGui::CollapsingHeader("travelling pulses")) {
                    ui::Checkbox("pulses on", &R.pulse.enabled);
                    ui::SliderFloat("pulse speed", &R.pulse.speed, 0.0f, 40.0f);
                    ui::SliderFloat("pulse spacing", &R.pulse.spacing, 4.0f, 60.0f);
                    ui::SliderFloat("pulse width", &R.pulse.width, 0.5f, 12.0f);
                    ui::SliderFloat("pulse intensity", &R.pulse.intensity, 0.0f, 4.0f);
                    ui::ColorEdit3("pulse color", R.pulse.color);
                }
                ui::PopSection();
                ui::PushSection("face masks");
                if (ImGui::CollapsingHeader("face masks")) {
                    if (ui::Checkbox("show faces", &roots.showFace)) roots.rebuildFace();
                    if (ui::SliderFloat("face scale", &roots.faceScale, 0.3f, 1.5f))
                        roots.rebuildFace();
                    ui::SliderFloat("face light", &R.face.lightIntensity, 0.0f, 8.0f);
                    ui::SliderFloat("face falloff", &R.face.lightFalloff, 0.001f, 0.1f);
                    ui::SliderFloat("face spec", &R.face.specStrength, 0.0f, 3.0f);
                    ui::ColorEdit3("vein color", R.face.veinColor);
                    ui::SliderFloat("vein scale", &R.face.veinScale, 0.1f, 2.0f);
                    ui::SliderFloat("vein strength", &R.face.veinStrength, 0.0f, 1.0f);
                }
                ui::PopSection();
                ui::PushSection("cached field: LOD & culling");
                if (ImGui::CollapsingHeader("cached field: LOD & culling")) {
                    ui::SliderInt("grid NxN", &fieldGrid, 2, 20);
                    if (ImGui::Button("tile field")) roots.buildField(fieldGrid, 30.0f);
                    ImGui::SameLine();
                    if (ImGui::Button("clear field")) { R.clearInstances(); roots.regrow(); }
                    ui::Checkbox("frustum cull", &R.cullInstances); ImGui::SameLine();
                    ui::Checkbox("sub-pixel cull", &R.subpixelCull);
                    ui::SliderFloat("cull below px", &R.instanceCullPx, 0.5f, 20.0f);
                    ui::SliderFloat("LOD bias (>1 coarser)", &R.lodBias, 0.1f, 4.0f);
                    ImGui::Text("instances %d   visible %d   culled %d",
                                R.instanceCount(), R.lastVisibleInstances, R.lastCulledInstances);
                    ImGui::Text("capsules drawn: %ld", R.lastDrawnSegments);
                }
                ui::PopSection();
                ui::PushSection("overlays");
                if (ImGui::CollapsingHeader("overlays")) {
                    ui::Checkbox("axes", &R.overlay.showAxes); ImGui::SameLine();
                    ui::Checkbox("grid", &R.overlay.showGrid);
                    ui::SliderFloat("grid spacing", &R.overlay.gridSpacing, 1.0f, 20.0f);
                }
                ui::PopSection();
                ui::PopSection();          // "roots"
            }

            // --- settings: MIDI and presets -------------------------------
            //
            // Last in the panel on purpose. It is the section that is set up
            // once and then left alone, and putting it above the controls it
            // binds would push those further from the top every session.
            ImGui::Separator();
            if (ImGui::CollapsingHeader("settings")) {
                ImGui::Text("MIDI");
                ImGui::SameLine();
                if (g_midi.isOpen()) {
                    ImGui::TextColored(ImVec4(0.6f, 1.f, 0.7f, 1.f), "%s",
                                       g_midi.deviceInfo().c_str());
                } else {
                    ImGui::TextDisabled("closed");
                }
                ImGui::SameLine();
                ImGui::TextDisabled("| %llu msgs",
                                    (unsigned long long)g_midi.received());

                if (ImGui::Button(g_midi.isOpen() ? "close MIDI" : "open MIDI")) {
                    if (g_midi.isOpen()) {
                        g_midi.close();
                    } else {
                        std::string merr;
                        if (!g_midi.open(merr)) g_midi_err = merr;
                        else g_midi_err.clear();
                    }
                }
                ImGui::SameLine();
                if (ImGui::Button("rescan devices")) g_midi.rescan();
                ImGui::SameLine();
                ImGui::TextDisabled("%d bound", ui::BindingCount());
                if (!g_midi_err.empty())
                    ImGui::TextColored(ImVec4(1.f, 0.5f, 0.5f, 1.f), "%s",
                                       g_midi_err.c_str());

                if (!ui::LearnTarget().empty()) {
                    ImGui::TextColored(ImVec4(1.f, 0.85f, 0.4f, 1.f),
                                       "learning: %s -- move a control",
                                       ui::LearnTarget().c_str());
                    ImGui::SameLine();
                    if (ImGui::SmallButton("cancel")) ui::SetLearnTarget("");
                } else {
                    ImGui::TextDisabled("right-click any slider to bind it");
                }

                // The last few messages, so a controller that is sending
                // something other than what you expect can be seen doing it --
                // "nothing is bound" and "nothing is arriving" look identical
                // otherwise.
                if (ImGui::TreeNode("incoming")) {
                    const auto& r = ui::RecentMessages();
                    if (r.empty()) ImGui::TextDisabled("(nothing yet)");
                    for (const auto& m : r)
                        ImGui::TextDisabled("ch %2d  cc %3d  %3d", m.channel + 1,
                                            m.cc, m.value);
                    ImGui::TreePop();
                }
                if (ImGui::TreeNode("bindings")) {
                    struct Row { std::string path; int ch, cc; };
                    static std::vector<Row> rows;
                    rows.clear();
                    ui::ForEachBinding(&rows, [](void* u, const char* p, int ch, int cc) {
                        static_cast<std::vector<Row>*>(u)->push_back({p, ch, cc});
                    });
                    if (rows.empty()) ImGui::TextDisabled("(none)");
                    for (const Row& r : rows) {
                        ImGui::PushID(r.path.c_str());
                        if (ImGui::SmallButton("x")) ui::ClearBinding(r.path);
                        ImGui::SameLine();
                        ImGui::TextDisabled("ch%d cc%-3d  %s", r.ch + 1, r.cc,
                                            r.path.c_str());
                        ImGui::PopID();
                    }
                    if (!rows.empty() && ImGui::SmallButton("clear all"))
                        ui::ClearAllBindings();
                    ImGui::TreePop();
                }

                ImGui::Separator();
                ImGui::Text("settings presets");
                ImGui::SameLine();
                ImGui::TextDisabled("(%d params)", ui::DeclaredCount());
                static std::vector<std::string> sets = ui::ListPresets();
                static char set_name[128] = "default";
                static std::string set_msg;
                ImGui::PushItemWidth(-90);
                if (ImGui::BeginCombo("load##set", "choose...")) {
                    for (const std::string& nm : sets) {
                        if (ImGui::Selectable(nm.c_str())) {
                            std::string e;
                            if (ui::LoadPreset(ui::PresetDir() + "/" + nm + ".set", e)) {
                                snprintf(set_name, sizeof(set_name), "%s", nm.c_str());
                                set_msg = "loaded " + nm;
                            } else {
                                set_msg = e;
                            }
                        }
                    }
                    ImGui::EndCombo();
                }
                ImGui::InputText("name##set", set_name, sizeof(set_name));
                ImGui::PopItemWidth();
                if (ImGui::Button("save##set")) {
                    std::string e;
                    set_msg = ui::SavePreset(ui::PresetDir() + "/" + set_name + ".set", e)
                                  ? ("saved " + std::string(set_name))
                                  : e;
                    sets = ui::ListPresets();
                }
                ImGui::SameLine();
                if (ImGui::Button("rescan##set")) sets = ui::ListPresets();
                if (!set_msg.empty()) ImGui::TextDisabled("%s", set_msg.c_str());
                // A value only reaches a control when that control next draws,
                // so anything inside a collapsed header is applied the moment
                // it is opened. Saying so beats it looking like a partial load.
                ImGui::TextDisabled("collapsed sections apply when opened");
                if (!ui::UnclaimedKeys().empty()) {
                    ImGui::TextColored(ImVec4(1.f, 0.85f, 0.4f, 1.f),
                                       "%zu key(s) no control claimed",
                                       ui::UnclaimedKeys().size());
                    if (ImGui::IsItemHovered()) {
                        std::string t;
                        for (const std::string& k : ui::UnclaimedKeys()) t += k + "\n";
                        ImGui::SetTooltip("%s", t.c_str());
                    }
                }
            }
            ImGui::End();

            // --- camera mask: the rectangle, on the frame ------------------
            //
            // Drawn over the scene rather than in the control panel: the mask
            // is a piece of set dressing aimed at a real room, and placing it
            // by numbers in a list means looking away from the thing being
            // aimed at.
            if (scene == (int)Scene::CamMask) {
                const ImGuiViewport* vp = ImGui::GetMainViewport();
                ImGui::SetNextWindowPos(vp->WorkPos);
                ImGui::SetNextWindowSize(vp->WorkSize);
                ImGui::Begin("##cammask_overlay", nullptr,
                             ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoBackground |
                             ImGuiWindowFlags_NoFocusOnAppearing);
                ImDrawList* dl = ImGui::GetWindowDrawList();
                const ImVec2 o = vp->WorkPos, sz = vp->WorkSize;
                auto toScreen = [&](float u, float v) {
                    return ImVec2(o.x + u * sz.x, o.y + v * sz.y);
                };

                float* xs[2] = {&g_cam_x0, &g_cam_x1};
                float* ys[2] = {&g_cam_y0, &g_cam_y1};
                const ImU32 col = g_cam_mask_on ? IM_COL32(255, 210, 120, 230)
                                                : IM_COL32(150, 150, 150, 140);
                dl->AddRect(toScreen(*xs[0], *ys[0]), toScreen(*xs[1], *ys[1]), col,
                            0.f, 0, 2.f);

                // One handle per corner. Corners rather than edges because a
                // rectangle has four degrees of freedom and four handles is the
                // fewest that reach all of them without a mode.
                const float grab = 12.f;
                for (int i = 0; i < 4; ++i) {
                    float* px = xs[i & 1];
                    float* py = ys[i >> 1];
                    const ImVec2 c = toScreen(*px, *py);
                    ImGui::SetCursorScreenPos(ImVec2(c.x - grab, c.y - grab));
                    ImGui::PushID(i);
                    ImGui::InvisibleButton("h", ImVec2(grab * 2, grab * 2));
                    const bool hot = ImGui::IsItemHovered() || ImGui::IsItemActive();
                    if (ImGui::IsItemActive()) {
                        const ImVec2 d = ImGui::GetIO().MouseDelta;
                        *px = std::min(1.f, std::max(0.f, *px + d.x / sz.x));
                        *py = std::min(1.f, std::max(0.f, *py + d.y / sz.y));
                    }
                    dl->AddCircleFilled(c, hot ? 8.f : 5.f, col);
                    ImGui::PopID();
                }

                // Drag the body to move the whole rectangle.
                const ImVec2 a = toScreen(std::min(g_cam_x0, g_cam_x1),
                                          std::min(g_cam_y0, g_cam_y1));
                const ImVec2 b = toScreen(std::max(g_cam_x0, g_cam_x1),
                                          std::max(g_cam_y0, g_cam_y1));
                ImGui::SetCursorScreenPos(ImVec2(a.x + grab, a.y + grab));
                ImGui::InvisibleButton("##body",
                                       ImVec2(std::max(1.f, b.x - a.x - grab * 2),
                                              std::max(1.f, b.y - a.y - grab * 2)));
                if (ImGui::IsItemActive()) {
                    const ImVec2 d = ImGui::GetIO().MouseDelta;
                    const float du = d.x / sz.x, dv = d.y / sz.y;
                    g_cam_x0 += du; g_cam_x1 += du;
                    g_cam_y0 += dv; g_cam_y1 += dv;
                }
                ImGui::End();
            }

            // --- source overlay: the raw frame, in a corner ----------------
            //
            // Answers one question the mirror's own output cannot: is anything
            // actually arriving, and is it what the fit is being pointed at.
            if (g_show_source && srcTex && srcTexW > 0) {
                const ImGuiViewport* vp = ImGui::GetMainViewport();
                const float pad = 16.f;
                const bool right = (g_source_corner == 1 || g_source_corner == 3);
                const bool bottom = (g_source_corner == 2 || g_source_corner == 3);
                ImGui::SetNextWindowPos(
                    ImVec2(vp->WorkPos.x + (right ? vp->WorkSize.x - pad : pad),
                           vp->WorkPos.y + (bottom ? vp->WorkSize.y - pad : pad)),
                    ImGuiCond_Always,
                    ImVec2(right ? 1.f : 0.f, bottom ? 1.f : 0.f));
                ImGui::SetNextWindowBgAlpha(0.35f);
                ImGui::Begin("##source_pip", nullptr,
                             ImGuiWindowFlags_NoDecoration |
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoFocusOnAppearing |
                             ImGuiWindowFlags_NoNav);
                const float iw = (float)g_source_pip_w;
                const float ih = iw * (float)srcTexH / (float)srcTexW;
                const ImVec2 p0 = ImGui::GetCursorScreenPos();
                ImGui::Image((ImTextureID)(intptr_t)(__bridge void*)srcTex,
                             ImVec2(iw, ih));
                // Landmarks on top, in the overlay's own coordinates: they are
                // normalised, so this is the same mapping the mask uses -- if
                // they sit off the face here, they sit off the face there.
                if (g_pip_landmarks && g_track_on && g_face.valid) {
                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    for (const mirror::FaceLandmark& L : g_face.landmarks) {
                        dl->AddRectFilled(
                            ImVec2(p0.x + L.x * iw, p0.y + L.y * ih),
                            ImVec2(p0.x + L.x * iw + 1.5f, p0.y + L.y * ih + 1.5f),
                            IM_COL32(120, 255, 170, 200));
                    }
                    // The crop the fit is supervised on, drawn where its pixels
                    // are *taken from* -- so in the centred mode this stays on
                    // the head even though the fit places it in the middle.
                    // Drawn from the smoothed box the mask is built from, which
                    // makes the smoothing itself visible: if this lags the face
                    // badly, that is the setting to turn up.
                    if (g_mask_fit && g_head_valid &&
                        g_mask_shape == (int)MaskShape::Box) {
                        const float px = fit_w > 0 ? float(g_mask_dilate) / fit_w : 0.f;
                        const float py = fit_h > 0 ? float(g_mask_dilate) / fit_h : 0.f;
                        dl->AddRect(
                            ImVec2(p0.x + (g_head_cx - g_head_hx - px) * iw,
                                   p0.y + (g_head_cy - g_head_hy - py) * ih),
                            ImVec2(p0.x + (g_head_cx + g_head_hx + px) * iw,
                                   p0.y + (g_head_cy + g_head_hy + py) * ih),
                            IM_COL32(255, 210, 120, 220));
                    }
                }
                const bool live = g_fit_live && mirror.pond().fitting() &&
                                  scene == (int)Scene::Mirror;
                ImGui::TextDisabled("%s%s", g_source == (int)Source::Photo
                                                ? "photo" : "sensor",
                                    srcFresh ? "" : "  (stale)");
                ImGui::SameLine();
#if MIRROR_HAVE_KINECT
                if (g_source == (int)Source::Kinect) {
                    ImGui::TextDisabled("| %llu frames",
                                        (unsigned long long)g_kinect.frames());
                    ImGui::SameLine();
                }
#endif
                if (live) {
                    ImGui::TextColored(ImVec4(0.6f, 1.f, 0.7f, 1.f), "| fitting this");
                } else {
                    ImGui::TextDisabled("| not the fit target");
                }
                ImGui::End();
            }

            ImGui::Render();

            // The text refracts through the ripples the mirror was *just*
            // rendered with, so the sources come from the pond rather than
            // being rebuilt from the clock here. In the other scenes there are
            // none, and the field is left unwarped and simply crisp.
            mirror::TextRipple tr;
            if (scene == (int)Scene::Mirror) {
                const mirror::PondParams& P = mirror.params();
                tr.k = P.ring_freq;
                tr.decay = P.decay;
                tr.core_r2 = P.core_rolloff ? P.core_radius * P.core_radius : 0.f;
                const auto& srcs = mirror.pond().lastSources();
                tr.n = int(std::min(srcs.size(), size_t(16)));
                for (int i = 0; i < tr.n; ++i)
                    for (int j = 0; j < 4; ++j) tr.src[i][j] = srcs[i][j];
            }
            text.update(textp);
            const float texAsp =
                sceneTex ? float(sceneTex.width) / float(sceneTex.height) : 1.f;
            const mirror::TextUniforms tu =
                text.uniforms(textp, texAsp, tr, glfwGetTime());

            id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rpd];
            if (sceneTex) present.encode(re, sceneTex, text.texture(), tu);
            ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), cb, re);
            [re endEncoding];

            [cb presentDrawable:drawable];
            [cb commit];
        }

        double now = glfwGetTime();
        fpsAccum += now - lastTime; lastTime = now; fpsFrames++;
        if (fpsAccum >= 0.5) { fpsShown = fpsFrames / fpsAccum; fpsAccum = 0; fpsFrames = 0; }
    }

    ImGui_ImplMetal_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
