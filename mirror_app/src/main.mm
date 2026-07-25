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
#if MIRROR_HAVE_KINECT
#include "kinect_target.h"
#endif
#include "root_scene.h"
#include "fullscreen_present.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
static int growshot(const char* path, int steps, float az, float el, float rad) {
    MetalContext ctx;
    if (!ctx.device()) { fprintf(stderr, "growshot: no Metal device\n"); return 1; }
    const int W = 960, H = 540;
    RootScene roots(ctx, W, H);
    if (!roots.valid()) { fprintf(stderr, "growshot: root scene invalid\n"); return 1; }
    printf("growshot: sim active = %d\n", roots.simActive() ? 1 : 0);
    roots.autoOrbit = false;
    if (rad > 0) roots.radius = rad;
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
int   g_fit_steps_per_frame = 1;
float g_fit_lr = 3e-3f;
int   g_fit_downscale = 2;      // fit grid = display size / this
bool  g_fit_live = false;       // retarget from the camera every frame
#if MIRROR_HAVE_KINECT
mirror::KinectFitTarget g_kinect;
#endif

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
        CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), cg);
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

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--selftest") return selftest();
        if (a == "--roottest") return roottest();
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
            return growshot(path, steps, az, el, rad);
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
    FullscreenPresent present(ctx, std::string(MIRROR_APP_SHADER_DIR) + "/present.metal",
                              layer.pixelFormat);

    enum class Scene { Mirror = 0, Roots = 1 };
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
            id<MTLTexture> sceneTex = nil;
            if (scene == (int)Scene::Mirror && mirror.valid()) {
                mirror.ensureSize(fbw / std::max(1, downscale), fbh / std::max(1, downscale));
                mirror.advance(dt);
                // Training runs here, not inside render(): one place, once per
                // frame, so the cost is attributable and the displayed frame is
                // always the post-step state.
                if (mirror.pond().fitting()) {
#if MIRROR_HAVE_KINECT
                    // Live feed: swap the target, keeping weights and Adam
                    // state. beginFit() here would reset the optimiser every
                    // frame and the fit would never build enough momentum to
                    // follow motion (measured 677x worse tracking error).
                    if (g_fit_live && g_kinect.isOpen()) {
                        static std::vector<float> live_rgb;
                        const int fw = std::max(8, mirror.lowW() / g_fit_downscale);
                        const int fh = std::max(8, mirror.lowH() / g_fit_downscale);
                        if (g_kinect.poll(fw, fh, live_rgb)) {
                            mirror.pond().updateFitTarget(live_rgb, fh, fw);
                        }
                    }
#endif
                    mirror.fitSteps(g_fit_steps_per_frame, g_fit_lr);
                }
                sceneTex = mirror.render();
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
                roots.ensureSize(fbw / effDs, fbh / effDs);
                roots.advance(dt);
                sceneTex = roots.render(cb);   // encodes geometry + fog passes into cb
            }

            MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
            rpd.colorAttachments[0].texture = drawable.texture;
            rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
            rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.05, 0.05, 0.06, 1.0);
            rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

            ImGui_ImplMetal_NewFrame(rpd);
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(340, 0), ImGuiCond_FirstUseEver);
            ImGui::Begin("neuromirror — controls");
            ImGui::Text("%.0f fps   t=%5.1fs", fpsShown, mirror.clock());
            ImGui::TextUnformatted("scene:"); ImGui::SameLine();
            ImGui::RadioButton("mirror", &scene, (int)Scene::Mirror); ImGui::SameLine();
            ImGui::RadioButton("roots",  &scene, (int)Scene::Roots);
            ImGui::Separator();
            if (scene == (int)Scene::Mirror) {
                mirror::PondParams& P = mirror.params();
                // ripples
                ImGui::SliderFloat("ring freq", &P.ring_freq, 0.3f, 10.0f);
                ImGui::SliderFloat("ripple decay", &P.decay, 0.0f, 5.0f);
                ImGui::SliderFloat("ripple speed", &P.speed, 0.0f, 6.0f);
                ImGui::SliderFloat("ripple phase", &P.ripple_offset, 0.0f, 2.0f * (float)M_PI);
                ImGui::SliderFloat("refraction (warp)", &P.warp, 0.0f, 1.0f);
                ImGui::SliderInt("raindrops", &P.drops, 0, 12);
                ImGui::Checkbox("moving ripple", &P.orbit_on);
                ImGui::Checkbox("soft centers (anti-alias)", &P.core_rolloff);
                if (P.core_rolloff) {
                    ImGui::SameLine(); ImGui::SetNextItemWidth(120);
                    ImGui::SliderFloat("radius", &P.core_radius, 0.02f, 0.5f);
                }
                ImGui::Separator();
                // --- live fitting -----------------------------------------
                {
                    static char fit_path[512] =
                        "/Users/erichan/Documents/Development/neuromirror/"
                        "emotion/_track_frames/f0016.png";
                    int&   fit_res = g_fit_downscale;
                    int&   fit_steps = g_fit_steps_per_frame;
                    float& fit_lr = g_fit_lr;
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
                        ImGui::BeginDisabled(!open || !mirror.pond().fitted());
                        if (ImGui::Checkbox("track live feed", &g_fit_live)) {
                            if (g_fit_live) mirror.pond().beginFit(
                                std::vector<float>(size_t(std::max(8, mirror.lowW() / fit_res)) *
                                                   std::max(8, mirror.lowH() / fit_res) * 3, 0.f),
                                std::max(8, mirror.lowH() / fit_res),
                                std::max(8, mirror.lowW() / fit_res), P);
                        }
                        ImGui::EndDisabled();
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip(
                                "Retarget from the camera every frame. The fit\n"
                                "never finishes -- it tracks, running a few\n"
                                "hundred ms behind whoever is in front of the\n"
                                "sensor. That lag is the effect.");
                        }
                        if (open) {
                            bool mir = g_kinect.mirrored();
                            if (ImGui::Checkbox("mirror image", &mir)) g_kinect.setMirrored(mir);
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip(
                                    "A mirror should put your left hand on your\n"
                                    "left. The sensor does not.");
                            }
                        }
                        ImGui::Separator();
                    }
#endif
                    ImGui::PushItemWidth(110);
                    ImGui::SliderInt("fit downscale", &fit_res, 1, 8);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Fit resolution as a divisor of the display size.\n"
                            "A step costs roughly 3x a render over the same\n"
                            "points, so fitting at display resolution cannot\n"
                            "share a frame with drawing. 2 (quarter area) is\n"
                            "the usual choice; the network is continuous, so\n"
                            "the result still renders at full size.");
                    }
                    ImGui::SameLine();
                    ImGui::SliderInt("steps/frame", &fit_steps, 1, 8);
                    ImGui::SameLine();
                    ImGui::SliderFloat("lr", &fit_lr, 1e-4f, 2e-2f, "%.4f",
                                       ImGuiSliderFlags_Logarithmic);
                    ImGui::PopItemWidth();

                    if (ImGui::Button(mirror.pond().fitting() ? "stop" : "fit")) {
                        if (mirror.pond().fitting()) {
                            mirror.pond().stopFit();
                        } else {
                            const int fw = std::max(8, mirror.lowW() / fit_res);
                            const int fh = std::max(8, mirror.lowH() / fit_res);
                            std::vector<float> rgb;
                            fit_err.clear();
                            if (LoadImageRGB(fit_path, fw, fh, rgb, fit_err)) {
                                mirror.pond().beginFit(rgb, fh, fw, P);
                            }
                        }
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
                // --- hybrid sine/tanh -------------------------------------
                ImGui::SliderInt("sine layers (0 = tanh only)", &P.sine_layers,
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
                ImGui::SliderFloat("sine w0 (composition)", &P.sine_w0, 1.0f, 60.0f,
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
                ImGui::SliderFloat("detail (w hidden)", &P.detail, 0.5f, 10.0f);
                ImGui::SliderFloat("gain tilt (front<->back)", &P.gain_tilt, -3.0f, 3.0f);
                ImGui::SliderFloat("w shape (gauss<->uniform)", &P.uniform_mix, 0.0f, 1.0f);
                ImGui::SliderFloat("contrast (w out)", &P.contrast, 1.0f, 12.0f);
                ImGui::Checkbox("sRGB fix", &P.srgb_fix); ImGui::SameLine();
                if (ImGui::Button("reset color")) { P.srgb_fix = false; P.gamma = 1.0f; }
                ImGui::SliderFloat("gamma (>1 darkens)", &P.gamma, 0.3f, 2.0f);
                ImGui::SliderFloat("color mix (0 grey -> 1 RGB)", &P.color_mix, 0.0f, 1.0f);
                ImGui::SameLine(); ImGui::SetNextItemWidth(90);
                const char* greyItems[] = {"R", "G", "B"};
                ImGui::Combo("grey ch", &P.grey_channel, greyItems, 3);
                ImGui::Checkbox("ripple amp -> color", &P.amp_drives_color);
                if (P.amp_drives_color) {
                    ImGui::SameLine(); ImGui::SetNextItemWidth(120);
                    ImGui::SliderFloat("amp gain", &P.amp_gain, 0.2f, 6.0f);
                }
                ImGui::SliderInt("downscale", &downscale, 1, 10);
                ImGui::Separator();
                // z latent
                ImGui::Text("z phase = %6.2f  (circular morph)", P.z);
                ImGui::DragFloat("z", &P.z, 0.02f);
                ImGui::SliderFloat("z amplitude", &P.z_amp, 0.0f, 3.0f);
                ImGui::SliderFloat("z auto-rate /s", &P.z_rate, -2.0f, 2.0f);
                ImGui::SliderFloat("z step size", &P.z_step, 0.01f, 1.0f);
                if (ImGui::Button("z - step")) P.z -= P.z_step; ImGui::SameLine();
                if (ImGui::Button("z + step")) P.z += P.z_step; ImGui::SameLine();
                if (ImGui::Button("z = 0")) P.z = 0.0f;
                ImGui::Separator();
                // time
                ImGui::SliderFloat("ripple time scale", &P.time_scale, 0.0f, 4.0f);
                ImGui::Checkbox("pause", &P.paused); ImGui::SameLine();
                ImGui::Checkbox("swap R/B", &P.swap_rb);
                ImGui::Checkbox("color travel (palette follows orbit)", &P.color_travel);
                if (ImGui::Button("reseed network")) mirror.reseed();
                ImGui::Text("render %d x %d -> %d x %d", mirror.lowW(), mirror.lowH(), fbw, fbh);
                ImGui::Separator();
                // mask emergence transition
                if (ImGui::CollapsingHeader("mask emergence (transition)")) {
                    ImGui::SliderFloat("transition (0 pond -> 1 mask)", &P.transition, 0.0f, 1.0f);
                    ImGui::Checkbox("auto-play", &P.trans_auto); ImGui::SameLine();
                    if (ImGui::Button("reset t")) { P.transition = 0.0f; P.trans_auto = false; }
                    ImGui::SliderFloat("play rate /s", &P.trans_rate, 0.05f, 1.0f);
                    ImGui::SliderFloat("relief height", &P.relief_h, 0.0f, 1.5f);
                    ImGui::SliderFloat("mask width", &P.mask_ax, 0.2f, 1.0f);
                    ImGui::SliderFloat("mask height", &P.mask_ay, 0.2f, 1.2f);
                    ImGui::SliderFloat("light azimuth", &P.light_az, -(float)M_PI, (float)M_PI);
                    ImGui::SliderFloat("light elevation", &P.light_elev, 0.1f, (float)M_PI / 2.0f);
                    ImGui::SliderFloat("wet sheen (spec)", &P.spec_amt, 0.0f, 1.5f);
                    ImGui::SliderFloat("sheen tightness", &P.shininess, 4.0f, 96.0f);
                    ImGui::SliderFloat("background dim", &P.bg_dim, 0.0f, 1.0f);
                }
            } else {
                MetalRootRenderer& R = roots.renderer();
                ImGui::Text("%.0f fps   t=%5.1fs", fpsShown, roots.clock());
                ImGui::Text("render %d x %d -> %d x %d  (overdraw-bound)",
                            roots.width(), roots.height(), fbw, fbh);
                ImGui::Checkbox("auto render-scale", &rootAutoScale);
                if (rootAutoScale) {
                    ImGui::SameLine(); ImGui::SetNextItemWidth(120);
                    ImGui::SliderInt("target px", &rootTargetDim, 720, 3840);
                } else {
                    ImGui::SliderInt("root downscale", &rootDownscale, 1, 6);
                }
                ImGui::Separator();
                // camera
                ImGui::Checkbox("auto-orbit", &roots.autoOrbit); ImGui::SameLine();
                ImGui::SetNextItemWidth(120);
                ImGui::SliderFloat("orbit rate", &roots.orbitRate, -1.0f, 1.0f);
                ImGui::SliderFloat("azimuth", &roots.azimuth, -(float)M_PI, (float)M_PI);
                ImGui::SliderFloat("elevation", &roots.elevation, -1.5f, 1.5f);
                ImGui::SliderFloat("radius", &roots.radius, 5.0f, 120.0f);
                ImGui::SliderFloat("fov", &roots.fov, 0.2f, 1.2f);
                if (ImGui::Button("reseed roots")) roots.reseed((uint32_t)(++rootSeed));
                ImGui::Separator();
                // shading
                const char* modes[] = {"Phong", "PBR", "Invert (approx)"};
                int sm = (int)R.shaderMode;
                if (ImGui::Combo("shader", &sm, modes, 3)) R.shaderMode = (MetalRootRenderer::ShaderMode)sm;
                ImGui::ColorEdit3("base color", R.mat.baseColor);
                ImGui::ColorEdit3("base color 2", R.mat.baseColor2);
                ImGui::SliderFloat("color noise", &R.mat.colorNoiseStrength, 0.0f, 1.0f);
                ImGui::SliderFloat("ambient", &R.mat.ambient, 0.0f, 0.5f);
                ImGui::SliderFloat("diffuse", &R.mat.diffuse, 0.0f, 1.5f);
                ImGui::SliderFloat("shininess", &R.mat.shininess, 4.0f, 300.0f);
                if (sm == 1) {
                    ImGui::SliderFloat("metallic", &R.pbr.metallic, 0.0f, 1.0f);
                    ImGui::SliderFloat("roughness", &R.pbr.roughness, 0.05f, 1.0f);
                }
                ImGui::SliderFloat("radius scale", &R.radiusScale, 0.2f, 4.0f);
                ImGui::Separator();
                // fog
                if (ImGui::CollapsingHeader("fog & atmosphere")) {
                    ImGui::ColorEdit3("fog color", R.fog.color);
                    ImGui::SliderFloat("fog density", &R.fog.density, 0.0f, 0.06f);
                    ImGui::SliderFloat("fog falloff", &R.fog.falloff, 0.0f, 0.3f);
                    ImGui::SliderFloat("fog noise", &R.fog.noiseStrength, 0.0f, 1.0f);
                    ImGui::SliderFloat("fog refDist", &R.fog.refDist, 0.0f, 120.0f);
                    ImGui::SliderFloat("wisp glow", &R.wispGlowStrength, 0.0f, 3.0f);
                    ImGui::SliderInt("wisps", &R.wispCount, 0, 8);
                }
                // pulses
                if (ImGui::CollapsingHeader("travelling pulses")) {
                    ImGui::Checkbox("pulses on", &R.pulse.enabled);
                    ImGui::SliderFloat("pulse speed", &R.pulse.speed, 0.0f, 40.0f);
                    ImGui::SliderFloat("pulse spacing", &R.pulse.spacing, 4.0f, 60.0f);
                    ImGui::SliderFloat("pulse width", &R.pulse.width, 0.5f, 12.0f);
                    ImGui::SliderFloat("pulse intensity", &R.pulse.intensity, 0.0f, 4.0f);
                    ImGui::ColorEdit3("pulse color", R.pulse.color);
                }
                if (ImGui::CollapsingHeader("face masks")) {
                    if (ImGui::Checkbox("show faces", &roots.showFace)) roots.rebuildFace();
                    if (ImGui::SliderFloat("face scale", &roots.faceScale, 0.3f, 1.5f))
                        roots.rebuildFace();
                    ImGui::SliderFloat("face light", &R.face.lightIntensity, 0.0f, 8.0f);
                    ImGui::SliderFloat("face falloff", &R.face.lightFalloff, 0.001f, 0.1f);
                    ImGui::SliderFloat("face spec", &R.face.specStrength, 0.0f, 3.0f);
                    ImGui::ColorEdit3("vein color", R.face.veinColor);
                    ImGui::SliderFloat("vein scale", &R.face.veinScale, 0.1f, 2.0f);
                    ImGui::SliderFloat("vein strength", &R.face.veinStrength, 0.0f, 1.0f);
                }
                if (ImGui::CollapsingHeader("cached field: LOD & culling")) {
                    ImGui::SliderInt("grid NxN", &fieldGrid, 2, 20);
                    if (ImGui::Button("tile field")) roots.buildField(fieldGrid, 30.0f);
                    ImGui::SameLine();
                    if (ImGui::Button("clear field")) { R.clearInstances(); roots.regrow(); }
                    ImGui::Checkbox("frustum cull", &R.cullInstances); ImGui::SameLine();
                    ImGui::Checkbox("sub-pixel cull", &R.subpixelCull);
                    ImGui::SliderFloat("cull below px", &R.instanceCullPx, 0.5f, 20.0f);
                    ImGui::SliderFloat("LOD bias (>1 coarser)", &R.lodBias, 0.1f, 4.0f);
                    ImGui::Text("instances %d   visible %d   culled %d",
                                R.instanceCount(), R.lastVisibleInstances, R.lastCulledInstances);
                    ImGui::Text("capsules drawn: %ld", R.lastDrawnSegments);
                }
                if (ImGui::CollapsingHeader("overlays")) {
                    ImGui::Checkbox("axes", &R.overlay.showAxes); ImGui::SameLine();
                    ImGui::Checkbox("grid", &R.overlay.showGrid);
                    ImGui::SliderFloat("grid spacing", &R.overlay.gridSpacing, 1.0f, 20.0f);
                }
            }
            ImGui::End();

            ImGui::Render();
            id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rpd];
            if (sceneTex) present.encode(re, sceneTex);   // fullscreen scene
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
