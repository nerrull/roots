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
#include "fullscreen_present.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

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

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--selftest") return selftest();
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
    FullscreenPresent present(ctx, std::string(MIRROR_APP_SHADER_DIR) + "/present.metal",
                              layer.pixelFormat);

    enum class Scene { Mirror = 0, Roots = 1 };
    int scene = (int)Scene::Mirror;
    int downscale = 4;   // mirror render-resolution divisor (low-res + upsample)

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
                sceneTex = mirror.render();
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
                ImGui::TextDisabled("roots scene: Metal port pending");
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
