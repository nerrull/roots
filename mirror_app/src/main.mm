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
#include <cstdio>
#include <string>

// Headless check of the MLX→Metal texture path (no window): render a few mirror
// frames and read back a pixel. Used to smoke-test without a GUI.
static int selftest() {
    MetalContext ctx;
    if (!ctx.device()) { fprintf(stderr, "selftest: no Metal device\n"); return 1; }
    MirrorScene mirror(ctx, MIRROR_APP_ASSET_DIR);
    if (!mirror.valid()) { fprintf(stderr, "selftest: mirror invalid\n"); return 1; }
    id<MTLTexture> tex = nil;
    for (int i = 0; i < 5; ++i) tex = mirror.render(i / 60.0);
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

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--selftest") return selftest();

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
    MirrorScene mirror(ctx, MIRROR_APP_ASSET_DIR);
    FullscreenPresent present(ctx, std::string(MIRROR_APP_SHADER_DIR) + "/present.metal",
                              layer.pixelFormat);

    enum class Scene { Mirror = 0, Roots = 1 };
    int scene = (int)Scene::Mirror;
    int downscale = 4;   // mirror render-resolution divisor (low-res + upsample)

    const double startTime = glfwGetTime();
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
            id<MTLTexture> sceneTex = nil;
            if (scene == (int)Scene::Mirror && mirror.valid()) {
                mirror.ensureSize(fbw / std::max(1, downscale), fbh / std::max(1, downscale));
                sceneTex = mirror.render(glfwGetTime() - startTime);
            }

            MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
            rpd.colorAttachments[0].texture = drawable.texture;
            rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
            rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.05, 0.05, 0.06, 1.0);
            rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

            ImGui_ImplMetal_NewFrame(rpd);
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::SetNextWindowSize(ImVec2(320, 0), ImGuiCond_FirstUseEver);
            ImGui::Begin("mirror_app");
            ImGui::Text("%.1f fps", fpsShown);
            ImGui::Text("drawable %d x %d", (int)fbw, (int)fbh);
            ImGui::Separator();
            ImGui::TextUnformatted("scene");
            ImGui::RadioButton("mirror", &scene, (int)Scene::Mirror); ImGui::SameLine();
            ImGui::RadioButton("roots",  &scene, (int)Scene::Roots);
            ImGui::Separator();
            if (scene == (int)Scene::Mirror) {
                ImGui::TextUnformatted("neural mirror (pond)");
                ImGui::Text("render %d x %d  (÷%d)", mirror.lowW(), mirror.lowH(), downscale);
                ImGui::SliderInt("downscale", &downscale, 1, 8);
                ImGui::SliderFloat("speed", &mirror.speed, 0.05f, 4.0f, "%.2f");
                ImGui::SliderFloat("ring freq", &mirror.ringFreq, 1.0f, 8.0f, "%.2f");
                ImGui::SliderFloat("decay", &mirror.decay, 0.4f, 3.0f, "%.2f");
                ImGui::SliderFloat("warp", &mirror.warp, 0.0f, 0.5f, "%.3f");
                ImGui::SliderFloat("core damp", &mirror.core, 0.0f, 0.6f, "%.3f");
                ImGui::Checkbox("animate z", &mirror.animateZ);
                if (mirror.animateZ) {
                    ImGui::SliderFloat("z speed", &mirror.zSpeed, 0.0f, 1.5f, "%.2f");
                } else {
                    ImGui::SliderFloat("z", &mirror.z, -1.0f, 1.0f, "%.2f");
                    ImGui::SliderFloat("z cos", &mirror.zCos, -1.0f, 1.0f, "%.2f");
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
