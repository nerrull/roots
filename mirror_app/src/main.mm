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

#include <algorithm>
#include <cstdio>

int main(int, char**) {
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

    // Placeholder scene selector until MirrorScene / RootScene land.
    enum class Scene { Mirror = 0, Roots = 1 };
    int scene = (int)Scene::Mirror;

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
            ImGui::TextDisabled("scaffold: scenes not yet wired");
            ImGui::End();

            ImGui::Render();
            id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rpd];
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
