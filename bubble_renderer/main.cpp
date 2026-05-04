#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "BubbleRenderer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

static void glfwError(int, const char* d) { fprintf(stderr, "GLFW: %s\n", d); }

int main() {
    glfwSetErrorCallback(glfwError);
    if (!glfwInit()) return 1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE,        GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* win = glfwCreateWindow(1280, 800, "Bubble SDF Renderer", nullptr, nullptr);
    if (!win) { glfwTerminate(); return 1; }

    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { fprintf(stderr, "glewInit failed\n"); return 1; }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 410");

    BubbleRenderer renderer(800, 600);
    BubbleRenderer::Params params;

    float  azimuth     = 0.3f;
    float  elevation   = 0.2f;
    float  orbitRadius = 5.0f;
    bool   autoLight   = true;
    float  lightAngle  = 0.0f;
    float  time        = 0.0f;
    double lastT       = glfwGetTime();

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        double now = glfwGetTime();
        float  dt  = (float)(now - lastT);
        lastT = now;
        time += dt;

        if (autoLight) lightAngle += dt * 0.4f;
        params.lightAzimuth = lightAngle;

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGuiIO& io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0, 0),        ImGuiCond_Always);
        ImGui::SetNextWindowSize(io.DisplaySize,     ImGuiCond_Always);

        constexpr ImGuiWindowFlags kFlags =
            ImGuiWindowFlags_NoDecoration |
            ImGuiWindowFlags_NoMove       |
            ImGuiWindowFlags_NoBringToFrontOnFocus;
        ImGui::Begin("##main", nullptr, kFlags);

        // ---- Controls ----
        ImGui::BeginChild("##ctrl", ImVec2(220, 0), true);

        ImGui::Text("Film");
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##thick", &params.filmThickness, 100.0f, 1200.0f, "%.0f nm");

        ImGui::Spacing();
        ImGui::Text("Bubble");
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##radius", &params.bubbleRadius, 0.3f, 4.0f, "%.2f");

        ImGui::Spacing();
        ImGui::Text("Wobble");
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##wfreq", &params.wobbleFreq, 0.3f, 5.0f, "freq %.2f");
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##wamp",  &params.wobbleAmp,  0.0f, 0.3f, "amp  %.3f");

        ImGui::Spacing();
        ImGui::Text("Light");
        ImGui::Checkbox("Auto-rotate", &autoLight);
        if (!autoLight) {
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderFloat("Azimuth##L",   &params.lightAzimuth,   0.0f, 6.28f, "%.2f");
        }
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("Elevation##L", &params.lightElevation, -1.5f, 1.5f, "%.2f rad");

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextDisabled("Left-drag : orbit");
        ImGui::TextDisabled("Scroll    : zoom");

        ImGui::EndChild();
        ImGui::SameLine();

        // ---- Viewport ----
        ImVec2 vpSize = ImGui::GetContentRegionAvail();
        int vpW = std::max(1, (int)vpSize.x);
        int vpH = std::max(1, (int)vpSize.y);

        renderer.resize(vpW, vpH);
        renderer.render(time, azimuth, elevation, orbitRadius, params);

        ImGui::Image((ImTextureID)(uintptr_t)renderer.colorTex(), vpSize);

        if (ImGui::IsItemHovered()) {
            if (io.MouseDown[0]) {
                azimuth   -= io.MouseDelta.x * 0.005f;
                elevation += io.MouseDelta.y * 0.005f;
                elevation  = std::fmax(-1.4f, std::fmin(1.4f, elevation));
            }
            if (io.MouseWheel != 0.f) {
                orbitRadius *= powf(0.9f, io.MouseWheel);
                orbitRadius  = std::fmax(0.3f, std::fmin(30.f, orbitRadius));
            }
        }

        ImGui::End();
        ImGui::Render();

        int fbW, fbH;
        glfwGetFramebufferSize(win, &fbW, &fbH);
        glViewport(0, 0, fbW, fbH);
        glClearColor(0.08f, 0.08f, 0.1f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(win);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
