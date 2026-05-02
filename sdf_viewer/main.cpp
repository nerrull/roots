#include <GL/glew.h>         // must come before any other GL/GLFW headers
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "RootRenderer.h"

#include "RootSystem.h"
#include "SegmentAnalyser.h"

#include <climits>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

static void glfwError(int, const char* desc) {
    fprintf(stderr, "GLFW error: %s\n", desc);
}

int main(int argc, char** argv) {
    glfwSetErrorCallback(glfwError);
    if (!glfwInit()) return 1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE,        GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);   // required on macOS

    GLFWwindow* window = glfwCreateWindow(1280, 800, "SDF Root System Viewer", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);   // vsync

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "glewInit failed\n");
        return 1;
    }

    // ------------------------------------------------------------------
    // ImGui
    // ------------------------------------------------------------------
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 410");

    // ------------------------------------------------------------------
    // Root system
    // ------------------------------------------------------------------
    std::string paramFile;
    if (argc > 1) {
        paramFile = argv[1];
    } else {
        paramFile = PARAM_DIR "/structural/rootsystem/Zea_mays_6_Leitner_2014.xml";
    }

    auto rs = std::make_shared<CPlantBox::RootSystem>();
    rs->readParameters(paramFile, "plant", true, false);
    rs->initialize(false);

    // ------------------------------------------------------------------
    // Renderer
    // ------------------------------------------------------------------
    RootRenderer renderer(800, 600);

    // ------------------------------------------------------------------
    // Camera state
    // ------------------------------------------------------------------
    float azimuth     = 0.5f;    // radians
    float elevation   = 0.3f;    // radians, clamped to ±~80°
    float orbitRadius = 30.0f;   // cm
    float target[3]   = {0.f, 0.f, 0.f};
    float fov         = 0.5236f; // 30° half-angle in radians

    // ------------------------------------------------------------------
    // Simulation state
    // ------------------------------------------------------------------
    bool   playing      = false;
    float  speed        = 1.0f;   // days per second
    double simTime      = 0.0;    // accumulated simulation days
    int    prevSegCount = INT_MIN; // forces initial upload

    double lastFrameTime = glfwGetTime();

    // ------------------------------------------------------------------
    // Render loop
    // ------------------------------------------------------------------
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        double now = glfwGetTime();
        float  dt  = static_cast<float>(now - lastFrameTime);
        lastFrameTime = now;

        // 1. Advance simulation
        if (playing) {
            rs->simulate(static_cast<double>(dt * speed), false);
            simTime += static_cast<double>(dt * speed);
        }

        // 2. Sync TBOs whenever segment count changes (cheap O(1) poll)
        int newSegCount = rs->getNumberOfSegments();
        if (newSegCount != prevSegCount) {
            prevSegCount = newSegCount;

            CPlantBox::SegmentAnalyser ana(*rs);
            auto radii = ana.getParameter("radius");
            renderer.uploadSegments(ana.nodes, ana.segments, radii);

            // Update orbit target to centroid of all nodes
            if (!ana.nodes.empty()) {
                float sx = 0, sy = 0, sz = 0;
                for (auto& n : ana.nodes) {
                    sx += static_cast<float>(n.x);
                    sy += static_cast<float>(n.y);
                    sz += static_cast<float>(n.z);
                }
                float inv = 1.f / static_cast<float>(ana.nodes.size());
                target[0] = sx * inv;
                target[1] = sy * inv;
                target[2] = sz * inv;
            }
        }

        // 3. ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGuiIO& io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(io.DisplaySize, ImGuiCond_Always);

        constexpr ImGuiWindowFlags kMainFlags =
            ImGuiWindowFlags_NoDecoration     |
            ImGuiWindowFlags_NoMove           |
            ImGuiWindowFlags_NoBringToFrontOnFocus;
        ImGui::Begin("##main", nullptr, kMainFlags);

        // ---- Left control panel ----
        ImGui::BeginChild("##controls", ImVec2(230, 0), true);

        if (ImGui::Button(playing ? "  Pause  " : "  Play  "))
            playing = !playing;
        ImGui::SameLine();
        if (ImGui::Button("  Reset  ")) {
            rs->reset();
            rs->initialize(false);
            simTime      = 0.0;
            prevSegCount = INT_MIN;   // force re-upload
            playing      = false;
        }

        ImGui::Spacing();
        ImGui::SliderFloat("Speed (days/s)", &speed, 0.1f, 20.0f, "%.1f");
        ImGui::Spacing();
        ImGui::Text("Day:      %.2f", simTime);
        ImGui::Text("Segments: %d",   std::max(0, newSegCount));
        ImGui::Separator();
        ImGui::Text("Camera");
        ImGui::SliderFloat("FOV (rad)",    &fov,         0.1f, 1.3f,   "%.2f");
        ImGui::SliderFloat("Orbit radius", &orbitRadius, 1.0f, 300.0f, "%.0f cm");
        ImGui::Spacing();
        ImGui::TextDisabled("Left-drag: orbit");
        ImGui::TextDisabled("Scroll:    zoom");

        ImGui::EndChild();
        ImGui::SameLine();

        // ---- Viewport ----
        ImVec2 vpSize = ImGui::GetContentRegionAvail();
        int vpW = std::max(1, static_cast<int>(vpSize.x));
        int vpH = std::max(1, static_cast<int>(vpSize.y));

        renderer.resize(vpW, vpH);
        renderer.render(azimuth, elevation, orbitRadius, target, fov);

        ImGui::Image(static_cast<ImTextureID>(renderer.colorTex()), vpSize);

        // Orbit camera — only when the image is hovered
        if (ImGui::IsItemHovered()) {
            if (io.MouseDown[0]) {
                azimuth   -= io.MouseDelta.x * 0.005f;
                elevation += io.MouseDelta.y * 0.005f;
                elevation  = std::fmaxf(-1.4f, std::fminf(1.4f, elevation));
            }
            if (io.MouseWheel != 0.f) {
                orbitRadius *= powf(0.9f, io.MouseWheel);
                orbitRadius  = std::fmaxf(0.5f, std::fminf(500.f, orbitRadius));
            }
        }

        ImGui::End();

        ImGui::Render();

        int fbW, fbH;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        glViewport(0, 0, fbW, fbH);
        glClearColor(0.1f, 0.1f, 0.1f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
