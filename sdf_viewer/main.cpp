#include <GL/glew.h>         // must come before any other GL/GLFW headers
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "RootRenderer.h"

#include "RootSystem.h"
#include "SegmentAnalyser.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

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
    // Root system presets — enumerate all .xml files in rootsystem dir
    // ------------------------------------------------------------------
    std::vector<std::string> rsNames, rsPaths;
    {
        namespace fs = std::filesystem;
        std::vector<std::pair<std::string, std::string>> entries;
        for (auto& e : fs::directory_iterator(PARAM_DIR "/structural/rootsystem"))
            if (e.path().extension() == ".xml")
                entries.push_back({e.path().stem().string(), e.path().string()});
        std::sort(entries.begin(), entries.end());
        for (auto& [n, p] : entries) { rsNames.push_back(n); rsPaths.push_back(p); }
    }
    int selectedRsIdx = 0;
    for (int i = 0; i < (int)rsPaths.size(); i++)
        if (rsPaths[i] == paramFile) { selectedRsIdx = i; break; }

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

    // ------------------------------------------------------------------
    // Light orbit state
    // ------------------------------------------------------------------
    float lightOrbitSpeed = 0.4f;   // rad/s
    float lightElevation  = 0.9f;   // radians above horizon
    float lightAngle      = 0.0f;   // accumulated azimuth

    // ------------------------------------------------------------------
    // Load-dialog state
    // ------------------------------------------------------------------
    char        loadPathBuf[2048] = {};
    std::string loadError;

    double lastFrameTime = glfwGetTime();

    // ------------------------------------------------------------------
    // Render loop
    // ------------------------------------------------------------------
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        double now = glfwGetTime();
        float  dt  = static_cast<float>(now - lastFrameTime);
        lastFrameTime = now;

        // 1. Advance simulation and light orbit
        if (playing) {
            rs->simulate(static_cast<double>(dt * speed), false);
            simTime += static_cast<double>(dt * speed);
        }
        lightAngle += lightOrbitSpeed * dt;

        float lCos = cosf(lightElevation), lSin = sinf(lightElevation);
        float lightDir[3] = {
            lCos * sinf(lightAngle),
            lSin,
            lCos * cosf(lightAngle)
        };
        float lLen = sqrtf(lightDir[0]*lightDir[0] + lightDir[1]*lightDir[1] + lightDir[2]*lightDir[2]);
        lightDir[0] /= lLen; lightDir[1] /= lLen; lightDir[2] /= lLen;

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

        // ---- Root system picker ----
        ImGui::Spacing();
        ImGui::Text("Root system:");
        ImGui::SetNextItemWidth(-1);
        const char* preview = (!rsNames.empty() && selectedRsIdx < (int)rsNames.size())
                              ? rsNames[selectedRsIdx].c_str() : "";
        if (ImGui::BeginCombo("##rscombo", preview)) {
            for (int i = 0; i < (int)rsNames.size(); i++) {
                bool sel = (i == selectedRsIdx);
                if (ImGui::Selectable(rsNames[i].c_str(), sel) && i != selectedRsIdx) {
                    selectedRsIdx = i;
                    try {
                        auto newRs = std::make_shared<CPlantBox::RootSystem>();
                        newRs->readParameters(rsPaths[i], "plant", true, false);
                        newRs->initialize(false);
                        rs = newRs; paramFile = rsPaths[i];
                        simTime = 0.0; prevSegCount = INT_MIN; playing = false;
                        loadError.clear();
                    } catch (const std::exception& e) { loadError = e.what(); }
                }
                if (sel) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        if (!loadError.empty())
            ImGui::TextColored(ImVec4(1.f, 0.35f, 0.35f, 1.f), "%s", loadError.c_str());

        // ---- Load arbitrary file ----
        ImGui::Spacing();
        if (ImGui::Button("  Load file...  ")) {
            strncpy(loadPathBuf, paramFile.c_str(), sizeof(loadPathBuf) - 1);
            loadPathBuf[sizeof(loadPathBuf) - 1] = '\0';
            loadError.clear();
            ImGui::OpenPopup("Load spec file");
        }

        if (ImGui::BeginPopupModal("Load spec file", nullptr,
                                   ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Root system spec file (.xml):");
            ImGui::SetNextItemWidth(500.0f);

            bool doLoad = ImGui::InputText("##specpath", loadPathBuf,
                                           sizeof(loadPathBuf),
                                           ImGuiInputTextFlags_EnterReturnsTrue);
            doLoad |= ImGui::Button("Load");
            if (doLoad) {
                try {
                    auto newRs = std::make_shared<CPlantBox::RootSystem>();
                    newRs->readParameters(std::string(loadPathBuf), "plant", true, false);
                    newRs->initialize(false);
                    rs = newRs; paramFile = loadPathBuf;
                    simTime = 0.0; prevSegCount = INT_MIN; playing = false;
                    loadError.clear();
                    ImGui::CloseCurrentPopup();
                } catch (const std::exception& e) { loadError = e.what(); }
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) { loadError.clear(); ImGui::CloseCurrentPopup(); }
            ImGui::EndPopup();
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

        ImGui::Separator();
        ImGui::Text("Material");
        ImGui::ColorEdit3("Base color",  renderer.mat.baseColor);
        ImGui::SliderFloat("Ambient",    &renderer.mat.ambient,   0.0f,  1.0f,   "%.2f");
        ImGui::SliderFloat("Diffuse",    &renderer.mat.diffuse,   0.0f,  1.0f,   "%.2f");
        ImGui::ColorEdit3("Spec color",  renderer.mat.specColor);
        ImGui::SliderFloat("Shininess",  &renderer.mat.shininess, 1.0f, 512.0f,  "%.0f",
                           ImGuiSliderFlags_Logarithmic);

        ImGui::Separator();
        ImGui::Text("Light");
        ImGui::SliderFloat("Orbit speed", &lightOrbitSpeed, 0.0f, 3.14f, "%.2f rad/s");
        ImGui::SliderFloat("Elevation",   &lightElevation, -1.5f,  1.5f, "%.2f rad");

        ImGui::Separator();
        ImGui::Text("Fog");
        ImGui::ColorEdit3("Fog color",       renderer.fog.color);
        ImGui::SliderFloat("Density",        &renderer.fog.density,       0.0f,  0.10f, "%.4f");
        ImGui::SliderFloat("Falloff",        &renderer.fog.falloff,       0.0f,  0.30f, "%.3f");
        ImGui::SliderFloat("Noise scale",    &renderer.fog.noiseScale,    0.01f, 0.5f,  "%.3f");
        ImGui::SliderFloat("Noise strength", &renderer.fog.noiseStrength, 0.0f,  1.0f,  "%.2f");
        ImGui::SliderFloat("Drift speed",    &renderer.fog.driftSpeed,   0.0f,  5.0f,  "%.2f");
        {
            const char* noiseTypes[] = { "Value", "Simplex" };
            ImGui::Combo("Noise type", &renderer.fog.noiseType, noiseTypes, 2);
        }

        ImGui::Separator();
        ImGui::Text("Overlay");
        ImGui::Checkbox("Show axes", &renderer.overlay.showAxes);
        if (renderer.overlay.showAxes)
            ImGui::SliderFloat("Axis length", &renderer.overlay.axisLength, 1.0f, 50.0f, "%.0f cm");
        ImGui::Checkbox("Show grid", &renderer.overlay.showGrid);
        if (renderer.overlay.showGrid)
            ImGui::SliderFloat("Grid spacing", &renderer.overlay.gridSpacing, 1.0f, 20.0f, "%.0f cm");

        ImGui::EndChild();
        ImGui::SameLine();

        // ---- Viewport ----
        ImVec2 vpSize = ImGui::GetContentRegionAvail();
        int vpW = std::max(1, static_cast<int>(vpSize.x));
        int vpH = std::max(1, static_cast<int>(vpSize.y));

        renderer.fog.driftTime += dt * renderer.fog.driftSpeed;
        renderer.resize(vpW, vpH);
        renderer.render(azimuth, elevation, orbitRadius, target, fov, lightDir);

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
