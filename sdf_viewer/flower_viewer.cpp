// Flower L-system viewer: drives FlowerLSystem.h into RootRenderer, with
// ImGui controls for the sunflower's phyllotaxis / petal parameters and the
// generic L-system presets. Same renderer (fog, wisps, pulses) as main.cpp.
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "RootRenderer.h"
#include "FlowerLSystem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <vector>

static void glfwError(int, const char* desc) { fprintf(stderr, "GLFW error: %s\n", desc); }

int main() {
    glfwSetErrorCallback(glfwError);
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(1280, 800, "Flower L-System Viewer", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { fprintf(stderr, "glewInit failed\n"); return 1; }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 410");

    RootRenderer renderer(800, 600);
    // Warm, backlit palette that flatters the monochrome geometry.
    renderer.mat.baseColor[0]  = 0.85f; renderer.mat.baseColor[1]  = 0.62f; renderer.mat.baseColor[2]  = 0.18f;
    renderer.mat.baseColor2[0] = 0.30f; renderer.mat.baseColor2[1] = 0.20f; renderer.mat.baseColor2[2] = 0.08f;
    renderer.mat.colorNoiseStrength = 0.5f;
    renderer.mat.colorNoiseScale    = 0.06f;
    renderer.mat.ambient = 0.28f;
    renderer.mat.diffuse = 0.95f;
    renderer.mat.specColor[0] = 0.14f; renderer.mat.specColor[1] = 0.13f; renderer.mat.specColor[2] = 0.12f;
    renderer.mat.shininess = 18.0f;    // soft, matte petals
    renderer.fog.density = 0.0035f;
    renderer.wispCount = 2;
    renderer.wisps[0].basePos[1] = 40.f; renderer.wisps[0].color[0] = 1.0f;
    renderer.wisps[0].color[1] = 0.85f;  renderer.wisps[0].color[2] = 0.4f;
    renderer.wisps[0].intensity = 2.0f;  renderer.wisps[0].driftRadius = 10.f;
    renderer.wisps[1].basePos[0] = -15.f; renderer.wisps[1].basePos[1] = 20.f;
    renderer.wisps[1].color[0] = 0.4f; renderer.wisps[1].color[1] = 0.7f; renderer.wisps[1].color[2] = 1.0f;
    renderer.wisps[1].intensity = 1.5f; renderer.wisps[1].driftRadius = 12.f;

    // Camera
    float azimuth = 0.5f, elevation = 0.15f, orbitRadius = 90.0f;
    float target[3] = {0.f, 25.f, 0.f}, fov = 0.5236f;

    float lightAngle = 0.f, lightOrbitSpeed = 0.3f, lightElevation = 0.9f;

    // Flower state
    // 0 Sunflower, 1 Chrysanthemum, 2 Witch hazel, 3 Daisy stem, 4 Fern, 5 Bush
    int   flowerType = 0;
    flower::SunflowerParams  sp;
    flower::ChrysanthParams  cp;
    flower::WitchHazelParams wp;
    bool  dirty = true;
    bool  growPlay = false;   // auto-advance decorative growth (bud -> open -> loop)

    // Palette rows are indexed by flower::Group (STEM, LEAF, DISK, PETAL, ACCENT).
    auto setPalette = [&](std::initializer_list<std::array<float,3>> cols) {
        int i = 0;
        for (auto& c : cols) {
            if (i >= RootRenderer::MAX_GROUPS) break;
            renderer.palette[i][0] = c[0]; renderer.palette[i][1] = c[1]; renderer.palette[i][2] = c[2];
            ++i;
        }
        renderer.paletteCount = i;
    };

    auto regenerate = [&]() {
        flower::FlowerMesh mesh;
        renderer.paletteTipCount = 0;   // default: no gradient (tips = base)
        switch (flowerType) {
            case 0:
                mesh = flower::buildSunflower(sp);
                setPalette({ {0.30f,0.45f,0.12f},   // stem
                             {0.28f,0.42f,0.14f},   // leaf
                             {0.32f,0.16f,0.05f},   // disk (brown)
                             {0.98f,0.74f,0.12f},   // petal (yellow)
                             {0.85f,0.42f,0.06f} });// accent (orange)
                break;
            case 1: {
                mesh = flower::buildChrysanthemum(cp);
                // Per-form petal base (throat) + tip (edge) colours + green eye.
                std::array<float,3> pBase, pTip, eye{0.70f,0.82f,0.28f};
                switch (cp.form) {
                    case flower::CHRYS_REFLEX:  pBase={0.98f,0.78f,0.28f}; pTip={0.86f,0.18f,0.06f}; eye={0.95f,0.75f,0.20f}; break;
                    case flower::CHRYS_INCURVE: pBase={0.99f,0.82f,0.34f}; pTip={0.92f,0.48f,0.08f}; eye={0.98f,0.80f,0.30f}; break;
                    case flower::CHRYS_POMPOM:  pBase={0.99f,0.99f,0.96f}; pTip={0.86f,0.88f,0.80f}; break;
                    case flower::CHRYS_QUILL:   pBase={0.99f,0.82f,0.20f}; pTip={0.93f,0.42f,0.10f}; break;
                    default:                    pBase={0.98f,0.80f,0.90f}; pTip={0.85f,0.16f,0.55f}; eye={0.92f,0.90f,0.55f}; break;
                }
                std::array<float,3> grn{0.14f,0.26f,0.11f}, lgrn{0.24f,0.38f,0.16f};
                setPalette({ grn, grn, eye, pBase, {0.80f,0.20f,0.40f} });
                std::array<float,3> tips[5] = { grn, lgrn, eye, pTip, {0.90f,0.30f,0.10f} };
                for (int i = 0; i < 5; ++i) { renderer.paletteTip[i][0]=tips[i][0];
                    renderer.paletteTip[i][1]=tips[i][1]; renderer.paletteTip[i][2]=tips[i][2]; }
                renderer.paletteTipCount = 5;
            } break;
            case 2:
                mesh = flower::buildWitchHazel(wp);
                setPalette({ {0.34f,0.22f,0.12f},   // woody twig
                             {0.30f,0.40f,0.16f},
                             {0.90f,0.55f,0.10f},
                             {0.98f,0.72f,0.06f},   // strap petal (yellow)
                             {0.70f,0.10f,0.05f} });// calyx (dark red)
                break;
            default: {
                flower::Preset presets[] = { flower::Preset::DaisyStem,
                                             flower::Preset::FernFrond,
                                             flower::Preset::Bush };
                mesh = flower::buildFromPreset(presets[flowerType - 3]);
                setPalette({ {0.30f,0.45f,0.14f}, {0.42f,0.55f,0.18f} });
            } break;
        }
        renderer.uploadSegments(mesh.nodes, mesh.segments, mesh.radii, &mesh.groups,
                                &mesh.prims, &mesh.frames, &mesh.aux);
        // Recentre orbit target on the mesh centroid height.
        if (!mesh.nodes.empty()) {
            double sy = 0; for (auto& n : mesh.nodes) sy += n.y;
            target[1] = float(sy / mesh.nodes.size());
        }
    };

    double lastFrameTime = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        double now = glfwGetTime();
        float dt = float(now - lastFrameTime);
        lastFrameTime = now;

        // Animate decorative growth: ease bud -> open, hold, then loop.
        if (growPlay && flowerType == 1 && cp.form == flower::CHRYS_DECORATIVE) {
            cp.growth += dt * 0.20f;                    // builder clamps to [0,1];
            if (cp.growth > 1.35f) cp.growth = 0.0f;    // 1.0-1.35 holds open, then loops
            dirty = true;
        }
        if (dirty) { regenerate(); dirty = false; }

        lightAngle += lightOrbitSpeed * dt;
        float lc = cosf(lightElevation), ls = sinf(lightElevation);
        float lightDir[3] = { lc * sinf(lightAngle), ls, lc * cosf(lightAngle) };

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGuiIO& io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(io.DisplaySize, ImGuiCond_Always);
        ImGui::Begin("##main", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);

        ImGui::BeginChild("##controls", ImVec2(270, 0), true);

        ImGui::Text("Flower");
        const char* types[] = { "Sunflower", "Chrysanthemum", "Witch hazel",
                                "Daisy stem", "Fern frond", "Bush" };
        if (ImGui::Combo("Type", &flowerType, types, 6)) dirty = true;

        if (flowerType == 1) {
            ImGui::SeparatorText("Chrysanthemum");
            const char* forms[] = { "Reflex", "Regular incurve", "Pompom", "Quill", "Decorative" };
            dirty |= ImGui::Combo("Form", &cp.form, forms, 5);
            dirty |= ImGui::SliderFloat("Stem height", (float*)&cp.stemHeight, 10.f, 90.f, "%.0f cm");
            dirty |= ImGui::SliderFloat("Bloom radius", (float*)&cp.radius, 4.f, 22.f, "%.0f cm");
            dirty |= ImGui::SliderInt  ("Petals (0=auto)", &cp.petalCount, 0, 1200);
            dirty |= ImGui::SliderInt  ("Leaves##c",    &cp.stemLeaves, 0, 6);
            if (cp.form == flower::CHRYS_DECORATIVE) {
                dirty |= ImGui::SliderFloat("Growth", (float*)&cp.growth, 0.f, 1.f, "%.2f");
                ImGui::SameLine();
                ImGui::Checkbox("Grow", &growPlay);
            }
            int s = (int)cp.seed;
            if (ImGui::SliderInt("Seed##c", &s, 0, 64)) { cp.seed = (unsigned)s; dirty = true; }
        } else if (flowerType == 2) {
            ImGui::SeparatorText("Witch hazel");
            dirty |= ImGui::SliderFloat("Twig scale",   (float*)&wp.branchScale, 0.6f, 2.5f, "%.2f");
            dirty |= ImGui::SliderInt  ("Iterations",   &wp.iterations, 2, 5);
            dirty |= ImGui::SliderInt  ("Clusters",     &wp.clusters, 2, 60);
            dirty |= ImGui::SliderInt  ("Straps/flower",&wp.strapsPerFlower, 3, 6);
            dirty |= ImGui::SliderFloat("Strap length", (float*)&wp.strapLength, 2.f, 14.f, "%.0f cm");
            dirty |= ImGui::SliderFloat("Strap width",  (float*)&wp.strapWidth, 0.1f, 0.6f, "%.2f cm");
            dirty |= ImGui::SliderFloat("Crimp",        (float*)&wp.crimp, 0.f, 1.5f, "%.2f");
            dirty |= ImGui::SliderFloat("Spread",       (float*)&wp.spread, 0.2f, 1.5f, "%.2f rad");
            int s = (int)wp.seed;
            if (ImGui::SliderInt("Seed##w", &s, 0, 64)) { wp.seed = (unsigned)s; dirty = true; }
        }

        if (flowerType == 0) {
            ImGui::SeparatorText("Stem");
            dirty |= ImGui::SliderFloat("Height",   (float*)&sp.stemHeight, 10.f, 100.f, "%.0f cm");
            dirty |= ImGui::SliderFloat("Radius",   (float*)&sp.stemRadius, 0.2f, 3.0f,  "%.2f cm");
            dirty |= ImGui::SliderFloat("Waver",    (float*)&sp.stemWaver,  0.f,  12.f,  "%.1f cm");
            dirty |= ImGui::SliderInt  ("Leaves",   &sp.stemLeaves, 0, 8);

            ImGui::SeparatorText("Head");
            dirty |= ImGui::SliderFloat("Head radius", (float*)&sp.headRadius, 4.f, 30.f, "%.0f cm");
            dirty |= ImGui::SliderFloat("Tilt",        (float*)&sp.headTilt,  -1.2f, 1.2f, "%.2f rad");
            dirty |= ImGui::SliderFloat("Dome",        (float*)&sp.headDome,  -0.4f, 0.4f, "%.2f");

            ImGui::SeparatorText("Florets (phyllotaxis)");
            dirty |= ImGui::SliderInt  ("Count",        &sp.floretCount, 50, 2500);
            dirty |= ImGui::SliderFloat("Floret radius", (float*)&sp.floretRadius, 0.1f, 1.0f, "%.2f cm");
            dirty |= ImGui::SliderFloat("Floret height", (float*)&sp.floretHeight, 0.1f, 2.5f, "%.2f cm");

            ImGui::SeparatorText("Petals");
            dirty |= ImGui::SliderInt  ("Petal count", &sp.petalCount, 5, 89);
            dirty |= ImGui::SliderInt  ("Petal rings", &sp.petalRings, 1, 4);
            dirty |= ImGui::SliderFloat("Petal length", (float*)&sp.petalLength, 3.f, 25.f, "%.0f cm");
            dirty |= ImGui::SliderFloat("Petal width",  (float*)&sp.petalWidth,  0.5f, 6.f, "%.1f cm");
            dirty |= ImGui::SliderFloat("Petal droop",  (float*)&sp.petalDroop, -0.3f, 1.0f, "%.2f rad");
            dirty |= ImGui::SliderFloat("Petal lift",   (float*)&sp.petalLift,  -0.3f, 0.8f, "%.2f rad");

            int seedInt = (int)sp.seed;
            if (ImGui::SliderInt("Seed", &seedInt, 0, 64)) { sp.seed = (unsigned)seedInt; dirty = true; }
        }

        ImGui::SeparatorText("Material");
        ImGui::ColorEdit3("Base",   renderer.mat.baseColor);
        ImGui::ColorEdit3("Base 2", renderer.mat.baseColor2);
        ImGui::SliderFloat("Color noise", &renderer.mat.colorNoiseStrength, 0.f, 1.f, "%.2f");
        ImGui::SliderFloat("Thickness",   &renderer.radiusScale, 0.2f, 4.f, "%.2f x");

        ImGui::SeparatorText("Fog");
        ImGui::ColorEdit3("Fog color",  renderer.fog.color);
        ImGui::SliderFloat("Density",   &renderer.fog.density, 0.f, 0.05f, "%.4f");
        ImGui::SliderFloat("Falloff",   &renderer.fog.falloff, 0.f, 0.3f, "%.3f");

        ImGui::SeparatorText("Light");
        ImGui::SliderFloat("Orbit spd", &lightOrbitSpeed, 0.f, 3.f, "%.2f");
        ImGui::SliderFloat("Elevation", &lightElevation, -1.5f, 1.5f, "%.2f");

        ImGui::TextDisabled("Left-drag: orbit / Scroll: zoom");
        ImGui::EndChild();
        ImGui::SameLine();

        ImVec2 vpSize = ImGui::GetContentRegionAvail();
        int vpW = std::max(1, (int)vpSize.x), vpH = std::max(1, (int)vpSize.y);
        renderer.fog.driftTime += dt * renderer.fog.driftSpeed;
        renderer.wispTime += dt;
        renderer.resize(vpW, vpH);
        renderer.render(azimuth, elevation, orbitRadius, target, fov, lightDir);
        ImGui::Image((ImTextureID)renderer.colorTex(), vpSize);

        if (ImGui::IsItemHovered()) {
            if (io.MouseDown[0]) {
                azimuth   -= io.MouseDelta.x * 0.005f;
                elevation += io.MouseDelta.y * 0.005f;
                elevation  = std::fmaxf(-1.4f, std::fminf(1.4f, elevation));
            }
            if (io.MouseWheel != 0.f) {
                orbitRadius *= powf(0.9f, io.MouseWheel);
                orbitRadius  = std::fmaxf(2.f, std::fminf(400.f, orbitRadius));
            }
        }

        ImGui::End();
        ImGui::Render();

        int fbW, fbH; glfwGetFramebufferSize(window, &fbW, &fbH);
        glViewport(0, 0, fbW, fbH);
        glClearColor(0.08f, 0.07f, 0.06f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
