// Offscreen flower renderer: builds each flower, renders one frame into the
// RootRenderer FBO, and writes a PNG. Headless-friendly (hidden GLFW window) so
// it works without screen-recording permissions. Usage:
//   flower_shot [outDir]
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "RootRenderer.h"
#include "FlowerLSystem.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <vector>

static void writePPM(const std::string& path, int w, int h, const std::vector<unsigned char>& rgba) {
    // glReadPixels/GetTexImage origin is bottom-left; flip vertically.
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return; }
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    for (int y = h - 1; y >= 0; --y)
        for (int x = 0; x < w; ++x) {
            const unsigned char* px = &rgba[(y * w + x) * 4];
            fputc(px[0], f); fputc(px[1], f); fputc(px[2], f);
        }
    fclose(f);
}

static void setPalette(RootRenderer& r, std::initializer_list<std::array<float,3>> cols) {
    int i = 0;
    for (auto& c : cols) { if (i >= RootRenderer::MAX_GROUPS) break;
        r.palette[i][0]=c[0]; r.palette[i][1]=c[1]; r.palette[i][2]=c[2]; ++i; }
    r.paletteCount = i;
}

int main(int argc, char** argv) {
    std::string outDir = argc > 1 ? argv[1] : "/tmp";
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);   // hidden -> no permissions needed
    GLFWwindow* win = glfwCreateWindow(900, 900, "shot", nullptr, nullptr);
    if (!win) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return 1;

    const int W = 900, H = 900;
    RootRenderer renderer(W, H);
    renderer.mat.colorNoiseStrength = 0.5f;
    renderer.mat.colorNoiseScale    = 0.06f;
    renderer.mat.ambient = 0.28f;
    renderer.mat.diffuse = 0.95f;
    // Petals are soft/matte, not glossy plastic -- keep specular low & broad.
    renderer.mat.specColor[0] = 0.14f; renderer.mat.specColor[1] = 0.13f; renderer.mat.specColor[2] = 0.12f;
    renderer.mat.shininess = 18.0f;
    renderer.fog.density = 0.0035f;
    renderer.wispCount = 2;
    renderer.wisps[0].basePos[1]=40.f; renderer.wisps[0].color[0]=1.f; renderer.wisps[0].color[1]=0.85f; renderer.wisps[0].color[2]=0.4f;
    renderer.wisps[0].intensity=2.f; renderer.wisps[0].driftRadius=10.f;
    renderer.wisps[1].basePos[0]=-15.f; renderer.wisps[1].basePos[1]=20.f;
    renderer.wisps[1].color[0]=0.4f; renderer.wisps[1].color[1]=0.7f; renderer.wisps[1].color[2]=1.f;
    renderer.wisps[1].intensity=1.5f; renderer.wisps[1].driftRadius=12.f;

    // Focus: the five chrysanthemum bloom forms, each with a fitting palette.
    // Palette order: STEM, LEAF, DISK(eye), PETAL, ACCENT.
    // petalBase = colour at the petal throat/base, petalTip = at the edge/tip.
    struct Shot { const char* name; int form; float radius;
                  std::array<float,3> petalBase; std::array<float,3> petalTip; std::array<float,3> eye; };
    const std::array<float,3> grn{0.14f,0.26f,0.11f};
    Shot shots[] = {
        {"chrys_reflex",     flower::CHRYS_REFLEX,     40.f, {0.98f,0.78f,0.28f}, {0.86f,0.18f,0.06f}, {0.95f,0.75f,0.20f}},
        {"chrys_incurve",    flower::CHRYS_INCURVE,    36.f, {0.99f,0.82f,0.34f}, {0.92f,0.48f,0.08f}, {0.98f,0.80f,0.30f}},
        {"chrys_pompom",     flower::CHRYS_POMPOM,     34.f, {0.99f,0.99f,0.96f}, {0.86f,0.88f,0.80f}, {0.70f,0.82f,0.28f}},
        {"chrys_quill",      flower::CHRYS_QUILL,      40.f, {0.99f,0.82f,0.20f}, {0.93f,0.42f,0.10f}, {0.66f,0.80f,0.26f}},
        {"chrys_decorative", flower::CHRYS_DECORATIVE, 36.f, {0.98f,0.80f,0.90f}, {0.85f,0.16f,0.55f}, {0.92f,0.90f,0.55f}},
    };

    for (auto& s : shots) {
        flower::ChrysanthParams cp; cp.form = s.form;
        flower::FlowerMesh mesh = flower::buildChrysanthemum(cp);
        // Base palette (throat colours) + tip palette (edge colours) per group.
        setPalette(renderer, {grn, grn, s.eye, s.petalBase, {0.80f,0.20f,0.40f}});
        std::array<float,3> lgrn{0.24f,0.38f,0.16f};
        std::array<float,3> tips[5] = {grn, lgrn, s.eye, s.petalTip, {0.90f,0.30f,0.10f}};
        for (int i = 0; i < 5; ++i) { renderer.paletteTip[i][0]=tips[i][0];
            renderer.paletteTip[i][1]=tips[i][1]; renderer.paletteTip[i][2]=tips[i][2]; }
        renderer.paletteTipCount = 5;
        renderer.uploadSegments(mesh.nodes, mesh.segments, mesh.radii, &mesh.groups,
                                &mesh.prims, &mesh.frames, &mesh.aux);
        // Frame on the bloom head (near its top), not the whole-plant centroid.
        float bloomY = 0.f, maxY = -1e9f;
        for (auto& n : mesh.nodes) maxY = std::max(maxY, (float) n.y);
        bloomY = maxY - flower::ChrysanthParams{}.radius * 0.5f;
        float target[3] = {0, bloomY, 0};

        // Two views: side (~15 deg above head-on) and top (~15 deg off vertical).
        struct View { const char* suffix; float elev; float lx, ly, lz; };
        View views[] = {
            {"_side", 0.26f, 0.4f, 0.55f, 0.5f},   // 15 deg, side light
            {"_top",  1.31f, 0.25f, 0.93f, 0.25f}, // 75 deg, overhead light
        };
        std::vector<unsigned char> pix(W * H * 4);
        for (auto& v : views) {
            float lightDir[3] = {v.lx, v.ly, v.lz};
            renderer.render(0.6f, v.elev, s.radius, target, 0.5236f, lightDir);
            glBindTexture(GL_TEXTURE_2D, renderer.colorTex());
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pix.data());
            writePPM(outDir + "/" + s.name + v.suffix + ".ppm", W, H, pix);
        }
        printf("wrote %s/%s _side/_top  (segs=%zu)\n", outDir.c_str(), s.name, mesh.segments.size());
    }

    // --- Decorative growth sequence: bud -> fully open ---
    {
        setPalette(renderer, {grn, grn, {0.92f,0.90f,0.55f}, {0.98f,0.80f,0.90f}, {0.80f,0.20f,0.40f}});
        std::array<float,3> lgrn{0.24f,0.38f,0.16f};
        std::array<float,3> tips[5] = {grn, lgrn, {0.92f,0.90f,0.55f}, {0.85f,0.16f,0.55f}, {0.90f,0.30f,0.10f}};
        for (int i=0;i<5;i++){ renderer.paletteTip[i][0]=tips[i][0];
            renderer.paletteTip[i][1]=tips[i][1]; renderer.paletteTip[i][2]=tips[i][2]; }
        renderer.paletteTipCount = 5;
        std::vector<unsigned char> pix(W * H * 4);
        const int STEPS = 6;
        for (int k = 0; k < STEPS; ++k) {
            flower::ChrysanthParams cp; cp.form = flower::CHRYS_DECORATIVE;
            cp.growth = (float) k / (STEPS - 1);
            flower::FlowerMesh mesh = flower::buildChrysanthemum(cp);
            renderer.uploadSegments(mesh.nodes, mesh.segments, mesh.radii, &mesh.groups,
                                    &mesh.prims, &mesh.frames, &mesh.aux);
            float maxY = -1e9f; for (auto& n : mesh.nodes) maxY = std::max(maxY, (float) n.y);
            float target[3] = {0, maxY - 8.f, 0};
            float lightDir[3] = {0.35f, 0.8f, 0.5f};
            renderer.render(0.6f, 0.35f, 40.0f, target, 0.5236f, lightDir);
            glBindTexture(GL_TEXTURE_2D, renderer.colorTex());
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pix.data());
            char name[128]; snprintf(name, sizeof(name), "%s/chrys_grow_%d.ppm", outDir.c_str(), k);
            writePPM(name, W, H, pix);
            printf("wrote %s/chrys_grow_%d.ppm  (growth=%.2f)\n", outDir.c_str(), k, cp.growth);
        }

        // Smooth video sequence: eased growth 0->1 with a hold, slow orbit.
        const int VF = 110, OPENF = 84;   // open over OPENF frames, then hold
        for (int fr = 0; fr < VF; ++fr) {
            double gg = fr < OPENF ? double(fr) / OPENF : 1.0;
            gg = gg * gg * (3.0 - 2.0 * gg);            // ease in/out
            flower::ChrysanthParams cp; cp.form = flower::CHRYS_DECORATIVE;
            cp.growth = (float) gg;
            flower::FlowerMesh mesh = flower::buildChrysanthemum(cp);
            renderer.uploadSegments(mesh.nodes, mesh.segments, mesh.radii, &mesh.groups,
                                    &mesh.prims, &mesh.frames, &mesh.aux);
            float maxY = -1e9f; for (auto& n : mesh.nodes) maxY = std::max(maxY, (float) n.y);
            float target[3] = {0, maxY - 8.f, 0};
            float az = 0.3f + 0.6f * (float) fr / VF;   // slow orbit
            float lightDir[3] = {0.35f, 0.8f, 0.5f};
            renderer.render(az, 0.34f, 40.0f, target, 0.5236f, lightDir);
            glBindTexture(GL_TEXTURE_2D, renderer.colorTex());
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pix.data());
            char name[128]; snprintf(name, sizeof(name), "%s/gv_%03d.ppm", outDir.c_str(), fr);
            writePPM(name, W, H, pix);
        }
        printf("wrote %d video frames %s/gv_###.ppm\n", VF, outDir.c_str());
    }

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
