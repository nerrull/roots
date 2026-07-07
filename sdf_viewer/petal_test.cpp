// Petal test harness: renders ONE petal in isolation across a range of curl
// (lengthwise) and latCup (cross-section) values, each from several angles, so
// the petal SDF can be inspected on its own -- curve direction, cup depth, and
// any renderer artefacts -- without the whole bloom on top. Output: PPMs named
// petal_c<ci>_l<li>_a<ai>.ppm which compare_petals.py montages into a grid.
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "RootRenderer.h"
#include "FlowerLSystem.h"

#include <cstdio>
#include <string>
#include <vector>

static void writePPM(const std::string& path, int w, int h, const std::vector<unsigned char>& rgba) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    for (int y = h - 1; y >= 0; --y)
        for (int x = 0; x < w; ++x) {
            const unsigned char* px = &rgba[(y * w + x) * 4];
            fputc(px[0], f); fputc(px[1], f); fputc(px[2], f);
        }
    fclose(f);
}

int main(int argc, char** argv) {
    std::string outDir = argc > 1 ? argv[1] : "/tmp";
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* win = glfwCreateWindow(500, 500, "petal", nullptr, nullptr);
    if (!win) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return 1;

    const int W = 500, H = 500;
    RootRenderer renderer(W, H);
    renderer.mat.ambient = 0.30f;
    renderer.mat.diffuse = 0.95f;
    renderer.mat.specColor[0]=0.14f; renderer.mat.specColor[1]=0.13f; renderer.mat.specColor[2]=0.12f;
    renderer.mat.shininess = 18.0f;
    renderer.fog.density = 0.0f;                 // no fog -- clean look at geometry
    renderer.wispCount = 0;
    // Palette: petal group (index 3) pale base -> magenta tip; grid overlay off.
    for (int i=0;i<5;i++){ renderer.palette[i][0]=0.4f; renderer.palette[i][1]=0.4f; renderer.palette[i][2]=0.4f; }
    renderer.palette[3][0]=0.98f; renderer.palette[3][1]=0.80f; renderer.palette[3][2]=0.90f;
    renderer.paletteCount = 5;
    for (int i=0;i<5;i++){ renderer.paletteTip[i][0]=renderer.palette[i][0];
        renderer.paletteTip[i][1]=renderer.palette[i][1]; renderer.paletteTip[i][2]=renderer.palette[i][2]; }
    renderer.paletteTip[3][0]=0.85f; renderer.paletteTip[3][1]=0.16f; renderer.paletteTip[3][2]=0.55f;
    renderer.paletteTipCount = 5;
    renderer.overlay.showAxes = true;            // axes: +X red-ish, +Y green, +Z blue in fog.frag
    renderer.overlay.axisLength = 6.0f;

    float curls[]   = {-0.7f, 0.0f, 0.7f};
    // latCup as a roll angle (radians): 0 flat, ~1.6 deep cup, ~3.1(=PI) tube.
    float latCups[] = {1.0f, 2.0f, 3.0f};
    // angle 0: side view down -Z (see the X/Y profile -> lengthwise curl up/down)
    // angle 1: view down the petal axis-ish (see the cross-section cup)
    // angle 2: 3/4
    struct Ang { float az, el; };
    Ang angs[] = {{0.0f, 0.0f}, {1.571f, 0.15f}, {0.7f, 0.5f}};

    std::vector<unsigned char> pix(W * H * 4);
    for (int ci = 0; ci < 3; ++ci)
        for (int li = 0; li < 3; ++li) {
            flower::FlowerMesh m;
            m.curGroup = flower::G_PETAL;
            // Petal along +X, world up +Y. length 10, width 4, thin.
            m.curvedPetal(flower::Vector3d(0,0,0), flower::Vector3d(1,0,0), flower::Vector3d(0,1,0),
                          10.0, 4.0, curls[ci], 0.18, flower::PRIM_PETAL, latCups[li], 0.0);
            renderer.uploadSegments(m.nodes, m.segments, m.radii, &m.groups, &m.prims, &m.frames, &m.aux);
            float target[3] = {5.0f, 0.0f, 0.0f};
            for (int ai = 0; ai < 3; ++ai) {
                float lightDir[3] = {0.3f, 0.8f, 0.5f};
                renderer.render(angs[ai].az, angs[ai].el, 24.0f, target, 0.5236f, lightDir);
                glBindTexture(GL_TEXTURE_2D, renderer.colorTex());
                glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pix.data());
                char name[128];
                snprintf(name, sizeof(name), "%s/petal_c%d_l%d_a%d.ppm", outDir.c_str(), ci, li, ai);
                writePPM(name, W, H, pix);
            }
        }
    // Rotation sweep of ONE aggressively-cupped petal (curl 0.7, latCup 2.5) to
    // hunt clipping across all viewing angles.
    {
        flower::FlowerMesh m;
        m.curGroup = flower::G_PETAL;
        m.curvedPetal(flower::Vector3d(0,0,0), flower::Vector3d(1,0,0), flower::Vector3d(0,1,0),
                      10.0, 4.0, 0.7, 0.18, flower::PRIM_PETAL, 2.5, 0.0);
        renderer.uploadSegments(m.nodes, m.segments, m.radii, &m.groups, &m.prims, &m.frames, &m.aux);
        float target[3] = {5.0f, 0.0f, 0.0f};
        for (int ai = 0; ai < 8; ++ai) {
            float az = 6.2831853f * ai / 8.0f;
            float lightDir[3] = {0.3f, 0.8f, 0.5f};
            renderer.render(az, 0.35f, 24.0f, target, 0.5236f, lightDir);
            glBindTexture(GL_TEXTURE_2D, renderer.colorTex());
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pix.data());
            char name[128];
            snprintf(name, sizeof(name), "%s/petal_rot_%d.ppm", outDir.c_str(), ai);
            writePPM(name, W, H, pix);
        }
    }
    printf("done: grid + rotation sweep\n");
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
