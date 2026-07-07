// Leaf test harness: renders ONE chrysanthemum leaf (emitLobedLeaf / PRIM_LEAF)
// broadside and from a few angles, so the lobed+serrated leaf outline can be
// inspected on its own. Output: leaf_a<ai>.ppm. Mirrors petal_test.cpp.
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
    GLFWwindow* win = glfwCreateWindow(500, 500, "leaf", nullptr, nullptr);
    if (!win) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return 1;

    const int W = 500, H = 500;
    RootRenderer renderer(W, H);
    renderer.mat.ambient = 0.30f; renderer.mat.diffuse = 0.95f;
    renderer.mat.specColor[0]=0.14f; renderer.mat.specColor[1]=0.13f; renderer.mat.specColor[2]=0.12f;
    renderer.mat.shininess = 18.0f;
    renderer.fog.density = 0.0f; renderer.wispCount = 0;
    // Leaf group (index 1) dark green base -> lighter green tip.
    for (int i=0;i<5;i++){ renderer.palette[i][0]=0.4f; renderer.palette[i][1]=0.4f; renderer.palette[i][2]=0.4f; }
    renderer.palette[1][0]=0.14f; renderer.palette[1][1]=0.26f; renderer.palette[1][2]=0.11f;
    renderer.paletteCount = 5;
    for (int i=0;i<5;i++){ renderer.paletteTip[i][0]=renderer.palette[i][0];
        renderer.paletteTip[i][1]=renderer.palette[i][1]; renderer.paletteTip[i][2]=renderer.palette[i][2]; }
    renderer.paletteTip[1][0]=0.24f; renderer.paletteTip[1][1]=0.38f; renderer.paletteTip[1][2]=0.16f;
    renderer.paletteTipCount = 5;

    // One leaf: midrib along +X in the X/Z plane (so a top-down view sees the
    // full outline), world up +Y. length 12, width 9.
    flower::FlowerMesh m;
    m.curGroup = flower::G_LEAF;
    flower::emitLobedLeaf(m, flower::Vector3d(0,0,0), flower::Vector3d(1,0,0),
                          flower::Vector3d(0,1,0), 12.0, 9.0, 0.25);
    renderer.uploadSegments(m.nodes, m.segments, m.radii, &m.groups, &m.prims, &m.frames, &m.aux);
    float target[3] = {6.0f, 0.0f, 0.0f};

    // Views: broadside from above (see the outline), 3/4, and edge-on.
    struct Ang { const char* s; float az, el; };
    Ang angs[] = {{"_face", 0.0f, 1.45f}, {"_34", 0.6f, 0.7f}, {"_edge", 1.571f, 0.12f}};
    std::vector<unsigned char> pix(W * H * 4);
    for (auto& an : angs) {
        float lightDir[3] = {0.3f, 0.85f, 0.4f};
        renderer.render(an.az, an.el, 26.0f, target, 0.5236f, lightDir);
        glBindTexture(GL_TEXTURE_2D, renderer.colorTex());
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pix.data());
        writePPM(outDir + "/leaf" + an.s + ".ppm", W, H, pix);
    }
    printf("done: leaf views\n");
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
