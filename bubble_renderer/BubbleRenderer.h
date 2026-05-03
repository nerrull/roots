#pragma once
#include <GL/glew.h>
#include <memory>

class BubbleRenderer {
public:
    struct Params {
        float filmThickness  = 450.0f;   // nm
        float bubbleRadius   = 1.5f;
        float lightAzimuth   = 0.8f;
        float lightElevation = 0.7f;
    };

    BubbleRenderer(int w, int h);
    ~BubbleRenderer();

    void   resize(int w, int h);
    void   render(float time, float azimuth, float elevation,
                  float orbitRadius, const Params& p);
    GLuint colorTex() const { return _glTex; }

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
    int    _w = 0, _h = 0;
    GLuint _glTex = 0;
};
