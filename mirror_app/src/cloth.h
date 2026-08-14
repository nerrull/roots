// Position-based-dynamics cloth on a rectangular grid with a hole cut out.
//
// The hole's rim vertices are *pinned* to the back rim of the mask (a ring in
// the z=0 plane); gravity points behind the mask (-z), so the sheet — the
// hydro-dip film / neural texture — sags backward and falls away, unveiling the
// face that sits in the hole.
//
// Header-only + SIMD (Apple <simd/simd.h>, hardware float3 vectors) so the
// solver builds and is testable with plain clang++, no Metal/GPU needed.
//
// Ported from neuromirror/cloth_cpp/src/cloth.h essentially unchanged -- it was
// already GPU-free and self-contained, which is exactly why it moved without a
// rewrite. `pinTo` is the one addition: cloth_cpp pinned the rim to a fixed
// ellipse because its face never moved, and here the face is fitted per person
// and turns with the head, so the rim has to be re-pinned to wherever the mesh
// actually is. See transition_scene.h.
#pragma once

#include <simd/simd.h>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

// simd_float3 is an ext-vector type: no aggregate brace-init in C++, so wrap the
// maker for concise construction.
static inline simd_float3 v3(float x, float y, float z) { return simd_make_float3(x, y, z); }

struct Cloth {
    struct Edge { int a, b; float rest; };

    int nx = 0, ny = 0;                 // grid resolution
    std::vector<simd_float3> pos;       // current positions
    std::vector<simd_float3> prev;      // previous (Verlet)
    std::vector<simd_float3> nrm;       // per-vertex normals (for shading)
    std::vector<uint8_t> active;        // 0 = removed (inside the hole)
    std::vector<uint8_t> pinned;        // 1 = fixed to the mask rim
    std::vector<simd_float3> pin;       // pin target position
    std::vector<Edge> edges;            // distance constraints
    std::vector<uint32_t> tris;         // render index buffer (active quads)

    simd_float3 gravity = v3(0.f, -0.3f, -9.8f);   // -z: behind the mask
    float damping = 0.99f;              // velocity retention (1 = none)
    int iterations = 24;                // constraint solve passes / step

    int idx(int i, int j) const { return j * nx + i; }

    // Build an (nx*ny) sheet of world size (w x h) centered at origin in z=0.
    // An elliptical ring of radii (hrx, hry) is pinned to the mask's back rim.
    // ``carve`` cuts a hole inside the ring (rim pinned); otherwise the sheet is
    // solid and an interior ring is pinned (the center hides behind the face).
    void build(int nx_, int ny_, float w, float h, float hrx, float hry, bool carve = true) {
        nx = nx_; ny = ny_;
        const int N = nx * ny;
        pos.assign(N, v3(0,0,0)); prev.assign(N, v3(0,0,0)); nrm.assign(N, v3(0,0,0));
        active.assign(N, 1); pinned.assign(N, 0); pin.assign(N, v3(0,0,0));
        edges.clear(); tris.clear();

        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                float x = (i / float(nx - 1) - 0.5f) * w;
                float y = (j / float(ny - 1) - 0.5f) * h;
                int k = idx(i, j);
                pos[k] = prev[k] = v3(x, y, 0.f);
                float e = (x * x) / (hrx * hrx) + (y * y) / (hry * hry);
                if (carve && e < 1.f) active[k] = 0;   // carve the hole
            }

        if (carve) {
            // rim = active vertex touching the hole -> pin to the hole boundary
            auto inHole = [&](int i, int j) {
                return i >= 0 && i < nx && j >= 0 && j < ny && !active[idx(i, j)];
            };
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    int k = idx(i, j);
                    if (!active[k]) continue;
                    if (inHole(i-1,j) || inHole(i+1,j) || inHole(i,j-1) || inHole(i,j+1) ||
                        inHole(i-1,j-1) || inHole(i+1,j-1) || inHole(i-1,j+1) || inHole(i+1,j+1)) {
                        float x = pos[k].x, y = pos[k].y;
                        float s = std::sqrt((x*x)/(hrx*hrx) + (y*y)/(hry*hry));
                        if (s > 1e-6f) { x /= s; y /= s; }
                        pinned[k] = 1;
                        pin[k] = pos[k] = prev[k] = v3(x, y, 0.f);
                    }
                }
        } else {
            // solid sheet: pin the ~1-cell-thick ring nearest the ellipse boundary
            float band = 1.5f * std::max(w / (nx - 1) / hrx, h / (ny - 1) / hry);
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    int k = idx(i, j);
                    float x = pos[k].x, y = pos[k].y;
                    float s = std::sqrt((x*x)/(hrx*hrx) + (y*y)/(hry*hry));
                    if (std::fabs(s - 1.f) < band) { pinned[k] = 1; pin[k] = pos[k] = prev[k] = v3(x, y, 0.f); }
                }
        }

        // distance constraints: structural (4-nbr), shear (diagonals), bend (2-away)
        auto add = [&](int ai, int aj, int bi, int bj) {
            if (ai < 0 || ai >= nx || aj < 0 || aj >= ny ||
                bi < 0 || bi >= nx || bj < 0 || bj >= ny) return;
            int a = idx(ai, aj), b = idx(bi, bj);
            if (!active[a] || !active[b]) return;
            edges.push_back({a, b, simd_distance(pos[a], pos[b])});
        };
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                if (!active[idx(i, j)]) continue;
                add(i, j, i+1, j);   add(i, j, i, j+1);         // structural
                add(i, j, i+1, j+1); add(i+1, j, i, j+1);       // shear
                add(i, j, i+2, j);   add(i, j, i, j+2);         // bend
            }

        // triangles for quads whose 4 corners are all active
        for (int j = 0; j < ny - 1; ++j)
            for (int i = 0; i < nx - 1; ++i) {
                int a = idx(i, j), b = idx(i+1, j), c = idx(i, j+1), d = idx(i+1, j+1);
                if (!(active[a] && active[b] && active[c] && active[d])) continue;
                tris.insert(tris.end(), {(uint32_t)a,(uint32_t)c,(uint32_t)b,
                                         (uint32_t)b,(uint32_t)c,(uint32_t)d});
            }
    }

    // Move the pinned ring onto a new set of target positions, supplied in the
    // same order the pins were created. cloth_cpp had no need for this: its face
    // was a baked asset at a fixed place, so the rim ellipse was decided at build
    // time and never moved. A fitted face is a different shape per person and
    // turns with the head, so the rim has to follow it or the sheet detaches
    // from the mask it is supposed to be hanging off.
    //
    // Positions only -- the rest lengths stay as built, so re-pinning stretches
    // the sheet toward the new rim rather than rebuilding the constraint graph.
    void pinTo(const std::vector<simd_float3>& targets) {
        size_t t = 0;
        for (size_t k = 0; k < pos.size() && t < targets.size(); ++k) {
            if (!pinned[k]) continue;
            pin[k] = pos[k] = prev[k] = targets[t++];
        }
    }

    // The pinned vertices in build order, so a caller can see how many targets
    // pinTo() wants and where they currently sit.
    std::vector<int> pinIndices() const {
        std::vector<int> out;
        for (size_t k = 0; k < pos.size(); ++k) if (pinned[k]) out.push_back(int(k));
        return out;
    }

    // One Verlet step + constraint projection. dt in seconds.
    void step(float dt) {
        const float dt2 = dt * dt;
        for (size_t k = 0; k < pos.size(); ++k) {
            if (!active[k] || pinned[k]) continue;
            simd_float3 tmp = pos[k];
            pos[k] += (pos[k] - prev[k]) * damping + gravity * dt2;
            prev[k] = tmp;
        }
        // Gauss-Seidel projection. Alternating the sweep direction each iteration
        // cancels the ordering bias that otherwise makes the sheet far stiffer
        // along one grid axis -- which on a regular grid seeds axis-aligned
        // buckling (the sheet gathers on one axis and stays taut on the other).
        const int M = (int)edges.size();
        for (int it = 0; it < iterations; ++it) {
            bool fwd = (it & 1) == 0;
            for (int n = 0; n < M; ++n) {
                const Edge& e = edges[fwd ? n : M - 1 - n];
                simd_float3 d = pos[e.b] - pos[e.a];
                float len = simd_length(d);
                if (len < 1e-8f) continue;
                float mA = pinned[e.a] ? 0.f : 1.f;
                float mB = pinned[e.b] ? 0.f : 1.f;
                float s = mA + mB;
                if (s == 0.f) continue;
                simd_float3 corr = d * ((len - e.rest) / len);
                pos[e.a] += corr * (mA / s);
                pos[e.b] -= corr * (mB / s);
            }
        }
        for (size_t k = 0; k < pos.size(); ++k)
            if (pinned[k]) pos[k] = prev[k] = pin[k];       // hold the rim
    }

    // Per-vertex normals from the triangle mesh (call before rendering).
    void computeNormals() {
        for (auto& n : nrm) n = v3(0.f, 0.f, 0.f);
        for (size_t t = 0; t + 2 < tris.size(); t += 3) {
            uint32_t a = tris[t], b = tris[t+1], c = tris[t+2];
            simd_float3 fn = simd_cross(pos[b] - pos[a], pos[c] - pos[a]);
            nrm[a] += fn; nrm[b] += fn; nrm[c] += fn;
        }
        for (size_t k = 0; k < nrm.size(); ++k) {
            float l = simd_length(nrm[k]);
            nrm[k] = l > 1e-8f ? nrm[k] / l : v3(0.f, 0.f, 1.f);
        }
    }

    // Diagnostics for the headless test.
    int activeCount() const { int c = 0; for (auto a : active) c += a; return c; }
    int pinnedCount() const { int c = 0; for (auto p : pinned) c += p; return c; }
    bool finite() const {
        for (const auto& p : pos)
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) return false;
        return true;
    }
    float minZ() const {
        float m = 0;
        for (size_t k = 0; k < pos.size(); ++k) if (active[k]) m = std::min(m, pos[k].z);
        return m;
    }
};
