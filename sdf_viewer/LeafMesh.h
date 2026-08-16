#pragma once
// ---------------------------------------------------------------------------
// Triangulated chrysanthemum leaf.
//
// WHY A MESH. The blade SDF builds a leaf as a union of capsules, which forces
// two things that read as moulded plastic no matter how the outline is tuned:
// every lobe ends in a circular arc of its capsule's radius, and the blade keeps
// a near-constant thickness with a rounded edge all the way around its margin.
// Real leaves are the opposite -- the margin goes to a knife edge, and the only
// thick parts are the midrib and the veins. A mesh can do exactly that: the
// cross-section here is a lens that thins to nothing at the boundary, swelling
// only where a vein runs under it.
//
// The SILHOUETTE is still the pinnate construction from the SDF (a union of
// oriented, tapering, toothed lobes evaluated in the leaf's own 2D plane),
// because that part was right -- lobes swept forward toward the tip, separated
// by deep narrow sinuses. Here it is used as a boundary to mesh up to rather
// than as a solid to march, so it costs nothing at render time.
//
// Renderer-agnostic on purpose: plain C++ with no GL/Metal and no CPlantBox
// types, emitting interleaved triangles. Output layout is 12 floats per vertex
// -- pos3, normal3, colour3, param3 -- matching the mid-geometry mesh vertex
// both backends already take (mirror_app's root_face.metal / sdf_viewer's
// face.vert). param is (s along the midrib, t across, vein proximity).
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <vector>

namespace leafmesh {

struct V3 {
    float x = 0, y = 0, z = 0;
    V3() = default;
    V3(float a, float b, float c) : x(a), y(b), z(c) {}
    V3 operator+(const V3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    V3 operator-(const V3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    V3 operator*(float s)     const { return {x * s, y * s, z * s}; }
};
inline float dot(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(const V3& a, const V3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline V3 normalize(const V3& v) {
    float l = std::sqrt(dot(v, v));
    return l < 1e-9f ? V3(0, 0, 1) : v * (1.0f / l);
}

struct LeafParams {
    float length      = 12.0f;   // cm, petiole junction to tip
    float halfWidth   = 4.6f;    // cm, half the widest span
    float thickness   = 0.075f;  // cm, half-thickness along the midrib at the base
    float cup         = 0.16f;   // cross-channel depth (fraction of halfWidth)
    float curl        = 0.30f;   // lengthwise droop of the blade
    float veinRelief  = 0.70f;   // rib swelling over a vein, as a multiple of thickness
    float veinWidth   = 0.055f;  // vein rib width, as a fraction of halfWidth
    int   sSteps      = 56;      // stations along the midrib
    int   tSteps      = 16;      // stations from midrib to margin (per side)
    float baseColor[3] = {0.13f, 0.25f, 0.10f};
    float tipColor[3]  = {0.26f, 0.42f, 0.17f};
};

// --- 2D silhouette, in the leaf's own plane: x along the midrib in [0,L], y
//     across, folded to |y| so one set of lobes serves both margins. ---------

// One lobe: a 2D capsule a->b whose radius lerps r0->r1, nibbled by `teeth`
// marginal serrations at `tf` cycles along its length.
inline float sdLobe2(float px, float py, float ax, float ay, float bx, float by,
                     float r0, float r1, float teeth, float tf) {
    float pax = px - ax, pay = py - ay;
    float bax = bx - ax, bay = by - ay;
    float bb  = bax * bax + bay * bay;
    float h   = bb > 1e-8f ? (pax * bax + pay * bay) / bb : 0.0f;
    h = h < 0 ? 0 : (h > 1 ? 1 : h);
    float r = (r0 + (r1 - r0) * h)
            * (1.0f - teeth * (0.5f + 0.5f * std::cos(6.2831853f * h * tf)));
    float dx = pax - bax * h, dy = pay - bay * h;
    return std::sqrt(dx * dx + dy * dy) - r;
}

// <0 inside the leaf. L = length, W = half width.
inline float leafOutline(float px, float py, float L, float W) {
    py = std::fabs(py);
    // Central body: a wedge widening from the cuneate base toward the tip. It
    // stays substantial -- the sinuses cut about half way in on a real leaf, and
    // a thin body turns the lobes into the arms of a starfish.
    float d = sdLobe2(px, py, 0.02f * L, 0.0f, 0.46f * L, 0.0f, W * 0.09f, W * 0.34f, 0.0f, 1.0f);
    d = std::min(d, sdLobe2(px, py, 0.46f * L, 0.0f, 0.78f * L, 0.0f, W * 0.34f, W * 0.30f, 0.0f, 1.0f));
    // Terminal lobe, tapering to a blunt point.
    d = std::min(d, sdLobe2(px, py, 0.60f * L, 0.0f, 0.99f * L, 0.0f, W * 0.30f, W * 0.07f, 0.16f, 3.0f));
    // Three pairs of lateral lobes, swept forward, each with its own teeth.
    for (int i = 0; i < 3; ++i) {
        float f   = (float) i;
        float ax  = (0.15f + 0.225f * f) * L, ay = 0.0f;
        float len = W * (0.46f + 0.30f * std::sin(3.14159265f * (f + 0.85f) / 3.0f));
        float ang = (58.0f - 7.0f * f) * 3.14159265f / 180.0f;
        d = std::min(d, sdLobe2(px, py, ax, ay,
                                ax + std::cos(ang) * len, ay + std::sin(ang) * len,
                                W * 0.30f, W * 0.13f, 0.22f, 2.5f));
    }
    return d;
}

// Distance to the venation: the midrib plus one vein into each lateral lobe.
inline float veinDist(float px, float py, float L, float W) {
    py = std::fabs(py);
    auto seg = [&](float ax, float ay, float bx, float by) {
        float pax = px - ax, pay = py - ay, bax = bx - ax, bay = by - ay;
        float bb = bax * bax + bay * bay;
        float h  = bb > 1e-8f ? (pax * bax + pay * bay) / bb : 0.0f;
        h = h < 0 ? 0 : (h > 1 ? 1 : h);
        float dx = pax - bax * h, dy = pay - bay * h;
        return std::sqrt(dx * dx + dy * dy);
    };
    float d = seg(0.02f * L, 0.0f, 0.97f * L, 0.0f);
    for (int i = 0; i < 3; ++i) {
        float f   = (float) i;
        float ax  = (0.15f + 0.225f * f) * L;
        float len = W * (0.46f + 0.30f * std::sin(3.14159265f * (f + 0.85f) / 3.0f));
        float ang = (58.0f - 7.0f * f) * 3.14159265f / 180.0f;
        d = std::min(d, seg(ax, 0.0f, ax + std::cos(ang) * len * 0.88f, std::sin(ang) * len * 0.88f));
    }
    return d;
}

// Half-width of the leaf at station x: the largest |y| still inside. Scanned
// then bisected, so a station that falls in a sinus (where the boundary dives
// toward the midrib) is found correctly rather than assumed convex.
inline float boundaryAt(float x, float L, float W) {
    if (leafOutline(x, 0.0f, L, W) > 0.0f) return 0.0f;    // past the tip / before the base
    const int N = 24;
    float lastIn = 0.0f, firstOut = -1.0f;
    for (int i = 1; i <= N; ++i) {
        float y = W * 1.25f * (float) i / N;
        if (leafOutline(x, y, L, W) <= 0.0f) lastIn = y;
        else { firstOut = y; break; }
    }
    if (firstOut < 0.0f) return lastIn;
    float lo = lastIn, hi = firstOut;
    for (int it = 0; it < 18; ++it) {
        float mid = 0.5f * (lo + hi);
        if (leafOutline(x, mid, L, W) <= 0.0f) lo = mid; else hi = mid;
    }
    return 0.5f * (lo + hi);
}

// A leaf's placement: origin at the blade base, `axis` along the midrib toward
// the tip, `up` the direction the leaf's face turns toward. They are
// orthonormalised here, so a caller can hand over any reasonable pair.
struct Frame {
    V3 origin{0, 0, 0};
    V3 axis{1, 0, 0};
    V3 up{0, 1, 0};
};

// Append one leaf to `out` as interleaved triangles (12 floats/vertex:
// pos3, normal3, colour3, param3 = (s, t, vein)).
inline void emitLeaf(std::vector<float>& out, const LeafParams& P, const Frame& F) {
    const int   NS = std::max(8, P.sSteps), NT = std::max(4, P.tSteps);
    const float L = P.length, W = P.halfWidth;

    V3 ax = normalize(F.axis);
    V3 up = normalize(F.up);
    V3 wd = cross(up, ax);                       // across the blade
    if (std::sqrt(dot(wd, wd)) < 1e-5f) wd = cross(V3(0, 0, 1), ax);
    wd = normalize(wd);
    V3 nz = normalize(cross(ax, wd));            // the face's own normal

    // --- Mid-surface, half-thickness and colour on an (s,t) grid. t runs the
    //     full span -1..1 so the two margins are grid borders, where the
    //     thickness goes to zero and the sheets close on a sharp edge.
    const int NW = 2 * NT + 1;
    std::vector<V3>    pos((size_t) (NS + 1) * NW);
    std::vector<float> half((size_t) (NS + 1) * NW);
    std::vector<float> vprox((size_t) (NS + 1) * NW);
    auto idx = [&](int i, int j) { return (size_t) i * NW + j; };

    for (int i = 0; i <= NS; ++i) {
        float s = (float) i / NS;
        float x = s * L;
        float ymax = boundaryAt(x, L, W);
        // Lengthwise droop, and a channel that is deepest near the base.
        float bend = -P.curl * 0.6f * L * s * s;
        for (int j = 0; j < NW; ++j) {
            float t = (float) (j - NT) / NT;            // -1 .. 1
            float y = t * ymax;
            float u = ymax > 1e-5f ? std::fabs(y) / ymax : 0.0f;
            // Channel measured in ABSOLUTE width, not in the normalised u: tied
            // to u it compresses in the sinuses and stretches over the lobes, and
            // the lamina ends up rippling lobe to lobe like poured wax.
            float cupz = -P.cup * (y * y) / std::max(W, 1e-4f) * (1.0f - 0.5f * s);
            float dv   = veinDist(x, y, L, W) / std::max(P.veinWidth * W, 1e-4f);
            float vp   = std::exp(-dv * dv);
            // Lens cross-section: full thickness on the midrib, zero at the
            // margin. This is the whole point of meshing the leaf.
            float lens = std::sqrt(std::max(0.0f, 1.0f - u * u));
            float th   = P.thickness * (1.0f - 0.35f * s) * lens * (1.0f + P.veinRelief * vp);
            if (i == 0 || i == NS) th *= 0.25f;          // close base and tip
            pos[idx(i, j)]   = F.origin + ax * x + wd * y + nz * (bend + cupz + vp * P.thickness * 0.30f);
            half[idx(i, j)]  = th;
            vprox[idx(i, j)] = vp;
        }
    }

    // --- Mid-surface normals by central differences on the grid. ---
    std::vector<V3> nrm((size_t) (NS + 1) * NW);
    for (int i = 0; i <= NS; ++i) {
        for (int j = 0; j < NW; ++j) {
            int i0 = std::max(0, i - 1), i1 = std::min(NS, i + 1);
            int j0 = std::max(0, j - 1), j1 = std::min(NW - 1, j + 1);
            V3 du = pos[idx(i1, j)] - pos[idx(i0, j)];
            V3 dv = pos[idx(i, j1)] - pos[idx(i, j0)];
            V3 n  = normalize(cross(du, dv));
            if (dot(n, nz) < 0.0f) n = n * -1.0f;        // keep the face outward
            nrm[idx(i, j)] = n;
        }
    }

    // Emit both sheets. Winding is set so the top sheet faces +n; the mid-
    // geometry pass draws with culling off, so a flipped triangle still shades.
    auto vertex = [&](int i, int j, int sideSign) {
        float s = (float) i / NS;
        float t = (float) (j - NT) / NT;
        V3 n = nrm[idx(i, j)] * (float) sideSign;
        V3 p = pos[idx(i, j)] + n * half[idx(i, j)];
        float vp = vprox[idx(i, j)];
        out.push_back(p.x); out.push_back(p.y); out.push_back(p.z);
        out.push_back(n.x); out.push_back(n.y); out.push_back(n.z);
        for (int c = 0; c < 3; ++c) {
            float col = P.baseColor[c] + (P.tipColor[c] - P.baseColor[c]) * s;
            col *= (sideSign > 0 ? 1.0f : 0.88f);
            out.push_back(col + 0.04f * vp);
        }
        out.push_back(s); out.push_back(t); out.push_back(vp);
    };

    for (int side = 0; side < 2; ++side) {
        int sgn = side == 0 ? +1 : -1;
        for (int i = 0; i < NS; ++i) {
            for (int j = 0; j < NW - 1; ++j) {
                // Skip quads with no area -- stations past the tip collapse onto
                // the midrib, and a fan of zero-area triangles there is just
                // wasted vertices.
                V3 e0 = pos[idx(i + 1, j)] - pos[idx(i, j)];
                V3 e1 = pos[idx(i, j + 1)] - pos[idx(i, j)];
                if (dot(cross(e0, e1), cross(e0, e1)) < 1e-12f) continue;
                if (sgn > 0) {
                    vertex(i, j, sgn);     vertex(i + 1, j, sgn); vertex(i + 1, j + 1, sgn);
                    vertex(i, j, sgn);     vertex(i + 1, j + 1, sgn); vertex(i, j + 1, sgn);
                } else {
                    vertex(i, j, sgn);     vertex(i + 1, j + 1, sgn); vertex(i + 1, j, sgn);
                    vertex(i, j, sgn);     vertex(i, j + 1, sgn); vertex(i + 1, j + 1, sgn);
                }
            }
        }
    }
}

// A petiole is a swept tube, which the capsule renderer already draws well, so
// it stays on that path; this header is only the blade.

}  // namespace leafmesh
