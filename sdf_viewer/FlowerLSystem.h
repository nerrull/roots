#pragma once
// Flower L-systems -> capsule geometry for RootRenderer.
//
// RootRenderer draws a list of constant-radius capsules (one radius per
// segment) sphere-traced on the GPU -- the same node/segment/radius arrays a
// CPlantBox SegmentAnalyser produces. This header generates those arrays
// directly from an L-system turtle plus a couple of hand-built flower forms,
// so the existing renderer (fog, pulses, wisps, PBR/phong) draws flowers with
// no shader changes.
//
// Two layers:
//   1. A generic 3D bracketed L-system (Turtle + LSystem) -- expands a string
//      axiom under production rules and walks it with a turtle, emitting tubes.
//      Good for stems, branching sprays, ferny leaves.
//   2. buildSunflower() -- a stem grown by the L-system, topped with a
//      phyllotaxis capitulum: golden-angle (137.5 deg) disk florets in a Vogel
//      spiral, ringed by lanceolate ray petals. This is the part that reads as
//      an actual sunflower; pure string rewriting doesn't give you the seed
//      spiral, so it's built parametrically.
//
// NOTE: the renderer applies ONE material to every capsule, so the whole plant
// is a single colour (tune baseColor/baseColor2 + colour noise for a
// disk-vs-petal-ish gradient). Per-part colour would need a shader change; see
// buildSunflower()'s comments for where a colour attribute would hook in.

#include "mymath.h"

#include <cmath>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace flower {

using CPlantBox::Vector2i;
using CPlantBox::Vector3d;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Golden angle -- consecutive florets/leaves rotate by this for phyllotaxis.
static constexpr double kGoldenAngle = M_PI * (3.0 - 2.23606797749979);  // ~137.507 deg

// Colour-group indices -- a segment's `group` is an index into the renderer's
// palette (RootRenderer::palette). Keep these in sync with the palette the
// caller uploads; buildSunflower/etc. tag segments with these and the viewer
// installs a matching palette.
enum Group {
    G_STEM   = 0,   // stalk, branches
    G_LEAF   = 1,   // foliage
    G_DISK   = 2,   // seed-head / disk florets
    G_PETAL  = 3,   // ray petals / ligules
    G_ACCENT = 4,   // petal tips, calyx, secondary tint
};

// Primitive types, matched to shader.frag's `prim` switch.
//   CAPSULE = tube/quill; BLADE = pointed lanceolate leaf; PETAL = broad
//   round-tipped spoon petal (same SDF, rounder outline).
enum Prim { PRIM_CAPSULE = 0, PRIM_BLADE = 1, PRIM_PETAL = 2, PRIM_LEAF = 3 };

// Rodrigues rotation of v about a unit axis k (defined below; forward-declared
// here because FlowerMesh::curvedPetal uses it).
inline Vector3d rotAxis(const Vector3d& v, const Vector3d& k, double angle);

// ---------------------------------------------------------------------------
// Output mesh: exactly what RootRenderer::uploadSegments() wants, plus
// parallel per-segment arrays for palette group, primitive type, and (for
// blades) a frame vec4 = half-width vector xyz + curl w.
// ---------------------------------------------------------------------------
struct FlowerMesh {
    std::vector<Vector3d> nodes;
    std::vector<Vector2i> segments;   // (parentNodeIdx, childNodeIdx)
    std::vector<double>   radii;      // one per segment (capsule radius / blade half-thickness)
    std::vector<int>      groups;     // one per segment (palette index)
    std::vector<int>      prims;      // one per segment (Prim)
    std::vector<float>    frames;     // four per segment (blade half-width xyz + curl)
    std::vector<float>    aux;        // four per segment (s0,s1,grad0,grad1)

    int curGroup = G_STEM;            // group applied to subsequently added segments

    int addNode(const Vector3d& p) {
        nodes.push_back(p);
        return static_cast<int>(nodes.size()) - 1;
    }
    void addSeg(int a, int b, double r) {
        segments.push_back(Vector2i(a, b));
        radii.push_back(r);
        groups.push_back(curGroup);
        prims.push_back(PRIM_CAPSULE);
        frames.insert(frames.end(), {0.f, 0.f, 0.f, 0.f});
        aux.insert(aux.end(), {0.f, 1.f, 0.f, 0.f});   // whole span, no cup/bias
    }

    // A blade/petal primitive: one flattened surface spanning base->tip.
    // widthVec is perpendicular to the axis, length = half-width; thickness is
    // the half-thickness; curl folds/droops it. `prim` picks the outline
    // (PRIM_BLADE = pointed leaf, PRIM_PETAL = round spoon). s0/s1 give this
    // segment's fractional span of the whole petal (for multi-segment curved
    // strips); `curl` is the lengthwise bend, `latCup` the cross-section spoon
    // scoop (independent); `gradBias` shifts the base->tip colour gradient.
    void blade(const Vector3d& base, const Vector3d& tip, const Vector3d& widthVec,
               double thickness, double curl, int prim = PRIM_BLADE,
               double s0 = 0.0, double s1 = 1.0, double latCup = 0.0, double gradBias = 0.0) {
        int ia = addNode(base), ib = addNode(tip);
        segments.push_back(Vector2i(ia, ib));
        radii.push_back(thickness);
        groups.push_back(curGroup);
        prims.push_back(prim);
        frames.insert(frames.end(), {(float) widthVec.x, (float) widthVec.y,
                                     (float) widthVec.z, (float) curl});
        aux.insert(aux.end(), {(float) s0, (float) s1, (float) latCup, (float) gradBias});
    }

    // A curved petal as a SINGLE primitive: the blade SDF bends along its length
    // (curl) and cups across its width (latCup) as smooth 2D functions, so there
    // are no segment seams and the whole surface has one exact bounding box.
    // `curl` is the lengthwise curl (+ curls toward the blade normal), `latCup`
    // the cross-section spoon depth. Width axis is horizontal (h x up) so the
    // face reads broadside and the normal points roughly up.
    void curvedPetal(const Vector3d& base, const Vector3d& dir, const Vector3d& up,
                     double length, double width, double curl, double thickness,
                     int prim = PRIM_PETAL, double latCup = 0.9, double gradBias = 0.0) {
        Vector3d h = dir.normalized();
        Vector3d wax = h.cross(up);                       // width axis (horizontal)
        if (wax.length() < 1e-5) wax = h.cross(Vector3d(1, 0, 0));
        wax = wax.normalized();
        Vector3d tip = base.plus(h.times(length));
        blade(base, tip, wax.times(width * 0.5), thickness, curl, prim, 0.0, 1.0, latCup, gradBias);
    }

    // Straight tube a->b, subdivided into `sub` capsules whose radius lerps
    // r0->r1 (so a single tapering stem is smooth under lighting, not a cone
    // of hard radius steps). Returns the index of the final node.
    int tube(const Vector3d& a, const Vector3d& b, double r0, double r1, int sub = 1) {
        sub = sub < 1 ? 1 : sub;
        int prev = addNode(a);
        for (int i = 1; i <= sub; ++i) {
            double t = static_cast<double>(i) / sub;
            Vector3d p = a.times(1.0 - t).plus(b.times(t));
            int cur = addNode(p);
            addSeg(prev, cur, r0 + (r1 - r0) * (t - 0.5 / sub));
            prev = cur;
        }
        return prev;
    }

    void append(const FlowerMesh& o) {
        int base = static_cast<int>(nodes.size());
        for (auto& n : o.nodes) nodes.push_back(n);
        for (size_t i = 0; i < o.segments.size(); ++i) {
            segments.push_back(Vector2i(o.segments[i].x + base, o.segments[i].y + base));
            radii.push_back(o.radii[i]);
            groups.push_back(o.groups[i]);
            prims.push_back(o.prims[i]);
            frames.insert(frames.end(), o.frames.begin() + i * 4, o.frames.begin() + i * 4 + 4);
            aux.insert(aux.end(), o.aux.begin() + i * 4, o.aux.begin() + i * 4 + 4);
        }
    }

    void centreXZ() {
        if (nodes.empty()) return;
        double sx = 0, sz = 0;
        for (auto& n : nodes) { sx += n.x; sz += n.z; }
        sx /= nodes.size(); sz /= nodes.size();
        for (auto& n : nodes) { n.x -= sx; n.z -= sz; }
    }
};

// Rodrigues rotation of v about a unit axis k by angle (radians).
inline Vector3d rotAxis(const Vector3d& v, const Vector3d& k, double angle) {
    double c = std::cos(angle), s = std::sin(angle);
    Vector3d term1 = v.times(c);
    Vector3d term2 = k.cross(v).times(s);
    Vector3d term3 = k.times(k.times(v) * (1.0 - c));   // k * (k·v)(1-c)
    return term1.plus(term2).plus(term3);
}

// ---------------------------------------------------------------------------
// Turtle: an oriented frame walking through space, emitting tubes as it goes.
//   H = heading (direction of travel), L = left, U = up. Kept orthonormal.
// ---------------------------------------------------------------------------
struct Turtle {
    Vector3d pos{0, 0, 0};
    Vector3d H{0, 1, 0};    // grow up (+Y) by default -- upright in the viewer
    Vector3d L{1, 0, 0};
    Vector3d U{0, 0, 1};
    double   width = 0.4;   // current tube radius (cm)
    double   step  = 2.0;   // forward step length (cm)

    void yaw(double a)   { H = rotAxis(H, U, a); L = rotAxis(L, U, a); }   // around up
    void pitch(double a) { H = rotAxis(H, L, a); U = rotAxis(U, L, a); }   // around left
    void roll(double a)  { L = rotAxis(L, H, a); U = rotAxis(U, H, a); }   // around heading
};

// ---------------------------------------------------------------------------
// Generic bracketed L-system.
//
// Alphabet (classic Lindenmayer turtle set):
//   F  forward, drawing a tube of length `step`, radius `width`
//   f  forward without drawing (a gap)
//   +  yaw left      -   yaw right     (about U, by `delta`)
//   &  pitch down    ^   pitch up      (about L)
//   \  roll left     /   roll right    (about H)
//   |  turn 180 deg (yaw by pi)
//   [  push state    ]   pop state
//   !  thin the tube (width *= widthScale)
//   L  emit a leaf (a small lanceolate blade of tubes) at the current frame
// Any other symbol is ignored (useful as a rewrite-only marker, e.g. X).
// ---------------------------------------------------------------------------
struct LSystem {
    std::string              axiom;
    std::map<char, std::string> rules;
    int    iterations = 4;
    double delta      = 25.0 * M_PI / 180.0;   // default branching angle
    double step       = 2.0;
    double width      = 0.4;
    double widthScale = 0.75;                  // '!' multiplier and per-branch taper
    double leafLength = 3.0;
    double leafWidth  = 1.4;
    double jitter     = 0.0;                    // radians of random angle noise per turn
    unsigned seed     = 1;

    std::string expand() const {
        std::string s = axiom;
        for (int it = 0; it < iterations; ++it) {
            std::string next;
            next.reserve(s.size() * 3);
            for (char c : s) {
                auto r = rules.find(c);
                if (r != rules.end()) next += r->second;
                else                  next += c;
            }
            s.swap(next);
        }
        return s;
    }

    // Interpret the expanded string, appending geometry into `mesh` starting
    // from turtle `t0`.
    void interpret(FlowerMesh& mesh, const Turtle& t0) const {
        std::string s = expand();
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> jd(-jitter, jitter);
        auto ja = [&](double a) { return jitter > 0.0 ? a + jd(rng) : a; };

        std::vector<Turtle> stack;
        Turtle t = t0;
        t.step = step;
        t.width = width;

        for (char c : s) {
            switch (c) {
                case 'F': {
                    mesh.curGroup = G_STEM;
                    Vector3d a = t.pos;
                    Vector3d b = t.pos.plus(t.H.times(t.step));
                    mesh.tube(a, b, t.width, t.width * 0.96, 2);
                    t.pos = b;
                    break;
                }
                case 'f': t.pos = t.pos.plus(t.H.times(t.step)); break;
                case '+': t.yaw(ja( delta)); break;
                case '-': t.yaw(ja(-delta)); break;
                case '&': t.pitch(ja( delta)); break;
                case '^': t.pitch(ja(-delta)); break;
                case '\\': t.roll(ja( delta)); break;
                case '/': t.roll(ja(-delta)); break;
                case '|': t.yaw(M_PI); break;
                case '!': t.width *= widthScale; break;
                case '[': stack.push_back(t); break;
                case ']': if (!stack.empty()) { t = stack.back(); stack.pop_back(); } break;
                case 'L': emitLeaf(mesh, t); break;
                default: break;   // rewrite-only symbols (X, A, ...) draw nothing
            }
        }
    }

    // A lanceolate leaf blade in the turtle's H/L plane, pointing along H,
    // cupped along its midrib.
    void emitLeaf(FlowerMesh& mesh, const Turtle& t) const {
        mesh.curGroup = G_LEAF;
        Vector3d tip = t.pos.plus(t.H.times(leafLength));
        mesh.blade(t.pos, tip, t.L.times(leafWidth * 0.5), leafWidth * 0.06, 0.5);
    }
};

// ---------------------------------------------------------------------------
// Preset L-systems (stems / sprays / ferny forms).
// ---------------------------------------------------------------------------
enum class Preset { Sunflower, DaisyStem, FernFrond, Bush };

inline LSystem presetLSystem(Preset p) {
    LSystem ls;
    switch (p) {
        case Preset::FernFrond:
            // Classic 3D ferny frond (Prusinkiewicz-style), leaves on the tips.
            ls.axiom = "A";
            ls.rules['A'] = "!F[&+A][&-A]F[^\\A][^/A]FA";
            ls.rules['F'] = "F";
            ls.iterations = 4;
            ls.delta = 22.0 * M_PI / 180.0;
            ls.step = 1.6; ls.width = 0.35; ls.widthScale = 0.85;
            ls.jitter = 0.05;
            break;
        case Preset::Bush:
            ls.axiom = "F";
            ls.rules['F'] = "FF-[-F+F+FL]+[+F-F-FL]";
            ls.iterations = 3;
            ls.delta = 24.0 * M_PI / 180.0;
            ls.step = 1.2; ls.width = 0.3; ls.jitter = 0.08;
            break;
        case Preset::DaisyStem:
            ls.axiom = "F";
            ls.rules['F'] = "F[&+FL][&-FL]F";
            ls.iterations = 3;
            ls.delta = 30.0 * M_PI / 180.0;
            ls.step = 2.2; ls.width = 0.4; ls.leafLength = 4.0; ls.leafWidth = 2.0;
            ls.jitter = 0.06;
            break;
        case Preset::Sunflower:
        default:
            // A tall, slightly wavering stalk with a couple of big leaves --
            // the capitulum is added by buildSunflower(), not the string.
            ls.axiom = "F!F!F!F!F";
            ls.rules['F'] = "F";
            ls.iterations = 1;
            ls.step = 6.0; ls.width = 0.9; ls.widthScale = 0.94;
            ls.jitter = 0.03;
            break;
    }
    return ls;
}

inline FlowerMesh buildFromPreset(Preset p) {
    LSystem ls = presetLSystem(p);
    FlowerMesh m;
    ls.interpret(m, Turtle{});
    m.centreXZ();
    return m;
}

// ---------------------------------------------------------------------------
// Sunflower.
// ---------------------------------------------------------------------------
// float (not double) fields so the viewer's ImGui sliders can bind to them
// directly; buildSunflower() promotes to double in its math anyway.
struct SunflowerParams {
    // Stem
    float stemHeight = 55.0f;   // cm to the base of the head
    float stemRadius = 0.9f;    // cm at the base, tapers up
    float stemWaver  = 3.0f;    // cm lateral sway amplitude
    int   stemLeaves = 4;       // big leaves marching up the stalk

    // Head / capitulum
    float headRadius   = 14.0f; // cm radius of the disk
    float headTilt     = 0.35f; // radians the face tips toward the viewer/up-forward
    float headDome     = 0.10f; // dome curvature (bowl if negative)
    int   floretCount  = 900;   // disk florets in the Vogel spiral
    float floretRadius = 0.35f; // cm radius of a floret stub
    float floretHeight = 0.9f;  // cm each floret stands proud of the disk

    // Ray petals
    int   petalCount  = 34;     // outer petals (34/55/89 read as Fibonacci-ish)
    int   petalRings  = 2;      // rows of petals for a fuller flower
    float petalLength = 12.0f;  // cm
    float petalWidth  = 2.4f;   // cm at the widest
    float petalDroop  = 0.30f;  // radians the petals dip below the disk plane
    float petalLift   = 0.15f;  // radians the petals lift above the disk plane

    unsigned seed = 7;
};

// One ray petal / blade, rooted at `base`, pointing along `dir` (already
// tilted), lying in the plane spanned by dir and `side`. A single blade
// primitive: lanceolate outline, cupped/curled by `curl`.
inline void addPetal(FlowerMesh& m, const Vector3d& base, const Vector3d& dir,
                     const Vector3d& side, double length, double width, double thickness,
                     double curl = 0.45) {
    Vector3d tip = base.plus(dir.times(length));
    m.blade(base, tip, side.times(width * 0.5), thickness, curl, PRIM_BLADE, 0.0, 1.0, curl, 0.0);
}

// A lobed chrysanthemum leaf: a short petiole then a central pointed lobe with
// two pairs of forward-swept side lobes, all coplanar -- the distinctive
// oak-like chrysanthemum leaf silhouette. `attach` is the point on the stem,
// `dir` the outward midrib direction, `up` world up (defines the leaf plane
// with dir). Uses the current mesh group (call with G_LEAF).
inline void emitLobedLeaf(FlowerMesh& m, const Vector3d& attach, const Vector3d& dir,
                          const Vector3d& up, double length, double width, double curl) {
    Vector3d d = dir.normalized();
    Vector3d side = d.cross(up);
    if (side.length() < 1e-5) side = d.cross(Vector3d(1, 0, 0));
    side = side.normalized();
    Vector3d leafN = d.cross(side).normalized();          // leaf-plane normal
    Vector3d wAx = leafN.cross(d).normalized();           // in-plane width axis
    if (wAx.length() < 1e-6) wAx = side;
    // Petiole (thin stalk), then ONE lobed blade: PRIM_LEAF carves pinnate
    // lobes + marginal teeth into the blade outline in the SDF (see shader.frag
    // sdBlade `lobe`), so a single primitive reads as a deeply-lobed, serrated
    // chrysanthemum leaf instead of a coplanar fan of sub-blades (a blob).
    Vector3d base = attach.plus(d.times(length * 0.14));
    m.tube(attach, base, width * 0.05, width * 0.04, 2);
    Vector3d tip = base.plus(d.times(length * 0.86));
    // latCup 0.28 gives the leaf a shallow central channel; curl droops it.
    m.blade(base, tip, wAx.times(width * 0.5), width * 0.045, curl, PRIM_LEAF, 0.0, 1.0, 0.28, 0.0);
}

inline FlowerMesh buildSunflower(const SunflowerParams& P) {
    FlowerMesh m;
    std::mt19937 rng(P.seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);

    // --- Stem: a wavering, tapering stalk up +Y. Sample a gentle sine sway in
    //     a random vertical plane so it isn't a ruler-straight pole. ---
    m.curGroup = G_STEM;
    const int stemSub = 40;
    double swayPhase = u(rng) * M_PI;
    Vector3d swayDir(std::cos(swayPhase), 0.0, std::sin(swayPhase));
    Vector3d prev(0, 0, 0);
    int prevIdx = m.addNode(prev);
    Vector3d headBase;
    for (int i = 1; i <= stemSub; ++i) {
        double t = static_cast<double>(i) / stemSub;
        double y = t * P.stemHeight;
        double sway = std::sin(t * M_PI * 1.3) * P.stemWaver * t;   // more sway up top
        Vector3d p = swayDir.times(sway).plus(Vector3d(0, y, 0));
        int idx = m.addNode(p);
        double r = P.stemRadius * (1.0 - 0.55 * t);                 // taper upward
        m.addSeg(prevIdx, idx, r);
        prevIdx = idx; prev = p;
        headBase = p;
    }

    // --- Stem leaves: alternating up the stalk at the golden angle, drooping. ---
    m.curGroup = G_LEAF;
    for (int i = 0; i < P.stemLeaves; ++i) {
        double t = 0.25 + 0.6 * (P.stemLeaves > 1 ? double(i) / (P.stemLeaves - 1) : 0.0);
        double y = t * P.stemHeight;
        double ang = i * kGoldenAngle;
        Vector3d out(std::cos(ang), 0, std::sin(ang));
        double droop = -0.5 - 0.3 * u(rng);
        Vector3d dir = out.plus(Vector3d(0, droop, 0)).normalized();
        Vector3d side = dir.cross(Vector3d(0, 1, 0)).normalized();
        Vector3d base = swayDir.times(std::sin(t * M_PI * 1.3) * P.stemWaver * t)
                            .plus(Vector3d(0, y, 0));
        double len = P.headRadius * (1.1 + 0.3 * u(rng));
        double wid = len * 0.5;
        // Reuse the petal builder as a leaf blade (broader, blunter).
        addPetal(m, base, dir, side, len, wid, 0.12, 0.55);   // leaf: cupped blade
    }

    // --- Head frame: the disk faces `faceN`, tilted off vertical so we see
    //     the spiral instead of edge-on. u/v span the disk plane. ---
    Vector3d faceN = rotAxis(Vector3d(0, 1, 0), Vector3d(1, 0, 0), P.headTilt).normalized();
    Vector3d diskU = faceN.cross(Vector3d(0, 0, 1));
    if (diskU.length() < 1e-6) diskU = faceN.cross(Vector3d(1, 0, 0));
    diskU = diskU.normalized();
    Vector3d diskV = faceN.cross(diskU).normalized();
    Vector3d centre = headBase.plus(faceN.times(P.stemRadius * 0.5));

    auto diskPoint = [&](double r, double ang) {
        // Dome the disk: florets near the rim sit lower (a shallow bowl/dome).
        double lift = P.headDome * (P.headRadius * P.headRadius - r * r) / P.headRadius;
        return centre.plus(diskU.times(r * std::cos(ang)))
                     .plus(diskV.times(r * std::sin(ang)))
                     .plus(faceN.times(lift));
    };

    // --- Disk florets: Vogel's model. r = c*sqrt(i), theta = i*goldenAngle.
    //     Each floret is a short stub standing along faceN -- en masse the
    //     stubs read as the packed seed spiral. Florets thicken toward the
    //     centre (young, tufted) and flatten toward the rim (open seeds). ---
    m.curGroup = G_DISK;
    double c = P.headRadius / std::sqrt(std::max(1, P.floretCount));
    for (int i = 0; i < P.floretCount; ++i) {
        double r = c * std::sqrt(double(i) + 0.5);
        if (r > P.headRadius) continue;
        double ang = i * kGoldenAngle;
        double edge = r / P.headRadius;                 // 0 centre .. 1 rim
        Vector3d p0 = diskPoint(r, ang);
        double h = P.floretHeight * (0.5 + 0.8 * (1.0 - edge));   // taller in the middle
        Vector3d p1 = p0.plus(faceN.times(h));
        double fr = P.floretRadius * (0.7 + 0.6 * (1.0 - edge));
        m.tube(p0, p1, fr, fr * 0.6, 1);
    }

    // --- Ray petals: rings of lanceolate blades just outside the disk rim,
    //     each tilted between petalLift (up) and petalDroop (down). Offset
    //     successive rings by half a step so they interleave. ---
    m.curGroup = G_PETAL;
    for (int ring = 0; ring < P.petalRings; ++ring) {
        double ringR = P.headRadius * (0.94 - 0.10 * ring);
        double tilt  = P.petalLift - (P.petalDroop + P.petalLift)
                                        * (P.petalRings > 1 ? double(ring) / (P.petalRings - 1) : 0.0);
        double len = P.petalLength * (1.0 - 0.12 * ring);
        double offset = (ring & 1) ? M_PI / P.petalCount : 0.0;
        for (int i = 0; i < P.petalCount; ++i) {
            double ang = 2.0 * M_PI * i / P.petalCount + offset;
            Vector3d radial = diskU.times(std::cos(ang)).plus(diskV.times(std::sin(ang)));
            Vector3d base = centre.plus(radial.times(ringR));
            // Tilt the petal off the disk plane toward/away from faceN.
            Vector3d dir = radial.times(std::cos(tilt)).plus(faceN.times(std::sin(tilt)));
            dir = dir.normalized();
            Vector3d side = faceN.cross(radial).normalized();
            double jl = 1.0 + 0.12 * u(rng);
            addPetal(m, base, dir, side, len * jl, P.petalWidth, 0.08, 0.30);   // ray petal
        }
    }

    m.centreXZ();
    return m;
}

// ---------------------------------------------------------------------------
// Chrysanthemum -- five bloom architectures matched to the reference photos,
// selected by ChrysForm. They differ in how ray florets are arranged and
// shaped:
//   REFLEX     broad spoon petals over a globe, arcing DOWN & OUT (shaggy mop)
//   INCURVE    broad spoon petals over a tight globe, arcing UP & IN (ball)
//   POMPOM     dense globe of SHORT blunt florets, near-uniform, green eye
//   QUILL      straight tubular quills radiating flat (starburst), green eye
//   DECORATIVE flat rosette of broad flat spoon petals in concentric rings
// The petals reuse the blade SDF (PRIM_PETAL = round-tipped spoon); quills are
// thin capsules. Curvature is baked into each petal's base->tip chord (the C
// direction) plus the blade's own cup/curl for the surface fold.
// ---------------------------------------------------------------------------
enum ChrysForm { CHRYS_REFLEX = 0, CHRYS_INCURVE = 1, CHRYS_POMPOM = 2,
                 CHRYS_QUILL = 3, CHRYS_DECORATIVE = 4 };

struct ChrysanthParams {
    int   form        = CHRYS_INCURVE;
    float stemHeight  = 42.0f;
    float stemRadius  = 0.7f;
    float radius      = 11.0f;   // bloom radius
    int   petalCount  = 0;       // 0 = use the form's tuned default
    int   stemLeaves  = 3;
    float growth      = 1.0f;    // 0 = tight bud, 1 = fully open (decorative)
    unsigned seed     = 3;
};

// The green involucre: a COMPACT cup of small overlapping bracts (phyllaries)
// that hugs the underside of the bloom, plus a receptacle knob the petals sit
// on. This replaces the old artichoke sepals, whose "open" pose reflexed the
// bracts DOWN & OUT into a wide flat lime-green skirt under the flower. Here a
// bract always aims from its base at the bloom base toward a target ON the bloom
// surface: at growth 0 the target rides up over the crown so the bracts close
// into a green ovoid (only the petal tips poke out); at growth 1 the target
// drops to just below the equator so they cradle the underside without ever
// splaying flat. `centre` is the bloom centre, `up` world up, `R` bloom radius.
inline void emitInvolucre(FlowerMesh& m, const Vector3d& centre, const Vector3d& up,
                          double R, double growth, unsigned seed) {
    std::mt19937 rng(seed * 71u + 13u);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    auto lp = [](double a, double b, double t){ return a + (b - a) * t; };
    double g = std::min(1.0, std::max(0.0, growth));
    Vector3d apex = centre.minus(up.times(R * 0.34));   // base of the bloom

    // Receptacle: a compact green knob of short scales so the thin stem never
    // visually disconnects from the wide bloom. Small and tight (not a frill).
    m.curGroup = G_LEAF;
    for (int i = 0; i < 20; ++i) {
        double ang = i * kGoldenAngle;
        Vector3d radial(std::cos(ang), 0, std::sin(ang));
        Vector3d dir  = up.times(0.55).plus(radial.times(1.0)).normalized();
        Vector3d side = dir.cross(up).normalized();
        Vector3d b    = apex.plus(radial.times(R * 0.05)).plus(up.times(R * 0.03));
        addPetal(m, b, dir, side, R * 0.12, R * 0.11, 0.05, 0.5);
    }

    // Involucral bracts: 2 shingled rings (outer ring half-offset & slightly
    // lower/longer). Convex curl makes each bract follow the bloom's surface.
    const int rings = 2, per = 16;
    for (int ring = 0; ring < rings; ++ring) {
        double off = (ring & 1) ? M_PI / per : 0.0;
        for (int i = 0; i < per; ++i) {
            double ang = 2.0 * M_PI * i / per + off + 0.12 * u(rng);
            Vector3d radial(std::cos(ang), 0, std::sin(ang));
            Vector3d base = apex.plus(radial.times(R * (0.05 + 0.03 * ring)))
                                .plus(up.times(R * 0.02));
            // Target on the bloom surface: (ty = height along up, tr = radial
            // reach). Open bracts stay SHORT and tucked under the bloom overhang
            // (small green cup); bud bracts sweep up over the crown to enclose.
            double ty_o = -R * (0.10 + 0.05 * ring);   // open: at/below the base
            double tr_o =  R * (0.40 - 0.05 * ring);   // well inside the R overhang
            // Bud: a rounded green lower cup whose bracts reach up to the bud's
            // equator (a stable up-and-out aim, not straight up), letting the
            // furled pink petal knob poke out the top (see buds_ref).
            double ty_b =  R * (0.08 - 0.05 * ring);
            double tr_b =  R * (0.40 - 0.04 * ring);
            double ty   = lp(ty_b, ty_o, g), tr = lp(tr_b, tr_o, g);
            Vector3d tgt = centre.plus(up.times(ty)).plus(radial.times(tr));
            Vector3d dir = tgt.minus(base);
            double len   = dir.length() * lp(1.02, 1.00, g);
            dir = dir.normalized();
            double wid  = R * lp(0.19, 0.15, g);       // broad, overlapping into a collar
            // The blade normal (ax x W) of an up-and-out bract points OUTWARD &
            // DOWN, and +curl bends toward the normal -- so hugging the convex
            // bloom surface (tips wrapping up & IN over it) needs NEGATIVE curl.
            // Positive curl here was the old flat-green-skirt bug.
            double curl = lp(-1.2, -0.7, g);
            double cup  = lp(-1.5, -0.9, g);           // rolled at bud, shallow cup open
            m.curvedPetal(base, dir, up, len, wid, curl, wid * 0.10, PRIM_PETAL, cup, 0.0);
        }
    }
}

inline FlowerMesh buildChrysanthemum(const ChrysanthParams& P) {
    FlowerMesh m;
    std::mt19937 rng(P.seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const Vector3d up(0, 1, 0);
    const double R = P.radius;

    // ---- shared stem + low leaves ----
    m.curGroup = G_STEM;
    const int stemSub = 30;
    int prevIdx = m.addNode(Vector3d(0, 0, 0));
    for (int i = 1; i <= stemSub; ++i) {
        double t = double(i) / stemSub;
        Vector3d p(std::sin(t * 3.0) * 0.6 * t, t * P.stemHeight, 0);
        int idx = m.addNode(p);
        m.addSeg(prevIdx, idx, P.stemRadius * (1.0 - 0.4 * t));
        prevIdx = idx;
    }
    Vector3d centre(0, P.stemHeight + R * 0.3, 0);

    // Alternate lobed leaves marching up the stem on petioles: larger and more
    // horizontal low down, smaller and more upright toward the bloom.
    m.curGroup = G_LEAF;
    for (int i = 0; i < P.stemLeaves; ++i) {
        double t = 0.22 + 0.55 * (P.stemLeaves > 1 ? double(i) / (P.stemLeaves - 1) : 0.0);
        double ang = i * kGoldenAngle;
        double pitch = -0.15 - 0.35 * (1.0 - t);     // droop more near the base
        Vector3d dir = Vector3d(std::cos(ang), pitch, std::sin(ang)).normalized();
        Vector3d attach(std::sin(t * 3.0) * 0.6 * t, t * P.stemHeight, 0);
        double leafLen = R * (1.25 - 0.4 * t);
        emitLobedLeaf(m, attach, dir, up, leafLen, leafLen * 0.8, 0.2);   // broad, fairly flat
    }

    // --- Green involucre (receptacle + calyx bracts) at the stem top ---
    // A compact cup that hugs the bloom's underside (see emitInvolucre): closes
    // over the crown at bud, cradles the base when open -- never a flat skirt.
    emitInvolucre(m, centre, up, R, (double) P.growth, P.seed);

    // Emit one curved petal: base->tip carries the arc, width across the
    // azimuthal tangent so petals shingle rather than fan from a point.
    auto petalTo = [&](const Vector3d& base, const Vector3d& tip,
                       double width, double curl, int prim) {
        Vector3d axis = tip.minus(base);
        if (axis.length() < 1e-6) return;
        axis = axis.normalized();
        Vector3d side = axis.cross(up);
        if (side.length() < 1e-5) side = axis.cross(Vector3d(1, 0, 0));
        side = side.normalized();
        // Thin petals; cup carried by latCup (=curl) now that curl only bends.
        m.blade(base, tip, side.times(width * 0.5), width * 0.08, curl, prim, 0.0, 1.0, curl, 0.0);
    };
    // Fibonacci-sphere direction i of n, over the top `cover` fraction.
    auto sphereDir = [&](int i, int n, double cover) {
        double f = (n > 1) ? double(i) / (n - 1) : 0.0;
        double y = 1.0 - f * (2.0 * cover);
        double rxz = std::sqrt(std::max(0.0, 1.0 - y * y));
        double ang = i * kGoldenAngle;
        return Vector3d(std::cos(ang) * rxz, y, std::sin(ang) * rxz).normalized();
    };
    auto horiz = [&](const Vector3d& d) {
        Vector3d h(d.x, 0, d.z);
        return h.length() < 1e-6 ? Vector3d(1, 0, 0) : h.normalized();
    };

    m.curGroup = G_PETAL;
    switch (P.form) {
        case CHRYS_REFLEX: {
            // A shaggy reflexed globe: broad petals whose tips arc outward then
            // curl DOWN & BACK. Bases ride a fattened globe (like incurve) and
            // reach/droop are BOUNDED so tips stay on a ~1.15R shaggy sphere and
            // never hang as isolated streamers. The lowest band tucks back
            // toward the axis so the underside closes (the involucre backs it).
            int n = P.petalCount > 0 ? P.petalCount : 520;
            for (int i = 0; i < n; ++i) {
                Vector3d d = sphereDir(i, n, 0.96);
                double band = 0.5 * (1.0 - d.y);                 // 0 crown .. 1 base
                double fat  = 0.46 + 0.10 * (1.0 - d.y * d.y);   // round, not egg
                Vector3d base = centre.plus(d.times(R * fat));
                double reach = R * (0.42 + 0.24 * band) * (1.0 + 0.08 * u(rng));
                double droop = R * (0.06 + 0.34 * band);         // more droop low down
                double tuck  = R * (0.12 - 0.34 * band);         // out at crown, IN at base
                Vector3d tip = base.plus(d.times(reach))
                                   .minus(up.times(droop))
                                   .plus(horiz(d).times(tuck));
                petalTo(base, tip, R * 0.16, -0.55, PRIM_PETAL); // reflex (down/back) curl
            }
        } break;

        case CHRYS_INCURVE: {
            // Broad petals over a full globe, tips arcing UP & IN toward the
            // crown -- a tight, smooth ball. Short reach so petals shingle.
            int n = P.petalCount > 0 ? P.petalCount : 780;
            for (int i = 0; i < n; ++i) {
                Vector3d d = sphereDir(i, n, 1.0);
                // Bulge the mid-latitudes outward so the globe is round, not
                // egg-pointed: bases ride a sphere fattened near the equator.
                double fat = 0.42 + 0.10 * (1.0 - d.y * d.y);
                Vector3d base = centre.plus(d.times(R * fat));
                // Longer reach + moderate width so florets shingle as incurved
                // spoons; low length jitter keeps the globe surface smooth.
                double reach = R * 0.55 * (1.0 + 0.05 * u(rng));
                Vector3d tip = base.plus(d.times(reach))
                                   .plus(up.times(R * 0.45))            // curl up
                                   .minus(horiz(d).times(R * 0.24));    // and in
                petalTo(base, tip, R * 0.15, 0.8, PRIM_PETAL);          // shingled spoons
            }
            // Crown fill: extra florets over the top cap, tips converging inward,
            // so looking straight down the axis shows a closed centre.
            int cn = n / 4;
            for (int i = 0; i < cn; ++i) {
                Vector3d d = sphereDir(i, cn, 0.28);            // top cap only
                Vector3d base = centre.plus(d.times(R * 0.5));
                Vector3d tip = base.plus(d.times(R * 0.42)).plus(up.times(R * 0.28))
                                   .minus(horiz(d).times(R * 0.34));   // strong inward
                petalTo(base, tip, R * 0.11, 0.95, PRIM_PETAL);
            }
        } break;

        case CHRYS_POMPOM: {
            // Dense globe of SHORT blunt florets, near-uniform, with a green
            // immature eye. Bases sit near the surface; short curled tips.
            int n = P.petalCount > 0 ? P.petalCount : 900;
            for (int i = 0; i < n; ++i) {
                Vector3d d = sphereDir(i, n, 0.98);
                Vector3d base = centre.plus(d.times(R * 0.62));
                double reach = R * 0.22 * (1.0 + 0.12 * u(rng));
                Vector3d tip = base.plus(d.times(reach)).plus(up.times(R * 0.10))
                                   .minus(horiz(d).times(R * 0.06));
                petalTo(base, tip, R * 0.12, 0.9, PRIM_PETAL);   // narrow, strongly cupped
            }
            m.curGroup = G_DISK;                                 // green eye
            for (int i = 0; i < 70; ++i) {
                Vector3d d = sphereDir(i, 220, 0.5);             // crown cap only
                Vector3d p0 = centre.plus(d.times(R * 0.66));
                m.tube(p0, p0.plus(d.times(R * 0.10)), R * 0.05, R * 0.03, 1);
            }
        } break;

        case CHRYS_QUILL: {
            // Straight tubular quills radiating over a shallow DOME cap (not a
            // single plane): each quill points at its own elevation from the
            // face axis out to the rim, so the starburst has real depth. Green
            // immature eye at the crown.
            int n = P.petalCount > 0 ? P.petalCount : 150;
            Vector3d faceN = rotAxis(up, Vector3d(1, 0, 0), 0.12).normalized();
            Vector3d fU = faceN.cross(Vector3d(0, 0, 1));
            if (fU.length() < 1e-5) fU = faceN.cross(Vector3d(1, 0, 0));
            fU = fU.normalized();
            Vector3d fV = faceN.cross(fU).normalized();
            const double capMax = 1.28;                          // ~73 deg from the axis
            const double cosCap = std::cos(capMax);
            for (int i = 0; i < n; ++i) {
                // Fibonacci over the spherical cap: polar angle from the axis
                // grows with i, azimuth by the golden angle.
                double f = (n > 1) ? double(i) / (n - 1) : 0.0;
                double cosT = 1.0 - f * (1.0 - cosCap);
                double sinT = std::sqrt(std::max(0.0, 1.0 - cosT * cosT));
                double ang = i * kGoldenAngle;
                Vector3d d = faceN.times(cosT)
                                 .plus(fU.times(std::cos(ang) * sinT))
                                 .plus(fV.times(std::sin(ang) * sinT))
                                 .normalized();
                Vector3d base = centre.plus(d.times(R * 0.10));
                double len = R * (0.80 + 0.20 * cosT) * (1.0 + 0.07 * u(rng)); // longer near crown
                Vector3d tip = base.plus(d.times(len));
                m.tube(base, tip, R * 0.032, R * 0.016, 3);      // thin tapering quill
            }
            m.curGroup = G_DISK;                                 // green immature centre
            for (int i = 0; i < 90; ++i) {
                Vector3d d = sphereDir(i, 260, 0.32);            // small crown cap
                Vector3d p0 = centre.plus(d.times(R * 0.06));
                m.tube(p0, p0.plus(d.times(R * 0.12)), R * 0.05, R * 0.03, 1);
            }
        } break;

        case CHRYS_DECORATIVE: {
            // Domed cushion rosette: broad spoon petals in a Vogel spiral over a
            // ROUNDED dome (not flat), each a curved petal cupping upward, base
            // pale -> edge saturated (gradient). Outer petals splay out & down,
            // inner ones short & upright -- the "Cheryl Pink" decorative look.
            int n = P.petalCount > 0 ? P.petalCount : 300;
            double g = std::min(1.0, std::max(0.0, (double) P.growth));
            auto lerp = [](double a, double b, double t){ return a + (b - a) * t; };
            for (int i = 0; i < n; ++i) {
                double f = double(i) / n;                        // 0 centre .. 1 rim
                double ang = i * kGoldenAngle + 0.20 * u(rng);
                double jr  = 1.0 + 0.10 * u(rng);
                double jl  = 1.0 + 0.10 * u(rng);
                double jc  = 1.0 + 0.15 * u(rng);

                // Per-petal opening: outer petals (f->1) open first, centre last.
                // A wave of `op` sweeping inward as growth `g` rises.
                double gstart = (1.0 - f) * 0.45;                // rim starts at 0, centre at 0.45
                double op = (g - gstart) / (1.0 - 0.45);
                op = op < 0 ? 0 : (op > 1 ? 1 : op);
                op = op * op * (3.0 - 2.0 * op);                 // smoothstep

                // --- OPEN pose: a rounded DOME, not an over-splayed flat disc.
                //     Outer petals reach out but stay above the horizontal. ---
                double theta_o  = f * 0.85 * (1.0 + 0.08 * u(rng));  // less polar spread
                double dome_o   = R * 0.55 * std::cos(theta_o);
                double baseR_o  = R * std::sin(theta_o) * 0.95 * jr;
                double outMix_o = 0.22 + 0.48 * f;               // more up -> domed, not drooping
                double len_o    = R * (0.36 + 0.30 * f) * jl;
                double curl_o   = -(0.35 + 0.25 * f) * jc;       // gentle up-curl
                double latCup_o = -(2.8 - 2.3 * f) * jc;         // deep centre, gentle rim

                // --- BUD pose: petals furl into a rounded PINK KNOB that pokes
                //     up out of the green calyx cup (see buds_ref) ---
                double baseR_b  = R * 0.06 * f * jr;             // clustered on the axis
                double dome_b   = R * (0.08 + 0.16 * (1.0 - f)); // inner stacked higher -> rounded crown
                double outMix_b = 0.03;                          // straight up
                double len_b    = R * (0.30 + 0.08 * f) * jl;    // short: only tips clear the bract rim
                double curl_b   = -0.9 * jc;                     // up/in furl -- tips lean in, no flip-over
                double latCup_b = -2.8 * jc;                     // tightly rolled (quilled) in the bud

                // --- interpolate bud -> open by this petal's openness ---
                double theta   = lerp(0.05, theta_o, op);
                double dome    = lerp(dome_b, dome_o, op);
                double baseR   = lerp(baseR_b, baseR_o, op);
                double outMix  = lerp(outMix_b, outMix_o, op);
                double len     = lerp(len_b, len_o, op);
                double curl    = lerp(curl_b, curl_o, op);
                double latCup  = lerp(latCup_b, latCup_o, op);
                double width   = R * (0.19 + 0.11 * f);

                Vector3d radial(std::cos(ang), 0, std::sin(ang));
                Vector3d base = centre.plus(radial.times(baseR)).plus(up.times(dome - R * 0.25));
                Vector3d dir = up.times(1.0 - outMix).plus(radial.times(outMix)).normalized();
                m.curvedPetal(base, dir, up, len, width, curl, width * 0.06,
                              PRIM_PETAL, latCup, 0.0);
            }
            // Tight raised central boss of tiny incurved florets (the eye).
            // Hidden while budding (a real bud shows no eye -- only furled petal
            // tips poke from the bract cup); grows in as the centre opens.
            if (g > 0.5) {
                double eye = (g - 0.5) / 0.5;
                m.curGroup = G_DISK;
                for (int i = 0; i < 70; ++i) {
                    double ang = i * kGoldenAngle;
                    double r = R * 0.14 * std::sqrt(double(i) / 70.0);
                    Vector3d radial(std::cos(ang), 0, std::sin(ang));
                    Vector3d base = centre.plus(radial.times(r)).plus(up.times(R * 0.32 * eye));
                    Vector3d tip = base.plus(up.times(R * 0.12 * eye)).minus(radial.times(R * 0.03));
                    petalTo(base, tip, R * 0.07 * (0.4 + 0.6 * eye), 0.9, PRIM_PETAL);
                }
            }
        } break;
    }

    m.centreXZ();
    return m;
}

// ---------------------------------------------------------------------------
// Witch hazel (Hamamelis): spidery winter flowers -- clusters of four long,
// thin, crinkled ribbon petals borne directly on bare woody twigs, with a
// small cup-shaped calyx at each cluster's base. Built as an L-system twig
// whose stem segments get studded with strap-petal clusters.
// ---------------------------------------------------------------------------
struct WitchHazelParams {
    float branchScale   = 1.0f;  // overall twig size multiplier
    int   iterations    = 3;
    int   clusters      = 22;    // flower clusters scattered on the twigs
    int   strapsPerFlower = 4;   // witch hazel is reliably 4-petalled
    float strapLength   = 9.0f;  // cm -- long spidery ribbons
    float strapWidth    = 0.5f;  // cm
    float crimp         = 0.6f;  // crinkle amplitude
    float spread        = 0.9f;  // radians the 4 straps fan apart
    unsigned seed       = 5;
};

// A crinkled ribbon-strap petal: a long, narrow blade from base along dir,
// with a curl so it reads as a curled ribbon rather than a straight spoke.
inline void addStrap(FlowerMesh& m, const Vector3d& base, const Vector3d& dir,
                     const Vector3d& side, double length, double width,
                     double crimp, unsigned salt) {
    std::mt19937 rng(salt);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    // Droop the tip and give each strap a slightly different curl for variety.
    Vector3d tip = base.plus(dir.times(length)).minus(Vector3d(0, 0.2 * length, 0));
    m.blade(base, tip, side.times(width * 0.5), width * 0.2, 0.7 + crimp * (0.6 + 0.3 * u(rng)));
}

inline FlowerMesh buildWitchHazel(const WitchHazelParams& P) {
    // Bare woody twig from an L-system (no leaves -- witch hazel blooms on bare
    // wood). A sparse, angular branching pattern.
    LSystem ls;
    ls.axiom = "F";
    ls.rules['F'] = "F[+F]F[-F][^F]F";
    ls.iterations = P.iterations;
    ls.delta = 30.0 * M_PI / 180.0;
    ls.step  = 3.0 * P.branchScale;
    ls.width = 0.55 * P.branchScale;
    ls.widthScale = 0.9;
    ls.jitter = 0.12;
    ls.seed = P.seed;

    FlowerMesh m;
    ls.interpret(m, Turtle{});

    // Collect stem segments to hang flowers on.
    std::vector<int> stemSegs;
    for (size_t i = 0; i < m.segments.size(); ++i)
        if (m.groups[i] == G_STEM) stemSegs.push_back((int)i);

    std::mt19937 rng(P.seed * 131 + 7);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::uniform_real_distribution<double> u11(-1.0, 1.0);

    int clusters = std::min<int>(P.clusters, (int)stemSegs.size());
    for (int ci = 0; ci < clusters; ++ci) {
        // Pick a stem segment (spread across the twig) and attach at its child.
        int si = stemSegs[(size_t)(u01(rng) * (stemSegs.size() - 1))];
        Vector2i seg = m.segments[si];
        Vector3d a = m.nodes[seg.x], b = m.nodes[seg.y];
        Vector3d base = b;
        Vector3d branchDir = b.minus(a);
        if (branchDir.length() < 1e-6) branchDir = Vector3d(0, 1, 0);
        branchDir = branchDir.normalized();
        // Outward frame perpendicular to the branch.
        Vector3d out = branchDir.cross(Vector3d(0, 1, 0));
        if (out.length() < 1e-6) out = branchDir.cross(Vector3d(1, 0, 0));
        out = out.normalized();
        Vector3d side = branchDir.cross(out).normalized();

        // Small dark calyx cup (accent) at the base.
        m.curGroup = G_ACCENT;
        for (int q = 0; q < 4; ++q) {
            double ang = 2.0 * M_PI * q / 4.0;
            Vector3d d = out.times(std::cos(ang)).plus(side.times(std::sin(ang)));
            m.tube(base, base.plus(d.times(0.6)).minus(branchDir.times(0.3)),
                   0.35 * P.branchScale, 0.15 * P.branchScale, 1);
        }

        // Four crinkled strap petals fanning out around `out`.
        m.curGroup = G_PETAL;
        double roll0 = u01(rng) * 2.0 * M_PI;
        for (int s = 0; s < P.strapsPerFlower; ++s) {
            double roll = roll0 + 2.0 * M_PI * s / P.strapsPerFlower;
            // Fan direction: a spoke in the out/side plane, tilted to lift off
            // the branch toward `out` by `spread`.
            Vector3d radial = out.times(std::cos(roll)).plus(side.times(std::sin(roll)));
            Vector3d dir = radial.times(std::sin(P.spread) + 0.4)
                               .plus(out.times(std::cos(P.spread)));
            dir = dir.normalized();
            Vector3d strapSide = dir.cross(branchDir).normalized();
            double len = P.strapLength * (0.85 + 0.3 * u01(rng));
            addStrap(m, base, dir, strapSide, len, P.strapWidth, P.crimp,
                     P.seed * 977u + (unsigned)(ci * 31 + s));
        }
    }

    m.centreXZ();
    return m;
}

}  // namespace flower
