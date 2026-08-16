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

#include <algorithm>
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
    // s0/s1 are this capsule's fractional span of the colour gradient it belongs
    // to. Defaulting them to the whole 0..1 range is right for a lone capsule but
    // wrong for a chain: every link would ramp base->tip colour over its own
    // length, banding a stem like a bamboo pole. Chains pass their own sub-range.
    void addSeg(int a, int b, double r, double s0 = 0.0, double s1 = 1.0) {
        segments.push_back(Vector2i(a, b));
        radii.push_back(r);
        groups.push_back(curGroup);
        prims.push_back(PRIM_CAPSULE);
        frames.insert(frames.end(), {0.f, 0.f, 0.f, 0.f});
        aux.insert(aux.end(), {(float) s0, (float) s1, 0.f, 0.f});
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
    int tube(const Vector3d& a, const Vector3d& b, double r0, double r1, int sub = 1,
             double s0 = 0.0, double s1 = 1.0) {
        sub = sub < 1 ? 1 : sub;
        int prev = addNode(a);
        for (int i = 1; i <= sub; ++i) {
            double t = static_cast<double>(i) / sub;
            Vector3d p = a.times(1.0 - t).plus(b.times(t));
            int cur = addNode(p);
            addSeg(prev, cur, r0 + (r1 - r0) * (t - 0.5 / sub),
                   s0 + (s1 - s0) * (t - 1.0 / sub), s0 + (s1 - s0) * t);
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
// Tapered capsule chains without the banding.
//
// Where two collinear capsules of radii r and r-dr meet, the wider one's
// spherical cap emerges from inside the narrower one's cylinder, and the union
// has a crease whose normal kinks by about sqrt(2*dr/r) radians. The specular
// lights every one of those creases, which is what made stems and petioles look
// like bamboo or a string of beads -- it is a shading artifact of the taper, not
// of the curve or the link count as such. A probe of four bare stalks (constant
// vs tapered, 40 vs 220 links; see chrys_shot's CHRYS_STEMTEST) confirmed it:
// constant-radius chains are clean at any density, tapered ones band until the
// per-link step is small.
//
// So subdivision has to be chosen from the RADIUS RATIO a chain spans, not
// picked by eye: taperSub() returns the number of links that keeps each crease
// under `kink` radians.
inline int taperSub(double r0, double r1, double kink = 0.085) {
    double a = std::max(std::abs(r0), 1e-6), b = std::max(std::abs(r1), 1e-6);
    double lr = std::abs(std::log(b / a));
    int n = (int) std::ceil(lr / std::max(0.5 * kink * kink, 1e-6));
    return std::max(2, std::min(n, 420));
}

inline double smoothstep01(double e0, double e1, double x) {
    double t = (e1 - e0) != 0.0 ? (x - e0) / (e1 - e0) : 0.0;
    t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
    return t * t * (3.0 - 2.0 * t);
}

// ---------------------------------------------------------------------------
// Stem as a chain of INTERNODES.
//
// A stalk is not a curve with an amplitude, it is a stack of segments, each of
// which elongates on its own schedule and shifts its angle a little while it
// does. Building it that way is what makes growth read as growth: the tip
// wanders because the segments under it are still extending and settling, and
// the wander never repeats, because it comes from a walk rather than from a
// waveform. The previous version added a sine to an analytic bow, and looked
// exactly like a sine.
//
// The walk is CORRELATED (each internode's lean is mostly its neighbour's, with
// some new randomness mixed in), so the stalk curves in long arcs instead of
// zig-zagging joint to joint.
//
// StemGrowth is the plan; buildStem() evaluates it at a growth value into a
// StemPath, which is what everything that has to meet the stem (leaves, the
// receptacle, the head) queries.
// ---------------------------------------------------------------------------
struct StemGrowth {
    int      internodes = 16;
    double   height     = 52.0;   // total length when mature
    double   grown      = 1.0;    // 0 = nothing, 1 = fully extended
    double   baseR      = 0.85;
    double   topR       = 0.45;
    double   lean       = 0.13;   // typical mature tilt of one internode (radians)
    double   swing      = 0.30;   // extra tilt while an internode is still young
    double   window     = 0.38;   // fraction of the timeline one internode takes
    double   persist    = 0.62;   // how much of its neighbour's lean each one keeps
    double   basalFlare = 0.55;
    double   neckSwell  = 1.10;
    double   neckStart  = 0.86;
    unsigned seed       = 3;
};

// A grown stem: the joints, plus smooth queries along them. Positions are
// interpolated Catmull-Rom through the joints, so a 16-segment stalk still reads
// as a smooth stem rather than as a bent wire.
struct StemPath {
    std::vector<Vector3d> node;    // joints, base first
    std::vector<double>   arc;     // cumulative length at each joint
    double len = 0.0;
    double baseR = 0.85, topR = 0.45;
    double basalFlare = 0.55, neckSwell = 1.10, neckStart = 0.86;

    Vector3d pos(double t) const {
        if (node.empty()) return Vector3d(0, 0, 0);
        if (node.size() == 1 || len <= 1e-9) return node[0];
        double s = (t < 0 ? 0 : (t > 1 ? 1 : t)) * len;
        size_t i = 1;
        while (i + 1 < node.size() && arc[i] < s) ++i;
        double a0 = arc[i - 1], a1 = arc[i];
        double u = (a1 - a0) > 1e-9 ? (s - a0) / (a1 - a0) : 0.0;
        const Vector3d& p1 = node[i - 1];
        const Vector3d& p2 = node[i];
        const Vector3d& p0 = node[i >= 2 ? i - 2 : 0];
        const Vector3d& p3 = node[i + 1 < node.size() ? i + 1 : node.size() - 1];
        double u2 = u * u, u3 = u2 * u;
        return p0.times(-0.5 * u3 + u2 - 0.5 * u)
            .plus(p1.times(1.5 * u3 - 2.5 * u2 + 1.0))
            .plus(p2.times(-1.5 * u3 + 2.0 * u2 + 0.5 * u))
            .plus(p3.times(0.5 * u3 - 0.5 * u2));
    }
    Vector3d dir(double t) const {
        const double h = 2e-3;
        Vector3d d = pos(std::min(1.0, t + h)).minus(pos(std::max(0.0, t - h)));
        return d.length() < 1e-9 ? Vector3d(0, 1, 0) : d.normalized();
    }
    double radius(double t) const {
        double gt = t < 0 ? 0 : (t > 1 ? 1 : t);
        double r = baseR + (topR - baseR) * std::pow(gt, 0.75);
        r *= 1.0 + basalFlare * std::exp(-gt * 16.0);
        double n = smoothstep01(neckStart, 1.0, gt);
        return r * (1.0 + neckSwell * n * n);
    }
    Vector3d top() const { return node.empty() ? Vector3d(0, 0, 0) : node.back(); }
};

inline StemPath buildStem(const StemGrowth& G) {
    StemPath P;
    P.baseR = G.baseR; P.topR = G.topR;
    P.basalFlare = G.basalFlare; P.neckSwell = G.neckSwell; P.neckStart = G.neckStart;

    const int N = std::max(3, G.internodes);
    std::mt19937 rng(G.seed * 7919u + 11u);
    std::uniform_real_distribution<double> u(-1.0, 1.0);

    // --- Plan (independent of growth, so a plant does not re-roll its shape as
    //     it grows): each internode's final length, its lean, and when it starts.
    std::vector<double>   len(N), startG(N);
    std::vector<Vector3d> latMature(N), latYoung(N);
    double totalShape = 0.0;
    Vector3d lat(0, 0, 0);                       // the correlated walk's state
    for (int i = 0; i < N; ++i) {
        double f = (N > 1) ? double(i) / (N - 1) : 0.0;
        // Internodes are short at the base, longest around mid-stem, short again
        // under the head -- the usual profile of a flowering stalk.
        double shape = 0.55 + 0.75 * std::sin(M_PI * std::pow(f, 0.85));
        len[i] = shape; totalShape += shape;
        // Windows overlap, so several internodes are extending at any moment.
        startG[i] = f * (1.0 - G.window);
        // Correlated walk: mostly the previous lean, plus a new random push.
        double a = 2.0 * M_PI * (0.5 + 0.5 * u(rng));
        Vector3d push(std::cos(a), 0, std::sin(a));
        lat = lat.times(G.persist).plus(push.times(1.0 - G.persist));
        double tilt = G.lean * (0.55 + 0.75 * std::fabs(u(rng)));
        latMature[i] = lat.times(tilt);
        // While young an internode is more upright but swings further off-axis;
        // it settles onto its mature lean as it finishes extending.
        double b = a + 1.7 + 0.6 * u(rng);
        latYoung[i] = lat.times(tilt * 0.35)
                         .plus(Vector3d(std::cos(b), 0, std::sin(b)).times(G.swing));
    }
    for (int i = 0; i < N; ++i) len[i] *= G.height / totalShape;

    // --- Evaluate at this growth: walk up the chain, accumulating joints. ---
    const Vector3d up(0, 1, 0);
    Vector3d p(0, 0, 0);
    P.node.push_back(p);
    P.arc.push_back(0.0);
    double g = std::min(1.0, std::max(0.0, G.grown));
    for (int i = 0; i < N; ++i) {
        double e = smoothstep01(startG[i], startG[i] + G.window, g);
        if (e <= 1e-4) break;                    // this internode has not started
        Vector3d d = up.plus(latYoung[i].times(1.0 - e)).plus(latMature[i].times(e)).normalized();
        double l = len[i] * e;
        if (l <= 1e-6) break;
        p = p.plus(d.times(l));
        P.len += l;
        P.node.push_back(p);
        P.arc.push_back(P.len);
    }
    return P;
}

// ---------------------------------------------------------------------------
// Stem centre-line (analytic).
//
// An ANALYTIC curve rather than a baked point list, so everything that has to
// meet the stem -- leaves, the receptacle, the bloom -- can ask for the exact
// position, tangent and radius at a height instead of guessing at a formula the
// stem builder used. Guessed attachments were why leaves met the stalk in a bare
// T-junction: the leaf was placed on a re-derived approximation of the curve.
//
// The profile is a taper with two departures from a cone: a flare at the ground
// (a stalk is buttressed where it leaves the soil) and a swelling into the neck
// (the peduncle thickens into the receptacle, so the head is carried rather than
// skewered on a pole).
// ---------------------------------------------------------------------------
struct StemShape {
    double height     = 42.0;
    double baseR      = 0.70;   // cm at the ground, before the flare
    double topR       = 0.40;   // cm just below the neck swelling
    double sway       = 1.7;    // cm of lateral bow at the top
    double swayPhase  = 0.0;    // azimuth of the bow plane
    double basalFlare = 0.55;   // extra radius fraction at the very base
    double neckSwell  = 1.10;   // extra radius fraction entering the receptacle
    double neckStart  = 0.84;   // t where that swelling begins

    // How much of `height` has actually been grown. The curve below is defined
    // over the FULL height and does not depend on this -- growth only reveals
    // more of a path that is fixed in space. Scaling the whole curve instead (the
    // obvious way) drags the lower stem and its leaves sideways as the plant
    // grows, which never happens: a stem elongates at its tip, and what is
    // already built stays put.
    double grown = 1.0;

    // Circumnutation: a growing tip does not track a straight line, it circles.
    // Two out-of-phase turns of different period give a path that wanders without
    // repeating, and because the amplitude climbs with height, the tip sweeps
    // further as it goes -- so during growth the tip is visibly moving around.
    double wander     = 0.0;    // cm of tip wander at full height
    double wanderTurn = 1.7;    // turns of the wander over the whole stem
    double wanderPhase = 0.0;

    Vector3d pos(double t) const {
        t = t < 0 ? 0 : t;
        // A lean that grows with height plus a slight secondary wobble -- a stalk
        // carrying a heavy head bows, it does not stand like a ruler.
        double bow = std::sin(t * M_PI * 0.80) * t;
        double wob = std::sin(t * M_PI * 2.4 + 0.7) * t * 0.16;
        double off = (bow + wob) * sway;
        double x = std::cos(swayPhase) * off, z = std::sin(swayPhase) * off;
        if (wander != 0.0) {
            double a = wander * std::pow(t, 1.35);
            double th = t * wanderTurn * 2.0 * M_PI + wanderPhase;
            x += a * std::sin(th);
            z += a * std::cos(th * 0.77 + 1.1);
        }
        return Vector3d(x, t * height, z);
    }
    Vector3d dir(double t) const {
        const double h = 1e-3;
        Vector3d d = pos(std::min(1.0, t + h)).minus(pos(std::max(0.0, t - h)));
        return d.length() < 1e-9 ? Vector3d(0, 1, 0) : d.normalized();
    }
    double radius(double t) const {
        // The taper and the neck swelling are relative to the GROWN length, not
        // the eventual one: a half-grown stalk is a whole stalk of its own size,
        // tapering into its own tip, not the bottom half of a mature one.
        double gt = grown > 1e-6 ? std::min(1.0, t / grown) : 0.0;
        double r = baseR + (topR - baseR) * std::pow(gt, 0.75);
        r *= 1.0 + basalFlare * std::exp(-gt * 16.0);
        double n = smoothstep01(neckStart, 1.0, gt);
        return r * (1.0 + neckSwell * n * n);
    }
    Vector3d top() const { return pos(grown); }
};

// Emit the stem as a capsule chain, sampled ADAPTIVELY.
//
// Uniform sampling spends links where the stalk is a near-cylinder and starves
// the two places the radius actually moves -- the basal flare and the neck
// swelling -- so those bulges banded even at high link counts. Here the links
// are placed at equal intervals of a cost that mixes arc-length with |dr|/r, so
// density follows the taper. `sub` is the budget, not a fixed spacing.
//
// The colour gradient is also spread across the whole chain (see addSeg), so a
// stem with distinct base/tip colours ramps once instead of once per link.
// Emit a grown StemPath. Same adaptive-by-radius-ratio sampling as the analytic
// version below (see taperSub for why the density is chosen and not guessed).
inline void emitStemPath(FlowerMesh& m, const StemPath& S, int sub = 200) {
    if (S.node.size() < 2 || S.len <= 1e-6) return;
    const int F = 1200;
    std::vector<double> cost(F + 1, 0.0);
    for (int i = 1; i <= F; ++i) {
        double r0 = S.radius(double(i - 1) / F), r1 = S.radius(double(i) / F);
        double dr = std::abs(std::log(std::max(r1, 1e-6) / std::max(r0, 1e-6)));
        cost[i] = cost[i - 1] + (1.0 / F) + dr * 2.2;
    }
    double total = cost[F];
    int n = std::max(sub, std::min(600, (int) std::ceil(total / 0.0045)));
    std::vector<double> ts(n + 1);
    ts[0] = 0.0; ts[n] = 1.0;
    int fi = 1;
    for (int k = 1; k < n; ++k) {
        double target = total * k / n;
        while (fi < F && cost[fi] < target) ++fi;
        ts[k] = double(fi) / F;
    }
    int prev = m.addNode(S.pos(ts[0]));
    for (int k = 1; k <= n; ++k) {
        int cur = m.addNode(S.pos(ts[k]));
        m.addSeg(prev, cur, S.radius(0.5 * (ts[k - 1] + ts[k])), ts[k - 1], ts[k]);
        prev = cur;
    }
}

inline void emitStem(FlowerMesh& m, const StemShape& S, int sub = 200) {
    sub = sub < 2 ? 2 : sub;
    const double G = std::min(1.0, std::max(0.0, S.grown));
    if (G < 1e-4) return;
    // Cost curve over a fine sampling: uniform term + relative radius change.
    const int F = 2000;
    std::vector<double> cost(F + 1, 0.0);
    for (int i = 1; i <= F; ++i) {
        double t0 = G * double(i - 1) / F, t1 = G * double(i) / F;
        double r0 = S.radius(t0), r1 = S.radius(t1);
        double dr = std::abs(std::log(std::max(r1, 1e-6) / std::max(r0, 1e-6)));
        cost[i] = cost[i - 1] + (1.0 / F) + dr * 2.2;
    }
    double total = cost[F];
    // Give the chain enough links that the busiest stretch stays under the crease
    // bound, then place them at equal cost.
    int n = std::max(sub, std::min(600, (int) std::ceil(total / 0.0045)));
    std::vector<double> ts(n + 1);
    ts[0] = 0.0; ts[n] = G;
    int fi = 1;
    for (int k = 1; k < n; ++k) {
        double target = total * k / n;
        while (fi < F && cost[fi] < target) ++fi;
        ts[k] = G * double(fi) / F;
    }
    int prev = m.addNode(S.pos(ts[0]));
    for (int k = 1; k <= n; ++k) {
        int cur = m.addNode(S.pos(ts[k]));
        m.addSeg(prev, cur, S.radius(0.5 * (ts[k - 1] + ts[k])), ts[k - 1], ts[k]);
        prev = cur;
    }
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

// A chrysanthemum leaf, built to the reference photograph rather than to the
// idea of "a leaf with teeth":
//
//   * BROAD -- nearly as wide as it is long, widest below the middle, with a
//     blunt rounded tip. (The outline itself is carved by the PRIM_LEAF branch
//     of sdBlade; this builder just supplies the frame and proportions.)
//   * SHORT PETIOLE, and a blade that is already wide where it starts, so the
//     leaf sits close to the stalk instead of on the end of a stick.
//   * HELD, NOT HANGING -- it leaves the stem angled up and arcs over gently.
//     A steep arc turns the blade's plane on edge, and edge-on a lobed, cupped
//     blade reads as a bumpy tube rather than a leaf.
//   * ROLLED about its own midrib, and UNDULATING -- successive links of the
//     blade cup in alternating directions, so the margins wave the way the
//     reference's do instead of lying dead flat. The roll also means a leaf held
//     near horizontal still shows its face to a camera near eye level.
//
// `attach` is the point on the stem, `dir` the outward midrib direction, `up`
// world up. Uses the current mesh group (call with G_LEAF).
inline void emitLobedLeaf(FlowerMesh& m, const Vector3d& attach, const Vector3d& dir,
                          const Vector3d& up, double length, double width, double curl,
                          double stemRad = 0.0, unsigned salt = 0u) {
    Vector3d d = dir.normalized();
    Vector3d side = d.cross(up);
    if (side.length() < 1e-5) side = d.cross(Vector3d(1, 0, 0));
    side = side.normalized();

    std::mt19937 rng(salt * 2654435761u + 17u);
    std::uniform_real_distribution<double> u(-1.0, 1.0);

    // Petiole thickness is set by the stem it grows from, so the two meet at a
    // matching diameter and taper away -- a fillet, not a butt joint. Its first
    // sample starts INSIDE the stem (behind the surface) so there is no seam.
    double sr   = stemRad > 0.0 ? stemRad : width * 0.055;
    double petR = sr * 0.50;
    double ribR = sr * 0.30;                                     // petioles barely taper

    // The petiole lifts away from the stalk, and the blade carries on from where
    // it ends -- so the leaf is held out and up, not pinned on at right angles.
    // The caller now hands over a dir that is already pitched up, so the extra
    // lift here is small -- doubling it up stands the leaf on end.
    Vector3d liftDir = d.plus(up.times(0.12)).normalized();
    double   petLen  = length * 0.13;                            // short: the blade sits close in
    Vector3d bladeBase = attach.plus(liftDir.times(petLen));

    // --- Petiole: sunk into the stem, tapering out. Subdivision comes from the
    //     radius ratio (taperSub), not a fixed count -- a short stalk narrowing
    //     2:1 over a handful of links is exactly the case that beads.
    {
        Vector3d sunk = attach.minus(d.times(sr * 0.75));        // start inside the stem
        m.tube(sunk, bladeBase, petR, ribR, taperSub(petR, ribR));
    }

    // --- Sheathing leaf base: two small blades wrapping the stem at the node, so
    //     the petiole appears to grow out of the stalk rather than be pinned to
    //     it. Kept small and pressed against the stalk -- oversized ones read as
    //     odd green buds. Skipped when there is no stem (isolated leaf harness).
    if (stemRad > 0.0) {
        for (int k = -1; k <= 1; k += 2) {
            Vector3d sd = up.times(0.92).plus(d.times(0.34)).plus(side.times(0.40 * k)).normalized();
            Vector3d sb = attach.minus(d.times(sr * 0.35)).minus(up.times(sr * 0.55));
            // Strong negative cup rolls the sheath around the stalk.
            m.curvedPetal(sb, sd, up, sr * 2.1, sr * 1.5, -0.30, sr * 0.10, PRIM_BLADE, -2.2, 0.25);
        }
    }

    // --- Blade: ONE primitive for the whole leaf.
    //
    //     A chain of links was tried, to arc the leaf along a Bezier. It works for
    //     a long narrow blade but not for this one: a chrysanthemum leaf is nearly
    //     as wide as it is long, so every link is wider than it is long, and the
    //     union's silhouette is dominated by the links' own straight side edges --
    //     a row of overlapping ovals instead of a lobed outline. sdBlade shapes the
    //     outline as a function of the fraction along the WHOLE blade, so the lobes
    //     only read when one primitive spans the whole thing. The lengthwise arc
    //     comes from `curl` instead, which bends the surface toward the blade
    //     normal, and the midrib below is sampled from that same bend.
    const double thick = width * 0.026;                 // blade half-thickness
    const double bladeLen = length * 0.86;
    // Roll the blade about its own midrib. Leaves held near horizontal are
    // otherwise seen exactly edge-on from a camera near eye level; a little roll
    // (alternating side to side between leaves) keeps a face turned outward.
    const double roll = (0.55 + 0.18 * u(rng)) * ((salt & 1u) ? 1.0 : -1.0);

    Vector3d ax = d.normalized();                        // blade axis follows the given dir
    Vector3d wAx = ax.cross(up);
    if (wAx.length() < 1e-5) wAx = ax.cross(Vector3d(1, 0, 0));
    wAx = rotAxis(wAx.normalized(), ax, roll).normalized();
    Vector3d nrm = ax.cross(wAx).normalized();           // sdBlade's normal axis (points down)
    // Positive curl bends toward `nrm`, i.e. the tip arcs downward -- a leaf held
    // out and curving over under its own weight.
    double curlB = 0.42 + 0.22 * curl + 0.08 * u(rng);
    double cup   = -0.22;                                // shallow channel along the midrib

    m.blade(bladeBase, bladeBase.plus(ax.times(bladeLen)), wAx.times(width * 0.5),
            thick, curlB, PRIM_LEAF, 0.0, 1.0, cup, 0.0);

    // --- Midrib: a thin tube tracing the blade's own bent midline (the same
    //     curl*0.6*L*s^2 sdBlade applies), tucked just under the surface, so the
    //     petiole carries on into the leaf instead of ending against a bare sheet.
    //     Sized off the BLADE, not the stem: a rib fatter than the leaf is thick
    //     rides on top of the surface as a caterpillar.
    double ribR0 = std::min(ribR, thick * 1.4);
    const int ribSteps = 5;
    Vector3d ribPrev = bladeBase;
    for (int i = 1; i <= ribSteps; ++i) {
        double s0 = double(i - 1) / ribSteps, s1 = double(i) / ribSteps;
        double rr0 = ribR0 * (1.0 - 0.82 * s0), rr1 = ribR0 * (1.0 - 0.82 * s1);
        Vector3d p = bladeBase.plus(ax.times(bladeLen * s1))
                              .plus(nrm.times(curlB * 0.6 * bladeLen * s1 * s1 + rr1 * 0.6));
        m.tube(ribPrev, p, rr0, rr1, taperSub(rr0, rr1));
        ribPrev = p;
    }
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
    float stemHeight  = 52.0f;
    float stemRadius  = 0.62f;
    // Bloom radius. A chrysanthemum head is roughly a sixth of the plant's
    // height, not a third: at 11 on a 42 cm stalk the flower was half the plant
    // and the leaves looked like a garnish.
    float radius      = 6.4f;
    int   petalCount  = 0;       // 0 = use the form's tuned default
    int   stemLeaves  = 9;
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
// `up` here is the HEAD AXIS, not world up, and cupH/crownH are distances along
// it from the stem tip -- so a nodding flower carries its receptacle and bracts
// with it instead of leaving them behind, pinned to vertical.
inline void emitInvolucre(FlowerMesh& m, const Vector3d& stemTop, const Vector3d& up,
                          double stemR, double cupR, double cupH, double crownH,
                          double growth, unsigned seed) {
    std::mt19937 rng(seed * 71u + 13u);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    auto lp = [](double a, double b, double t){ return a + (b - a) * t; };
    double g = std::min(1.0, std::max(0.0, growth));
    // Radials are taken in the plane perpendicular to the HEAD AXIS, so the cup
    // stays square to the flower when the head nods rather than shearing.
    Vector3d BU = up.cross(Vector3d(0, 0, 1));
    if (BU.length() < 1e-5) BU = up.cross(Vector3d(1, 0, 0));
    BU = BU.normalized();
    Vector3d BV = up.cross(BU).normalized();

    // --- Receptacle: a trumpet that flares the stem out to the width the bloom
    //     is carried on. This is the piece that was missing: the old receptacle
    //     was a small knob of scales sitting at the bloom's own centre, so a thin
    //     stalk ran up and stopped, and the head began somewhere above it. Built
    //     from the ACTUAL stem top and radius, densely subdivided (the taper is
    //     steep, and sparse links would crease it into rings).
    //     The trumpet SCALES WITH GROWTH. Sized only for the open bloom it stays
    //     a full-width green egg under a bud a fraction of its size -- the head
    //     has to widen as the flower opens, not spring to final width at growth 0.
    double neckH = std::max(cupH, cupR * 0.30) * lp(0.55, 1.0, g);
    const double rimR = cupR * lp(0.26, 0.42, g);      // trumpet mouth
    m.curGroup = G_STEM;
    {
        // The trumpet spans a large radius ratio over a short rise, so it is the
        // most banding-prone chain in the plant: subdivide it by radius ratio.
        const int sub = 26;
        Vector3d prev = stemTop.minus(up.times(neckH * 0.10));   // start below, no seam
        double   pr   = stemR * 1.02;
        for (int i = 1; i <= sub; ++i) {
            double q = double(i) / sub;
            // Concave flare (q^1.8): hugs the stalk low, opens fast at the mouth.
            double y = -neckH * 0.10 + neckH * 0.72 * q;
            double r = stemR * 1.02 + (rimR - stemR * 1.02) * std::pow(q, 1.8);
            Vector3d p = stemTop.plus(up.times(y));
            m.tube(prev, p, pr, r, taperSub(pr, r), (i - 1.0) / sub, double(i) / sub);
            prev = p; pr = r;
        }
    }
    Vector3d apex = stemTop.plus(up.times(neckH * 0.62));   // top of the trumpet

    // --- Involucral bracts: 3 shingled rings rising off the trumpet, aimed at
    //     the ring where the outermost petals are ROOTED (cupR, cupY) rather than
    //     at a fixed fraction of the bloom radius. That reach is the whole point:
    //     stopping short at 0.4R left every petal base hanging in the open under
    //     the flower -- the ragged bright collar between bloom and stem.
    m.curGroup = G_LEAF;
    const int rings = 3, per = 18;
    for (int ring = 0; ring < rings; ++ring) {
        double rf  = double(ring) / (rings - 1);          // 0 inner .. 1 outer
        double off = (ring & 1) ? M_PI / per : 0.0;
        for (int i = 0; i < per; ++i) {
            double ang = 2.0 * M_PI * i / per + off + 0.10 * u(rng);
            Vector3d radial = BU.times(std::cos(ang)).plus(BV.times(std::sin(ang)));
            // Bases climb the trumpet: outer rings start lower and wider.
            double bq = 0.80 - 0.30 * rf;
            double baseR = (stemR * 1.02 + (rimR - stemR * 1.02) * std::pow(bq, 1.8)) * 0.85;
            Vector3d base = stemTop.plus(up.times(neckH * 0.72 * bq))
                                  .plus(radial.times(baseR));
            // OPEN: tips reach just under the petal ring -- the outer ring all the
            // way out to it, inner rings tucked progressively further in, so the
            // three rings shingle into a closed green underside.
            // The outer ring reaches slightly PAST the petal-base ring: stopping
            // exactly on it still showed the bases in silhouette at the widest
            // azimuths, which is where the bright torn collar was left over.
            double tr_o = cupR * (0.58 + 0.50 * rf);
            double ty_o = cupH - cupH * 0.10 * (1.0 - rf);
            // BUD: the bracts CONVERGE on the axis just over the furled petals,
            // closing the head into a green ovoid. Aimed wide (a third of the cup)
            // and high they instead stand off the flower as a ring of vertical
            // spikes with the petals showing between them -- which is what the
            // first growth frame used to be.
            double tr_b = cupR * (0.10 + 0.05 * rf);
            double ty_b = crownH * (0.88 + 0.08 * rf);
            double tr = lp(tr_b, tr_o, g), ty = lp(ty_b, ty_o, g);
            Vector3d tgt = stemTop.plus(up.times(ty)).plus(radial.times(tr));
            Vector3d dir = tgt.minus(base);
            double len = dir.length() * lp(1.04, 1.02, g);
            if (len < 1e-5) continue;
            dir = dir.normalized();
            // Width comes from the ring the bracts EMERGE on, not the ring they
            // end on. Sized off a converging tip they turn into needles, and a bud
            // built from needles is a fringe rather than a shell.
            double ringR = std::max(std::max(baseR, tr), cupR * 0.16);
            double wid = 2.0 * M_PI * ringR / per * lp(2.30, 1.30, g);
            // The blade normal (ax x W) of an up-and-out bract points outward and
            // down, and +curl bends toward that normal -- so following the convex
            // underside (tips wrapping up & IN) needs NEGATIVE curl. Positive curl
            // here was the old flat-green-skirt bug.
            // sdBlade's bend is curl*0.6*L, so a LONG bract swings its tip much
            // further than a short one for the same curl. Closed-bud bracts are
            // near-vertical and long, and at the -1.1 the open pose wants, the tip
            // is thrown a third of the bloom's radius sideways -- the bud came out
            // as a green starburst. Scale the curl down by the bract's length so
            // the bend stays a fixed fraction of it.
            double curlRef = std::max(cupR * 0.55, 1e-3);
            double curl = lp(-0.9, -0.55, g) * std::min(1.0, curlRef / len);
            double cup  = lp(-1.5, -0.75, g);
            m.curvedPetal(base, dir, up, len, wid, curl, wid * 0.09, PRIM_PETAL, cup, 0.12);
        }
    }
}

inline FlowerMesh buildChrysanthemum(const ChrysanthParams& P) {
    FlowerMesh m;
    std::mt19937 rng(P.seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const Vector3d up(0, 1, 0);
    const double Rfull = P.radius;
    const double g = std::min(1.0, std::max(0.0, (double) P.growth));

    // `growth` drives the WHOLE plant, not just the bloom, and the phases are
    // SEQUENTIAL: the stalk goes up and its leaves unfurl behind it, and only
    // once it has reached its height does the head open. Overlapping the two
    // reads as a flower inflating while it is still being carried upward.
    const double gStem  = smoothstep01(0.0, 0.62, g);
    const double gBloom = smoothstep01(0.66, 1.0, g);
    // The head swells as it opens; a bud is a fraction of the open bloom's size.
    const double R = Rfull * (0.34 + 0.66 * gBloom);

    // ---- shared stem + low leaves ----
    // The stalk carries a head several times its own diameter, so it is thicker
    // than a wire, bows under the load, and swells into the receptacle.
    StemGrowth SG;
    SG.height = P.stemHeight;
    SG.grown  = 0.04 + 0.96 * gStem;
    SG.baseR  = P.stemRadius * 1.35 * (0.45 + 0.55 * gStem);
    SG.topR   = P.stemRadius * 0.80 * (0.45 + 0.55 * gStem);
    SG.seed   = P.seed;
    StemPath S = buildStem(SG);
    m.curGroup = G_STEM;
    emitStemPath(m, S);
    Vector3d stemTop = S.top();

    // The head sits on the tip and takes the tip's DIRECTION, so a stalk that
    // wanders as it grows carries a flower that nods with it -- the head is not
    // pinned to world up. Blended toward vertical so a heavy bloom still faces
    // mostly upward however far the tip has leant.
    Vector3d tipDir = S.dir(1.0);
    Vector3d N = tipDir.times(0.62).plus(up.times(0.38)).normalized();
    Vector3d HU = N.cross(Vector3d(0, 0, 1));
    if (HU.length() < 1e-5) HU = N.cross(Vector3d(1, 0, 0));
    HU = HU.normalized();
    Vector3d HV = N.cross(HU).normalized();
    // Map a direction expressed in the world-Y frame onto the head's frame, so
    // the bloom builders below can keep working in a canonical upright frame.
    Vector3d rotAx = up.cross(N);
    double   rotAng = std::asin(std::min(1.0, rotAx.length()));
    if (rotAx.length() > 1e-6) rotAx = rotAx.normalized(); else { rotAx = Vector3d(1,0,0); rotAng = 0; }
    auto toHead = [&](const Vector3d& v) { return rotAng > 1e-6 ? rotAxis(v, rotAx, rotAng) : v; };

    // How high the bloom rides, and the ring its outermost petals are rooted on
    // (cupR/cupY) -- the involucre aims at that ring, so it closes the underside
    // whatever shape the form is. A globe form has to sit a full half-radius
    // clear of the stem top, or the stalk ends up buried in the ball.
    Vector3d centre;
    double cupR, cupH, crownH, centreH;
    switch (P.form) {
        case CHRYS_DECORATIVE:
            centreH = R * 0.34; cupR = R * 0.74; cupH = centreH + R * 0.05; break;
        case CHRYS_QUILL:
            centreH = R * 0.20; cupR = R * 0.22; cupH = centreH - R * 0.02; break;
        default:                                    // REFLEX / INCURVE / POMPOM
            centreH = R * 0.62; cupR = R * 0.50; cupH = centreH - R * 0.30; break;
    }
    centre = stemTop.plus(N.times(centreH));
    // Where the bracts meet when the bud is shut. This has to sit just over the
    // furled petal knob (which stands about 0.3R above centre in the decorative
    // form) -- aimed higher, the bracts overshoot the flower entirely and stand
    // around it as a fringe instead of closing over it.
    crownH = centreH + R * 0.30;

    // Alternate lobed leaves marching up the stem on petioles: larger and more
    // horizontal low down, smaller and more upright toward the bloom. Attachment
    // point, stem direction and stem radius all come from the SAME analytic curve
    // the stalk was emitted from, so the petiole meets it exactly.
    m.curGroup = G_LEAF;
    // One leaf per internode JOINT, which is where a leaf actually grows, and
    // which also makes them appear in step with the stem: a joint only exists
    // once the internode below it has extended. A leaf then unfurls over the
    // stretch of growth after its joint appears.
    const int joints = (int) S.node.size() - 1;      // joint 0 is the ground end
    for (int i = 0; i < P.stemLeaves; ++i) {
        // Spread the leaves over ALL the joints rather than taking the first few,
        // which bunched every leaf onto the bottom of the stalk.
        int ni = 1 + (int) std::llround((joints - 1) * (P.stemLeaves > 1
                        ? double(i) / (P.stemLeaves - 1) : 0.0));
        if (ni < 1 || ni >= (int) S.node.size()) continue;
        double t     = S.len > 1e-6 ? S.arc[ni] / S.len : 0.0;
        double tFull = t;
        // How long this joint has existed, in units of the growth timeline.
        double le = smoothstep01(0.0, 0.16 * S.len,
                                 S.len - S.arc[ni] > 0 ? S.len - S.arc[ni] : 0.0);
        double ang = i * kGoldenAngle + 0.15 * u(rng);
        // Chrysanthemum leaves are held UP and in, closer to the stalk than to the
        // horizontal -- a positive pitch, not a droop. Splayed out at right angles
        // (a negative pitch) they read as a fern or a palm. Still varied leaf to
        // leaf: at one pitch they all sit in a plane and a camera near eye level
        // catches every one of them edge-on at once.
        double pitch = 0.62 - 0.26 * tFull + 0.20 * u(rng);
        Vector3d dir = Vector3d(std::cos(ang), pitch, std::sin(ang)).normalized();
        Vector3d attach = S.pos(t);
        // Leaf size is set by the PLANT, not by the bloom. Tied to the bloom
        // radius they shrank with it when the head was scaled down, and a
        // chrysanthemum's leaves are not a garnish -- the biggest are about a
        // quarter of the stalk's height.
        double leafLen = P.stemHeight * (0.27 - 0.09 * tFull) * (1.0 + 0.10 * u(rng))
                       * (0.16 + 0.84 * le);
        // Broad: a chrysanthemum leaf is nearly as wide as it is long.
        emitLobedLeaf(m, attach, dir, up, leafLen, leafLen * 0.70, 0.35,
                      S.radius(t), P.seed * 101u + (unsigned) i);
    }

    // --- Green involucre (receptacle trumpet + calyx bracts) at the stem top ---
    // Flares the stalk out to the width the bloom is carried on and shingles
    // bracts up to the petal-base ring: closes over the crown at bud, cradles the
    // underside when open -- never a flat skirt, never a gap.
    emitInvolucre(m, stemTop, N, S.radius(1.0), cupR, cupH, crownH,
                  gBloom, P.seed);

    // Emit one curved petal: base->tip carries the arc, width across the
    // azimuthal tangent so petals shingle rather than fan from a point.
    auto petalTo = [&](const Vector3d& base, const Vector3d& tip,
                       double width, double curl, int prim) {
        Vector3d axis = tip.minus(base);
        if (axis.length() < 1e-6) return;
        axis = axis.normalized();
        Vector3d side = axis.cross(N);
        if (side.length() < 1e-5) side = axis.cross(Vector3d(1, 0, 0));
        side = side.normalized();
        // Thin petals; cup carried by latCup (=curl) now that curl only bends.
        m.blade(base, tip, side.times(width * 0.5), width * 0.08, curl, prim, 0.0, 1.0, curl, 0.0);
    };
    // Fibonacci-sphere direction i of n, over the top `cover` fraction, rotated
    // into the head's frame so the globe forms nod with the tip too.
    auto sphereDir = [&](int i, int n, double cover) {
        double f = (n > 1) ? double(i) / (n - 1) : 0.0;
        double y = 1.0 - f * (2.0 * cover);
        double rxz = std::sqrt(std::max(0.0, 1.0 - y * y));
        double ang = i * kGoldenAngle;
        return toHead(Vector3d(std::cos(ang) * rxz, y, std::sin(ang) * rxz).normalized());
    };
    // The component of d perpendicular to the head axis (was the world-XZ part,
    // which shears once the head is not vertical).
    auto horiz = [&](const Vector3d& d) {
        Vector3d h = d.minus(N.times(N.times(d)));
        return h.length() < 1e-6 ? HU : h.normalized();
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
            const double g = gBloom;   // the head opens on the bloom phase
            auto lerp = [](double a, double b, double t){ return a + (b - a) * t; };
            for (int i = 0; i < n; ++i) {
                double f = double(i) / n;                        // 0 centre .. 1 rim
                double ang = i * kGoldenAngle + 0.20 * u(rng);
                double jr  = 1.0 + 0.10 * u(rng);
                double jl  = 1.0 + 0.10 * u(rng);
                double jc  = 1.0 + 0.15 * u(rng);

                // Per-petal opening: outer petals (f->1) open first, centre last.
                // A wave of `op` sweeping inward as growth `g` rises.
                // The opening wave sweeps inward across the WHOLE range: the rim
                // starts at 0 and the centre only begins at 0.65, so each petal
                // takes the last 35% to open. Packed into 0..0.45 (as it was), the
                // bloom reached full width by growth 0.4 and the back half of the
                // sequence had nothing left to show.
                double gstart = (1.0 - f) * 0.65;
                double op = (g - gstart) / 0.35;
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

                // --- organic wobble ---------------------------------------
                // A rosette laid out on exact golden-angle spokes with one length
                // per ring is too regular to read as grown: the rim comes out a
                // clean circle and the dome a clean paraboloid. Two low-frequency
                // waves around the head (deliberately non-integer and out of
                // phase, so they never line up into a symmetry) push each petal's
                // reach, height and tilt around, and every petal also leans a
                // little out of its own radial plane.
                double wob1 = std::sin(ang * 2.7 + 0.9), wob2 = std::sin(ang * 4.3 + 2.4);
                double wobR = 1.0 + (0.10 * wob1 + 0.05 * wob2) * op;
                double wobY = (0.055 * wob2 - 0.035 * wob1) * R * op;
                double lean = (0.10 * wob1 + 0.06 * u(rng)) * op;

                Vector3d radial = HU.times(std::cos(ang)).plus(HV.times(std::sin(ang)));
                Vector3d tang   = N.cross(radial).normalized();
                Vector3d base = centre.plus(radial.times(baseR * wobR))
                                      .plus(N.times(dome - R * 0.25 + wobY));
                Vector3d dir = N.times(1.0 - outMix)
                                .plus(radial.times(outMix))
                                .plus(tang.times(lean))
                                .normalized();
                m.curvedPetal(base, dir, N, len * (1.0 + 0.06 * wob2), width, curl,
                              width * 0.06, PRIM_PETAL, latCup, 0.0);
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
                    Vector3d radial = HU.times(std::cos(ang)).plus(HV.times(std::sin(ang)));
                    Vector3d base = centre.plus(radial.times(r)).plus(N.times(R * 0.32 * eye));
                    Vector3d tip = base.plus(N.times(R * 0.12 * eye)).minus(radial.times(R * 0.03));
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
