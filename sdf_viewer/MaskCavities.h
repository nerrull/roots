#pragma once
// Mask-cavity conditioning for the root system.
//
// Idea: lay out "mask nodes" in a phyllotactic spiral on the surface of a
// downward cone, then make each node an ellipsoidal obstacle that the roots must
// grow *around*. Feeding container \ (union of ellipsoids) to
// RootSystem::setGeometry() confines growth to "inside the cone AND outside every
// cavity", so the root mass leaves clean oval pockets where the face masks sit.
//
// CPlantBox convention: SignedDistanceFunction::getDist(v) < 0 means "inside the
// allowed region". A standalone solid (ellipsoid/cone) returns < 0 inside itself.

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "sdf.h"
#include "mymath.h"

namespace maskcav {

using CPlantBox::Vector3d;
using CPlantBox::SignedDistanceFunction;

// --- ellipsoidal cavity (the pocket a mask sits in) ------------------------
class SDF_Ellipsoid : public SignedDistanceFunction {
public:
    SDF_Ellipsoid(const Vector3d& c, const Vector3d& a0, const Vector3d& a1, const Vector3d& a2,
                  double r0, double r1, double r2)
        : c_(c), a0_(a0), a1_(a1), a2_(a2), r0_(r0), r1_(r1), r2_(r2) {}

    double getDist(const Vector3d& v) const override {
        Vector3d d = v.minus(c_);
        double p0 = d.times(a0_) / r0_, p1 = d.times(a1_) / r1_, p2 = d.times(a2_) / r2_;
        double k = std::sqrt(p0 * p0 + p1 * p1 + p2 * p2);
        double rmin = std::min({r0_, r1_, r2_});
        return (k - 1.0) * rmin;                 // < 0 inside the ellipsoid
    }
    std::string toString() const override { return "SDF_Ellipsoid"; }
private:
    Vector3d c_, a0_, a1_, a2_;
    double r0_, r1_, r2_;
};

// --- finite cylinder (a "keep-clear" tube in front of a mask, so roots can
// ball up around/behind it but not obscure the face from the front) --------
class SDF_Cylinder : public SignedDistanceFunction {
public:
    SDF_Cylinder(const Vector3d& center, const Vector3d& axis, double radius, double halfLength)
        : c_(center), axis_(axis.normalized()), r_(radius), hl_(halfLength) {}

    double getDist(const Vector3d& v) const override {
        Vector3d w = v.minus(c_);
        double axial = w.times(axis_);
        Vector3d radialVec = w.minus(axis_.times(axial));
        double radial = radialVec.length();
        return std::max(radial - r_, std::fabs(axial) - hl_);   // < 0 inside the cylinder
    }
    std::string toString() const override { return "SDF_Cylinder"; }
private:
    Vector3d c_, axis_;
    double r_, hl_;
};

// --- downward cone/frustum container: the seed grows from z=0, widening to
// base radius R0 at z=-h (deepest). tipRadius (default 0) lets the seed start
// somewhere already-open on the cone rather than pinched to a literal point --
// a frustum whose *virtual* apex is below z=0, so there's room right at the
// root's start instead of forcing everything through a single point.
// taperPower bends the radius profile away from a straight cone: 1.0 = linear
// cone (radius grows steadily from tip to base). <1.0 opens up fast near the
// tip and stays close to full width for most of the length -- closer to a
// cylinder. >1.0 stays narrow for most of the length and flares sharply near
// the base -- more spike-like.
class SDF_Cone : public SignedDistanceFunction {
public:
    // apex: world position of the cone's tip (z=0 point of its own frame).
    // Defaults to the origin. Needed because the root system grows in a local
    // frame translated from the global cone frame -- pass apex = (globalOrigin
    // - localOffset) so the same global cone can be evaluated on local coords.
    SDF_Cone(double baseRadius, double height, double tipRadius = 0.0,
            const Vector3d& axisXY = Vector3d(0, 0, 0), double taperPower = 1.0,
            const Vector3d& apex = Vector3d(0, 0, 0))
        : R0_(baseRadius), h_(height), tipR_(tipRadius), cxy_(axisXY), taper_(taperPower), apex_(apex) {}

    double getDist(const Vector3d& v) const override {
        double t = -(v.z - apex_.z);              // 0 at tip .. h at base
        double frac = std::pow(std::clamp(t / h_, 0.0, 1.0), taper_);
        double allowedR = tipR_ + (R0_ - tipR_) * frac;
        double dx = v.x - cxy_.x - apex_.x, dy = v.y - cxy_.y - apex_.y;
        double radial = std::sqrt(dx * dx + dy * dy);
        return std::max({radial - allowedR, t - h_, -t});   // < 0 inside the cone
    }
    std::string toString() const override { return "SDF_Cone"; }
private:
    double R0_, h_, tipR_, taper_;
    Vector3d cxy_, apex_;
};

// A thin shell straddling the lateral surface of the cone the masks sit on
// (radius profile tipRadius..baseRadius over height, with taperPower): the
// region between an outer and inner cone offset +/- thickness/2 from that
// surface. Confining travel to this shell makes roots crawl ALONG the cone
// surface between masks instead of cutting straight through the interior --
// the straight chord between two surface points dips inside the cone, and the
// shell pushes the path back out onto the surface. apex positions it in the
// grower's local frame (see SDF_Cone::apex).
inline std::shared_ptr<SignedDistanceFunction>
buildConeShell(double baseRadius, double height, double tipRadius, double taperPower,
               double thickness, const Vector3d& apex = Vector3d(0, 0, 0)) {
    double hh = thickness * 0.5;
    auto outer = std::make_shared<SDF_Cone>(baseRadius + hh, height, tipRadius + hh,
                                            Vector3d(0, 0, 0), taperPower, apex);
    auto inner = std::make_shared<SDF_Cone>(baseRadius - hh, height, std::max(0.0, tipRadius - hh),
                                            Vector3d(0, 0, 0), taperPower, apex);
    return std::make_shared<CPlantBox::SDF_Difference>(
        std::static_pointer_cast<SignedDistanceFunction>(outer),
        std::static_pointer_cast<SignedDistanceFunction>(inner));
}

// --- a mask node = a cavity + the frame the mask should be placed in --------
struct MaskNode {
    Vector3d pos;        // cavity centre (cm)
    Vector3d normal;     // outward surface normal (the mask faces this way)
    Vector3d tangent;    // horizontal, around the cone
    Vector3d bitangent;  // up-the-surface
    double r_depth, r_width, r_height;   // ellipsoid radii along (normal, tangent, bitangent)
};

// Phyllotaxis on the lateral surface of the downward cone/frustum (seed at z=0).
//   radius grows from tipRadius at z=0 to baseRadius R0 at z=-height (deepest).
//
// angleStepRad: angular offset between consecutive masks (default: the golden
// angle, ~137.5deg, for the classic non-repeating phyllotactic spiral -- pass
// something else for a more regular/repeating pattern).
// distStepFrac: how far along the cone (as a fraction of total height) each
// successive mask advances (default: spreads the requested count evenly
// across [startFrac, endFrac], the old behavior). Set explicitly to make
// mask count and mask spacing independent -- e.g. a fixed distStepFrac means
// requesting more masks just continues the spiral further rather than
// compressing the existing ones closer together.
// taperPower: see SDF_Cone -- must match whatever confinement geometry the
// masks are actually being placed on, or masks won't sit on the real surface.
inline std::vector<MaskNode> conePhyllotaxis(int n, double baseRadius, double height,
                                             double maskR, double startFrac = 0.12,
                                             double endFrac = 0.9, double tipRadius = 0.0,
                                             double angleStepRad = -1.0, double distStepFrac = -1.0,
                                             double taperPower = 1.0) {
    const double golden = M_PI * (3.0 - std::sqrt(5.0));   // ~137.5 deg
    double angStep = angleStepRad > 0.0 ? angleStepRad : golden;
    double dStep = distStepFrac > 0.0 ? distStepFrac : (n > 0 ? (endFrac - startFrac) / n : 0.0);
    std::vector<MaskNode> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i) {
        double t = std::min(1.0, startFrac + dStep * (i + 0.5));   // 0(tip)..1(deep/wide)
        double tp = std::pow(std::max(t, 1e-6), taperPower);
        double z = -t * height;
        double radius = tipRadius + (baseRadius - tipRadius) * tp;
        double phi = i * angStep;
        double cp = std::cos(phi), sp = std::sin(phi);

        // local surface slope dradius/dz, accounting for the taper curve --
        // a straight cone (taperPower=1) reduces to the old constant slope.
        double slope = taperPower * (baseRadius - tipRadius) * std::pow(std::max(t, 1e-6), taperPower - 1.0) / height;

        MaskNode m;
        m.pos = Vector3d(radius * cp, radius * sp, z);
        // cone surface normal: radial component + slope back up toward the tip.
        m.normal = Vector3d(cp, sp, slope).normalized();
        m.tangent = Vector3d(-sp, cp, 0.0);                       // horizontal, around
        m.bitangent = m.normal.cross(m.tangent).normalized();     // up the surface
        m.r_depth = maskR * 0.55;    // shallow into the surface
        m.r_width = maskR * 1.0;     // tangential half-width
        m.r_height = maskR * 1.25;   // taller (oval)
        out.push_back(m);
    }
    return out;
}

// Build container \ (union of node cavities) for RootSystem::setGeometry().
// `nodes` may be empty (e.g. before the first mask is revealed in sequential
// growth) -- handled explicitly since SDF_Union/SDF_Difference index sdfs[0]
// unconditionally and would crash on an empty cavity list.
//
// viewCylLen > 0 adds a "keep-clear" cylinder projecting outward from each
// mask along its normal -- roots can still ball up around the sides/back of
// the cavity, but stay out of the tube directly in front of it, so the face
// remains visible instead of getting buried under a wrapped knot.
inline std::shared_ptr<SignedDistanceFunction>
buildCavityGeometry(const std::vector<MaskNode>& nodes, double baseRadius, double height,
                    bool coneContainer = true, double coneMargin = 2.0, double tipRadius = 0.0,
                    double viewCylLen = 0.0, double viewCylRadiusMult = 0.9, double taperPower = 1.0) {
    std::shared_ptr<SignedDistanceFunction> cone = coneContainer
        ? std::make_shared<SDF_Cone>(baseRadius + coneMargin, height + coneMargin, tipRadius,
                                     Vector3d(0, 0, 0), taperPower)
        : nullptr;
    if (nodes.empty())
        return cone ? cone : std::make_shared<SignedDistanceFunction>();   // unconstrained
    std::vector<std::shared_ptr<SignedDistanceFunction>> cavities;
    for (const auto& m : nodes) {
        cavities.push_back(std::make_shared<SDF_Ellipsoid>(
            m.pos, m.normal, m.tangent, m.bitangent, m.r_depth, m.r_width, m.r_height));
        if (viewCylLen > 0) {
            double r = viewCylRadiusMult * std::max(m.r_width, m.r_height);
            Vector3d cylCenter = m.pos.plus(m.normal.times(viewCylLen * 0.5));
            cavities.push_back(std::make_shared<SDF_Cylinder>(cylCenter, m.normal, r, viewCylLen * 0.5));
        }
    }
    auto obstacles = std::make_shared<CPlantBox::SDF_Union>(cavities);
    if (cone)
        return std::make_shared<CPlantBox::SDF_Difference>(cone, obstacles);
    return std::make_shared<CPlantBox::SDF_Complement>(obstacles);   // just avoid cavities
}

}  // namespace maskcav
