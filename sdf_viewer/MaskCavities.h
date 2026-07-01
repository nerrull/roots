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

// --- downward cone container: apex (tip) at z=0 -- where the seed grows from --
// -- widening to base radius R0 at z=-h (deepest). The root system starts as a
// single point at the tip and gets more room to spread out the deeper it grows,
// matching how a taproot naturally fans out with depth.
class SDF_Cone : public SignedDistanceFunction {
public:
    SDF_Cone(double baseRadius, double height, const Vector3d& axisXY = Vector3d(0, 0, 0))
        : R0_(baseRadius), h_(height), cxy_(axisXY) {}

    double getDist(const Vector3d& v) const override {
        double t = -v.z;                          // 0 at apex/tip (z=0) .. h at base (z=-h)
        double allowedR = R0_ * (t / h_);
        double dx = v.x - cxy_.x, dy = v.y - cxy_.y;
        double radial = std::sqrt(dx * dx + dy * dy);
        return std::max({radial - allowedR, t - h_, -t});   // < 0 inside the cone
    }
    std::string toString() const override { return "SDF_Cone"; }
private:
    double R0_, h_;
    Vector3d cxy_;
};

// --- a mask node = a cavity + the frame the mask should be placed in --------
struct MaskNode {
    Vector3d pos;        // cavity centre (cm)
    Vector3d normal;     // outward surface normal (the mask faces this way)
    Vector3d tangent;    // horizontal, around the cone
    Vector3d bitangent;  // up-the-surface
    double r_depth, r_width, r_height;   // ellipsoid radii along (normal, tangent, bitangent)
};

// Phyllotaxis on the lateral surface of the downward cone (apex/seed at z=0).
//   radius grows from 0 at the tip to baseRadius R0 at z=-height (deepest).
inline std::vector<MaskNode> conePhyllotaxis(int n, double baseRadius, double height,
                                             double maskR, double startFrac = 0.12,
                                             double endFrac = 0.9) {
    const double golden = M_PI * (3.0 - std::sqrt(5.0));   // ~137.5 deg
    std::vector<MaskNode> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i) {
        double t = startFrac + (endFrac - startFrac) * ((i + 0.5) / n);   // 0(tip)..1(deep/wide)
        double z = -t * height;
        double radius = baseRadius * t;
        double phi = i * golden;
        double cp = std::cos(phi), sp = std::sin(phi);

        MaskNode m;
        m.pos = Vector3d(radius * cp, radius * sp, z);
        // cone surface normal: radial component + slope back up toward the tip.
        m.normal = Vector3d(cp, sp, baseRadius / height).normalized();
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
inline std::shared_ptr<SignedDistanceFunction>
buildCavityGeometry(const std::vector<MaskNode>& nodes, double baseRadius, double height,
                    bool coneContainer = true, double coneMargin = 2.0) {
    std::vector<std::shared_ptr<SignedDistanceFunction>> cavities;
    for (const auto& m : nodes) {
        cavities.push_back(std::make_shared<SDF_Ellipsoid>(
            m.pos, m.normal, m.tangent, m.bitangent, m.r_depth, m.r_width, m.r_height));
    }
    auto obstacles = std::make_shared<CPlantBox::SDF_Union>(cavities);
    if (coneContainer) {
        auto cone = std::make_shared<SDF_Cone>(baseRadius + coneMargin, height + coneMargin);
        return std::make_shared<CPlantBox::SDF_Difference>(cone, obstacles);
    }
    return std::make_shared<CPlantBox::SDF_Complement>(obstacles);   // just avoid cavities
}

}  // namespace maskcav
