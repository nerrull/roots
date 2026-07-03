#pragma once
// Point attractors for root growth: the counterpart to MaskCavities.h's
// repulsion. Where a mask cavity pushes roots away (via confining geometry),
// an attractor pulls a root's heading toward a point (via a custom Tropism
// objective), so roots can be made to reach for / wrap around / converge on
// specific locations -- e.g. the rim of a mask niche, so the roots look like
// they're cradling the face rather than growing past it.
//
// Mechanism: CPlantBox's Tropism::getHeading() dices candidate headings and
// keeps the one minimizing tropismObjective() in [0,1] (see Gravitropism for
// the canonical example: 0.5*(heading.z+1), minimized when pointing down).
// AttractionTropism computes the Gaussian-weighted pull direction from every
// nearby attractor and returns 0.5*(1 - heading·pull), minimized when the
// candidate heading points at the (weighted) attractor.

#include <cmath>
#include <memory>
#include <vector>

#include "tropism.h"
#include "mymath.h"
#include "MaskCavities.h"   // for MaskNode, reused to place attractors around cavities

namespace maskcav {

using CPlantBox::Vector3d;
using CPlantBox::Matrix3d;
using CPlantBox::Tropism;
using CPlantBox::Organism;
using CPlantBox::Organ;

struct Attractor {
    Vector3d pos;
    double strength = 1.0;   // relative weight vs. other attractors
    double radius = 6.0;     // cm, Gaussian falloff -- influence ~0 beyond ~2*radius
};

class AttractionTropism : public Tropism {
public:
    AttractionTropism(std::shared_ptr<Organism> plant, double n, double sigma,
                      std::vector<Attractor> attractors)
        : Tropism(plant, n, sigma), attractors_(std::move(attractors)) {}

    std::shared_ptr<Tropism> copy(std::shared_ptr<Organism> plant) override {
        auto nt = std::make_shared<AttractionTropism>(*this);
        nt->plant = plant;
        return nt;
    }

    double tropismObjective(const Vector3d& pos, const Matrix3d& old, double a, double b,
                            double dx, const std::shared_ptr<Organ> o = nullptr) override {
        Vector3d pull(0, 0, 0);
        double wsum = 0.0;
        for (const auto& at : attractors_) {
            Vector3d d = at.pos.minus(pos);
            double dist = d.length();
            if (dist < 1e-6) continue;
            double w = at.strength * std::exp(-(dist * dist) / (2.0 * at.radius * at.radius));
            pull = pull.plus(d.times(w / dist));   // w * unit(d)
            wsum += w;
        }
        if (wsum < 1e-9) return 0.5;               // no attractor nearby -> neutral
        Vector3d dir = pull.times(1.0 / pull.length());
        Vector3d heading = old.times(Vector3d::rotAB(a, b));
        double align = heading.times(dir);          // -1..1, 1 = heading straight at the pull
        return 0.5 * (1.0 - align);
    }

private:
    std::vector<Attractor> attractors_;
};

// Blend gravitropism + attraction, the usual combination (roots still fall,
// but bend toward attractors as they pass nearby). weight in [0,1]: how much
// of the objective is attraction vs. the base tropism.
//
// IMPORTANT: RootSystem::setTropism() (used to install the result) replaces
// rp->f_tf directly -- it does *not* re-propagate the organism's confining
// geometry the way RootSystem::initCallbacks() does for the default tropisms
// (tropism->setGeometry(geometry) is only called there). Tropism::getHeading()
// only respects geometry that was set on *that instance*, so pass `geometry`
// here (the same SDF given to RootSystem::setGeometry()) or the cavities will
// silently stop repelling roots the moment a custom tropism is installed.
inline std::shared_ptr<Tropism> combinedAttraction(
    std::shared_ptr<Organism> plant, std::shared_ptr<Tropism> base,
    std::vector<Attractor> attractors, double n, double sigma, double weight = 0.5,
    std::shared_ptr<SignedDistanceFunction> geometry = nullptr) {
    auto att = std::make_shared<AttractionTropism>(plant, n, sigma, std::move(attractors));
    auto combined = std::make_shared<CPlantBox::CombinedTropism>(plant, n, sigma, base, 1.0 - weight, att, weight);
    if (geometry) combined->setGeometry(geometry);
    return combined;
}

// Attractors ringing the rim of each mask cavity, just outside the ellipsoid --
// pulls nearby root tips to curl around the cavity mouth like fingers cradling
// the mask, instead of ignoring it. n_per_ring points around the rim, offset
// outward along the normal-tangent-bitangent frame already computed for the mask.
inline std::vector<Attractor> rimAttractors(const std::vector<MaskNode>& masks,
                                            int n_per_ring = 8, double strength = 1.0,
                                            double radius = 3.0, double rim_margin = 1.15) {
    std::vector<Attractor> out;
    out.reserve(masks.size() * n_per_ring);
    for (const auto& m : masks) {
        for (int i = 0; i < n_per_ring; ++i) {
            double ang = 2.0 * M_PI * i / n_per_ring;
            double ct = std::cos(ang) * m.r_width * rim_margin;
            double cb = std::sin(ang) * m.r_height * rim_margin;
            Vector3d p = m.pos.plus(m.tangent.times(ct)).plus(m.bitangent.times(cb));
            out.push_back(Attractor{p, strength, radius});
        }
    }
    return out;
}

// Delegates tropismObjective() to one of two sub-tropisms based on the
// growing organ's order (0 = the primary/main axis grown straight from the
// seed; 1+ = laterals/offshoots branching off it) -- lets the main root and
// its offshoots be pulled toward attractors with different strength, e.g. a
// weaker travel weight for the main axis so it wanders more organically
// between masks, while laterals stay strongly attracted for dense wrapping.
//
// Note: dicing intensity (n trials, sigma angular spread) is this instance's
// own -- inherited from Tropism -- since Tropism::getHeading()/getUCHeading()
// use `this->n`/`this->sigma` regardless of which sub-tropism's objective
// ends up being evaluated. Only the objective *shape* (and thus which
// direction is preferred) differs between main and lateral.
// Delegates the ENTIRE heading computation (not just the attraction
// objective) to one of two sub-tropisms based on the growing organ's order --
// 0 (the primary/main axis grown straight from the seed) vs 1+ (laterals/
// offshoots branching off it).
//
// Earlier version only overrode tropismObjective(), which looked right but
// wasn't: CPlantBox's confining-geometry repair loop (the hard "don't grow
// through a cavity" guarantee, as opposed to the soft attraction pull) lives
// inside Tropism::getHeading(), and that's only called on the OUTER
// OrderSplitTropism instance -- so it always used *this* instance's own
// geometry regardless of which sub-tropism's objective got delegated to,
// meaning a per-order geometry difference (e.g. a travel corridor that should
// only constrain the main root) was silently impossible to express that way.
// Delegating the whole call fixes that: each sub-tropism's own geometry
// (and own n/sigma dicing intensity, as a side benefit) genuinely applies
// per-order.
class OrderSplitTropism : public Tropism {
public:
    OrderSplitTropism(std::shared_ptr<Organism> plant, double n, double sigma,
                      std::shared_ptr<Tropism> mainTropism, std::shared_ptr<Tropism> lateralTropism)
        : Tropism(plant, n, sigma), main_(mainTropism), lateral_(lateralTropism) {}

    std::shared_ptr<Tropism> copy(std::shared_ptr<Organism> plant) override {
        auto nt = std::make_shared<OrderSplitTropism>(*this);
        nt->plant = plant;
        nt->main_ = main_->copy(plant);
        nt->lateral_ = lateral_->copy(plant);
        return nt;
    }

    CPlantBox::Vector2d getHeading(const Vector3d& pos, const Matrix3d& old, double dx,
                                   const std::shared_ptr<Organ> o = nullptr, int nodeIdx = -1) override {
        int order = o ? (int) o->getParameter("order") : 0;
        return (order <= 0 ? main_ : lateral_)->getHeading(pos, old, dx, o, nodeIdx);
    }

    // kept for completeness/direct callers; getHeading() above no longer
    // routes through this on its way to a sub-tropism.
    double tropismObjective(const Vector3d& pos, const Matrix3d& old, double a, double b, double dx,
                            const std::shared_ptr<Organ> o = nullptr) override {
        int order = o ? (int) o->getParameter("order") : 0;
        return (order <= 0 ? main_ : lateral_)->tropismObjective(pos, old, a, b, dx, o);
    }

private:
    std::shared_ptr<Tropism> main_, lateral_;
};

// Delegates the whole heading computation by an organ's EMERGENCE TIME (the
// creation time of its first node, getNodeCT(0)): organs that emerged at or
// before thresholdTime use early_, later ones use late_. Same delegation
// trick as OrderSplitTropism (see the long note there on why the entire call
// must be delegated, not just the objective).
//
// Purpose: an offshoot that spawned while the main axis was still travelling
// toward a mask should keep its loose travel attraction even after the main
// root arrives and the rest of the system switches to tight dwell wrapping --
// otherwise the moment the hop reaches the mask, every previously-dangling
// travel offshoot gets yanked inward by the strong dwell pull. Splitting on
// emergence time lets travel-spawned offshoots stay loose while only offshoots
// that emerge during dwell wrap densely.
class EmergenceTimeSplitTropism : public Tropism {
public:
    EmergenceTimeSplitTropism(std::shared_ptr<Organism> plant, double n, double sigma,
                              double thresholdTime, std::shared_ptr<Tropism> early,
                              std::shared_ptr<Tropism> late)
        : Tropism(plant, n, sigma), threshold_(thresholdTime), early_(early), late_(late) {}

    std::shared_ptr<Tropism> copy(std::shared_ptr<Organism> plant) override {
        auto nt = std::make_shared<EmergenceTimeSplitTropism>(*this);
        nt->plant = plant;
        nt->early_ = early_->copy(plant);
        nt->late_ = late_->copy(plant);
        return nt;
    }

    CPlantBox::Vector2d getHeading(const Vector3d& pos, const Matrix3d& old, double dx,
                                   const std::shared_ptr<Organ> o = nullptr, int nodeIdx = -1) override {
        double ct = (o && o->getNumberOfNodes() > 0) ? o->getNodeCT(0) : 0.0;
        return (ct <= threshold_ ? early_ : late_)->getHeading(pos, old, dx, o, nodeIdx);
    }

    double tropismObjective(const Vector3d& pos, const Matrix3d& old, double a, double b, double dx,
                            const std::shared_ptr<Organ> o = nullptr) override {
        double ct = (o && o->getNumberOfNodes() > 0) ? o->getNodeCT(0) : 0.0;
        return (ct <= threshold_ ? early_ : late_)->tropismObjective(pos, old, a, b, dx, o);
    }

private:
    double threshold_;
    std::shared_ptr<Tropism> early_, late_;
};

// Convenience: builds two combinedAttraction() tropisms (one per weight,
// optionally one per geometry) and wraps them in an OrderSplitTropism, so the
// caller doesn't have to. lateralGeometry defaults to mainGeometry if not
// given -- pass a looser (or absent) one to let offshoots roam beyond
// whatever hard bound the main root is held to (e.g. a travel corridor).
// mainN/lateralN: dicing trial counts, independent per order -- meaningful
// now that OrderSplitTropism::getHeading() delegates the whole call (each
// sub-tropism's own n actually gets used, unlike before that fix). A higher
// mainN makes the main root's steering more reliably point at the target
// each step (more candidate headings tried, more likely one aligns well),
// which matters more than it might seem: at high attraction weight the
// dicing objective is almost purely "does this candidate point at the
// target", so trying more candidates directly improves how well it's
// followed, not just how much it's preferred.
inline std::shared_ptr<Tropism> combinedAttractionSplit(
    std::shared_ptr<Organism> plant, std::shared_ptr<Tropism> base, std::vector<Attractor> attractors,
    double mainN, double lateralN, double sigma, double mainWeight, double lateralWeight,
    std::shared_ptr<SignedDistanceFunction> mainGeometry = nullptr,
    std::shared_ptr<SignedDistanceFunction> lateralGeometry = nullptr) {
    if (!lateralGeometry) lateralGeometry = mainGeometry;
    auto mainT = combinedAttraction(plant, base, attractors, mainN, sigma, mainWeight, mainGeometry);
    auto latT = combinedAttraction(plant, base, attractors, lateralN, sigma, lateralWeight, lateralGeometry);
    return std::make_shared<OrderSplitTropism>(plant, mainN, sigma, mainT, latT);
}

// Like combinedAttractionSplit, but the lateral/offshoot branch is further
// split by emergence time (see EmergenceTimeSplitTropism): offshoots that
// emerged at or before thresholdTime pull with earlyLateralWeight, later ones
// with lateLateralWeight. Used for the dwell phase so travel-spawned offshoots
// keep their loose travel weight while dwell-spawned offshoots wrap tightly.
inline std::shared_ptr<Tropism> combinedAttractionSplitTimed(
    std::shared_ptr<Organism> plant, std::shared_ptr<Tropism> base, std::vector<Attractor> attractors,
    double mainN, double lateralN, double sigma, double mainWeight,
    double earlyLateralWeight, double lateLateralWeight, double thresholdTime,
    std::shared_ptr<SignedDistanceFunction> geometry = nullptr) {
    auto mainT     = combinedAttraction(plant, base, attractors, mainN, sigma, mainWeight, geometry);
    auto earlyLatT = combinedAttraction(plant, base, attractors, lateralN, sigma, earlyLateralWeight, geometry);
    auto lateLatT  = combinedAttraction(plant, base, attractors, lateralN, sigma, lateLateralWeight, geometry);
    auto latSplit  = std::make_shared<EmergenceTimeSplitTropism>(plant, lateralN, sigma, thresholdTime,
                                                                 earlyLatT, lateLatT);
    return std::make_shared<OrderSplitTropism>(plant, mainN, sigma, mainT, latSplit);
}

}  // namespace maskcav
