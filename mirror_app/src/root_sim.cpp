#include "root_sim.h"

#include "RootSystem.h"
#include "SegmentAnalyser.h"
#include "MaskCavities.h"
#include "RootAttractors.h"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace CPlantBox;
using namespace maskcav;

namespace rootsim {

// CPlantBox grow space -> render space (matches render_relay_gui.cpp::toYup).
static Vector3d toYup(const Vector3d& v) { return Vector3d(v.x, -v.z, v.y); }

static double minDist(const std::vector<Vector3d>& nodes, const Vector3d& p) {
    double best = std::numeric_limits<double>::max();
    for (const auto& n : nodes) best = std::min(best, n.minus(p).length());
    return best;
}

struct FrozenHop {
    std::vector<Vector3d> nodes;   // grow-global (offset applied)
    std::vector<Vector2i> segs;
    std::vector<double>   radii;
};

struct RootSim::Impl {
    SimParams p;
    std::string paramPath;
    double tipRadius = 0.0;

    std::vector<MaskNode> masks;      // grow space
    std::vector<MaskNode> revealed;   // grow space
    std::vector<SimMask>  revealedRender;
    std::vector<FrozenHop> frozen;

    // Current hop state.
    std::shared_ptr<RootSystem> rs;
    std::shared_ptr<Tropism>    base;
    int    hop = 0;
    Vector3d prevPos{0, 0, 0};
    Vector3d offset{0, 0, 0};
    Vector3d localTarget{0, 0, 0};
    MaskNode localTargetNode;
    std::vector<MaskNode> localRevealed;
    double hopLen = 0.0, hopMaxDays = 0.0;
    double day = 0.0, reachedDay = -1.0;
    bool   reached = false;
    bool   doneFlag = false;
    bool   ok = false;

    // Live snapshot (grow-global) refreshed each step.
    std::vector<Vector3d> liveNodes;
    std::vector<Vector2i> liveSegs;
    std::vector<double>   liveRadii;

    void rebuildTropism(double mainW, double latW, bool travel, double dwellThreshold) {
        auto geom = buildCavityGeometry(localRevealed, p.R0, p.Hh, false, 2.0,
                                        tipRadius, p.viewCylLen, 0.9, p.taperPower);
        rs->setGeometry(geom);
        std::vector<Attractor> attrs;
        if (travel) {
            double reach = std::max((double)p.travelPullReach * hopLen, (double)p.R0);
            attrs.push_back(Attractor{localTarget, 1.0, reach});
        } else {
            attrs = rimAttractors({localTargetNode}, 6, 1.0, 3.0, 1.15);
        }
        if (dwellThreshold >= 0.0) {
            rs->setTropism(combinedAttractionSplitTimed(
                rs, base, attrs, p.mainTravelTrials, 6.0, p.sigma,
                mainW, p.lateralWeight, latW, dwellThreshold, geom), -1);
        } else {
            rs->setTropism(combinedAttractionSplit(
                rs, base, attrs, p.mainTravelTrials, 6.0, p.sigma,
                mainW, latW, geom, geom), -1);
        }
    }

    void pushRevealedRender(const MaskNode& m) {
        Vector3d pos = toYup(m.pos), n = toYup(m.normal);
        Vector3d t = toYup(m.tangent), b = toYup(m.bitangent);
        SimMask sm;
        sm.pos[0] = (float)pos.x; sm.pos[1] = (float)pos.y; sm.pos[2] = (float)pos.z;
        sm.normal[0] = (float)n.x; sm.normal[1] = (float)n.y; sm.normal[2] = (float)n.z;
        sm.tangent[0] = (float)t.x; sm.tangent[1] = (float)t.y; sm.tangent[2] = (float)t.z;
        sm.bitangent[0] = (float)b.x; sm.bitangent[1] = (float)b.y; sm.bitangent[2] = (float)b.z;
        sm.rDepth = (float)m.r_depth; sm.rWidth = (float)m.r_width; sm.rHeight = (float)m.r_height;
        revealedRender.push_back(sm);
    }

    void initHop(int h) {
        Vector3d maskGlobal = masks[h].pos;
        Vector3d targetGlobal = maskGlobal.plus(Vector3d(0, 0, (double)p.targetLift));

        rs = std::make_shared<RootSystem>();
        rs->readParameters(paramPath, "plant", true, false);
        rs->setSeed(p.seed + (unsigned)h);
        rs->initialize(false);
        auto seedNodes = rs->getNodes();
        Vector3d localSeed = seedNodes.empty() ? Vector3d(0, 0, 0) : seedNodes[0];
        offset = prevPos.minus(localSeed);
        localTarget = targetGlobal.minus(offset);

        localRevealed.clear();
        for (const auto& m : revealed) {
            MaskNode lm = m; lm.pos = m.pos.minus(offset);
            localRevealed.push_back(lm);
        }
        localTargetNode = masks[h];
        localTargetNode.pos = maskGlobal.minus(offset);

        base = std::make_shared<Gravitropism>(rs, 1.0, p.sigma);
        hopLen = localTarget.minus(localSeed).length();
        rebuildTropism(p.weight, p.lateralWeight, true, -1.0);
        hopMaxDays = p.maxHopDays * std::max(1.0, hopLen / std::max(1.0, (double)p.R0));
        day = 0.0; reachedDay = -1.0; reached = false;
        snapshotLive();
    }

    void snapshotLive() {
        SegmentAnalyser ana(*rs);
        auto radii = ana.getParameter("radius");
        liveNodes.clear();
        liveNodes.reserve(ana.nodes.size());
        for (const auto& n : ana.nodes) liveNodes.push_back(n.plus(offset));
        liveSegs = ana.segments;
        liveRadii = radii;
    }

    void finalizeHop() {
        SegmentAnalyser ana(*rs);
        auto radii = ana.getParameter("radius");
        FrozenHop fh;
        fh.nodes.reserve(ana.nodes.size());
        for (const auto& n : ana.nodes) fh.nodes.push_back(n.plus(offset));
        fh.segs = ana.segments;
        fh.radii = radii;
        frozen.push_back(std::move(fh));

        prevPos = masks[hop].pos.minus(masks[hop].normal.times(
            masks[hop].r_depth + (double)p.spawnBehind));

        liveNodes.clear(); liveSegs.clear(); liveRadii.clear();
        hop++;
        if (hop >= p.N) { doneFlag = true; rs.reset(); }
        else initHop(hop);
    }

    void step() {
        if (doneFlag || !ok) return;
        double dt = std::max(0.02, (double)p.growthDt);
        rs->simulate(dt, false);
        day += dt;

        bool forced = !reached && day > 0.6 * hopMaxDays;
        if (!reached) {
            double d = minDist(rs->getNodes(), localTarget);
            double thr = p.reachMult * std::max(masks[hop].r_width, masks[hop].r_height);
            if (d < thr || forced) {
                reached = true; reachedDay = day;
                localRevealed.push_back(localTargetNode);
                revealed.push_back(masks[hop]);
                pushRevealedRender(masks[hop]);
                rebuildTropism(p.dwellWeight, p.dwellLateralWeight, false, reachedDay);
            }
        }
        snapshotLive();
        if ((reached && day - reachedDay > p.dwellDays) || day >= hopMaxDays)
            finalizeHop();
    }
};

RootSim::RootSim() : impl_(new Impl()) {}
RootSim::~RootSim() = default;

bool RootSim::reset(const SimParams& p) {
    impl_->p = p;
    impl_->paramPath = p.paramDir + p.speciesXml;
    impl_->tipRadius = 0.22 * p.R0;
    impl_->masks.clear();
    impl_->revealed.clear();
    impl_->revealedRender.clear();
    impl_->frozen.clear();
    impl_->liveNodes.clear(); impl_->liveSegs.clear(); impl_->liveRadii.clear();
    impl_->hop = 0;
    impl_->prevPos = Vector3d(0, 0, 0);
    impl_->doneFlag = false;
    impl_->ok = false;

    const double goldenRad = M_PI * (3.0 - std::sqrt(5.0));
    const double maskR = 2.6;
    impl_->masks = conePhyllotaxis(p.N, p.R0, p.Hh, maskR, p.startFrac, p.endFrac,
                                   impl_->tipRadius, p.angleStepGoldenMult * goldenRad,
                                   p.distStepFrac, p.taperPower);
    if (impl_->masks.empty()) return false;

    // Probe the parameter file: readParameters throws if the XML is missing.
    try {
        auto probe = std::make_shared<RootSystem>();
        probe->readParameters(impl_->paramPath, "plant", true, false);
        probe->initialize(false);
    } catch (...) {
        return false;
    }
    impl_->ok = true;
    impl_->initHop(0);
    return true;
}

void RootSim::step() { impl_->step(); }
bool RootSim::valid() const { return impl_->ok; }
bool RootSim::done()  const { return impl_->doneFlag; }

void RootSim::geometry(std::vector<float>& nodesXYZ,
                       std::vector<int>&   segs,
                       std::vector<float>& radii) const {
    nodesXYZ.clear(); segs.clear(); radii.clear();
    int base = 0;
    auto emit = [&](const std::vector<Vector3d>& ns, const std::vector<Vector2i>& ss,
                    const std::vector<double>& rs) {
        for (const auto& n : ns) {
            Vector3d y = toYup(n);
            nodesXYZ.push_back((float)y.x); nodesXYZ.push_back((float)y.y); nodesXYZ.push_back((float)y.z);
        }
        for (const auto& s : ss) { segs.push_back(s.x + base); segs.push_back(s.y + base); }
        for (double r : rs) radii.push_back((float)r);
        base += (int)ns.size();
    };
    for (const auto& fh : impl_->frozen) emit(fh.nodes, fh.segs, fh.radii);
    emit(impl_->liveNodes, impl_->liveSegs, impl_->liveRadii);
}

const std::vector<SimMask>& RootSim::revealedMasks() const { return impl_->revealedRender; }

}  // namespace rootsim
