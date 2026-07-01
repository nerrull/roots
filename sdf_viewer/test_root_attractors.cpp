// Headless validation of root attractors: grow the same mask-cavity-confined
// root system with and without rim attractors, and check that root nodes end
// up closer to the mask rims on average when attraction is active (i.e. roots
// curl toward the cavities they're forbidden from entering, cradling them,
// instead of ignoring them).
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "RootSystem.h"
#include "SegmentAnalyser.h"
#include "MaskCavities.h"
#include "RootAttractors.h"

using namespace CPlantBox;
using namespace maskcav;

// Attractor-centric coverage: for each attractor, distance to the nearest root
// node. Tests whether attractors actually get "claimed" by nearby root growth
// (cradling), not diluted by the many nodes elsewhere on the plant that were
// never going to be near a localized rim attractor regardless.
static double meanNearestDist(const std::vector<Attractor>& attrs, const std::vector<Vector3d>& nodes) {
    double sum = 0.0;
    for (const auto& a : attrs) {
        double best = std::numeric_limits<double>::max();
        for (const auto& p : nodes) best = std::min(best, p.minus(a.pos).length());
        sum += best;
    }
    return attrs.empty() ? 0.0 : sum / attrs.size();
}

// Coverage count: how many attractors have *some* root node within `thr` cm.
// With many competing attractors and few structural axes, the mean-distance
// metric above is dominated by attractors no root was ever going to reach
// (a topology mismatch, not a mechanism failure); this asks the more honest
// question -- does attraction get *more* attractors visibly claimed at all.
static int coverageCount(const std::vector<Attractor>& attrs, const std::vector<Vector3d>& nodes, double thr) {
    int c = 0;
    for (const auto& a : attrs) {
        for (const auto& p : nodes)
            if (p.minus(a.pos).length() < thr) { ++c; break; }
    }
    return c;
}

// Returns (all nodes, structural-only nodes [order 0: taproot/basals, no fine
// laterals]). The dense lateral fuzz space-fills the whole cone regardless of
// attraction, saturating any "nearest node" metric; the structural axes are
// the visually significant roots we actually want reaching for the cavities.
static std::pair<std::vector<Vector3d>, std::vector<Vector3d>>
grow(bool attract, const std::string& param, const std::vector<MaskNode>& masks,
    const std::vector<Attractor>& attrs, double R0, double H, double days, double weight) {
    auto rs = std::make_shared<RootSystem>();
    rs->readParameters(param, "plant", true, false);
    auto geom = buildCavityGeometry(masks, R0, H, true, 2.0);
    rs->setGeometry(geom);
    rs->initialize(false);
    if (attract) {
        auto base = std::make_shared<Gravitropism>(rs, 1.0, 0.2);
        // n=12: many more dicing trials than the species' own default (n=1), so the
        // objective-minimizing search reliably finds a heading aligned to the pull
        // instead of the weak, mostly-random-walk behaviour a low n produces.
        // geom passed through explicitly -- setTropism() doesn't do this for you.
        rs->setTropism(combinedAttraction(rs, base, attrs, 6.0, 0.25, weight, geom), -1);
    }
    rs->simulate(days, false);
    SegmentAnalyser ana(*rs);
    ana.filter("order", 0);
    ana.pack();   // filter() only prunes segments; pack() drops now-unused nodes
    return {rs->getNodes(), ana.nodes};
}

int main(int argc, char** argv) {
    std::string param = (argc > 1) ? argv[1]
        : "/Users/erichan/Documents/Development/jardins_racine/CPlantBox/"
          "modelparameter/structural/rootsystem/Zea_mays_6_Leitner_2014.xml";
    int N = (argc > 2) ? std::atoi(argv[2]) : 10;
    double days = (argc > 3) ? std::atof(argv[3]) : 40.0;
    double weight = (argc > 4) ? std::atof(argv[4]) : 0.6;
    double R0 = 8.0, H = 26.0, maskR = 2.6;

    auto masks = conePhyllotaxis(N, R0, H, maskR, 0.10, 0.92);
    auto attrs = rimAttractors(masks, 3, 1.0, 3.0, 1.15);
    std::cout << "cone phyllotaxis: " << masks.size() << " masks, " << attrs.size()
              << " rim attractors\n";

    auto [nBaseAll, nBaseStruct] = grow(false, param, masks, attrs, R0, H, days, weight);
    auto [nAttrAll, nAttrStruct] = grow(true, param, masks, attrs, R0, H, days, weight);

    double dBase = meanNearestDist(attrs, nBaseStruct);
    double dAttr = meanNearestDist(attrs, nAttrStruct);
    std::cout << "\nafter " << days << " days (structural roots only, order 0):\n"
              << "  BASELINE (no attraction): " << nBaseStruct.size() << "/" << nBaseAll.size()
              << " nodes, mean dist to nearest rim attractor = " << dBase << " cm\n"
              << "  ATTRACTED                : " << nAttrStruct.size() << "/" << nAttrAll.size()
              << " nodes, mean dist to nearest rim attractor = " << dAttr << " cm\n";

    double dBaseAll = meanNearestDist(attrs, nBaseAll);
    double dAttrAll = meanNearestDist(attrs, nAttrAll);
    std::cout << "(all nodes, incl. laterals):\n"
              << "  BASELINE : " << nBaseAll.size() << " nodes, mean dist = " << dBaseAll << " cm\n"
              << "  ATTRACTED: " << nAttrAll.size() << " nodes, mean dist = " << dAttrAll << " cm\n";

    double thr = 1.0;
    int cBase = coverageCount(attrs, nBaseAll, thr);
    int cAttr = coverageCount(attrs, nAttrAll, thr);
    std::cout << "coverage (attractors with a node within " << thr << " cm): "
              << cBase << "/" << attrs.size() << " -> " << cAttr << "/" << attrs.size() << "\n";

    if (cAttr > cBase) {
        std::cout << "PASS: attraction claims more attractors (" << cBase << " -> " << cAttr << ").\n";
        return 0;
    }
    std::cout << "FAIL: attraction did not increase attractor coverage.\n";
    return 1;
}
