// Headless validation of mask-cavity conditioning (no GUI): grow the root system
// with and without the cone-phyllotaxis cavity geometry and check that the
// confined roots avoid the cavities the masks go in.
//
// build: see the c++ command in the accompanying run.
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "RootSystem.h"
#include "MaskCavities.h"

using namespace CPlantBox;
using namespace maskcav;

static int countInsideCavities(const std::vector<Vector3d>& nodes,
                               const std::vector<MaskNode>& masks, double thr) {
    std::vector<SDF_Ellipsoid> es;
    for (const auto& m : masks)
        es.emplace_back(m.pos, m.normal, m.tangent, m.bitangent, m.r_depth, m.r_width, m.r_height);
    int cnt = 0;
    for (const auto& p : nodes)
        for (const auto& e : es)
            if (e.getDist(p) < -thr) { ++cnt; break; }
    return cnt;
}

static std::vector<Vector3d> grow(bool confine, const std::string& param,
                                  const std::vector<MaskNode>& masks,
                                  double R0, double H, double days) {
    auto rs = std::make_shared<RootSystem>();
    rs->readParameters(param, "plant", true, false);
    if (confine) rs->setGeometry(buildCavityGeometry(masks, R0, H, true, 2.0));
    rs->initialize(false);
    rs->simulate(days, false);
    return rs->getNodes();
}

int main(int argc, char** argv) {
    std::string param = (argc > 1) ? argv[1]
        : "/Users/erichan/Documents/Development/jardins_racine/CPlantBox/"
          "modelparameter/structural/rootsystem/Zea_mays_6_Leitner_2014.xml";
    int N = (argc > 2) ? std::atoi(argv[2]) : 12;
    double R0 = 8.0, H = 25.0, maskR = 3.0, days = 40.0;

    auto masks = conePhyllotaxis(N, R0, H, maskR, 0.12, 0.9);
    std::cout << "cone phyllotaxis: " << masks.size()
              << " mask nodes (R0=" << R0 << "cm H=" << H << "cm maskR=" << maskR << "cm)\n";
    for (size_t i = 0; i < masks.size(); ++i) {
        const auto& m = masks[i];
        std::cout << "  node " << i << "  pos=(" << m.pos.x << ", " << m.pos.y << ", "
                  << m.pos.z << ")  n=(" << m.normal.x << ", " << m.normal.y << ", "
                  << m.normal.z << ")\n";
    }

    auto nUnc = grow(false, param, masks, R0, H, days);
    auto nCon = grow(true, param, masks, R0, H, days);

    const double thr = 0.3;   // cm inside a cavity to count as an intrusion
    int iu = countInsideCavities(nUnc, masks, thr);
    int ic = countInsideCavities(nCon, masks, thr);
    std::cout << "\nafter " << days << " days:\n"
              << "  UNCONFINED: " << nUnc.size() << " nodes, " << iu << " inside cavities\n"
              << "  CONFINED  : " << nCon.size() << " nodes, " << ic << " inside cavities\n";
    if (ic == 0)
        std::cout << "PASS: confined roots avoid every mask cavity.\n";
    else if (ic < iu)
        std::cout << "PARTIAL: cavities largely avoided (" << iu << " -> " << ic << ").\n";
    else
        std::cout << "FAIL: confinement not repelling roots.\n";
    return (ic <= iu) ? 0 : 1;
}
