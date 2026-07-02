// Sequential mask reveal: instead of placing every mask cavity/attractor up
// front, grow toward ONE target mask at a time. Once the root system's growth
// front reaches it (a node lands within `reachMult` * mask-radius of its
// centre), that mask's cavity is folded into the confining geometry right away
// (so it can't be grown through) but attraction *keeps targeting it* for
// `dwellDays` more -- roots already in the area keep converging and wrapping
// around the now-solid cavity instead of instantly moving on. Only after the
// dwell period does attraction retarget to the next mask in the phyllotaxis
// sequence. Repeats until every mask has been revealed (plus a short tail so
// the last mask gets properly wrapped too).
//
// Output format (frame-per-simulate-step, so it can be rendered as a video
// where masks visibly pop in as they're reached):
//   FRAME <i> <day> <nMasksRevealed> <nNodes> <nSegs>
//   MASK x y z nx ny nz rd rw rh      (nMasksRevealed of these)
//   NODE x y z                        (nNodes of these)
//   SEG a b radius                    (nSegs of these)
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "RootSystem.h"
#include "SegmentAnalyser.h"
#include "MaskCavities.h"
#include "RootAttractors.h"

using namespace CPlantBox;
using namespace maskcav;

static double minDistToMask(const std::vector<Vector3d>& nodes, const MaskNode& m) {
    double best = std::numeric_limits<double>::max();
    for (const auto& p : nodes) best = std::min(best, p.minus(m.pos).length());
    return best;
}

int main(int argc, char** argv) {
    std::string param = (argc > 1) ? argv[1]
        : "/Users/erichan/Documents/Development/jardins_racine/CPlantBox/"
          "modelparameter/structural/rootsystem/Heliantus_Pages_2013.xml";
    std::string outPath = (argc > 2) ? argv[2] : "sequential.txt";
    int N = (argc > 3) ? std::atoi(argv[3]) : 8;
    double maxDays = (argc > 4) ? std::atof(argv[4]) : 60.0;
    double weight = (argc > 5) ? std::atof(argv[5]) : 0.55;
    double dt = (argc > 6) ? std::atof(argv[6]) : 0.75;    // simulate step (days) -> also frame rate
    double reachMult = (argc > 7) ? std::atof(argv[7]) : 1.3;
    double tailDays = (argc > 8) ? std::atof(argv[8]) : 6.0;   // extra growth after the last reveal
    double dwellDays = (argc > 9) ? std::atof(argv[9]) : 5.0;  // keep attracting after first contact
    int rimN = (argc > 10) ? std::atoi(argv[10]) : 6;          // rim attractor points (fuller wrap)

    double R0 = 9.0, H = 34.0, maskR = 2.6;
    double tipRadius = 0.22 * R0;   // frustum: open a bit at the seed instead of a bare point

    auto masks = conePhyllotaxis(N, R0, H, maskR, 0.10, 0.94, tipRadius);

    auto rs = std::make_shared<RootSystem>();
    rs->readParameters(param, "plant", true, false);
    rs->initialize(false);

    auto base = std::make_shared<Gravitropism>(rs, 1.0, 0.2);
    std::vector<MaskNode> revealed;   // cavities already added (roots must avoid)
    int target = 0;                   // index into masks: current attraction goal

    auto applyState = [&]() {
        auto geom = buildCavityGeometry(revealed, R0, H, true, 2.0, tipRadius);
        rs->setGeometry(geom);
        if (target < (int) masks.size()) {
            auto attrs = rimAttractors({masks[target]}, rimN, 1.0, 3.0, 1.15);
            rs->setTropism(combinedAttraction(rs, base, attrs, 6.0, 0.25, weight, geom), -1);
        } else {
            rs->setTropism(base, -1);   // all masks revealed -- just fall
        }
    };
    applyState();

    std::ofstream out(outPath);
    double day = 0.0, tailStart = -1.0;
    bool dwelling = false;
    double reachedDay = 0.0;
    int frame = 0;
    while (day < maxDays) {
        rs->simulate(dt, false);
        day += dt;

        if (target < (int) masks.size() && !dwelling) {
            double d = minDistToMask(rs->getNodes(), masks[target]);
            double thr = reachMult * std::max({masks[target].r_width, masks[target].r_height});
            if (d < thr) {
                std::cout << "day " << day << ": reached mask " << target << " (dist=" << d
                          << "), dwelling " << dwellDays << " days\n";
                revealed.push_back(masks[target]);   // solid now -- can't be grown through
                dwelling = true;
                reachedDay = day;
                applyState();                        // geometry updated; attraction stays on `target`
            }
        } else if (dwelling && day - reachedDay > dwellDays) {
            std::cout << "day " << day << ": done wrapping mask " << target << ", advancing\n";
            target++;
            dwelling = false;
            if (target >= (int) masks.size()) tailStart = day;
            applyState();                            // attraction retargets to the next mask
        }
        if (tailStart > 0 && day - tailStart > tailDays) break;

        SegmentAnalyser ana(*rs);
        auto radii = ana.getParameter("radius");
        out << "FRAME " << frame << " " << day << " " << revealed.size() << " "
            << ana.nodes.size() << " " << ana.segments.size() << "\n";
        for (const auto& m : revealed)
            out << "MASK " << m.pos.x << " " << m.pos.y << " " << m.pos.z << " "
                << m.normal.x << " " << m.normal.y << " " << m.normal.z << " "
                << m.r_depth << " " << m.r_width << " " << m.r_height << "\n";
        for (const auto& n : ana.nodes)
            out << "NODE " << n.x << " " << n.y << " " << n.z << "\n";
        for (size_t i = 0; i < ana.segments.size(); ++i)
            out << "SEG " << ana.segments[i].x << " " << ana.segments[i].y << " " << radii[i] << "\n";
        frame++;
    }
    out.close();
    std::cout << "wrote " << frame << " frames, " << revealed.size() << "/" << masks.size()
              << " masks revealed -> " << outPath << "\n";
    return 0;
}
