// Headless exporter for the "mask column" variant: grows a root system confined
// by cone-phyllotaxis mask cavities and dumps segments+radii+mask nodes to a
// plain text file so it can be rendered/inspected without a GL context.
//
// Format:
//   NODE x y z            (root system nodes, one per line)
//   SEG a b radius         (segment: node indices a,b + radius cm)
//   MASK x y z nx ny nz rd rw rh   (mask node: pos, normal, ellipsoid radii)
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "RootSystem.h"
#include "SegmentAnalyser.h"
#include "MaskCavities.h"

using namespace CPlantBox;
using namespace maskcav;

int main(int argc, char** argv) {
    std::string param = (argc > 1) ? argv[1]
        : "/Users/erichan/Documents/Development/jardins_racine/CPlantBox/"
          "modelparameter/structural/rootsystem/Zea_mays_6_Leitner_2014.xml";
    std::string outPath = (argc > 2) ? argv[2] : "mask_column.txt";
    int N = (argc > 3) ? std::atoi(argv[3]) : 14;
    double days = (argc > 4) ? std::atof(argv[4]) : 45.0;
    double R0 = 8.0, H = 26.0, maskR = 2.6;

    auto masks = conePhyllotaxis(N, R0, H, maskR, 0.10, 0.92);

    auto rs = std::make_shared<RootSystem>();
    rs->readParameters(param, "plant", true, false);
    rs->setGeometry(buildCavityGeometry(masks, R0, H, true, 2.0));
    rs->initialize(false);
    rs->simulate(days, false);

    SegmentAnalyser ana(*rs);
    auto radii = ana.getParameter("radius");

    std::ofstream out(outPath);
    for (const auto& n : ana.nodes)
        out << "NODE " << n.x << " " << n.y << " " << n.z << "\n";
    for (size_t i = 0; i < ana.segments.size(); ++i)
        out << "SEG " << ana.segments[i].x << " " << ana.segments[i].y << " " << radii[i] << "\n";
    for (const auto& m : masks)
        out << "MASK " << m.pos.x << " " << m.pos.y << " " << m.pos.z << " "
            << m.normal.x << " " << m.normal.y << " " << m.normal.z << " "
            << m.r_depth << " " << m.r_width << " " << m.r_height << "\n";
    out.close();

    std::cout << "wrote " << ana.nodes.size() << " nodes, " << ana.segments.size()
              << " segments, " << masks.size() << " mask nodes -> " << outPath << "\n";
    return 0;
}
