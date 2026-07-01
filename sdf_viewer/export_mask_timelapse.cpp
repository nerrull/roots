// Timelapse exporter: grows the mask-cavity/attractor root system incrementally
// and dumps a NODE/SEG snapshot after every step, so the growth can be rendered
// as an animation. MASK lines (fixed, don't grow) are written once at the top.
//
// Format: same as export_mask_column.cpp, plus "FRAME <i> <day>" markers
// separating snapshots.
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "RootSystem.h"
#include "SegmentAnalyser.h"
#include "MaskCavities.h"
#include "RootAttractors.h"

using namespace CPlantBox;
using namespace maskcav;

int main(int argc, char** argv) {
    std::string param = (argc > 1) ? argv[1]
        : "/Users/erichan/Documents/Development/jardins_racine/CPlantBox/"
          "modelparameter/structural/rootsystem/Zea_mays_6_Leitner_2014.xml";
    std::string outPath = (argc > 2) ? argv[2] : "mask_timelapse.txt";
    int N = (argc > 3) ? std::atoi(argv[3]) : 8;
    double totalDays = (argc > 4) ? std::atof(argv[4]) : 45.0;
    int nFrames = (argc > 5) ? std::atoi(argv[5]) : 60;
    bool attract = (argc > 6) ? (std::atoi(argv[6]) != 0) : true;
    double weight = (argc > 7) ? std::atof(argv[7]) : 0.5;
    double R0 = 9.0, H = 34.0, maskR = 2.6;

    auto masks = conePhyllotaxis(N, R0, H, maskR, 0.08, 0.94);

    auto rs = std::make_shared<RootSystem>();
    rs->readParameters(param, "plant", true, false);
    auto geom = buildCavityGeometry(masks, R0, H, true, 2.0);
    rs->setGeometry(geom);
    rs->initialize(false);
    if (attract) {
        auto attrs = rimAttractors(masks, 3, 1.0, 3.0, 1.15);
        auto base = std::make_shared<Gravitropism>(rs, 1.0, 0.2);
        rs->setTropism(combinedAttraction(rs, base, attrs, 6.0, 0.25, weight, geom), -1);
    }

    std::ofstream out(outPath);
    for (const auto& m : masks)
        out << "MASK " << m.pos.x << " " << m.pos.y << " " << m.pos.z << " "
            << m.normal.x << " " << m.normal.y << " " << m.normal.z << " "
            << m.r_depth << " " << m.r_width << " " << m.r_height << "\n";

    double dt = totalDays / nFrames;
    for (int f = 0; f < nFrames; ++f) {
        rs->simulate(dt, false);
        SegmentAnalyser ana(*rs);
        auto radii = ana.getParameter("radius");
        out << "FRAME " << f << " " << (dt * (f + 1)) << "\n";
        for (const auto& n : ana.nodes)
            out << "NODE " << n.x << " " << n.y << " " << n.z << "\n";
        for (size_t i = 0; i < ana.segments.size(); ++i)
            out << "SEG " << ana.segments[i].x << " " << ana.segments[i].y << " " << radii[i] << "\n";
        std::cout << "frame " << f << "/" << nFrames << " day " << (dt * (f + 1))
                  << " -- " << ana.nodes.size() << " nodes\n";
    }
    out.close();
    std::cout << "wrote " << nFrames << " frames -> " << outPath << "\n";
    return 0;
}
