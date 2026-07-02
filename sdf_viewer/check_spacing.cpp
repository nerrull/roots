// Quick diagnostic: print pairwise distances between mask cavity centers for
// given phyllotaxis params, so spacing can be tuned before a full growth run.
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include "MaskCavities.h"
using namespace maskcav;

int main(int argc, char** argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 8;
    double R0 = (argc > 2) ? std::atof(argv[2]) : 9.0;
    double H = (argc > 3) ? std::atof(argv[3]) : 34.0;
    double maskR = (argc > 4) ? std::atof(argv[4]) : 2.6;
    double startFrac = (argc > 5) ? std::atof(argv[5]) : 0.10;
    double endFrac = (argc > 6) ? std::atof(argv[6]) : 0.94;
    double tipRadius = 0.22 * R0;

    auto masks = conePhyllotaxis(N, R0, H, maskR, startFrac, endFrac, tipRadius);
    double minD = 1e18, sumNearest = 0;
    for (size_t i = 0; i < masks.size(); i++) {
        double nearest = 1e18;
        for (size_t j = 0; j < masks.size(); j++) {
            if (i == j) continue;
            double d = masks[i].pos.minus(masks[j].pos).length();
            minD = std::min(minD, d);
            nearest = std::min(nearest, d);
        }
        sumNearest += nearest;
        std::cout << "mask " << i << " nearest neighbor dist = " << nearest
                  << "  (cavity size ~" << std::max(masks[i].r_width, masks[i].r_height) << ")\n";
    }
    std::cout << "\nN=" << N << " R0=" << R0 << " H=" << H << " maskR=" << maskR
              << "\nglobal min pairwise dist = " << minD
              << "\nmean nearest-neighbor dist = " << sumNearest / masks.size()
              << "\n(cavity radius ~" << maskR << ", so dist should be well over 2x that to avoid crowding)\n";
    return 0;
}
