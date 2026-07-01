// Isolate AttractionTropism from cavity confinement: grow a plain root system
// toward a single point attractor with no other geometry, and check root nodes
// end up closer to it than an unattracted baseline. If this passes but the
// combined mask-cavity test doesn't, the interaction with the confining
// geometry (not the attraction math) is the problem.
#include <iostream>
#include <limits>
#include <memory>
#include <string>

#include "RootSystem.h"
#include "RootAttractors.h"

using namespace CPlantBox;
using namespace maskcav;

static double meanDist(const std::vector<Vector3d>& nodes, const Vector3d& target) {
    double sum = 0;
    for (const auto& p : nodes) sum += p.minus(target).length();
    return nodes.empty() ? 0.0 : sum / nodes.size();
}

int main(int argc, char** argv) {
    std::string param = argv[1];
    double days = (argc > 2) ? std::atof(argv[2]) : 20.0;
    double weight = (argc > 3) ? std::atof(argv[3]) : 0.7;
    double n = (argc > 4) ? std::atof(argv[4]) : 12.0;

    Vector3d target(6.0, 0.0, -15.0);   // off to the side and deep -- gravitropism alone won't reach x=6

    auto rsB = std::make_shared<RootSystem>();
    rsB->readParameters(param, "plant", true, false);
    rsB->initialize(false);
    rsB->simulate(days, false);
    auto nodesB = rsB->getNodes();

    auto rsA = std::make_shared<RootSystem>();
    rsA->readParameters(param, "plant", true, false);
    rsA->initialize(false);
    auto base = std::make_shared<Gravitropism>(rsA, 1.0, 0.2);
    std::vector<Attractor> attrs{Attractor{target, 1.0, 8.0}};
    rsA->setTropism(combinedAttraction(rsA, base, attrs, n, 0.25, weight), -1);
    rsA->simulate(days, false);
    auto nodesA = rsA->getNodes();

    double dB = meanDist(nodesB, target);
    double dA = meanDist(nodesA, target);
    std::cout << "target=(" << target.x << "," << target.y << "," << target.z << ")\n"
              << "BASELINE : " << nodesB.size() << " nodes, mean dist to target = " << dB << " cm\n"
              << "ATTRACTED: " << nodesA.size() << " nodes, mean dist to target = " << dA << " cm\n";
    std::cout << (dA < dB ? "PASS\n" : "FAIL\n");
    return dA < dB ? 0 : 1;
}
