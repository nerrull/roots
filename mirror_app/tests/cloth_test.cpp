// Headless validation of the cloth solver (no Metal/GPU). Steps the sim and
// checks: hole carved, rim pinned, stays finite, rim stays put, and the free
// sheet actually falls behind the mask (-z). Build: see CMake target cloth_test.
#include "cloth.h"
#include <cstdio>

int main() {
    Cloth c;
    c.build(/*nx*/64, /*ny*/64, /*w*/2.0f, /*h*/2.0f, /*hrx*/0.35f, /*hry*/0.45f);

    int nActive = c.activeCount(), nPin = c.pinnedCount();
    printf("built: %dx%d  active=%d  pinned=%d  edges=%zu  tris=%zu\n",
           c.nx, c.ny, nActive, nPin, c.edges.size(), c.tris.size() / 3);

    bool ok = true;
    if (nActive <= 0 || nActive >= c.nx * c.ny) { printf("FAIL: hole not carved\n"); ok = false; }
    if (nPin <= 0) { printf("FAIL: no rim pinned\n"); ok = false; }

    // snapshot pinned targets, then simulate
    auto pin0 = c.pin;
    for (int f = 0; f < 300; ++f) c.step(1.0f / 120.0f);

    if (!c.finite()) { printf("FAIL: non-finite after sim\n"); ok = false; }

    float rimDrift = 0.f;
    for (size_t k = 0; k < c.pos.size(); ++k)
        if (c.pinned[k]) rimDrift = std::max(rimDrift, simd_distance(c.pos[k], pin0[k]));
    if (rimDrift > 1e-4f) { printf("FAIL: rim drifted %.5f\n", rimDrift); ok = false; }

    float fell = c.minZ();
    printf("after 300 steps: finite=%d  rimDrift=%.2e  minZ=%.3f (should be < 0)\n",
           c.finite(), rimDrift, fell);
    if (fell > -0.05f) { printf("FAIL: sheet did not fall behind the mask\n"); ok = false; }

    printf(ok ? "cloth_test: PASS\n" : "cloth_test: FAIL\n");
    return ok ? 0 : 1;
}
