"""Dump a demo_panel PondState render to a u8 fixture for C++ parity checking.

Uses drops=0 + orbit_on so there is NO dependence on Python's random.Random drop
placement (which the C++ port can't reproduce bit-for-bit) — every other part of
the pipeline (bit-identical MLX weights, features, colour, transition) is covered.

    cd /Users/erichan/Documents/Development/neuromirror
    PYTHONPATH=. .venv/bin/python \
        ../jardins_racine/mirror_app/tests/dump_pond_fixture.py
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np

from mlx_fused_mlp.demo_panel import PondState

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures")

OUT_W, OUT_H, DOWNSCALE = 1920, 1080, 4    # → lw=480, lh=270
T, Z = 1.234, 0.5


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    s = PondState(seed=11)
    s.drops = 0
    s.orbit_on = True
    s.downscale = DOWNSCALE
    s.t = T
    s.z = Z
    img = s.render(OUT_W, OUT_H)          # (lh, lw, 3) uint8, RGB
    mx.synchronize()
    img = np.ascontiguousarray(img).astype(np.uint8)
    lh, lw, _ = img.shape
    img.tofile(os.path.join(OUT, "pond_u8.bin"))
    with open(os.path.join(OUT, "pond_u8.meta"), "w") as f:
        f.write(f"lh {lh}\nlw {lw}\nt {T}\nz {Z}\n")
    print(f"wrote pond_u8.bin ({lh}x{lw}x3) t={T} z={Z}")


if __name__ == "__main__":
    main()
