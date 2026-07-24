"""Dump fixed (coords, weights, expected-output) fixtures from the Python
reference fused MLP, so the C++/MLX port can be checked for parity against them.

Run from the neuromirror repo (its venv has mlx + the mlx_fused_mlp package):

    cd /Users/erichan/Documents/Development/neuromirror
    PYTHONPATH=. .venv/bin/python \
        ../jardins_racine/mirror_app/tests/dump_fixtures.py

Writes float32 raw little-endian arrays + a meta.txt into fixtures/ next to this
script. C++ reads them back, runs the ported kernel, and compares within the same
fp16 tolerance the Python tests use.
"""

from __future__ import annotations

import os

import mlx.core as mx

from mlx_fused_mlp.config import MLPConfig
from mlx_fused_mlp.forward import fused_mlp_forward
from mlx_fused_mlp.tests.helpers import make_coords, make_weights

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures")


def save_f32(name: str, arr: mx.array) -> None:
    import numpy as np
    a = np.array(arr.astype(mx.float32), copy=True).astype("<f4").ravel()
    a.tofile(os.path.join(OUT, name))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    # Representative "raw pond / mirror" config: 8-in, 64-wide, 6-layer, tanh→sigmoid.
    cfg = MLPConfig(in_dim=8, hidden_dim=64, out_dim=3, num_layers=6,
                    activation="tanh", output_activation="sigmoid")
    n = 1000  # not a multiple of TILE_ROWS=32, so the tail path is exercised
    coords = make_coords(n, cfg, seed=1)
    weights = make_weights(cfg, seed=0, scale=1.0)
    expected = fused_mlp_forward(coords, weights, cfg)
    mx.eval(coords, weights, expected)

    save_f32("coords.f32", coords)
    save_f32("weights.f32", weights)
    save_f32("expected.f32", expected)
    with open(os.path.join(OUT, "meta.txt"), "w") as f:
        f.write(f"n {n}\n")
        f.write(f"in_dim {cfg.in_dim}\n")
        f.write(f"hidden_dim {cfg.hidden_dim}\n")
        f.write(f"out_dim {cfg.out_dim}\n")
        f.write(f"num_layers {cfg.num_layers}\n")
        f.write(f"act {cfg.act_code}\n")
        f.write(f"out_act {cfg.out_act_code}\n")
        f.write(f"total_weights {cfg.total_weights}\n")
    print(f"wrote fixtures to {OUT} (n={n}, total_weights={cfg.total_weights})")


if __name__ == "__main__":
    main()
