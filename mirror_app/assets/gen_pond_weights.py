"""Dump the demo_pond MLP weights to a flat float32 asset the C++ app loads.

Reuses neuromirror's exact weight construction (make_weights + scale_weights) so
the C++ mirror shows the same "pond" the Python demo does. Run from neuromirror:

    cd /Users/erichan/Documents/Development/neuromirror
    PYTHONPATH=. .venv/bin/python \
        ../jardins_racine/mirror_app/assets/gen_pond_weights.py
"""

from __future__ import annotations

import os

import mlx.core as mx

from mlx_fused_mlp.config import MLPConfig
from mlx_fused_mlp.demo_video import scale_weights
from mlx_fused_mlp.features import ENRICHED_DIM
from mlx_fused_mlp.tests.helpers import make_weights

HERE = os.path.dirname(os.path.abspath(__file__))

# demo_pond defaults.
SEED, HIDDEN, LAYERS = 11, 64, 6
HIDDEN_SCALE, OUTPUT_SCALE = 3.0, 6.0


def main() -> None:
    cfg = MLPConfig(ENRICHED_DIM, HIDDEN, 3, LAYERS, "tanh", "sigmoid")
    wb = make_weights(cfg, seed=SEED, scale=1.0).astype(mx.float16)
    w = scale_weights(wb, HIDDEN_SCALE, cfg.weight_offsets[-1], OUTPUT_SCALE)
    mx.eval(w)

    import numpy as np
    np.array(w.astype(mx.float32), copy=True).astype("<f4").ravel().tofile(
        os.path.join(HERE, "pond_weights.f32"))
    with open(os.path.join(HERE, "pond_weights.meta"), "w") as f:
        f.write(f"in_dim {cfg.in_dim}\n")
        f.write(f"hidden_dim {cfg.hidden_dim}\n")
        f.write(f"out_dim {cfg.out_dim}\n")
        f.write(f"num_layers {cfg.num_layers}\n")
        f.write(f"act {cfg.act_code}\n")
        f.write(f"out_act {cfg.out_act_code}\n")
        f.write(f"total_weights {cfg.total_weights}\n")
    print(f"wrote pond_weights.f32 (total_weights={cfg.total_weights})")


if __name__ == "__main__":
    main()
