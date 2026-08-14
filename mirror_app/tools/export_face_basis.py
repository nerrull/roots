"""Export a morphable face basis to a flat binary for the C++ app.

Supersedes export_ict_basis.py, which could only emit ICT's own topology.

The file carries TWO bases that share one set of coefficients:

  * a **render mesh** -- whatever topology the mask should be drawn with, and
  * a **landmark basis** -- 68 points in dlib order, used only for fitting.

Keeping them separate is the whole point. Identity fitting needs landmarks in
exact dlib-68 order, and ICT-FaceKit is the model that has that ordering
right (from vertex_indices.json). NVIDIA's Maxine mesh (face_model2.nvf) has
the nicer render topology -- 2056 verts / 4048 tris, already cropped to a face
mask, no neck or scalp -- but its LMRK chunk stores landmarks in an internal
order that neuromirror had to recover by nearest-neighbour matching, so it is
the less trustworthy source for a fit.

So: fit against ICT's landmarks, draw with Maxine's triangles. That is the
bridge neuromirror/emotion/nvf_live.py validated live, and this exporter bakes
it in -- the render basis is ICT's identity/expression modes resampled onto the
NVF vertex positions by nearest neighbour, so one (alpha, expression) pair
drives both bases consistently and the C++ side needs no mapping table.

    # Maxine topology (default when the .nvf is present)
    python3 tools/export_face_basis.py --nvf ../../neuromirror/emotion/face_assets/face_model2.nvf

    # ICT topology, no Maxine needed
    python3 tools/export_face_basis.py

Both write external/face_basis.bin, which is gitignored -- generated, like the
MediaPipe dylib.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np

MAGIC = b"FBAS"
VERSION = 1


def nearest_map(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """For each dst point, the index of the nearest src point.

    Chunked because the full distance matrix is len(dst) x len(src); at
    2056 x 7801 that is fine, but a denser render mesh would not be.
    """
    out = np.empty(len(dst_pts), np.int64)
    step = 256
    for i in range(0, len(dst_pts), step):
        blk = dst_pts[i:i + step]
        d2 = ((blk[:, None, :] - src_pts[None, :, :]) ** 2).sum(-1)
        out[i:i + step] = d2.argmin(1)
    return out


def load_ict(path: str):
    if not os.path.isfile(path):
        raise SystemExit(
            f"error: {path} not found.\nBuild it first:  cd neuromirror && "
            f".venv/bin/python -c 'from emotion.ict_model import build_cache; build_cache()'")
    d = np.load(path)
    return dict(
        neutral=d["neutral"].astype(np.float32),
        faces=d["faces"].astype(np.int32),
        id_modes=d["id_modes"].astype(np.float32),
        ex_modes=d["ex_modes"].astype(np.float32),
        ex_names=[str(x) for x in d["ex_names"]],
        lm68=d["landmark68"].astype(np.int64),
        region=d["face_region"].astype(np.int64),
    )


def load_nvf(path: str, neuromirror_dir: str):
    """Parse face_model2.nvf via neuromirror's validated NVFModel."""
    if neuromirror_dir not in sys.path:
        sys.path.insert(0, neuromirror_dir)
    from emotion.nvf_model import NVFModel   # noqa: E402
    m = NVFModel(path)
    if m.faces is None:
        raise SystemExit(f"error: {path} has no TRNG chunk -- no triangles to render")
    return m.neutral.astype(np.float32), m.faces.astype(np.int32)


def ict_render_mesh(ict):
    """ICT's own front-face region as the render mesh (the no-Maxine fallback)."""
    keep = ict["region"]
    remap = np.full(len(ict["neutral"]), -1, np.int64)
    remap[keep] = np.arange(len(keep))
    fmask = np.all(remap[ict["faces"]] >= 0, axis=1)
    tris = remap[ict["faces"][fmask]].astype(np.int32)
    print(f"render mesh: ICT face region -- {len(keep)} verts, {len(tris)} of "
          f"{len(ict['faces'])} faces (a face straddling the region edge would "
          f"draw a stray triangle across the mask boundary)")
    return keep, tris


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    nm = os.path.abspath(os.path.join(here, "..", "..", "..", "neuromirror"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(nm, "emotion", "data", "ict_basis.npz"),
                    help="ICT basis npz -- always the source of the landmark basis")
    ap.add_argument("--nvf", default=os.path.join(nm, "emotion", "face_assets",
                                                  "face_model2.nvf"),
                    help="Maxine .nvf for the render topology; omit or point at a "
                         "missing file to fall back to ICT's own region mesh")
    ap.add_argument("--neuromirror", default=nm)
    ap.add_argument("--out", default=os.path.join(here, "..", "external", "face_basis.bin"))
    args = ap.parse_args()

    ict = load_ict(args.src)
    n_id, n_ex = len(ict["id_modes"]), len(ict["ex_modes"])

    # --- landmark basis: always ICT, always the full 68 -----------------------
    # 12 of the 68 sit outside ICT's face region. Extracting them here rather
    # than from the render mesh is what stops the region crop from silently
    # dropping them and skewing the identity fit.
    lm = ict["lm68"]
    lm_neutral = ict["neutral"][lm]
    lm_id = ict["id_modes"][:, lm, :]
    lm_ex = ict["ex_modes"][:, lm, :]
    outside = int((~np.isin(lm, ict["region"])).sum())
    print(f"landmark basis: ICT dlib-68 ({outside} of them outside the face region, "
          f"kept anyway)")

    # --- render mesh ---------------------------------------------------------
    if args.nvf and os.path.isfile(args.nvf):
        nvf_neutral, tris = load_nvf(args.nvf, args.neuromirror)
        # Both meshes are y-up and in the same units, aligned closely enough that
        # no transform is needed (neuromirror measured nose tips agreeing to
        # <0.02 mm), so nearest neighbour against ICT's face region is a
        # resample, not a registration.
        region = ict["region"]
        pick = region[nearest_map(ict["neutral"][region], nvf_neutral)]
        resid = np.linalg.norm(ict["neutral"][pick] - nvf_neutral, axis=1)
        print(f"render mesh: NVF (Maxine) -- {len(nvf_neutral)} verts, {len(tris)} tris")
        print(f"  ICT->NVF resample residual: mean {resid.mean():.3f} "
              f"max {resid.max():.3f} (model units, ~cm)")
        source = "nvf"
    else:
        pick, tris = ict_render_mesh(ict)
        source = "ict"

    neutral = ict["neutral"][pick]
    id_modes = ict["id_modes"][:, pick, :]
    ex_modes = ict["ex_modes"][:, pick, :]
    n_v = len(neutral)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<8i", VERSION, n_v, len(tris), n_id, n_ex, len(lm),
                            1 if source == "nvf" else 0, 0))
        for a in (neutral, tris, id_modes, ex_modes, lm_neutral, lm_id, lm_ex):
            np.ascontiguousarray(a).tofile(f)
        for n in ict["ex_names"]:
            b = n.encode()
            f.write(struct.pack("<i", len(b)))
            f.write(b)

    print(f"wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
    print(f"  render  neutral {neutral.shape}  tris {tris.shape}  "
          f"id {id_modes.shape}  ex {ex_modes.shape}")
    print(f"  landmark neutral {lm_neutral.shape}  id {lm_id.shape}  ex {lm_ex.shape}")
    print(f"  extent {neutral.min(0).round(2)} .. {neutral.max(0).round(2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
