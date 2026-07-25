"""Export the ICT-FaceKit morphable model to a flat binary for the C++ app.

Why ICT and not NVIDIA Maxine's own mesh: Maxine's 3DMM ships as face_model2.nvf
inside the AR SDK installer, which is EULA-gated and not redistributable. It is
not present in neuromirror -- emotion/TODO.md lists obtaining it as an open item
and the parser there (nvf_model.py) has only ever been validated against a
synthetic chunk file. Maxine's model *is* a modified ICT-FaceKit (MIT) with the
same 100 identity + 53 ARKit-blendshape structure, so ICT is a faithful
stand-in, and nvf_model.py can be pointed at the real file later without the
fitting code changing.

Source basis is neuromirror/emotion/data/ict_basis.npz (built by
emotion.ict_model.build_cache from the ICT .obj files).

The export is trimmed to the vertices the app actually needs: ICT's front-face
"fitting" region (the mask, no neck or scalp) plus any landmark vertices that
fall outside it -- 12 of the dlib-68 do, so restricting to the region alone
would silently drop landmarks and skew the identity fit.

    python3 tools/export_ict_basis.py [--out external/ict_basis.bin]
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np

MAGIC = b"ICTM"
VERSION = 1


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    default_src = os.path.join(here, "..", "..", "..", "neuromirror", "emotion",
                               "data", "ict_basis.npz")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=default_src)
    ap.add_argument("--out", default=os.path.join(here, "..", "external", "ict_basis.bin"))
    args = ap.parse_args()

    if not os.path.isfile(args.src):
        print(f"error: {args.src} not found.\n"
              f"Build it first:  cd neuromirror && "
              f".venv/bin/python -c 'from emotion.ict_model import build_cache; build_cache()'",
              file=sys.stderr)
        return 1

    d = np.load(args.src)
    neutral = d["neutral"].astype(np.float32)
    faces = d["faces"].astype(np.int32)
    id_modes = d["id_modes"].astype(np.float32)
    ex_modes = d["ex_modes"].astype(np.float32)
    ex_names = [str(x) for x in d["ex_names"]]
    lm68 = d["landmark68"].astype(np.int64)
    region = d["face_region"].astype(np.int64)

    # Keep the mask region plus every landmark vertex. 12 of the 68 sit outside
    # the region (jawline mostly), and dropping them would quietly bias the fit.
    keep = np.unique(np.concatenate([region, lm68]))
    remap = np.full(len(neutral), -1, np.int64)
    remap[keep] = np.arange(len(keep))
    outside = int((~np.isin(lm68, region)).sum())
    print(f"vertices: {len(keep)} kept of {len(neutral)} "
          f"({len(region)} region + {outside} landmarks outside it)")

    # Faces with all three corners kept. A face straddling the boundary would
    # render a stray triangle across the mask edge.
    fmask = np.all(remap[faces] >= 0, axis=1)
    tri = remap[faces[fmask]].astype(np.int32)
    print(f"faces: {len(tri)} of {len(faces)}")

    neutral_k = neutral[keep]
    id_k = id_modes[:, keep, :]
    ex_k = ex_modes[:, keep, :]
    lm_k = remap[lm68].astype(np.int32)
    assert (lm_k >= 0).all(), "a landmark vertex was dropped"

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(MAGIC)
        # counts, then arrays. Everything little-endian and tightly packed so
        # the C++ side can read it with a handful of fread calls.
        f.write(struct.pack("<7i", VERSION, len(keep), len(tri), len(id_k),
                            len(ex_k), len(lm_k), 0))
        neutral_k.tofile(f)
        tri.tofile(f)
        id_k.tofile(f)
        ex_k.tofile(f)
        lm_k.tofile(f)
        for n in ex_names:
            b = n.encode()
            f.write(struct.pack("<i", len(b)))
            f.write(b)

    mb = os.path.getsize(out) / 1e6
    print(f"wrote {out}  ({mb:.1f} MB)")
    print(f"  neutral {neutral_k.shape}  id {id_k.shape}  ex {ex_k.shape}  "
          f"lm {lm_k.shape}")
    print(f"  extent {neutral_k.min(0).round(2)} .. {neutral_k.max(0).round(2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
