#!/bin/bash
# Builds and runs the demo renderer, which writes four WAVs: a Drum Synth
# pattern, and that pattern run through each of the three effects.
#
# Usage: tests/run_demo.sh [output-dir]

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
MI="${MI_EURORACK_DIR:-$(cd "$ROOT/../.." && pwd)/eurorack}"
OUT="${1:-${TMPDIR:-/tmp}/mi_demo}"

if [ ! -d "$MI/stmlib" ]; then
    echo "eurorack checkout not found at $MI"
    echo "set MI_EURORACK_DIR to point at it"
    exit 1
fi

mkdir -p "$OUT"

clang++ -std=c++17 -O2 -DTEST -I"$ROOT" -I"$MI" -o "$OUT/mi_demo" "$HERE/mi_demo.cpp" \
    $MI/peaks/processors.cc $MI/peaks/drums/*.cc $MI/peaks/modulations/*.cc \
    $MI/peaks/number_station/*.cc $MI/peaks/pulse_processor/*.cc $MI/peaks/resources.cc \
    $MI/rings/dsp/part.cc $MI/rings/dsp/fm_voice.cc $MI/rings/dsp/resonator.cc \
    $MI/rings/dsp/string.cc $MI/rings/resources.cc \
    $MI/clouds/dsp/granular_processor.cc $MI/clouds/dsp/correlator.cc \
    $MI/clouds/dsp/mu_law.cc $MI/clouds/dsp/pvoc/*.cc $MI/clouds/resources.cc \
    $MI/elements/dsp/*.cc $MI/elements/resources.cc \
    $MI/stmlib/dsp/atan.cc $MI/stmlib/dsp/units.cc $MI/stmlib/utils/random.cc || exit 1

cd "$OUT" && ./mi_demo .
