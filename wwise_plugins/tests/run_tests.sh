#!/bin/bash
# Builds and runs the offline DSP harnesses. These drive the Mutable
# Instruments cores through exactly the same wrappers the Wwise plug-ins use
# (block adapter, resampler, FIFO), so they catch DSP regressions without
# needing to load Wwise.
#
# Usage: tests/run_tests.sh [output-dir]

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
MI="${MI_EURORACK_DIR:-$(cd "$ROOT/../.." && pwd)/eurorack}"
OUT="${1:-${TMPDIR:-/tmp}/mi_tests}"

if [ ! -d "$MI/stmlib" ]; then
    echo "eurorack checkout not found at $MI"
    echo "set MI_EURORACK_DIR to point at it"
    exit 1
fi

mkdir -p "$OUT"

# TEST selects stmlib's portable code paths (no Cortex-M asm, IN_RAM a no-op).
CXXFLAGS="-std=c++17 -O2 -DTEST -I$ROOT -I$ROOT/mi_common/patched -I$MI"

STMLIB="$MI/stmlib/dsp/atan.cc $MI/stmlib/dsp/units.cc $MI/stmlib/utils/random.cc"

fail=0

build_and_run() {
    local name="$1"; shift
    local src="$1"; shift
    echo "--- $name"
    if ! clang++ $CXXFLAGS -o "$OUT/$name" "$src" "$@" 2>&1 | grep -E "error"; then
        :
    fi
    if [ ! -x "$OUT/$name" ]; then
        echo "$name: BUILD FAILED"
        fail=1
        return
    fi
    if ! (cd "$OUT" && "./$name"); then
        fail=1
    fi
}

build_and_run resampler_test "$HERE/resampler_test.cpp"

build_and_run rings_test "$HERE/rings_render_test.cpp" \
    $MI/rings/dsp/part.cc $MI/rings/dsp/fm_voice.cc $MI/rings/dsp/resonator.cc \
    $MI/rings/dsp/string.cc $MI/rings/resources.cc $STMLIB

build_and_run plaits_test "$HERE/plaits_render_test.cpp" \
    $(find $MI/plaits/dsp -name "*.cc") $MI/plaits/resources.cc $STMLIB

build_and_run peaks_test "$HERE/peaks_render_test.cpp" \
    $MI/peaks/processors.cc $MI/peaks/drums/*.cc $MI/peaks/modulations/*.cc \
    $MI/peaks/number_station/*.cc $MI/peaks/pulse_processor/*.cc \
    $MI/peaks/resources.cc $MI/stmlib/utils/random.cc

build_and_run clouds_test "$HERE/clouds_render_test.cpp" \
    $MI/clouds/dsp/granular_processor.cc $MI/clouds/dsp/correlator.cc \
    $MI/clouds/dsp/mu_law.cc $MI/clouds/dsp/pvoc/*.cc $MI/clouds/resources.cc $STMLIB

build_and_run elements_test "$HERE/elements_render_test.cpp" \
    $MI/elements/dsp/*.cc $MI/elements/resources.cc $STMLIB

echo
if [ $fail -ne 0 ]; then
    echo "SOME TESTS FAILED"
    exit 1
fi
echo "ALL TESTS PASSED (audio written to $OUT)"
