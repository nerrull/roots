#!/usr/bin/env bash
# Build MediaPipe's C Tasks API as a shared library for mirror_app's face
# tracking. macOS / Apple Silicon.
#
# This is the "one-time Bazel build" PLAN.md refers to. It is deliberately NOT
# cpvrlab/libmediapipe: that wrapper pins MediaPipe v0.8.11, which predates the
# Tasks API entirely (legacy 468-point face_mesh, no blendshapes, no pose
# matrix) and is GPL-3.0. Upstream now ships an official C API for the tasks
# under Apache-2.0, which is what this builds:
#     mediapipe/tasks/c/vision/face_landmarker -> 478 landmarks, 52
#     blendshapes, 4x4 facial transformation matrix.
#
# external/ is gitignored and re-cloned, so the local fixes live in patches/
# and are re-applied here. Each one is a distinct failure mode -- see the patch
# headers for why, they are not cosmetic.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mp="$here/external/mediapipe"
model="$here/external/face_landmarker.task"
target="//mediapipe/tasks/c/vision/face_landmarker:libface_landmarker.dylib"

# --- prerequisites ----------------------------------------------------------
#
# Three of these are not optional and each fails in a way that does not name
# itself:
#   bazelisk  -- honours MediaPipe's .bazelversion (7.4.1)
#   openjdk   -- without a JDK, local_jdk resolves to a stub and the build dies
#                with a confusing "no such package '@@rules_java~//tools/jdk'"
#   opencv@4  -- the plain `opencv` formula is 5.x, whose headers live under
#                include/opencv5 and whose API MediaPipe does not build against
need_brew=()
command -v bazelisk >/dev/null 2>&1 || command -v bazel >/dev/null 2>&1 || need_brew+=(bazelisk)
[[ -d /opt/homebrew/opt/openjdk ]] || need_brew+=(openjdk)
[[ -d /opt/homebrew/opt/opencv@4 ]] || need_brew+=(opencv@4)
if (( ${#need_brew[@]} )); then
  echo "==> installing: ${need_brew[*]}"
  brew install "${need_brew[@]}"
fi

export JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home
export PATH="$JAVA_HOME/bin:$PATH"

if [[ ! -d "$mp" ]]; then
  echo "==> cloning mediapipe"
  git clone --depth 1 https://github.com/google-ai-edge/mediapipe.git "$mp"
fi

# --- local patches ----------------------------------------------------------
# Idempotent: a patch that is already applied is skipped, matching
# kinect_v2_validate/setup.sh.
for p in "$here"/patches/*.patch; do
  [[ -e "$p" ]] || continue
  if git -C "$mp" apply --reverse --check "$p" >/dev/null 2>&1; then
    echo "==> patch already applied: $(basename "$p")"
  elif git -C "$mp" apply "$p"; then
    echo "==> applied patch: $(basename "$p")"
  else
    echo "!! failed to apply $(basename "$p") -- upstream may have moved." >&2
    echo "   Without it the build fails; see the patch header for what it fixes." >&2
    exit 1
  fi
done

# --- build ------------------------------------------------------------------
# HERMETIC_PYTHON_VERSION: MediaPipe only ships requirements lockfiles for
# 3.9-3.12, and a newer system python3 (3.13+) aborts configuration outright.
echo "==> building $target (first build takes a while)"
(
  cd "$mp"
  # macos_minimum_os: bazel's default deployment target is old enough that
  # libc++ marks std::optional::value() unavailable, and flatbuffers (a
  # MediaPipe dependency) calls it -- ~20 "'value' is unavailable: introduced
  # in macOS 10.13" errors deep in binary_annotator.cpp, which name the SDK
  # rather than the flag actually responsible. host_ too: flatc is built for
  # the host and hits the same wall.
  bazel build --config darwin_arm64 -c opt \
    --define MEDIAPIPE_DISABLE_GPU=1 \
    --macos_minimum_os=11.0 \
    --host_macos_minimum_os=11.0 \
    --repo_env=HERMETIC_PYTHON_VERSION=3.12 \
    --repo_env=JAVA_HOME="$JAVA_HOME" \
    "$target"
)

lib="$mp/bazel-bin/mediapipe/tasks/c/vision/face_landmarker/libface_landmarker.dylib"
[[ -f "$lib" ]] || { echo "!! build reported success but $lib is missing" >&2; exit 1; }

# The C entry points must actually be exported -- patch 0003 exists precisely
# because they silently were not, producing a 14 MB library with none of them.
n=$(nm -gU "$lib" | grep -c "MpFaceLandmarker" || true)
if (( n == 0 )); then
  echo "!! $lib exports no MpFaceLandmarker* symbols; patch 0003 did not take" >&2
  exit 1
fi
echo "==> built, exporting $n MpFaceLandmarker* symbols"

# --- model ------------------------------------------------------------------
if [[ ! -f "$model" ]]; then
  src="$here/../../neuromirror/face_landmarker.task"
  if [[ -f "$src" ]]; then
    cp "$src" "$model"
    echo "==> copied face_landmarker.task from neuromirror"
  else
    echo "!! face_landmarker.task not found. Download it to:" >&2
    echo "   $model" >&2
    echo "   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" >&2
    exit 1
  fi
fi

echo
echo "done. Re-run cmake to pick it up:"
echo "  cmake -S . -B build   # should print 'MediaPipe face tracking enabled'"
