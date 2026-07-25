#!/usr/bin/env bash
# Build libfreenect2 (from source, no brew formula exists) into a local
# install prefix, then build the validator. Idempotent-ish: re-running
# rebuilds. macOS / Apple Silicon.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
f2="$here/external/libfreenect2"
prefix="$f2/install"

if [[ ! -d "$f2" ]]; then
  echo "cloning libfreenect2..."
  git clone --depth 1 https://github.com/OpenKinect/libfreenect2.git "$f2"
fi

# Local patches. external/ is gitignored and re-cloned, so they live in patches/
# and are (re)applied here. Idempotent: skipped if already applied.
for p in "$here"/patches/*.patch; do
  [[ -e "$p" ]] || continue
  if git -C "$f2" apply --reverse --check "$p" >/dev/null 2>&1; then
    echo "==> patch already applied: $(basename "$p")"
  elif git -C "$f2" apply "$p"; then
    echo "==> applied patch: $(basename "$p")"
  else
    echo "!! failed to apply $(basename "$p") -- upstream may have moved." >&2
    echo "   Audio and video/depth cannot run simultaneously without it." >&2
    exit 1
  fi
done

echo "==> build deps (libusb, jpeg-turbo, glfw)"
brew install libusb jpeg-turbo glfw >/dev/null || true

brew_prefix="$(brew --prefix)"

echo "==> configuring libfreenect2"
cmake -S "$f2" -B "$f2/build" \
  -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DENABLE_CXX11=ON \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_OPENNI2_DRIVER=OFF \
  -DLibUSB_ROOT="$brew_prefix" \
  -DTurboJPEG_ROOT="$brew_prefix/opt/jpeg-turbo" \
  -DGLFW3_ROOT="$brew_prefix"

echo "==> building libfreenect2"
cmake --build "$f2/build" -j"$(sysctl -n hw.ncpu)"
cmake --install "$f2/build"

echo "==> building validator"
cmake -S "$here" -B "$here/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$here/build" -j"$(sysctl -n hw.ncpu)"

echo
echo "done. run:  $here/build/kinect_v2_validate -t 10"
echo
echo "NOTE: Kinect v2 requires a udev-style permission on Linux; on macOS just"
echo "plug into a real USB3 port. If 'devices found: 0', unplug/replug the"
echo "USB3 adapter brick and re-run."
