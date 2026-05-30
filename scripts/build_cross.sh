#!/usr/bin/env bash
# Cross-compile star_detect on an x86 Linux box for the Pi Zero 2 W (aarch64).
# Produces a wheel under target/wheels/ to copy to the Pi and `pip install`.
# Run from the star_detect/ directory.
set -euo pipefail

# One-time prerequisites on the x86 host:
# rustup target add aarch64-unknown-linux-gnu
# sudo apt install gcc-aarch64-linux-gnu
# pip install --user maturin
#
# Tell cargo to use the cross linker (add to ~/.cargo/config.toml once):
#   [target.aarch64-unknown-linux-gnu]
#   linker = "aarch64-linux-gnu-gcc"

PYVER="${PYVER:-3.11}"

echo "=== Cross-building wheel for aarch64-unknown-linux-gnu (Python $PYVER) ==="
maturin build --release \
    --target aarch64-unknown-linux-gnu \
    --interpreter "python${PYVER}"

echo
echo "Wheel(s) produced:"
ls -1 target/wheels/

echo
echo "Copy and install on the Pi, e.g.:"
echo "  scp target/wheels/star_detect-*-cp${PYVER//./}-*aarch64*.whl pi@pi.local:"
echo "  ssh pi@pi.local pip install --user ./star_detect-*.whl"
