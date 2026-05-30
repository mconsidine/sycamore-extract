#!/usr/bin/env bash
# Build star_detect natively on the Raspberry Pi Zero 2 W and install into the
# current Python environment. Run this from the star_detect/ directory.
set -euo pipefail

# One-time toolchain prerequisites (uncomment if missing):
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# source "$HOME/.cargo/env"
# pip install --user maturin numpy pillow

echo "=== Building star_detect (release, native A53) ==="
maturin develop --release

echo
echo "=== Verifying the module imports ==="
python3 -c "import star_detect; print('OK:', star_detect.__file__)"

echo
echo "=== Spot-checking the generated assembly for NEON ==="
# scan_band's inner threshold scan is what should vectorize. Look for v-register
# byte loads (ld1) and unsigned compares (cmhs) — those mean NEON. Scalar code
# uses ldrb/cmp on w-registers.
cargo rustc --release --lib -- --emit asm -C "llvm-args=-x86-asm-syntax=intel" 2>/dev/null || true
ASM=$(find target -name '*.s' -path '*release*' | head -n1 || true)
if [ -n "${ASM:-}" ]; then
    echo "Asm at: $ASM"
    echo "NEON byte-load (ld1 ... .16b) occurrences in scan_band region:"
    awk '/scan_band/,/^_ZN/' "$ASM" | grep -cE 'ld1\s*\{[^}]*\.16b\}' || echo "0 (autovectorizer did NOT use NEON for the byte loop)"
else
    echo "(asm file not found; run: cargo rustc --release --lib -- --emit asm)"
fi

echo
echo "Done. Next: run tests/bench.py against real frames."
