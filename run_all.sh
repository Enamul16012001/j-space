#!/usr/bin/env bash
# Run everything in order.  Step 00 (computing the lens) is the slow one, and is
# skipped when its output already exists — delete the .pt file to force a redo.
set -e
cd "$(dirname "$0")"

LENS=$(python -c "import config; print(config.JLENS_PATH)")
if [ -f "$LENS" ]; then
    echo "== $LENS already exists, skipping step 00 (delete it to recompute) =="
else
    python experiments/00_compute_jlens.py
fi

for f in experiments/*.py; do
    case "$f" in *00_compute_jlens.py) continue ;; esac
    echo
    echo "===== $f ====="
    python "$f"
done
