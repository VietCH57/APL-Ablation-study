#!/usr/bin/env bash
# Ablation: AL, and P / PL / APL with phonetic embeddings from XLS-R-300M before (raw) and after
# (ft) phoneme-recognition fine-tuning. One run per configuration, seed 42 (configs/apl.yaml).
#
#   RAW_DIR=feats/xlsr_raw FT_DIR=feats/xlsr_ft bash scripts/run_ablation.sh
#
# GPU limits (50% VRAM, 50% GPU time) come from configs/apl.yaml; override with e.g.
#   EXTRA="--gpu_memory_fraction 0.5 --gpu_duty_pct 50"
# Runs that already have test_metrics.json are skipped, so the script can be re-launched.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RAW_DIR="${RAW_DIR:-feats/xlsr_raw}"
FT_DIR="${FT_DIR:-feats/xlsr_ft}"
RUNS="${RUNS:-runs}"
CONFIG="${CONFIG:-configs/apl.yaml}"
EXTRA="${EXTRA:-}"

run() {  # name variant phon_dir
    local out="$RUNS/$1"
    if [ -f "$out/test_metrics.json" ]; then echo "skip $out"; return; fi
    local phon=()
    if [ -n "$3" ]; then phon=(--phon_dir "$3"); fi
    echo "=== $out"
    # shellcheck disable=SC2086
    python scripts/train.py --config "$CONFIG" --variant "$2" ${phon[@]+"${phon[@]}"} --out_dir "$out" $EXTRA
}

run AL      AL  ""
run P-raw   P   "$RAW_DIR"
run PL-raw  PL  "$RAW_DIR"
run APL-raw APL "$RAW_DIR"
run P-ft    P   "$FT_DIR"
run PL-ft   PL  "$FT_DIR"
run APL-ft  APL "$FT_DIR"

python scripts/summarize.py --runs_dir "$RUNS" --out "$RUNS/results_test.md"
