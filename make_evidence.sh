#!/usr/bin/env bash
# Regenerate the execution evidence in evidence/. No API key or network needed.
#
#   ./make_evidence.sh
#
# Each command's output (stdout and stderr) is saved as a .txt file, and the
# log the runs wrote is copied to evidence/pawpal_log_sample.txt.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
OUT=evidence
mkdir -p "$OUT"

# Keep these runs out of the real logs/pawpal.log, and keep a real .env key
# out of them (nothing here should ever call Gemini).
LOG_DIR="$(mktemp -d)"
# PYTHONUNBUFFERED keeps stdout and stderr (log warnings) in their real order.
export PAWPAL_LOG_DIR="$LOG_DIR" PAWPAL_SKIP_DOTENV=1 PYTHONUNBUFFERED=1
unset GEMINI_API_KEY GOOGLE_API_KEY

run() {
  local name="$1"; shift
  echo "Running: $*"
  {
    echo "\$ ${SHOWN:-$*}"
    echo
    "$@" 2>&1
    echo
    echo "[exit code $?]"
  } > "$OUT/$name.txt"
}

# The tests get their own throwaway log, so the sample below shows only the demos.
TEST_LOG_DIR="$(mktemp -d)"
SHOWN="$PYTHON -m pytest -v" \
run tests        env PAWPAL_LOG_DIR="$TEST_LOG_DIR" "$PYTHON" -m pytest -v -p no:cacheprovider
rm -rf "$TEST_LOG_DIR"
run demo_offline "$PYTHON" demo_offline.py
run model_eval   "$PYTHON" -m pawpal_model
run scheduler    "$PYTHON" main.py

cp "$LOG_DIR/pawpal.log" "$OUT/pawpal_log_sample.txt"
rm -rf "$LOG_DIR"

# Show paths relative to the project, not this machine's home folder.
ROOT="$(pwd)"
for file in "$OUT"/*.txt; do
  sed "s|$ROOT|.|g" "$file" > "$file.tmp" && mv "$file.tmp" "$file"
done
echo "Wrote $(ls "$OUT" | wc -l | tr -d ' ') files to $OUT/"
