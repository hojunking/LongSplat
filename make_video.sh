#!/bin/bash

PNG_DIR="$1"
OUT="$2"
FPS="${3:-30}"

if ! command -v ffmpeg &> /dev/null; then
  echo "[ERROR] ffmpeg not found"
  exit 1
fi

cd "$PNG_DIR" || exit 1

ffmpeg -y \
  -framerate "$FPS" \
  -pattern_type glob \
  -i "*.png" \
  -vsync vfr \
  -c:v libx264 \
  -crf 18 \
  -pix_fmt yuv420p \
  "$OUT"

echo "Saved video to: $OUT"
