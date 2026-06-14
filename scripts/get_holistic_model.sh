#!/usr/bin/env bash
# Download the MediaPipe Holistic Landmarker model bundle into models/.
set -euo pipefail
cd "$(dirname "$0")/.."
URL="https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
mkdir -p models
echo "Downloading holistic_landmarker.task ..."
curl -sSL -o models/holistic_landmarker.task "$URL"
echo "Done: $(ls -lh models/holistic_landmarker.task | awk '{print $5}')"
