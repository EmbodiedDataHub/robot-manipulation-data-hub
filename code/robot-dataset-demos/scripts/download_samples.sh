#!/usr/bin/env bash
# Download small public samples for format learning.
# Usage: ./scripts/download_samples.sh [--lerobot-only|--all]
# Requires: pip install -r requirements.txt
# Optional: export HF_ENDPOINT=https://hf-mirror.com  (if huggingface.co times out)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAMPLES="$ROOT/samples"
MODE="${1:---lerobot-only}"

cd "$ROOT"
mkdir -p "$SAMPLES"

download_lerobot() {
  echo "==> LeRobot: aloha_sim (1 episode)"
  python3 - "$ROOT" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(sys.argv[1])
dest = root / "samples" / "lerobot_aloha_sim_hf"
dest.mkdir(parents=True, exist_ok=True)
path = snapshot_download(
    repo_id="lerobot/aloha_sim_transfer_cube_human",
    repo_type="dataset",
    local_dir=str(dest),
    # LeRobot v3: chunked parquet + mp4 (whole repo is small ~50MB)
)
print("OK:", path)
PY
}

download_libero() {
  echo "==> LIBERO: libero_spatial (HDF5, several GB)"
  if [ ! -d "$SAMPLES/LIBERO" ]; then
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "$SAMPLES/LIBERO"
  fi
  (cd "$SAMPLES/LIBERO" && python benchmark_scripts/download_libero_datasets.py \
    --datasets libero_spatial --use-huggingface)
}

download_rlds() {
  echo "==> RLDS: droid_100 (~2GB, needs gsutil)"
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m cp -r gs://gresearch/robotics/droid_100 "$SAMPLES/tensorflow_datasets/"
    echo "Inspect: python scripts/inspect_rlds.py --data-dir $SAMPLES/tensorflow_datasets/droid_100/1.0.0"
  else
    echo "gsutil not found: https://cloud.google.com/storage/docs/gsutil_install"
    echo "Or OXE colab: https://colab.research.google.com/github/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb"
  fi
}

case "$MODE" in
  --lerobot-only)
    download_lerobot
    ;;
  --all)
    download_lerobot
    download_libero
    download_rlds
    ;;
  *)
    echo "Usage: $0 [--lerobot-only|--all]"
    exit 1
    ;;
esac

echo ""
echo "Done. Offline minimal samples: python scripts/create_minimal_samples.py"
