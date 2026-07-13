#!/usr/bin/env python3
"""
Inspect RLDS / TFRecord robot data (OXE, DROID, Bridge).

Requires: pip install tensorflow tensorflow-datasets

Example (droid_100 sample):
  python scripts/inspect_rlds.py \\
    --data-dir samples/tensorflow_datasets/droid_100/1.0.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect(data_dir: Path, max_episodes: int = 1, max_steps: int = 3) -> None:
    import tensorflow_datasets as tfds

    data_dir = data_dir.resolve()
    if not (data_dir / "dataset_info.json").exists():
        raise FileNotFoundError(
            f"Missing dataset_info.json in {data_dir}. "
            "Point --data-dir to the version folder, e.g. .../droid_100/1.0.0"
        )

    info = json.loads((data_dir / "dataset_info.json").read_text(encoding="utf-8"))
    print(f"=== RLDS dataset: {info.get('name', data_dir.name)} ===")
    print(f"Path: {data_dir}")
    print(f"Version: {info.get('version')}")
    print(f"Format: {info.get('fileFormat')}")
    if features_path := data_dir / "features.json":
        if features_path.exists():
            features = json.loads(features_path.read_text(encoding="utf-8"))
            top_keys = list(features.get("featuresDict", features).keys())[:8]
            print(f"Feature keys (top-level): {top_keys}")

    builder = tfds.builder_from_directory(str(data_dir))
    ds = builder.as_dataset(split="train")

    for ep_idx, episode in enumerate(ds.take(max_episodes)):
        print(f"\n--- Episode {ep_idx} ---")
        if "episode_metadata" in episode:
            meta = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in episode["episode_metadata"].items()}
            print(f"episode_metadata: {meta}")

        steps = episode["steps"]
        for step_idx, step in enumerate(steps.take(max_steps)):
            print(f"\nStep {step_idx}:")
            action = step["action"].numpy()
            print(f"  action: shape={action.shape}, first4={action[:4]}")
            if "language_instruction" in step:
                lang = step["language_instruction"].numpy()
                if hasattr(lang, "decode"):
                    lang = lang.decode()
                print(f"  language: {lang}")
            obs = step["observation"]
            for k in sorted(obs.keys()):
                v = obs[k]
                if hasattr(v, "shape"):
                    print(f"  observation.{k}: shape={tuple(v.shape)}, dtype={v.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[1] / "samples/tensorflow_datasets/droid_100/1.0.0"
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default,
        help="Path to TFDS version dir containing dataset_info.json",
    )
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3)
    args = parser.parse_args()
    inspect(args.data_dir, args.max_episodes, args.max_steps)


if __name__ == "__main__":
    main()
