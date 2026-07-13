#!/usr/bin/env python3
"""Generate tiny local samples for learning robot dataset formats (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = REPO_ROOT / "samples"


def create_aloha_style_hdf5(out_dir: Path, num_steps: int = 30) -> Path:
    """ALOHA / ACT style: one episode per .hdf5 file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "episode_0.hdf5"
    cam_names = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    h, w = 64, 64

    with h5py.File(path, "w") as root:
        root.attrs["sim"] = True
        obs = root.create_group("observations")
        images = obs.create_group("images")
        for cam in cam_names:
            images.create_dataset(
                cam,
                data=np.random.randint(0, 255, (num_steps, h, w, 3), dtype=np.uint8),
            )
        obs.create_dataset("qpos", data=np.random.randn(num_steps, 14).astype(np.float32))
        obs.create_dataset("qvel", data=np.random.randn(num_steps, 14).astype(np.float32))
        root.create_dataset("action", data=np.random.randn(num_steps, 14).astype(np.float32))

    return path


def create_lerobot_style_sample(out_dir: Path, num_steps: int = 30) -> Path:
    """LeRobot v2 style: meta/info.json + parquet per episode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(exist_ok=True)

    info = {
        "codebase_version": "v2.0",
        "robot_type": "aloha",
        "total_episodes": 1,
        "total_frames": num_steps,
        "fps": 50,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [14]},
            "observation.state": {"dtype": "float32", "shape": [14]},
            "observation.images.top": {"dtype": "video", "shape": [64, 64, 3]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    rows = []
    for t in range(num_steps):
        rows.append(
            {
                "episode_index": 0,
                "frame_index": t,
                "timestamp": t / 50.0,
                "task_index": 0,
                "action": np.random.randn(14).astype(np.float32).tolist(),
                "observation.state": np.random.randn(14).astype(np.float32).tolist(),
            }
        )
    data_dir = out_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / "episode_000000.parquet"
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    return parquet_path


def main() -> None:
    hdf5_path = create_aloha_style_hdf5(SAMPLES / "aloha_hdf5")
    lerobot_path = create_lerobot_style_sample(SAMPLES / "lerobot_minimal")
    print(f"Created HDF5 sample: {hdf5_path}")
    print(f"Created LeRobot-style sample: {lerobot_path}")
    print("\nNext:")
    print("  python scripts/inspect_hdf5.py samples/aloha_hdf5/episode_0.hdf5")
    print("  python scripts/inspect_lerobot.py samples/lerobot_minimal")


if __name__ == "__main__":
    main()
