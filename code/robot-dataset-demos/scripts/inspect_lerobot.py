#!/usr/bin/env python3
"""Inspect LeRobot v2/v3 dataset directory (Parquet + MP4 + meta/info.json)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _find_data_parquet(root: Path, episode: int, info: dict) -> Path:
    # v2: data/chunk-XXX/episode_XXXXXX.parquet
    chunk = episode // info.get("chunks_size", 1000)
    v2_path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"
    if v2_path.exists():
        return v2_path

    # v3: data/chunk-XXX/file-YYY.parquet (multiple episodes per file)
    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"No data/ under {root}")

    files = sorted(data_dir.rglob("file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet under {data_dir}")

    for f in files:
        df = pd.read_parquet(f)
        if "episode_index" in df.columns and episode in df["episode_index"].values:
            return f
    return files[0]


def inspect(root: Path, episode: int = 0) -> None:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}. Is this a LeRobot dataset root?")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    print(f"=== LeRobot dataset: {root} ===\n")
    print("meta/info.json (summary):")
    for k in ["codebase_version", "robot_type", "total_episodes", "total_frames", "fps"]:
        if k in info:
            print(f"  {k}: {info[k]}")
    print(f"  features: {list(info.get('features', {}).keys())}")

    parquet = _find_data_parquet(root, episode, info)
    df = pd.read_parquet(parquet)
    if "episode_index" in df.columns:
        df = df[df["episode_index"] == episode]

    print(f"\nEpisode {episode}: {parquet.relative_to(root)}")
    print(f"  rows (frames): {len(df)}")
    print(f"  columns: {list(df.columns)}")
    if len(df) == 0:
        print("  (no rows for this episode_index)")
        return

    print("\nFirst row:")
    row = df.iloc[0].to_dict()
    for k, v in row.items():
        if isinstance(v, list):
            preview = v[:4] if len(v) > 4 else v
            print(f"  {k}: list(len={len(v)}) {preview}...")
        else:
            print(f"  {k}: {v}")

    videos = list((root / "videos").rglob("*.mp4")) if (root / "videos").exists() else []
    if videos:
        print(f"\nVideo chunks: {len(videos)} file(s), e.g. {videos[0].relative_to(root)}")
    else:
        print("\nNo MP4 (local minimal sample may omit video).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="LeRobot dataset root directory")
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    inspect(args.root, args.episode)


if __name__ == "__main__":
    main()
