#!/usr/bin/env python3
"""Inspect ALOHA / robomimic / LIBERO style HDF5 robot episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def print_tree(name: str, obj: h5py.HLObject, indent: int = 0) -> None:
    prefix = "  " * indent
    if isinstance(obj, h5py.Dataset):
        print(f"{prefix}{name}: shape={obj.shape}, dtype={obj.dtype}")
    else:
        print(f"{prefix}{name}/")
        for key in obj.keys():
            print_tree(key, obj[key], indent + 1)


def inspect(path: Path, step: int = 0) -> None:
    with h5py.File(path, "r") as f:
        print(f"=== HDF5: {path} ===\n")
        print("Attributes:", dict(f.attrs))
        print("\nTree:")
        for key in f.keys():
            print_tree(key, f[key], indent=1)

        # ALOHA layout
        if "observations" in f and "action" in f:
            action = f["action"]
            qpos = f["observations/qpos"]
            print(f"\nStep {step} sample (ALOHA-style):")
            print(f"  action[{step}] = {np.array(action[step])[:4]}... (dim={action.shape[1]})")
            print(f"  qpos[{step}]   = {np.array(qpos[step])[:4]}...")
            if "images" in f["observations"]:
                for cam in f["observations/images"].keys():
                    img = f[f"observations/images/{cam}"][step]
                    print(f"  image {cam}: shape={img.shape}, mean={img.mean():.1f}")

        # robomimic layout: data/demo_0/...
        if "data" in f:
            demos = list(f["data"].keys())
            print(f"\nrobomimic-style demos: {demos[:5]}{'...' if len(demos) > 5 else ''}")
            if demos:
                demo = f[f"data/{demos[0]}"]
                if "actions" in demo:
                    actions = demo["actions"]
                    print(f"  {demos[0]}/actions: shape={actions.shape}")
                if "obs" in demo:
                    print(f"  {demos[0]}/obs keys: {list(demo['obs'].keys())}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to .hdf5 file")
    parser.add_argument("--step", type=int, default=0, help="Timestep to print")
    args = parser.parse_args()
    inspect(args.path, args.step)


if __name__ == "__main__":
    main()
