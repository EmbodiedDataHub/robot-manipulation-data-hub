#!/usr/bin/env python3
"""Reorganize paper/ into a clearer taxonomy. Run from repo root."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path("/Users/rookie/Desktop/RoBot/paper")

# (old_relative, new_relative) — directories moved with all contents
DIR_MOVES: list[tuple[str, str]] = [
    # IL + action heads (from Algorithm)
    ("IL-Action-Head/DAgger", "IL-Action-Head/DAgger"),
    ("IL-Action-Head/Diffusion-Policy", "IL-Action-Head/Diffusion-Policy"),
    ("Algorithm/3D-Diffusion-Policy", "IL-Action-Head/3D-Diffusion-Policy"),
    ("IL-Action-Head/Flow-Matching", "IL-Action-Head/Flow-Matching"),
    ("IL-Action-Head/IL-Analysis", "IL-Action-Head/IL-Analysis"),
    # VLA policies (from Algorithm)
    ("VLA/RT-1", "VLA/RT-1"),
    ("VLA/RT-2", "VLA/RT-2"),
    ("VLA/OpenVLA", "VLA/OpenVLA"),
    ("VLA/Octo", "VLA/Octo"),
    ("VLA/Pi0", "VLA/Pi0"),
    ("VLA/PaLM-E", "VLA/PaLM-E"),
    ("VLA/VIMA", "VLA/VIMA"),
    ("VLA/RoboCat", "VLA/RoboCat"),
    ("VLA/BC-Z", "VLA/BC-Z"),
    # Datasets (from Algorithm + Data Acquisition)
    ("Datasets/BridgeData-V2", "Datasets/BridgeData-V2"),
    ("Datasets/DROID", "Datasets/DROID"),
    ("Datasets/MimicGen", "Datasets/MimicGen"),
    # Foundation models (rename, keep flat per-paper subdirs)
    ("Foundation Models", "Foundation-Models/_flat"),
    # Data collection (rename + simplify)
    ("Data-Collection/ALOHA", "Data-Collection/ALOHA"),
    ("Data-Collection/VR-Humanoid", "Data-Collection/VR-Humanoid"),
    ("Data-Collection/Ego", "Data-Collection/Ego"),
    ("Data-Collection/Video", "Data-Collection/Video"),
    ("Data-Collection/Dexterous", "Data-Collection/Dexterous/_all"),
]

# Survey paper moved out of dexterous folder
SURVEY_MOVE = (
    "Data-Collection/Dexterous/_all/Survey of Learning Approaches for Robotic In-Hand Manipulation.pdf",
    "Surveys/In-Hand-Manipulation-Survey.pdf",
)
SURVEY_MOVE_ZH = [
    (
        "Data-Collection/Dexterous/_all/Survey of Learning Approaches for Robotic In-Hand Manipulation.zh.mono.pdf",
        "Surveys/In-Hand-Manipulation-Survey.zh.mono.pdf",
    ),
    (
        "Data-Collection/Dexterous/_all/Survey of Learning Approaches for Robotic In-Hand Manipulation.zh.dual.pdf",
        "Surveys/In-Hand-Manipulation-Survey.zh.dual.pdf",
    ),
]


def git_mv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        raise FileExistsError(dst)
    import subprocess

    r = subprocess.run(["git", "mv", str(src), str(dst)], cwd=ROOT.parent, capture_output=True, text=True)
    if r.returncode != 0:
        shutil.move(str(src), str(dst))


def move_dir(old: str, new: str) -> None:
    src, dst = ROOT / old, ROOT / new
    print(f"  {old} -> {new}")
    git_mv(src, dst)


def flatten_handheld_umi() -> None:
    """Merge Hand-Held Grippers Teleoperation + UMIs/ into Data-Collection/UMI/."""
    src_base = ROOT / "Data-Collection/UMI"
    dst = ROOT / "Data-Collection/UMI"
    if not src_base.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    print("  Hand-Held Grippers Teleoperation -> Data-Collection/UMI (flatten)")
    for item in sorted(src_base.rglob("*")):
        if not item.is_file():
            continue
        target = dst / item.name
        if target.exists():
            continue
        git_mv(item, target)
    # remove empty dirs
    for d in sorted(src_base.rglob("*"), reverse=True):
        if d.is_dir():
            d.rmdir()
    src_base.rmdir()


def flatten_foundation_models() -> None:
    """Foundation-Models/_flat/*.pdf -> Foundation-Models/<stem>/ per paper group."""
    flat = ROOT / "Foundation-Models/_flat"
    if not flat.exists():
        return
    print("  Foundation-Models: group by paper stem")
    groups: dict[str, list[Path]] = {}
    for f in flat.iterdir():
        if not f.is_file():
            continue
        stem = f.name
        if stem.endswith(".zh.mono.pdf"):
            base = stem[: -len(".zh.mono.pdf")]
        elif stem.endswith(".zh.dual.pdf"):
            base = stem[: -len(".zh.dual.pdf")]
        elif stem.endswith(".pdf"):
            base = stem[: -len(".pdf")]
        else:
            continue
        groups.setdefault(base, []).append(f)

    for base, files in groups.items():
        # folder name from base (short)
        folder = base.split("-")[0] if base.startswith("RDT") else base.replace(" ", "-")[:32]
        if base.startswith("RDT-1B"):
            folder = "RDT-1B"
        elif base == "RDT2":
            folder = "RDT2"
        elif base == "RDT2-en":
            folder = "RDT2-en"
        elif base == "Qwen-RobotManip":
            folder = "Qwen-RobotManip"
        elif base == "ABot-M0":
            folder = "ABot-M0"
        sub = ROOT / "Foundation-Models" / folder
        sub.mkdir(parents=True, exist_ok=True)
        for f in files:
            git_mv(f, sub / f.name)
    flat.rmdir()


def flatten_dexterous() -> None:
    """Data-Collection/Dexterous/_all -> Data-Collection/Dexterous."""
    src = ROOT / "Data-Collection/Dexterous/_all"
    dst = ROOT / "Data-Collection/Dexterous"
    if not src.exists():
        return
    print("  Dexterous/_all -> Dexterous")
    for f in list(src.iterdir()):
        if f.is_file():
            git_mv(f, dst / f.name)
    src.rmdir()


def cleanup_empty() -> None:
    for path in [ROOT / "Algorithm", ROOT / "Data Acquisition", ROOT / "Foundation Models"]:
        if path.exists():
            try:
                path.rmdir()
            except OSError:
                remaining = list(path.rglob("*"))
                print(f"  WARN: not empty {path}: {remaining[:5]}")


def move_survey() -> None:
    src = ROOT / SURVEY_MOVE[0]
    dst = ROOT / SURVEY_MOVE[1]
    if src.exists():
        print(f"  survey -> Surveys/In-Hand-Manipulation-Survey.pdf")
        git_mv(src, dst)
    for old, new in SURVEY_MOVE_ZH:
        s, d = ROOT / old, ROOT / new
        if s.exists():
            git_mv(s, d)


def main() -> None:
    print("Reorganizing paper/ ...")
    for old, new in DIR_MOVES:
        if (ROOT / old).exists():
            move_dir(old, new)
    flatten_handheld_umi()
    move_survey()
    flatten_foundation_models()
    flatten_dexterous()
    cleanup_empty()
    print("Done.")


if __name__ == "__main__":
    main()
