#!/usr/bin/env python3
"""Ensure every paper lives in its own subdir with EN + zh translations."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path("/Users/rookie/Desktop/RoBot/paper")


def git_mv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(dst)
    import subprocess

    r = subprocess.run(["git", "mv", str(src), str(dst)], cwd=ROOT.parent, capture_output=True, text=True)
    if r.returncode != 0:
        shutil.move(str(src), str(dst))


def folder_name_for_stem(stem: str) -> str:
    """Short slug for subdir; keep recognizable names."""
    special = {
        "ALOHA - Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware": "ALOHA",
        "ALOHA 2- An Enhanced Low-Cost Hardware for Bimanual Teleoperation": "ALOHA-2",
        "Mobile ALOHA- Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation": "Mobile-ALOHA",
        "GELLO- A General, Low-Cost, and Intuitive Teleoperation Framework for Robot Manipulators": "GELLO",
        "AirExo - Low-Cost Exoskeletons for Learning Whole-Arm Manipulation in the Wild": "AirExo",
        "Giving Robots a Hand- Learning Generalizable Manipulation with Eye-in-Hand Human Video Demonstrations": "Giving-Robots-a-Hand",
        "Dobb-E - On Bringing Robots Home": "Dobb-E",
        "UMI - Universal Manipulation Interface- In-The-Wild Robot Teaching Without In-The-Wild Robots": "UMI",
        "Grasping in the Wild- Learning 6DoF Closed-Loop Grasping from Low-Cost Demonstrations": "Grasping-in-the-Wild",
        "CDF-Glove_ A Cable-Driven Force Feedback Glove for Dexterous Teleoperation": "CDF-Glove",
        "TWIST2- Scalable, Portable, and Holistic Humanoid Data Collection System": "TWIST2",
        "DexCap- Scalable and Portable Mocap Data Collection System for Dexterous Manipulation": "DexCap",
        "H-InDex- Visual Reinforcement Learning with Hand-Informed Representations for Dexterous Manipulation": "H-InDex",
        "Immersive Demonstrations are the Key to Imitation Learning": "Immersive-Demonstrations",
        "Affordances from human videos as a versatile representation for robotics": "Affordances-from-Human-Videos",
        "AnyTeleop- A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System": "AnyTeleop",
        "DexPilot- Vision Based Teleoperation of Dexterous Robotic Hand-Arm System": "DexPilot",
        "Robotic Telekinesis- Learning a Robotic Hand Imitator by Watching Humans on YouTube": "Robotic-Telekinesis",
        "VideoDex- Learning Dexterity from Internet Videos": "VideoDex",
        "Single RGB-D Camera Teleoperation for General Robotic Manipulation": "Single-RGB-D-Camera-Teleoperation",
        "From One Hand to Multiple Hands Imitation Learning for Dexterous": "From-One-Hand-to-Multiple-Hands",
        "FrankMocap- Fast Monocular 3D Hand and Body Motion Capture by Regression and Integration": "FrankMocap",
        "MediaPipe Hands- On-device Real-time Hand Tracking": "MediaPipe-Hands",
        "Humanoid Policy ∼ Human Policy": "Humanoid-Policy",
        "Emergence of Human to Robot Transfer in Vision-Language-Action Models": "Human-to-Robot-Transfer",
        "MimicGen- A Data Generation System for Scalable Robot Learning using Human Demonstrations": "MimicGen",
        "BridgeData V2- A Dataset for Robot Learning at Scale": "BridgeData-V2",
        "DROID- A Large-Scale In-The-Wild Robot Manipulation Dataset": "DROID",
    }
    if stem in special:
        return special[stem]
    if stem.endswith(".pdf"):
        stem = stem[:-4]
    # default: first token or filename without spaces issues
    if ".pdf" in stem:
        stem = stem.replace(".pdf", "")
    # short names for simple titles
    if stem in {"TidyBot++", "FastUMI", "DexUMI", "DexWild", "Open-TeleVision", "OPEN TEACH", "MotionTrans",
                "EgoMimic", "EgoVLA", "Phantom", "EgoBridge", "EgoScale", "EgoZero", "EgoHumanoid", "EgoMI", "EMMA",
                "Visual Imitation Made Easy", "RDT2", "RDT2-en", "ABot-M0", "Qwen-RobotManip"}:
        return stem.replace(" ", "-")
    # use stem prefix before first " - " or " · "
    for sep in (" - ", " · "):
        if sep in stem:
            return stem.split(sep)[0].strip().replace(" ", "-")
    return stem.replace(" ", "-")[:48]


def group_flat_pdfs(flat_dir: Path) -> int:
    if not flat_dir.exists() or not flat_dir.is_dir():
        return 0
    # skip if already only subdirs (no loose pdfs)
    loose = [f for f in flat_dir.glob("*.pdf") if f.is_file()]
    if not loose:
        return 0
    stems: dict[str, list[Path]] = {}
    for f in flat_dir.glob("*.pdf"):
        if not f.is_file():
            continue
        name = f.name
        if name.endswith(".zh.mono.pdf"):
            stem = name[: -len(".zh.mono.pdf")]
        elif name.endswith(".zh.dual.pdf"):
            stem = name[: -len(".zh.dual.pdf")]
        elif name.endswith(".pdf"):
            stem = name[: -len(".pdf")]
        else:
            continue
        stems.setdefault(stem, []).append(f)

    moved = 0
    for stem, files in stems.items():
        sub_name = folder_name_for_stem(stem)
        sub = flat_dir / sub_name
        sub.mkdir(parents=True, exist_ok=True)
        for f in files:
            target = sub / f.name
            if target.exists():
                continue
            print(f"  {f.relative_to(ROOT)} -> {target.relative_to(ROOT)}")
            git_mv(f, target)
            moved += 1
    return moved


def main() -> None:
    total = 0
    # All category dirs that may contain loose PDFs
    for cat in sorted(ROOT.iterdir()):
        if not cat.is_dir() or cat.name.startswith("_"):
            continue
        if cat.name == "Surveys":
            continue  # already done
        # recurse one level: e.g. Data-Collection/ALOHA, VLA/VLM
        for sub in sorted(cat.iterdir()):
            if sub.is_dir():
                n = group_flat_pdfs(sub)
                # VLA/VLM needs inner grouping already done; VLA/RT-1 already per-paper
                total += n
        n = group_flat_pdfs(cat)
        total += n
    print(f"Grouped {total} files.")


if __name__ == "__main__":
    main()
