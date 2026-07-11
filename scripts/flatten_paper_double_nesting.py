#!/usr/bin/env python3
"""Flatten accidental double-nesting in single-paper directories."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path("/Users/rookie/Desktop/RoBot/paper")

# Dirs that hold MULTIPLE papers as subdirs — do not flatten their children
MULTI_PAPER = {
    "Surveys",
    "VLM",
    "ALOHA",
    "UMI",
    "Ego",
    "Video",
    "Dexterous",
    "VR-Humanoid",
}


def git_mv(src: Path, dst: Path) -> None:
    import subprocess

    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["git", "mv", str(src), str(dst)], cwd=ROOT.parent, capture_output=True, text=True)
    if r.returncode != 0:
        shutil.move(str(src), str(dst))


def flatten_paper_dir(paper_dir: Path) -> None:
    """If paper_dir has exactly one subdir with all PDFs, hoist files up."""
    if paper_dir.name in MULTI_PAPER:
        return
    subdirs = [d for d in paper_dir.iterdir() if d.is_dir()]
    loose_pdfs = list(paper_dir.glob("*.pdf"))
    if loose_pdfs or len(subdirs) != 1:
        return
    inner = subdirs[0]
    inner_pdfs = list(inner.glob("*.pdf"))
    if not inner_pdfs or any(d.is_dir() for d in inner.iterdir()):
        return
    print(f"  flatten {paper_dir.relative_to(ROOT)}")
    for f in inner_pdfs:
        git_mv(f, paper_dir / f.name)
    inner.rmdir()


def main() -> None:
    for cat in ROOT.iterdir():
        if not cat.is_dir():
            continue
        if cat.name == "Data-Collection":
            continue
        for paper_dir in cat.iterdir():
            if paper_dir.is_dir():
                flatten_paper_dir(paper_dir)
        flatten_paper_dir(cat)
    # Foundation-Models/RDT2/RDT2 -> RDT2/
    fm = ROOT / "Foundation-Models"
    if fm.exists():
        for paper_dir in fm.iterdir():
            flatten_paper_dir(paper_dir)
    ds = ROOT / "Datasets"
    if ds.exists():
        for paper_dir in ds.iterdir():
            flatten_paper_dir(paper_dir)
    print("Done flattening.")


if __name__ == "__main__":
    main()
