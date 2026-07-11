#!/usr/bin/env python3
"""Fix READING_SECTIONS paths in generate_paper_index.py by PDF basename lookup."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path("/Users/rookie/Desktop/RoBot/paper")
SCRIPT = Path("/Users/rookie/Desktop/RoBot/scripts/generate_paper_index.py")

# Build basename -> rel path map
pdf_map: dict[str, str] = {}
for p in ROOT.rglob("*.pdf"):
    if ".zh." in p.name:
        continue
    pdf_map[p.name] = p.relative_to(ROOT).as_posix()


def resolve(old_rel: str) -> str:
    name = Path(old_rel).name
    if (ROOT / old_rel).exists():
        return old_rel
    if name in pdf_map:
        return pdf_map[name]
    raise FileNotFoundError(f"Cannot resolve: {old_rel}")


def main() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    pattern = re.compile(
        r'"((?:Surveys|VLA|IL-Action-Head|Datasets|Foundation-Models|Data-Collection)/[^"]+\.pdf)"'
    )

    def repl(m: re.Match) -> str:
        old = m.group(1)
        if old == "*.pdf" or "*" in old:
            return m.group(0)
        new = resolve(old)
        if new != old:
            print(f"  {old}\n    -> {new}")
        return f'"{new}"'

    new_text = pattern.sub(repl, text)
    SCRIPT.write_text(new_text, encoding="utf-8")
    print("Updated generate_paper_index.py")


if __name__ == "__main__":
    main()
