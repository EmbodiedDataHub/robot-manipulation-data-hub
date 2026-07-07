#!/usr/bin/env python3
"""Batch translate missing PDFs under paper/ using pdf2zh_next + DeepSeek."""
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path("/Users/rookie/Desktop/RoBot/paper")
CONFIG = Path.home() / ".config/PDFMathTranslate/config.json"
LOG = ROOT / "_translation_log.txt"
INDEX = ROOT / "论文索引.md"


def get_api_key() -> str:
    cfg = json.loads(CONFIG.read_text())
    for t in cfg.get("translators", []):
        if t.get("name") == "deepseek":
            return t["envs"]["DEEPSEEK_API_KEY"]
    raise RuntimeError("DeepSeek API key not found in config")


def needs_translation(pdf: Path) -> bool:
    stem = pdf.stem
    if stem.endswith(("-mono", "-dual")) or ".zh." in pdf.name:
        return False
    return not (pdf.parent / f"{stem}.zh.mono.pdf").exists()


def find_pdfs() -> list[Path]:
    pdfs = [p for p in ROOT.rglob("*.pdf") if needs_translation(p)]
    # Smaller papers first, then surveys
    return sorted(pdfs, key=lambda p: (p.stat().st_size, str(p)))


def translate(pdf: Path, key: str) -> bool:
    cmd = [
        "pdf2zh_next",
        "--deepseek",
        f"--deepseek-api-key={key}",
        "--lang-in",
        "en",
        "--lang-out",
        "zh",
        pdf.name,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=pdf.parent)
    ok = r.returncode == 0 and (pdf.parent / f"{pdf.stem}.zh.mono.pdf").exists()
    with LOG.open("a", encoding="utf-8") as f:
        status = "OK" if ok else "FAIL"
        f.write(f"[{status}] {pdf.relative_to(ROOT)}\n")
        if not ok:
            f.write((r.stderr or r.stdout)[-3000:] + "\n")
    return ok


def md_link(path: Path, label: str) -> str:
    rel = path.relative_to(ROOT).as_posix()
    return f"[{label}]({quote(rel, safe='/')})"


def has_translation(stem: str, directory: Path):
    mono = directory / f"{stem}.zh.mono.pdf"
    dual = directory / f"{stem}.zh.dual.pdf"
    return mono.exists(), dual.exists()


def regenerate_index() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import generate_paper_index

    generate_paper_index.generate()


def main() -> None:
    key = get_api_key()
    pdfs = find_pdfs()
    if not pdfs:
        print("Nothing to translate.", flush=True)
        regenerate_index()
        return

    LOG.write_text(f"Starting batch: {len(pdfs)} files\n", encoding="utf-8")
    ok = fail = 0
    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf.name}", flush=True)
        if translate(pdf, key):
            ok += 1
        else:
            fail += 1

    regenerate_index()
    summary = f"\nDone: success={ok}, failed={fail}, total={len(pdfs)}\n"
    print(summary, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(summary)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
