#!/usr/bin/env python3
"""Generate ALOHA architecture SVG diagrams with valid UTF-8 encoding."""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "note" / "paper-note" / "ALOHA" / "assets"

STYLE = """
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0 0, 10 4, 0 8" fill="#475569"/>
    </marker>
    <filter id="shadow" x="-4%" y="-4%" width="108%" height="108%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.12"/>
    </filter>
  </defs>
"""

FONT = "PingFang SC, Microsoft YaHei, Helvetica Neue, Arial, sans-serif"


def wrap(title: str, width: int, height: int, body: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
{STYLE}
  <rect width="{width}" height="{height}" fill="#f8fafc"/>
  <style>
    text {{ font-family: {FONT}; }}
    .title {{ font-size: 22px; font-weight: 700; fill: #f8fafc; }}
    .h1 {{ font-size: 15px; font-weight: 700; }}
    .h2 {{ font-size: 14px; font-weight: 600; }}
    .body {{ font-size: 12px; fill: #334155; }}
    .muted {{ font-size: 11px; fill: #64748b; }}
  </style>
{body}
</svg>
"""


def box(x, y, w, h, fill, stroke, lines, rx=10, sw=2, cls="body", anchor="middle", cx=None):
    cx = cx if cx is not None else x + w / 2
    ys = [y + 28 + i * 20 for i in range(len(lines))]
    rects = f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" filter="url(#shadow)"/>\n'
    texts = ""
    for i, (line, weight) in enumerate(lines):
        fw = "700" if weight else "400"
        texts += f'  <text x="{cx}" y="{ys[i]}" text-anchor="{anchor}" class="{cls}" font-weight="{fw}">{line}</text>\n'
    return rects + texts


def arrow(x1, y1, x2, y2, dashed=False):
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    return f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#475569" stroke-width="2"{dash} marker-end="url(#arrow)"/>\n'


def comparison_svg() -> str:
    body = f"""
  <rect x="24" y="20" width="1032" height="52" rx="12" fill="#0f172a" filter="url(#shadow)"/>
  <text x="540" y="53" text-anchor="middle" class="title">CNNMLP vs ACT \u00b7 ALOHA \u57fa\u7ebf vs \u4e3b\u7b56\u7565</text>

  <rect x="40" y="92" width="480" height="580" rx="16" fill="#ffffff" stroke="#22c55e" stroke-width="3" filter="url(#shadow)"/>
  <rect x="40" y="92" width="480" height="48" rx="16" fill="#dcfce7"/>
  <rect x="40" y="124" width="480" height="16" fill="#dcfce7"/>
  <text x="280" y="124" text-anchor="middle" fill="#166534" font-size="20" font-weight="700">CNNMLP (baseline)</text>

  <text x="70" y="168" class="body" font-weight="700">\u6838\u5fc3\u601d\u8def Core idea</text>
  <text x="70" y="192" class="body">ResNet \u63d0\u7279\u5f81 + MLP \u56de\u5f52\u4e0b\u4e00\u6b65\u5173\u8282\u89d2</text>
  <text x="70" y="232" class="body" font-weight="700">\u89c6\u89c9 backbone</text>
  <text x="70" y="256" class="body">\u6bcf\u76f8\u673a\u72ec\u7acb ResNet18\uff08\u6743\u91cd\u4e0d\u5171\u4eab\uff09</text>
  <text x="70" y="296" class="body" font-weight="700">\u65f6\u5e8f\u5efa\u6a21</text>
  <text x="70" y="320" class="body">\u65e0 \u00b7 \u9a6c\u5c14\u53ef\u592b\uff1a\u53ea\u770b\u5f53\u524d\u5e27</text>
  <text x="70" y="360" class="body" font-weight="700">\u8f93\u51fa</text>
  <text x="70" y="384" class="body">\u5355\u6b65 a_hat \u2208 R^14</text>
  <text x="70" y="424" class="body" font-weight="700">\u8bad\u7ec3\u6807\u7b7e</text>
  <text x="70" y="448" class="body">actions[:, 0]\uff08demo \u7684\u4e0b\u4e00\u6b65\uff09</text>
  <text x="70" y="488" class="body" font-weight="700">\u635f\u5931</text>
  <text x="70" y="512" class="body">MSE(a_hat, a)</text>
  <text x="70" y="552" class="body" font-weight="700">\u63a8\u7406\u9891\u7387</text>
  <text x="70" y="576" class="body">\u6bcf\u4e2a control step \u90fd query</text>
  <text x="70" y="616" class="body" font-weight="700">\u968f\u673a\u6027</text>
  <text x="70" y="640" class="body">\u65e0 CVAE \u00b7 \u786c\u6620\u5c04</text>

  <rect x="560" y="92" width="480" height="580" rx="16" fill="#ffffff" stroke="#6366f1" stroke-width="3" filter="url(#shadow)"/>
  <rect x="560" y="92" width="480" height="48" rx="16" fill="#e0e7ff"/>
  <rect x="560" y="124" width="480" height="16" fill="#e0e7ff"/>
  <text x="800" y="124" text-anchor="middle" fill="#3730a3" font-size="20" font-weight="700">ACT (DETRVAE)</text>

  <text x="590" y="168" class="body" font-weight="700">\u6838\u5fc3\u601d\u8def Core idea</text>
  <text x="590" y="192" class="body">CVAE + Transformer \u9884\u6d4b action chunk</text>
  <text x="590" y="232" class="body" font-weight="700">\u89c6\u89c9 backbone</text>
  <text x="590" y="256" class="body">\u5171\u4eab ResNet18 + cross-attention \u878d\u5408</text>
  <text x="590" y="296" class="body" font-weight="700">\u65f6\u5e8f\u5efa\u6a21</text>
  <text x="590" y="320" class="body">100 \u6b65 chunk \u00b7 100 \u4e2a learned queries</text>
  <text x="590" y="360" class="body" font-weight="700">\u8f93\u51fa</text>
  <text x="590" y="384" class="body">100 \u6b65 a_hat \u2208 R^(100 x 14)</text>
  <text x="590" y="424" class="body" font-weight="700">\u8bad\u7ec3\u6807\u7b7e</text>
  <text x="590" y="448" class="body">\u5b8c\u6574 future 100 \u6b65 + padding mask</text>
  <text x="590" y="488" class="body" font-weight="700">\u635f\u5931</text>
  <text x="590" y="512" class="body">L1 + beta*KL(q(z|a) || N(0,I))</text>
  <text x="590" y="552" class="body" font-weight="700">\u63a8\u7406\u9891\u7387</text>
  <text x="590" y="576" class="body">\u6bcf 100 steps query \u4e00\u6b21\uff0c\u9010\u6b65\u6267\u884c chunk</text>
  <text x="590" y="616" class="body" font-weight="700">\u968f\u673a\u6027</text>
  <text x="590" y="640" class="body">CVAE \u9690\u53d8\u91cf z\uff1b\u63a8\u7406\u65f6 z=0</text>

  <circle cx="540" cy="380" r="36" fill="#0f172a"/>
  <text x="540" y="388" text-anchor="middle" fill="#f8fafc" font-size="18" font-weight="700">VS</text>

  <rect x="40" y="690" width="1000" height="20" rx="6" fill="#e2e8f0"/>
  <text x="540" y="704" text-anchor="middle" class="muted">Shared input: multi-cam RGB + qpos(14) | Task: behavior cloning | Entry: imitate_episodes.py -&gt; policy.py</text>
"""
    return wrap("comparison", 1080, 720, body)


def cnnmlp_svg() -> str:
    body = """
  <rect x="24" y="20" width="932" height="56" rx="12" fill="#0f172a" filter="url(#shadow)"/>
  <text x="490" y="55" text-anchor="middle" class="title">CNNMLPPolicy \u524d\u5411\u7ed3\u6784 Forward Pass</text>
"""
    body += box(40, 100, 420, 72, "#dbeafe", "#2563eb", [("image", True), ("(B, C, 3, 480, 640)  C=2 cameras", False)])
    body += box(520, 100, 420, 72, "#dbeafe", "#2563eb", [("qpos", True), ("(B, 14) \u53cc\u81c2\u5173\u8282 + \u5939\u722a", False)])
    body += box(170, 196, 160, 44, "#e2e8f0", "#64748b", [("ImageNet Normalize", True)], rx=8, sw=1.5)
    body += arrow(250, 172, 250, 196)
    body += '  <text x="490" y="268" text-anchor="middle" class="muted" font-weight="600">\u6bcf\u8def\u76f8\u673a\u72ec\u7acb ResNet18\uff08\u6743\u91cd\u4e0d\u5171\u4eab\uff09</text>\n'

    for ox, label in [(40, "cam_0 \u652f\u8def"), (520, "cam_1 \u652f\u8def")]:
        cx = ox + 210
        body += f'  <rect x="{ox}" y="286" width="420" height="300" rx="12" fill="#ffffff" stroke="#22c55e" stroke-width="2" filter="url(#shadow)"/>\n'
        body += f'  <rect x="{ox}" y="286" width="420" height="34" rx="12" fill="#dcfce7"/>\n'
        body += f'  <text x="{cx}" y="308" text-anchor="middle" fill="#166534" font-size="15" font-weight="700">{label}</text>\n'
        body += box(ox + 30, 336, 360, 48, "#f0fdf4", "#86efac", [("ResNet18 (ImageNet \u9884\u8bad\u7ec3, FrozenBN)", True), ("layer4 \u2192 (B, 512, 15, 20)", False)], cx=cx, rx=8, sw=1)
        body += box(ox + 30, 398, 360, 88, "#fef9c3", "#facc15", [("3\u00d7 Conv2d \u4e0b\u91c7\u6837", True), ("512\u2192128\u219264\u219232, kernel=5", False), ("\u8f93\u51fa (B, 32, 3, 8)", False), ("flatten \u2192 (B, 768)", False)], cx=cx, rx=8, sw=1)
        body += box(ox + 80, 504, 260, 56, "#dcfce7", "#22c55e", [(f"{label.split()[0]} feature", True), ("768-d", False)], cx=cx, rx=8)

    body += arrow(210, 240, 210, 286)
    body += arrow(290, 240, 290, 286)
    body += arrow(210, 240, 730, 286)
    body += box(300, 616, 380, 56, "#fef3c7", "#f59e0b", [("concat camera features", True), ("(B, 1536) = 768 x 2", False)])
    body += arrow(250, 560, 400, 616)
    body += arrow(730, 560, 580, 616)
    body += box(300, 696, 380, 56, "#fef3c7", "#f59e0b", [("concat qpos", True), ("(B, 1550) = 1536 + 14", False)])
    body += arrow(490, 672, 490, 696)
    body += arrow(730, 172, 730, 680, dashed=True)
    body += box(260, 776, 460, 88, "#fff7ed", "#fb923c", [("MLP (hidden_depth=2)", True), ("Linear 1550-1024-ReLU-1024-ReLU-1024-14", False), ("code: detr_vae.py CNNMLP.mlp", False)])
    body += arrow(490, 752, 490, 776)
    body += box(330, 884, 320, 24, "#fee2e2", "#ef4444", [("a_hat_t  (B, 14)  \u5355\u6b65\u5173\u8282\u76ee\u6807", True)], rx=8)

    body += box(40, 616, 220, 180, "#ffffff", "#6366f1", [
        ("\u8bad\u7ec3 (actions != None)", True),
        ("label = actions[:, 0]", False),
        ("\u53ea\u76d1\u7763\u300c\u4e0b\u4e00\u6b65\u300d", False),
        ("loss = MSE(a_hat, a)", False),
        ("policy.py L51-63", False),
        ("\u6bcf env step query", False),
    ], cx=150, anchor="middle")
    body += box(720, 616, 220, 180, "#ffffff", "#6366f1", [
        ("\u63a8\u7406 (actions = None)", True),
        ("forward \u4e0d\u7528 actions", False),
        ("\u6bcf\u63a7\u5236\u5468\u671f call \u4e00\u6b21", False),
        ("\u76f4\u63a5\u6267\u884c a_hat_t", False),
        ("imitate_episodes.py", False),
        ("\u65e0 chunk / \u65e0 CVAE", False),
    ], cx=830, anchor="middle")
    return wrap("cnnmlp", 980, 920, body)


def act_svg() -> str:
    body = """
  <rect x="24" y="20" width="1052" height="56" rx="12" fill="#4338ca" filter="url(#shadow)"/>
  <text x="550" y="55" text-anchor="middle" class="title">ACTPolicy / DETRVAE \u00b7 \u8bad\u7ec3 vs \u63a8\u7406</text>

  <rect x="40" y="96" width="1020" height="250" rx="14" fill="#faf5ff" stroke="#a855f7" stroke-width="2" stroke-dasharray="8 4" filter="url(#shadow)"/>
  <text x="68" y="124" fill="#6b21a8" font-size="15" font-weight="700">CVAE Encoder \u00b7 \u4ec5\u8bad\u7ec3\u65f6\u8fd0\u884c</text>
  <text x="920" y="124" text-anchor="end" fill="#9333ea" font-size="12" font-weight="600">\u63a8\u7406\u65f6\u8df3\u8fc7 \u2192 z = 0</text>
"""
    body += box(70, 142, 180, 56, "#dbeafe", "#2563eb", [("actions", True), ("(B, 100, 14)", False)], cx=160, rx=8)
    body += box(280, 142, 140, 56, "#dbeafe", "#2563eb", [("qpos", True), ("(B, 14)", False)], cx=350, rx=8)
    body += box(450, 142, 120, 56, "#ede9fe", "#8b5cf6", [("[CLS]", True)], cx=510, rx=8)
    body += box(600, 132, 420, 76, "#ffffff", "#c084fc", [
        ("concat -&gt; Transformer Encoder (4 layers)", True),
        ("+ sinusoid pos, mask padding", False),
        ("take CLS output -&gt; latent_proj", False),
    ], cx=810, rx=8, sw=1)
    body += arrow(250, 170, 280, 170)
    body += arrow(420, 170, 450, 170)
    body += arrow(570, 170, 600, 170)
    body += box(300, 230, 500, 96, "#f3e8ff", "#a855f7", [
        ("\u91cd\u53c2\u6570\u5316 reparametrize(\u03bc, log\u03c3\u00b2)", True),
        ("\u03bc, logvar: (B, 32) \u2192 latent_out_proj", False),
        ("\u63a8\u7406: z = zeros(32)", False),
        ("KL = \u03b2 \u00b7 KL(q(z|a) || N(0,I))", False),
    ], cx=550)
    body += box(860, 248, 160, 60, "#ddd6fe", "#7c3aed", [("latent z", True), ("32-d embedding", False)], cx=940)
    body += arrow(810, 208, 550, 230)
    body += arrow(800, 278, 860, 278)

    body += """
  <rect x="40" y="370" width="1020" height="420" rx="14" fill="#ffffff" stroke="#0ea5e9" stroke-width="2" filter="url(#shadow)"/>
  <text x="68" y="398" fill="#0369a1" font-size="15" font-weight="700">Transformer Decoder \u00b7 \u8bad\u7ec3 &amp; \u63a8\u7406\u5171\u7528</text>
"""
    body += box(70, 418, 300, 72, "#dbeafe", "#2563eb", [("image (B, C, 3, H, W)", True), ("ImageNet Normalize, C=2", False)], cx=220, rx=8)
    body += box(70, 508, 300, 64, "#dcfce7", "#22c55e", [("\u5171\u4eab ResNet18 backbone", True), ("backbones[0] \u5904\u7406\u6240\u6709\u76f8\u673a", False), ("\u2192 input_proj Conv1x1", False)], cx=220, rx=8, sw=1)
    body += box(70, 588, 300, 48, "#fef3c7", "#f59e0b", [("concat \u76f8\u673a\u7279\u5f81\uff08width \u7ef4\u62fc\u63a5\uff09", True)], cx=220, rx=8, sw=1)
    body += arrow(220, 490, 220, 508)
    body += arrow(220, 572, 220, 588)
    body += box(420, 418, 180, 56, "#dbeafe", "#2563eb", [("qpos (B, 14)", True), ("input_proj_robot_state", False)], cx=510, rx=8)
    body += box(420, 492, 180, 56, "#ddd6fe", "#7c3aed", [("latent z", True), ("+ additional_pos_embed", False)], cx=510, rx=8)
    body += '  <line x1="940" y1="308" x2="940" y2="340" stroke="#7c3aed" stroke-width="2"/>\n'
    body += '  <line x1="940" y1="340" x2="510" y2="340" stroke="#7c3aed" stroke-width="2"/>\n'
    body += arrow(510, 340, 510, 492)
    body += box(640, 418, 390, 218, "#eff6ff", "#0284c7", [
        ("Transformer Decoder", True),
        ("cross-attn: 100 queries x vision memory", False),
        ("+ proprio (qpos) + latent token", False),
        ("query_embed: Embedding(100, hidden_dim)", False),
        ("hidden_dim=512, nheads=8, 7 dec layers", False),
        ("detr_vae.py DETRVAE.transformer", False),
    ], cx=835)
    body += arrow(370, 612, 640, 560)
    body += arrow(510, 474, 640, 500)
    body += arrow(510, 548, 640, 540)
    body += box(660, 580, 350, 40, "#ffffff", "#94a3b8", [("100 action queries -&gt; 100 future steps (chunk)", False)], cx=835, rx=6, sw=1)
    body += box(640, 660, 390, 100, "#fff7ed", "#fb923c", [
        ("Output Heads", True),
        ("action_head: Linear(hidden, 14) -&gt; (B, 100, 14)", False),
        ("is_pad_head: Linear(hidden, 1)", False),
        ("train loss = L1 + beta*KL", False),
    ], cx=835)
    body += arrow(835, 636, 835, 660)
    body += box(70, 660, 520, 100, "#ecfdf5", "#10b981", [
        ("\u90e8\u7f72\u6267\u884c eval_bc", True),
        ("\u6bcf 100 env steps query \u4e00\u6b21 policy", False),
        ("\u9010\u6b65\u6267\u884c chunk[0], chunk[1], ...", False),
        ("\u53ef\u9009 temporal aggregation", False),
        ("CNNMLP: \u6bcf\u6b65 query\uff0c\u53ea\u8f93\u51fa 1 \u6b65", False),
    ], cx=330)

    body += """
  <rect x="40" y="810" width="1020" height="150" rx="12" fill="#ffffff" stroke="#cbd5e1" filter="url(#shadow)"/>
  <text x="68" y="838" fill="#0f172a" font-size="14" font-weight="700">\u56fe\u4f8b \u00b7 policy.py \u5206\u53c9</text>
  <rect x="70" y="854" width="14" height="14" fill="#dbeafe" stroke="#2563eb"/>
  <text x="92" y="866" class="body">\u8f93\u5165 tensor</text>
  <rect x="200" y="854" width="14" height="14" fill="#faf5ff" stroke="#a855f7" stroke-dasharray="4 2"/>
  <text x="222" y="866" class="body">\u4ec5\u8bad\u7ec3</text>
  <rect x="320" y="854" width="14" height="14" fill="#eff6ff" stroke="#0284c7"/>
  <text x="342" y="866" class="body">\u8bad\u7ec3+\u63a8\u7406</text>
  <rect x="480" y="854" width="14" height="14" fill="#fee2e2" stroke="#ef4444"/>
  <text x="502" y="866" class="body">\u8f93\u51fa / loss</text>
  <text x="70" y="896" class="body">\u8bad\u7ec3: policy(qpos, image, actions, is_pad) -&gt; {l1, kl, loss}</text>
  <text x="70" y="918" class="body">\u63a8\u7406: policy(qpos, image) -&gt; a_hat (B,100,14); Encoder \u5173\u95ed; z=0</text>
  <text x="70" y="940" class="muted">\u6e90\u7801: code/act/policy.py, code/act/detr/models/detr_vae.py</text>
"""
    return wrap("act", 1100, 980, body)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "cnnmlp-vs-act-comparison.svg": comparison_svg(),
        "cnnmlp-architecture.svg": cnnmlp_svg(),
        "act-detrvae-architecture.svg": act_svg(),
    }
    for name, content in files.items():
        path = OUT / name
        path.write_text(content, encoding="utf-8")
        # verify
        path.read_text(encoding="utf-8")
        print(f"Wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
