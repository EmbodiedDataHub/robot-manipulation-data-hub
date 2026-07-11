#!/usr/bin/env python3
"""Generate paper/论文索引.md ordered by recommended reading path."""
from __future__ import annotations

from pathlib import Path

ROOT = Path("/Users/rookie/Desktop/RoBot/paper")
INDEX = ROOT / "论文索引.md"

# (section_id, title, subtitle, weeks, [(rel_path, short_name, priority), ...])
READING_SECTIONS: list[tuple[str, str, str, str, list[tuple[str, str, str]]]] = [
    (
        "0-overview",
        "0 · 入门 · 综述与数据全景",
        "建立全局地图：数据从哪来、怎么存、VLA 瓶颈在哪。建议最先读。",
        "1–2 周",
        [
            (
                "Surveys/VLA-Datasets-Benchmarks-Data-Engines-2604.23001/VLA-Datasets-Benchmarks-Data-Engines-2604.23001.pdf",
                "VLA 数据 / Benchmark / Data Engine 综述",
                "⭐",
            ),
            (
                "Surveys/Open-X-Embodiment-2310.08864/Open-X-Embodiment-2310.08864.pdf",
                "Open X-Embodiment · 跨 embodiment 数据",
                "⭐",
            ),
            ("Surveys/In-Hand-Manipulation-Survey/In-Hand-Manipulation-Survey.pdf", "In-Hand Manipulation 综述", "⭐"),
            (
                "Datasets/MimicGen/MimicGen- A Data Generation System for Scalable Robot Learning using Human Demonstrations.pdf",
                "MimicGen · 200 demo → 50K 合成",
                "⭐",
            ),
            (
                "Surveys/Robo-DM-Data-Management-2505.15558/Robo-DM-Data-Management-2505.15558.pdf",
                "Robo-DM · 大规模数据管理",
                "选读",
            ),
            (
                "Surveys/VLA-Anatomy-Survey-2512.11362/VLA-Anatomy-Survey-2512.11362.pdf",
                "VLA Anatomy 综述",
                "选读",
            ),
            (
                "Surveys/VLA-Systematic-Review-2507.10672/VLA-Systematic-Review-2507.10672.pdf",
                "VLA 系统综述",
                "选读",
            ),
            (
                "Surveys/Towards-Unified-Robot-Manipulation-Survey-2510.10903/Towards-Unified-Robot-Manipulation-Survey-2510.10903.pdf",
                "操作全领域综述",
                "选读",
            ),
            (
                "Surveys/Embodied-Learning-Object-Centric-Manipulation-2408.11537/Embodied-Learning-Object-Centric-Manipulation-2408.11537.pdf",
                "物体中心具身学习综述",
                "选读",
            ),
            (
                "Surveys/Large-VLA-Models-Survey-2508.13133/Large-VLA-Models-Survey-2508.13133.pdf",
                "大 VLM-VLA 综述",
                "选读",
            ),
        ],
    ),
    (
        "0-vlm",
        "0 · VLA / VLM 底座",
        "VLM 是 VLA 的 **VL 部分**（Stage 0），目录位于 `VLA/VLM/`。对比学习 → 指令微调 → Prismatic/PaliGemma。配合 [VLM 概述](../note/paper-note/VLM/概述.md) 阅读。",
        "1 周",
        [
            (
                "VLA/VLM/CLIP/CLIP- Learning Transferable Visual Models From Natural Language Supervision.pdf",
                "CLIP · 图文对比学习开山",
                "⭐",
            ),
            (
                "VLA/VLM/SigLIP/SigLIP- Sigmoid Loss for Language Image Pre-Training.pdf",
                "SigLIP · OpenVLA/π0 视觉 encoder",
                "⭐",
            ),
            ("VLA/VLM/LLaVA/LLaVA- Visual Instruction Tuning.pdf", "LLaVA · 视觉指令微调范式", "⭐"),
            (
                "VLA/VLM/Prismatic-VLM/Prismatic VLMs- Investigating the Design Space of Visually-Conditioned Language Models.pdf",
                "Prismatic VLMs · OpenVLA 底座",
                "⭐",
            ),
            (
                "VLA/VLM/PaliGemma/PaliGemma- A versatile 3B VLM for transfer.pdf",
                "PaliGemma · π0 底座",
                "⭐",
            ),
        ],
    ),
    (
        "0-il",
        "0 · IL 与 Action Head",
        "模仿学习基础 + 连续动作生成（Diffusion / Flow）。配合 [IL 范式概述](../note/paper-note/IL-Paradigms/概述.md) 阅读。",
        "1–2 周",
        [
            (
                "IL-Action-Head/IL-Analysis/What Matters in Learning from Offline Human Demonstrations for Robot Manipulation.pdf",
                "What Matters in IL · 模仿学习经验结论",
                "选读",
            ),
            (
                "IL-Action-Head/DAgger/DAgger - A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.pdf",
                "DAgger · BC 误差累积解法",
                "○",
            ),
            (
                "IL-Action-Head/Diffusion-Policy/Diffusion Policy- Visuomotor Policy Learning via Action Diffusion.pdf",
                "Diffusion Policy",
                "⭐",
            ),
            (
                "IL-Action-Head/3D-Diffusion-Policy/3D Diffusion Policy- Generalizable Visuomotor Policy Learning via Simple 3D Representations.pdf",
                "3D Diffusion Policy",
                "选读",
            ),
            (
                "IL-Action-Head/Flow-Matching/Flow Matching for Generative Modeling - A Tutorial Introduction.pdf",
                "Flow Matching 教程 · π0 理论基础",
                "选读",
            ),
        ],
    ),
    (
        "0-vla",
        "0 · VLA 策略",
        "RT-1 → RT-2 → OpenVLA/Octo → π0。配合 [算法层清单](../note/VLA算法层学习路线与论文清单.md) 阅读。",
        "2–3 周",
        [
            ("VLA/RT-1/RT-1- Robotics Transformer for Real-World Control at Scale.pdf", "RT-1 · Robotics Transformer", "⭐"),
            (
                "VLA/RT-2/RT-2- Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.pdf",
                "RT-2 · VLA 开山",
                "⭐",
            ),
            ("VLA/PaLM-E/PaLM-E- An Embodied Multimodal Language Model.pdf", "PaLM-E · 具身多模态 LLM", "选读"),
            ("VLA/VIMA/VIMA- General Robot Manipulation with Multimodal Prompts.pdf", "VIMA · 多模态 prompt 操作", "选读"),
            (
                "VLA/BC-Z/BC-Z- Zero-Shot Task Generalization with Robotic Imitation Learning.pdf",
                "BC-Z · 语言条件 BC",
                "选读",
            ),
            ("VLA/OpenVLA/OpenVLA- An Open-Source Vision-Language-Action Model.pdf", "OpenVLA · 开源 7B VLA", "⭐"),
            ("VLA/Octo/Octo- An Open-Source Generalist Robot Policy.pdf", "Octo · 开源通用策略", "⭐"),
            (
                "VLA/RoboCat/RoboCat- A Self-Improving Generalist Agent for Robotic Manipulation.pdf",
                "RoboCat · 自改进 data flywheel",
                "选读",
            ),
            ("VLA/Pi0/Pi0- A Vision-Language-Action Flow Model for General Robot Control.pdf", "π0 · Flow Matching VLA", "⭐"),
        ],
    ),
    (
        "0-datasets",
        "0 · 机器人数据集",
        "开源 manipulation 数据集论文；与 OXE 综述配合阅读。",
        "选读",
        [
            (
                "Datasets/BridgeData-V2/BridgeData V2- A Dataset for Robot Learning at Scale.pdf",
                "BridgeData V2 · 低成本桌面数据集",
                "选读",
            ),
            (
                "Datasets/DROID/DROID- A Large-Scale In-The-Wild Robot Manipulation Dataset.pdf",
                "DROID · 野外大规模数据集",
                "选读",
            ),
        ],
    ),
    (
        "1-aloha",
        "1 · 实验室 Teleop — ALOHA 生态",
        "低成本双臂遥操作 + ACT。需要真实机器人，fidelity 最高。",
        "2–3 周",
        [
            (
                "Data-Collection/ALOHA/ALOHA/ALOHA - Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.pdf",
                "ALOHA",
                "⭐",
            ),
            (
                "Data-Collection/ALOHA/ALOHA-2/ALOHA 2- An Enhanced Low-Cost Hardware for Bimanual Teleoperation.pdf",
                "ALOHA 2",
                "⭐",
            ),
            (
                "Data-Collection/ALOHA/Mobile-ALOHA/Mobile ALOHA- Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation.pdf",
                "Mobile ALOHA",
                "⭐",
            ),
            (
                "Data-Collection/ALOHA/GELLO/GELLO- A General, Low-Cost, and Intuitive Teleoperation Framework for Robot Manipulators.pdf",
                "GELLO",
                "⭐",
            ),
            ("Data-Collection/ALOHA/TidyBot++/TidyBot++.pdf", "TidyBot++ · 全向移动平台", "⭐"),
            (
                "Data-Collection/ALOHA/AirExo/AirExo - Low-Cost Exoskeletons for Learning Whole-Arm Manipulation in the Wild.pdf",
                "AirExo · 外骨骼野外采集",
                "选读",
            ),
            (
                "Data-Collection/ALOHA/Giving-Robots-a-Hand/Giving Robots a Hand- Learning Generalizable Manipulation with Eye-in-Hand Human Video Demonstrations.pdf",
                "Giving Robots a Hand · eye-in-hand 人类视频",
                "选读",
            ),
            ("Data-Collection/UMI/Dobb-E/Dobb-E - On Bringing Robots Home.pdf", "Dobb-E · 家用机器人 Stick", "选读"),
        ],
    ),
    (
        "2-umi",
        "2 · 手持野外采集 — UMI 生态",
        "无需机器人即可采数据；relative trajectory + Diffusion Policy。",
        "2–3 周",
        [
            (
                "Data-Collection/UMI/UMI/UMI - Universal Manipulation Interface- In-The-Wild Robot Teaching Without In-The-Wild Robots.pdf",
                "UMI",
                "⭐",
            ),
            ("Data-Collection/UMI/FastUMI/FastUMI.pdf", "FastUMI", "⭐"),
            ("Data-Collection/UMI/DexUMI/DexUMI.pdf", "DexUMI · 灵巧手版 UMI", "⭐"),
            ("Data-Collection/UMI/DexWild/DexWild.pdf", "DexWild · 纯人手野外", "⭐"),
            ("Data-Collection/UMI/Visual-Imitation-Made-Easy/Visual Imitation Made Easy.pdf", "Visual Imitation Made Easy · 早期 handheld", "选读"),
            (
                "Data-Collection/UMI/Grasping-in-the-Wild/Grasping in the Wild- Learning 6DoF Closed-Loop Grasping from Low-Cost Demonstrations.pdf",
                "Grasping in the Wild · 6DoF 闭环抓取",
                "选读",
            ),
            (
                "Data-Collection/UMI/CDF-Glove/CDF-Glove_ A Cable-Driven Force Feedback Glove for Dexterous Teleoperation.pdf",
                "CDF-Glove · 力反馈手套",
                "选读",
            ),
        ],
    ),
    (
        "3-vr-humanoid",
        "3 · VR / 人形遥操作",
        "主动立体视觉、whole-body 采集、跨地域远程 teleop。",
        "2 周",
        [
            ("Data-Collection/VR-Humanoid/Open-TeleVision/Open-TeleVision.pdf", "Open-TeleVision", "⭐"),
            ("Data-Collection/VR-Humanoid/OPEN-TEACH/OPEN TEACH.pdf", "OPEN TEACH", "⭐"),
            (
                "Data-Collection/VR-Humanoid/TWIST2/TWIST2- Scalable, Portable, and Holistic Humanoid Data Collection System.pdf",
                "TWIST2",
                "⭐",
            ),
            ("Data-Collection/Ego/MotionTrans/MotionTrans.pdf", "MotionTrans · VR 人体 → 机器人迁移", "选读"),
        ],
    ),
    (
        "4-foundation",
        "4 · Foundation Models",
        "规模化多源数据 + 统一 Action 表示 + VLA 预训练。",
        "3–4 周",
        [
            (
                "Foundation-Models/RDT-1B/RDT-1B- a Diffusion Foundation Model for Bimanual Manipulation.pdf",
                "RDT-1B",
                "⭐",
            ),
            ("Foundation-Models/RDT2/RDT2.pdf", "RDT2", "⭐"),
            ("Foundation-Models/Qwen-RobotManip/Qwen-RobotManip.pdf", "Qwen-RobotManip", "⭐"),
            ("Foundation-Models/ABot-M0/ABot-M0.pdf", "ABot-M0 · UniACT 数据管线", "⭐"),
            ("Foundation-Models/RDT2-en/RDT2-en.pdf", "RDT2-en · 英文技术报告", "选读"),
        ],
    ),
    (
        "5-ego",
        "5 · 人类视频 / Ego 数据",
        "human-to-robot transfer；人类视频 scale-up 与 alignment。",
        "2–3 周",
        [
            ("Data-Collection/Ego/EgoMimic/EgoMimic.pdf", "EgoMimic", "⭐"),
            ("Data-Collection/Ego/EgoVLA/EgoVLA.pdf", "EgoVLA", "⭐"),
            ("Data-Collection/Ego/Phantom/Phantom.pdf", "Phantom · 零 robot 数据", "⭐"),
            ("Data-Collection/Ego/EgoBridge/EgoBridge.pdf", "EgoBridge · OT 对齐", "⭐"),
            ("Data-Collection/Ego/EgoScale/EgoScale.pdf", "EgoScale · 20k 小时 scaling law", "⭐"),
            ("Data-Collection/Ego/EgoZero/EgoZero.pdf", "EgoZero", "⭐"),
            ("Data-Collection/Ego/EgoHumanoid/EgoHumanoid.pdf", "EgoHumanoid", "选读"),
            ("Data-Collection/Ego/EgoMI/EgoMI.pdf", "EgoMI · 主动视觉", "选读"),
            ("Data-Collection/Ego/EMMA/EMMA.pdf", "EMMA · 移动操作", "选读"),
            ("Data-Collection/Ego/Humanoid-Policy/Humanoid Policy ∼ Human Policy.pdf", "HAT · Humanoid Policy", "选读"),
            (
                "Data-Collection/Ego/Human-to-Robot-Transfer/Emergence of Human to Robot Transfer in Vision-Language-Action Models.pdf",
                "Human→Robot Transfer 何时涌现",
                "选读",
            ),
        ],
    ),
    (
        "6-dexterous",
        "6 · 灵巧手专项",
        "dexterous teleop、mocap、in-hand manipulation 扩展。",
        "选读",
        [
            (
                "Data-Collection/Dexterous/DexCap/DexCap- Scalable and Portable Mocap Data Collection System for Dexterous Manipulation.pdf",
                "DexCap",
                "选读",
            ),
            (
                "Data-Collection/Dexterous/H-InDex/H-InDex- Visual Reinforcement Learning with Hand-Informed Representations for Dexterous Manipulation.pdf",
                "H-InDex",
                "选读",
            ),
            (
                "Data-Collection/Dexterous/Immersive-Demonstrations/Immersive Demonstrations are the Key to Imitation Learning.pdf",
                "Immersive Demonstrations · 力反馈 VR",
                "选读",
            ),
        ],
    ),
    (
        "7-video-tools",
        "7 · 视觉遥操作 · Internet 视频 · 工具",
        "视频 teleop 谱系、YouTube 模仿、手部 tracking 工具论文。",
        "按需",
        [
            (
                "Data-Collection/Video/Affordances-from-Human-Videos/Affordances from human videos as a versatile representation for robotics.pdf",
                "Affordances from Human Videos",
                "选读",
            ),
            (
                "Data-Collection/Video/AnyTeleop/AnyTeleop- A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System.pdf",
                "AnyTeleop",
                "选读",
            ),
            (
                "Data-Collection/Video/DexPilot/DexPilot- Vision Based Teleoperation of Dexterous Robotic Hand-Arm System.pdf",
                "DexPilot",
                "选读",
            ),
            (
                "Data-Collection/Video/Robotic-Telekinesis/Robotic Telekinesis- Learning a Robotic Hand Imitator by Watching Humans on YouTube.pdf",
                "Robotic Telekinesis · YouTube 模仿",
                "选读",
            ),
            ("Data-Collection/Video/VideoDex/VideoDex- Learning Dexterity from Internet Videos.pdf", "VideoDex", "选读"),
            (
                "Data-Collection/Video/Single-RGB-D-Camera-Teleoperation/Single RGB-D Camera Teleoperation for General Robotic Manipulation.pdf",
                "Single RGB-D Camera Teleoperation",
                "选读",
            ),
            (
                "Data-Collection/Video/From-One-Hand-to-Multiple-Hands/From One Hand to Multiple Hands Imitation Learning for Dexterous.pdf",
                "From One Hand to Multiple Hands",
                "选读",
            ),
            (
                "Data-Collection/Video/FrankMocap/FrankMocap- Fast Monocular 3D Hand and Body Motion Capture by Regression and Integration.pdf",
                "FrankMocap",
                "工具",
            ),
            (
                "Data-Collection/Video/MediaPipe-Hands/MediaPipe Hands- On-device Real-time Hand Tracking.pdf",
                "MediaPipe Hands",
                "工具",
            ),
        ],
    ),
]


def md_link(rel: str, label: str) -> str:
    return f"[{label}](<{rel}>)"


def row_for_pdf(rel: str, short: str, priority: str, global_idx: int) -> tuple[str, bool, bool]:
    pdf = ROOT / rel
    if not pdf.exists():
        raise FileNotFoundError(f"Missing PDF: {rel}")
    stem = pdf.stem
    mono = pdf.parent / f"{stem}.zh.mono.pdf"
    dual = pdf.parent / f"{stem}.zh.dual.pdf"
    mono_ok, dual_ok = mono.exists(), dual.exists()
    en = md_link(rel, "EN")
    zh = md_link(str(mono.relative_to(ROOT)), "中文") if mono_ok else "—"
    dual_l = md_link(str(dual.relative_to(ROOT)), "对照") if dual_ok else "—"
    line = f"| {global_idx} | {priority} | {short} | {en} | {zh} | {dual_l} |"
    return line, mono_ok, dual_ok


def generate() -> None:
    lines: list[str] = [
        "# 论文索引",
        "",
        "> 按 **推荐阅读顺序** 编排 · 目录 `paper/` · 更新：2026-07-11  ",
        "> 点击 **EN** / **中文** / **对照** 打开 PDF（链接相对本文件所在目录 `paper/`）。若预览里点不开，请 **Cmd+Click** 或在文件树中按路径打开。",
        "> 目录结构说明见 [README.md](./README.md)。",
        "",
        "## 阅读路线",
        "",
        "| 阶段 | 主题 | 建议时长 |",
        "|------|------|----------|",
    ]

    for sec_id, title, _, weeks, _ in READING_SECTIONS:
        short_title = title.split(" · ", 1)[-1] if " · " in title else title
        lines.append(f"| [{title}](#{sec_id}) | {short_title} | {weeks} |")

    lines.extend(["", "---", ""])

    global_idx = 0
    total = translated_mono = translated_dual = 0
    ordered_paths: set[str] = set()

    for sec_id, title, subtitle, weeks, papers in READING_SECTIONS:
        lines.append(f'<a id="{sec_id}"></a>')
        lines.append("")
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"{subtitle} · **{weeks}**")
        lines.append("")
        lines.append("| # | 优先级 | 论文 | EN | 中文 | 对照 |")
        lines.append("|---:|:------:|------|:--:|:--:|:--:|")

        for rel, short, priority in papers:
            ordered_paths.add(rel)
            global_idx += 1
            row, mono_ok, dual_ok = row_for_pdf(rel, short, priority, global_idx)
            lines.append(row)
            total += 1
            translated_mono += int(mono_ok)
            translated_dual += int(dual_ok)
        lines.append("")

    all_pdfs = sorted(
        p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.pdf") if ".zh." not in p.name
    )
    missing = [p for p in all_pdfs if p not in ordered_paths]
    if missing:
        lines.append("## 附录 · 未归类")
        lines.append("")
        lines.append("| # | 优先级 | 论文 | EN | 中文 | 对照 |")
        lines.append("|---:|:------:|------|:--:|:--:|:--:|")
        for rel in missing:
            global_idx += 1
            short = Path(rel).stem
            if len(short) > 48:
                short = short[:45] + "..."
            row, mono_ok, dual_ok = row_for_pdf(rel, short, "—", global_idx)
            lines.append(row)
            total += 1
            translated_mono += int(mono_ok)
            translated_dual += int(dual_ok)
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## 统计",
            "",
            f"- 论文总数：**{total}**",
            f"- 已有中文 mono：**{translated_mono}**",
            f"- 已有中英 dual：**{translated_dual}**",
            f"- 未翻译：**{total - translated_mono}**",
            "",
            "## 按目标速查",
            "",
            "| 你的目标 | 跳转到 |",
            "|---------|--------|",
            "| 建立全局观 | [0 · 综述](#0-overview) |",
            "| 学 VLM 底座 | [0 · VLM 底座](#0-vlm) |",
            "| 学 IL / Diffusion Action Head | [0 · IL 与 Action Head](#0-il) |",
            "| 学 VLA 策略主线 | [0 · VLA 策略](#0-vla) |",
            "| 机器人开源数据集 | [0 · 数据集](#0-datasets) |",
            "| 搭双臂 teleop 采数据 | [1 · ALOHA](#1-aloha) |",
            "| 低成本野外采数据 | [2 · UMI](#2-umi) |",
            "| 人形 / VR teleop | [3 · VR 人形](#3-vr-humanoid) |",
            "| 训练 Foundation Model | [4 · FM](#4-foundation) |",
            "| 人类视频 → 机器人 | [5 · Ego](#5-ego) |",
            "| 灵巧手 | [6 · 灵巧手](#6-dexterous) |",
            "",
            "## 相关文档",
            "",
            "- [目录结构说明](./README.md)",
            "- [VLM 概述](../note/paper-note/VLM/概述.md)",
            "- [VLA 算法层学习路线](../note/VLA算法层学习路线与论文清单.md)",
            "- [VLA 训练与数据全貌（深度版）](../note/VLA训练与数据全貌-深度版.md)",
            "- [机器人操作数据学习路线报告](../note/机器人操作数据学习路线报告.md)",
            "- [快速入门](../note/快速入门-最短学习路径.md)",
        ]
    )

    INDEX.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {INDEX} ({total} papers, {len(missing)} in appendix)")


if __name__ == "__main__":
    generate()
