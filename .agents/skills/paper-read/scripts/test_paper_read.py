#!/usr/bin/env python3
"""Offline tests for paper-read path resolution and output contracts."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from paper_read_common import ContractError, paper_paths, sha256_file, validate_outputs  # noqa: E402


VALID_TRANSLATION = """# 测试论文

## 摘要

本文研究视频时间定位问题，并提出一种用于查询与视频对齐的方法。该方法首先编码视频和文本，然后逐步预测时间边界，以改善复杂视频中的定位结果。

## 方法

给定视频 V 和查询 Q，模型计算表示 z = f(V, Q)，随后预测起点与终点。训练目标保持为 $L = L_s + L_e$，从而便于与原文公式对应。
"""

VALID_NOTE = """# Paper

一句话：
解决视频时刻检索中的边界定位问题，通过联合视频文本编码与边界解码，实现更准确的时间区间预测。

## Problem

- 查询与长视频内容的对齐困难。
- 起止边界容易受到相邻背景片段干扰。

## Method

Video / Query
↓
Video-Text Encoder
↓
Boundary Decoder
↓
Temporal Boundary

核心创新：
1. 联合建模视频与查询。
2. 显式预测起止边界。
3. 使用多阶段训练稳定优化。

## Training

- Training paradigm: finetune
- Loss / Reward: boundary loss
- 特殊训练策略: multistage

## Experiment

Dataset: QVHighlights
Metric: R@1; mAP
主要结果 / 结论: 在主要指标上优于论文报告的基线。

## Tags

task: task:mr
role: role:executor
training: train:finetune; opt:multistage
time: reason:boundary; out:regression
visual: modal:vt; setting:finetuned

## Key Figure / Table

- Fig.2：核心架构
- Table.1：主要结果

## Code

Repo: Not Checked
关键文件: N/A

## 我还没懂

1. 边界解码器对短事件是否稳定？
2. 多阶段训练各阶段的贡献如何？
3. 在跨数据集设置下能否保持性能？
"""


class PaperReadTests(unittest.TestCase):
    paper_id = "2026-test-paper"

    def make_workspace(self, root: Path) -> tuple[Path, Path]:
        source = root / "literature" / "sources" / "CVPR 2026" / f"{self.paper_id}.pdf"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"%PDF-1.4\nfixture\n%%EOF\n")
        return root, source

    def write_outputs(self, root: Path) -> dict[str, object]:
        paths = paper_paths(root, self.paper_id, create_dirs=True)
        Path(paths["translation"]).write_text(VALID_TRANSLATION, encoding="utf-8")
        Path(paths["note"]).write_text(VALID_NOTE, encoding="utf-8")
        return paths

    def test_resolve_paths_and_detect_reusable_cache(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root, source = self.make_workspace(Path(raw))
            paths = paper_paths(root, self.paper_id, create_dirs=True)
            self.assertEqual(Path(paths["translation"]).parent.name, "translations")
            self.assertEqual(Path(paths["note"]).parent.name, "notes")
            self.assertFalse(paths["cache_reusable"])

            cache = root / "literature" / "extracted" / self.paper_id
            cache.mkdir(parents=True)
            (cache / "paper.md").write_text("# extracted\n", encoding="utf-8")
            (cache / "source.sha256").write_text(sha256_file(source) + "\n", encoding="utf-8")
            self.assertTrue(paper_paths(root, self.paper_id)["cache_reusable"])

    def test_validate_good_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root, _ = self.make_workspace(Path(raw))
            self.write_outputs(root)
            result = validate_outputs(root, self.paper_id, SKILL_DIR)
            self.assertTrue(result["valid"])
            self.assertIn("task:mr", result["note"]["tags"])

    def test_reject_translation_formatting_and_unknown_tag(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root, _ = self.make_workspace(Path(raw))
            paths = self.write_outputs(root)
            translation = Path(paths["translation"])
            translation.write_text(VALID_TRANSLATION + "\n**错误加粗**\n", encoding="utf-8")
            with self.assertRaises(ContractError):
                validate_outputs(root, self.paper_id, SKILL_DIR)

            translation.write_text(VALID_TRANSLATION, encoding="utf-8")
            note = Path(paths["note"])
            note.write_text(VALID_NOTE.replace("task: task:mr", "task: task:unknown"), encoding="utf-8")
            with self.assertRaises(ContractError):
                validate_outputs(root, self.paper_id, SKILL_DIR)

    def test_cli_resolver_accepts_exact_pdf_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root, source = self.make_workspace(Path(raw))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "resolve_paper.py"),
                    "--workspace",
                    str(root),
                    "--paper",
                    str(source),
                    "--create-dirs",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertTrue((source.parent / "translations").is_dir())
            self.assertTrue((source.parent / "notes").is_dir())


if __name__ == "__main__":
    unittest.main()
