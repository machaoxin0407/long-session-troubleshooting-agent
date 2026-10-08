"""Export selected synthetic evaluation evidence without credentials or caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from session_experiment_data import score  # noqa: E402

FILES = (
    "dataset.json", "freeze.json", "comparison.json", "cost_and_path_audit.json",
    "automatic_conclusion.json", "automatic_closeout_policy.json",
    "retrieval_evidence_audit.json",
)
SECRET = re.compile(
    r"sk-[a-zA-Z0-9_-]{16,}|gh[pousr]_[a-zA-Z0-9]{20,}|"
    r"github_pat_[a-zA-Z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"
)
PRIVATE_IP = re.compile(
    r"(?<![\d.])(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?![\d.])"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_text(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if SECRET.search(text) or PRIVATE_IP.search(text) or re.search(r"[A-Z]:[\\/]Users[\\/]", text):
        raise ValueError(f"Potential private data in {path.name}; export stopped")
    for match in re.finditer(r"https?://[^\s\"<>]+", text):
        if urlsplit(match.group()).hostname not in {"github.com", "docs.github.com"}:
            raise ValueError(f"Non-allowlisted endpoint in {path.name}; export stopped")


def verify(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or digest(path) != expected:
            raise ValueError(f"Evidence hash mismatch: {name}")
        safe_text(path)
    dataset = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in dataset["cases"]}
    counts = {}
    for stage, expected in (("development", 70), ("test", 280), ("supplement", 98)):
        files = sorted((output / stage).glob("*/result.json"))
        if len(files) != expected:
            raise ValueError(f"Incomplete stage: {stage}")
        for path in files:
            row = json.loads(path.read_text(encoding="utf-8"))
            context = json.loads(path.with_name("context.json").read_text(encoding="utf-8"))
            if score(row["answer"], cases[row["case"]], context) != row["answer_check"]:
                raise ValueError(f"Frozen scoring mismatch: {stage}/{row['task_id']}")
        counts[stage] = len(files)
    return {"hashes_verified": len(manifest["files_sha256"]), "rescored_answers": counts}


def export(source: Path, output: Path) -> dict:
    if source.resolve() == output.resolve():
        raise ValueError("Export must not replace the private source campaign")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Export requires a fresh directory; use --verify for an existing export")
    selected = [source / name for name in FILES]
    for stage in ("development", "test", "supplement"):
        selected.extend(sorted((source / stage).glob("*/result.json")))
        selected.extend(sorted((source / stage).glob("*/context.json")))
    for path in selected:
        safe_text(path)
    output.mkdir(parents=True, exist_ok=True)
    for path in selected:
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for name in ("automatic_conclusion.json", "automatic_closeout_policy.json"):
        target = output / name
        document = json.loads(target.read_text(encoding="utf-8"))
        policy = document.get("policy", document)
        policy.pop("user_instruction", None)
        target.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    labels = {"window": "滑动窗口", "rolling": "滚动摘要＋窗口",
              "structured_recall": "结构化摘要＋原文回查", "collapsed": "全层树检索", "traversal": "逐层树检索"}
    report = ["# 长会话故障排查：自动评测报告", "",
              "56组冻结合成会话，覆盖40/80/160轮历史；五策略共280条主评测回答，另有70条重复和28条消融。",
              "70例标准答案已核对；实际回答未进行独立人工语义验收。结果为预设自动规则指标，不是生产准确率。", "",
              "| 策略 | 规则通过/任务数 | 正常路径通过/任务数 | 降级路径通过/任务数 |",
              "|---|---:|---:|---:|"]
    for strategy, label in labels.items():
        row = comparison["by_strategy"][strategy]
        report.append(f"| {label} | {row['rule_passed']}/{row['tasks']} | "
                      f"{row['normal_rule_passed']}/{row['tasks'] - row['fallback_tasks']} | "
                      f"{row['degraded_rule_passed']}/{row['fallback_tasks']} |")
    report += ["", "结构化摘要＋原文回查方案（含降级回退）由窗口基线15/56提升至45/56，"
               "自动规则通过率提高53.6个百分点。45次通过含正常路径2次、降级路径43次，不能归因于模型摘要本身。",
               "树相对单层回查的配对差异区间包含0，未显示明确额外收益；不证明树无效或方案等价。", "",
               "## 成本与复核", "",
               "comparison.json保留按策略、场景和历史长度的结果、配对bootstrap区间、重复与消融统计。",
               "cost_and_path_audit.json保留冷构建、查询、降级路径及usage观测；缺失usage不按零计费。",
               "重复调用不增加独立会话样本数；正常与降级路径分别统计，不能将词项回退计为语义路径成功。",
               "retrieval_evidence_audit.json记录独立历史视频检索实验，不代表手册检索或客服整体问答效果。", "",
               "## 离线复算", "", "```bash",
               "python deploy/export_public_session_evidence.py --verify reports/public_long_session_20261008",
               "```", "", "此命令核验文件哈希并按冻结规则复算448条回答，不调用模型。",
               "1273次HTTP尝试为归档统计；完整HTTP账本、向量缓存和服务器原始记录不在公开包内。", ""]
    (output / "FINAL_AUTOMATIC_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    original = json.loads((source / "regression_verified/verification.json").read_text(encoding="utf-8"))
    checks = json.loads((source / "automatic_closeout_checks.json").read_text(encoding="utf-8"))
    regression = {key: original[key] for key in ("started_utc", "elapsed_seconds", "exit_code", "summary", "counts")}
    regression.update(
        runtime_and_test_code_unchanged=checks["frozen_source_files_preserved"],
        documentation_updates_after_regression=checks.get("documentation_updates_after_regression", []),
        note="Existing full regression; publication adds documentation and an offline export helper only.",
    )
    (output / "regression.json").write_text(json.dumps(regression, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# 长会话故障排查：公开评测证据\n\n"
        "本目录仅包含冻结合成数据与自动评测证据，未包含真实用户对话、凭据、服务器原始记录、数据库或向量缓存。"
        "冻结数据、回答与上下文按字节复制；报告和说明按公开范围整理，当前文件哈希列于 manifest.json。\n\n"
        "- [最终报告](FINAL_AUTOMATIC_REPORT.md)。\n"
        "- dataset.json：70个独立合成会话及标准答案；freeze.json：冻结参数、模型与源码哈希。\n"
        "- development/：70条开发回答；test/：56会话×5策略的280条主回答。\n"
        "- supplement/：70条重复与28条消融。每个任务包含 result.json 和实际可见的 context.json。\n"
        "- comparison.json 与 cost_and_path_audit.json：原始规则结果、配对区间及成本。\n"
        "- regression.json：已有全量回归的摘要，源码与测试未改动；发布时仅更新文档并新增离线导出工具。\n\n"
        "45/56为含降级回退的整体方案自动规则通过数，正常路径2/2、降级路径43/54；窗口15/56。"
        "回答未进行独立人工语义验收，不等于生产准确率。树结构未显示明确额外增益，不证明树无效。\n\n"
        "在仓库根目录离线核验全部文件哈希并按冻结评分器复算448条回答（不发起模型请求）：\n\n"
        "```bash\npython deploy/export_public_session_evidence.py --verify reports/public_long_session_20261008\n```\n\n"
        "如需新实验，按 docs/LONG_SESSION_AGENT_EXPERIMENT_20261008.md 创建独立输出目录、配置自己的模型凭据和请求额度。"
        "新模型结果允许不同；禁止覆盖本冻结证据。报告中提及的原始 ledger、缓存和人工审阅页面不在公开包内，"
        "1273次请求总数属于归档统计，公开包不提供完整HTTP账本的独立复核。\n",
        encoding="utf-8",
    )
    hashes = {path.relative_to(output).as_posix(): digest(path)
              for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    (output / "manifest.json").write_text(json.dumps({
        "schema": 1, "synthetic_data_only": True, "new_model_requests": 0,
        "source_dataset_sha256": digest(source / "dataset.json"), "files_sha256": hashes,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return verify(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify(args.verify)
    elif args.source and args.output:
        result = export(args.source, args.output)
    else:
        parser.error("Use --verify DIR or --source DIR --output DIR")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
