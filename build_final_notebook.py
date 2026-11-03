"""构建自包含Colab交付本；只读冻结源码/日志，不运行模型或修改报告。

用法：
    python build_final_notebook.py
    python build_final_notebook.py --results-root PATH_TO_task1_assignment

传入真实日志后，重建的Markdown摘要和源码ZIP会包含可核对的真实结果；
所有代码格始终保持 execution_count=None / outputs=[]，不冒充已执行notebook。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
SOURCE_NAMES = ("runner.py", "cloud_driver.py", "analyze_results.py", "supplemental_runner.py",
                "supplemental_protocol.json", "supplemental_items.json", "analyze_supplemental.py",
                "test_supplemental_runner.py")
METHODS = ("zero_shot", "few_shot", "cot", "agent")
RESULT_NAMES = ("manifest.json", "source_data.json", "data_audit.json", "ledger.jsonl",
                "results.jsonl", "summary.json", "summary.csv")
ANALYSIS_NAMES = ("analysis.json", "case_candidates.json", "main_table.csv", "question_type_table.csv",
                  "agent_changes.csv", "accuracy_comparison.png", "accuracy_comparison.pdf",
                  "efficiency_comparison.png", "efficiency_comparison.pdf")
SUPPLEMENTAL_NAMES = ("protocol.json", "job_manifest.json", "ledger.jsonl", "results.jsonl", "summary.json")
SUPPLEMENTAL_ANALYSIS_NAMES = ("analysis.json", "metrics.csv", "paired_comparisons.json")
NOTEBOOK_NAME = "Task1_完整实验与结果.ipynb"
PACKAGE_NAME = "Task1_可复现源码与日志.zip"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip().splitlines(keepends=True)}


def _percent(value) -> str:
    return "未完成" if value is None else f"{value:.2%}"


def read_snapshot(results_root: Path | None, data_path: Path, source_hashes: dict):
    """只读校验。正式结果完整时，再调用只读分析器进行ledger一致性验证。"""
    if results_root is None:
        return (
            "## 实际运行摘要（等待日志）\n\n"
            "本次构建尚未指定实际日志目录，因此不填写任何成绩、费用或案例结论。"
            "完成实验并下载日志后，用构建器的 `--results-root` 参数重建即可。\n\n"
            "本notebook的代码格尚未执行，所有旧输出为空。",
            {"status": "no_logs_supplied", "phases": {}},
        )
    snapshot = {"status": "logs_supplied", "phases": {}}
    lines = [
        "## 实际运行摘要（只读导入）", "",
        "下表由已下载日志生成；它不是本notebook代码格的执行输出。"
        "日志、指纹及源文件哈希保存在配套源码ZIP中。所有代码格仍保持未执行状态。", "",
        "| 阶段 | 方法 | 终结结果数/有效输入数 | 正确数 | 主集合准确率 | 敏感性准确率 | API尝试次数 | 已知token |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for phase in ("development", "formal"):
        directory = results_root / phase
        summary_path, manifest_path = directory / "summary.json", directory / "manifest.json"
        if not summary_path.exists() or not manifest_path.exists():
            snapshot["phases"][phase] = {"status": "missing_summary_or_manifest"}
            lines.append(f"| {phase} | 尚未取得日志 | — | — | — | — | — | — |")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if summary.get("fingerprint") != manifest.get("fingerprint"):
            raise ValueError(f"{phase} summary与manifest指纹不一致。")
        if manifest["experiment"].get("runner_source_sha256") != source_hashes["runner.py"]:
            raise ValueError(f"{phase}日志使用的runner与当前冻结源码不一致，不能包装成同一版本。")
        methods = summary.get("methods", {})
        if set(methods) != set(METHODS):
            raise ValueError(f"{phase}不是四组实验的汇总。")
        all_finished = all(s.get("all_valid_inputs_finished") is True for s in methods.values())
        files = {name: sha((directory / name).read_bytes()) for name in RESULT_NAMES if (directory / name).exists()}
        snapshot["phases"][phase] = {
            "status": "finished" if all_finished else "partial",
            "fingerprint": summary["fingerprint"], "generated_at": summary.get("generated_at"),
            "summary_sha256": files["summary.json"], "files_sha256": files,
            "methods": methods,
        }
        for method in METHODS:
            row = methods[method]
            terminal = sum(row.get("status_counts", {}).get(s, 0) for s in ("completed", "api_error", "parse_error"))
            lines.append(
                f"| {phase} | {method} | {terminal}/{row['valid_input_count']} | {row['correct_count']} | "
                f"{_percent(row.get('accuracy_valid_inputs'))} | {_percent(row.get('sensitivity_accuracy'))} | "
                f"{row['api_attempts']} | {row['known_total_tokens']} |"
            )
        if phase == "formal" and all_finished:
            spec = importlib.util.spec_from_file_location("_task1_packaging_analysis", HERE / "analyze_results.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.load_run(directory, data_path)
            snapshot["phases"][phase]["ledger_validation"] = "passed"
    lines.extend([
        "", "开发题仅用于流程/格式验证，不估计药师考试能力。正式主集合为97条有文本选项的记录，"
        "预注册敏感性集合为90条；原始100条全部保留。API/解析失败计错，未完成项不会被写成最终准确率。",
        "", "已知token不等于账单。若存在未知用量，需要查阅日志并在成本说明中保留该限制。",
    ])
    supplemental_dir = results_root / "supplemental"
    if (supplemental_dir / "summary.json").is_file():
        from analyze_supplemental import load_supplemental
        checked = load_supplemental(supplemental_dir, results_root / "formal", HERE / "supplemental_protocol.json")
        supplemental_summary = checked.summary()
        snapshot["supplemental"] = {"status": "finished" if supplemental_summary["all_finished"] else "partial",
                                    "ledger_validation": "passed", "summary": supplemental_summary,
                                    "files_sha256": {name: sha((supplemental_dir / name).read_bytes()) for name in SUPPLEMENTAL_NAMES if (supplemental_dir / name).is_file()}}
        lines.extend(["", "### 主实验之后追加的研究（Post-hoc）", "",
                      "A为16道独立人工封闭世界协议检查（8道设计题、8道固定probe），B为95个原始合法初答状态上的单步反馈消融。它们不替代主实验，也不代表完整Agent消融。",
                      "", "| 研究 | 条件 | 集合 | 正确数/分母 | 准确率 | 格式合法率 | API次数 |", "|---|---|---|---:|---:|---:|---:|"])
        for study, conditions, subsets in (("A", ("cot_v1", "cot_v2"), ("all_16", "held_out_probe")),
                                           ("B", ("feedback", "null_feedback"), ("conditional_95", "overall_97"))):
            for condition in conditions:
                for subset in subsets:
                    value = supplemental_summary["study_" + study][condition][subset]
                    lines.append(f"| {study} | {condition} | {subset} | {value['correct']}/{value['n']} | {_percent(value['accuracy'])} | {_percent(value['format_valid_rate'])} | {value['api_attempts']} |")
        lines.extend(["", "两项新增预指定比较为A全部16题与B条件95题，使用双侧精确McNemar检验并在这两个比较内做Holm校正；probe8与overall97只是重叠的描述子集。",
                      f"补充实际调用：{supplemental_summary['api_attempts']}；已知token：{supplemental_summary['totals']['known_total_tokens']}；未知用量调用：{supplemental_summary['totals']['calls_with_unknown_usage']}。"])
        for name, comparison in supplemental_summary["supplementary_primary_comparisons"].items():
            if comparison.get("all_finished"):
                lines.append(f"- {name}: McNemar p={comparison['mcnemar_exact_two_sided_p']:.6g}；Holm2 p={comparison['holm_adjusted_p_two_comparison_family']:.6g}。")
    else:
        snapshot["supplemental"] = {"status": "no_logs_supplied"}
        lines.extend(["", "补充研究代码与冻结协议已附；尚未取得补充运行日志时不填写补充成绩。"])
    return "\n".join(lines), snapshot


def make_readme(has_logs: bool) -> str:
    return f"""Task 1 可复现实验交付说明

一、文件用途
1. {NOTEBOOK_NAME}：上传Colab即可使用的自包含notebook。包含冻结源码和正式100题。
2. assignment2_task1/runner.py：三种提示与反馈Agent、预算守卫、严格评分、逐请求ledger、断点恢复。
3. assignment2_task1/cloud_driver.py：独立开发检查→正式四组评测。
4. assignment2_task1/analyze_results.py：只读真实日志，生成统计表、图表、案例候选。
5. data/1.exam.json：此次固定的正式数据。标准答案/解析仅供本地评分。
6. build_manifest.json：代码/数据/日志哈希及构建记录。
7. results/：{'已附构建时取得的日志；请看manifest中的完整/部分状态。' if has_logs else '本模板尚未附真实日志，不存在虚构成绩。'}
8. supplemental_runner.py、supplemental_protocol.json、supplemental_items.json：主实验后冻结的16题提示比较与95状态单步反馈消融。
9. analyze_supplemental.py：只读校验补充请求、ledger、逐题结果并重新计算统计；不调用API。

二、Colab最短操作
1. 上传并打开{NOTEBOOK_NAME}，连接普通CPU运行时。
2. 依次运行安装依赖、写入冻结源码与数据的单元格。
3. 若只阅读结果，保持RUN_LIVE_EXPERIMENTS=False和RUN_SUPPLEMENTAL_LIVE=False，不调用模型、不需要Key。
4. 若要允许开始/恢复实验，将RUN_LIVE_EXPERIMENTS=True，并在getpass隐藏输入框中输入自己的DeepSeek官方Key。
   已附且已完成的同配置结果会跳过，不会强制重新付费。要独立重跑，必须同时：设置新的PROJECT_DIR、将RESTORE_EXISTING_RESULTS=False、将RUN_LIVE_EXPERIMENTS=True。保留旧目录。
5. 运行独立开发单元格：3题×4方法，最多24次API尝试。
6. 开发流程无API/格式失败后，运行正式单元格：主集合97题×4方法，2轮复核上限，正常最多776次请求。
   正式累计预算上限800保留给明确授权的恢复；程序不会隐性重试。
7. 运行离线分析单元格和ZIP导出单元格，下载结果。Colab临时文件可能随运行时回收而丢失。
8. 补充研究由独立RUN_SUPPLEMENTAL_LIVE开关控制。继续同目录时已尝试项一律跳过（包括失败）；不允许挑题重试。
   如需新补充研究重复运行，保留已附原formal日志，设置新的SUPPLEMENTAL_OUTPUT_DIR，再开启补充开关；SUPPLEMENTAL_FORMAL_DIR必须指向协议冻结的原始formal目录，不能使用新跑的主实验日志。

三、在本地仅复算既有结果（不需要API Key）
先安装Python 3.10+，从本源码包根目录执行：
  python -m pip install --index-url https://pypi.org/simple matplotlib
  python assignment2_task1/analyze_results.py --run-dir results/formal --data data/1.exam.json --output-dir analysis_reproduced
  python assignment2_task1/analyze_supplemental.py --run-dir results/supplemental --formal-dir results/formal --output-dir supplemental_analysis_reproduced
只有正式四组日志完整且与指纹一致时才会生成最终分析。未完成日志不会冒充完整实验。

四、在本地运行离线测试（不调用API）
  python -m pip install --index-url https://pypi.org/simple -r requirements.txt
  python assignment2_task1/test_runner.py
  python -m unittest discover -s assignment2_task1 -p test_analyze_results.py -v
  python -m unittest discover -s assignment2_task1 -p test_supplemental_runner.py -v
  python -m unittest discover -s assignment2_task1 -p test_analyze_supplemental.py -v
  python -m unittest discover -s assignment2_task1 -p test_final_notebook.py -v
测试所需正式数据副本位于data/1.exam.json。

五、复现与恢复约定
- notebook的代码格全部是未执行状态；只读成绩表来自results目录的日志，不是伪造的单元格输出。
- 不要改冻结runner.py后继续复用旧实验目录；代码、提示、数据或参数变化会触发指纹拒绝。
- 重跑同一配置时，已完成项会跳过，已成功的中间阶段可从ledger重放。
- 失败或未确认请求默认不重新发送；不要为了提高分数挑选答错题重跑。
- API Key不写入日志、ZIP或报告；请勿把客户端对象/密钥打印出来。
- 正式数据第21、26、66条所有选项为空，主评测统一排除；敏感性分析额外排除预注册的7条图示/上下文风险记录。
- 三个few-shot示例与三个开发题均为独立人工构造算术/单位换算；不等于专业药学知识训练，报告应讨论这个局限。
- 分析器中的费用是声明费率下的估计，不是账单；实际模型/价格可随时间变更。

此包不自动提交Blackboard，也不伪造尚未完成的实验或报告结论。
"""


def result_files(results_root: Path | None) -> dict[str, bytes]:
    """只收集显式白名单的日志/分析，不扫描整个目录。"""
    collected = {}
    if results_root is None:
        return collected
    for phase in ("development", "formal"):
        for name in RESULT_NAMES:
            path = results_root / phase / name
            if path.is_file():
                collected[f"task1_assignment/{phase}/{name}"] = path.read_bytes()
    for folder in (results_root / "analysis", results_root / "formal/analysis"):
        if folder.is_dir():
            for name in ANALYSIS_NAMES:
                path = folder / name
                if path.is_file():
                    collected[f"task1_assignment/analysis/{name}"] = path.read_bytes()
            break
    for name in SUPPLEMENTAL_NAMES:
        path = results_root / "supplemental" / name
        if path.is_file():
            collected[f"task1_assignment/supplemental/{name}"] = path.read_bytes()
    for folder in (results_root / "supplemental_analysis", results_root / "supplemental/analysis"):
        if folder.is_dir():
            for name in SUPPLEMENTAL_ANALYSIS_NAMES:
                path = folder / name
                if path.is_file():
                    collected[f"task1_assignment/supplemental_analysis/{name}"] = path.read_bytes()
            break
    return collected


def version_file(path: Path | None) -> bytes | None:
    if path is None:
        return None
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    # 只读取软件版本白名单；不保留任意环境变量、配置或客户端字段。
    selected = {key: raw[key] for key in ("python", "platform", "openai", "matplotlib", "numpy", "pandas", "requests", "scipy")
                if isinstance(raw.get(key), str)}
    if not selected:
        raise ValueError("指定的环境文件没有可识别的软件版本字段。")
    return json_bytes({"purpose": "software_versions_recorded_from_original_run", "versions": selected})


def build_notebook(sources: dict[str, bytes], data_raw: bytes, snapshot_md: str, readme: str,
                   quality_raw: bytes | None, restored_files: dict[str, bytes]):
    embedded = {name: {"base64": base64.b64encode(raw).decode("ascii"), "sha256": sha(raw)}
                for name, raw in sources.items()}
    embedded["task1_smoke/1.exam.json"] = {"base64": base64.b64encode(data_raw).decode("ascii"), "sha256": sha(data_raw)}
    if quality_raw is not None:
        embedded["data_quality_review.json"] = {"base64": base64.b64encode(quality_raw).decode("ascii"), "sha256": sha(quality_raw)}
    bundle_buffer = io.BytesIO()
    with zipfile.ZipFile(bundle_buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, raw in sorted(restored_files.items()):
            bundle.writestr(name, raw)
    bundle_raw = bundle_buffer.getvalue()
    result_bundle = {"base64_zip": base64.b64encode(bundle_raw).decode("ascii"),
                     "sha256": sha(bundle_raw),
                     "files_sha256": {name: sha(raw) for name, raw in restored_files.items()}}
    allowed_restore_paths = (
        [f"task1_assignment/{phase}/{name}" for phase in ("development", "formal") for name in RESULT_NAMES]
        + [f"task1_assignment/analysis/{name}" for name in ANALYSIS_NAMES]
        + [f"task1_assignment/supplemental/{name}" for name in SUPPLEMENTAL_NAMES]
        + [f"task1_assignment/supplemental_analysis/{name}" for name in SUPPLEMENTAL_ANALYSIS_NAMES]
        + ["recorded_environment.json"]
    )
    cells = [markdown("""
# 作业2 Task 1：完整实验与可复现结果

运行路线：**固定数据与代码 → 独立开发检查 → 三种提示与Agent正式对比 → 离线分析 → 下载结果**。

本notebook包含全部执行源码和此次正式数据，不依赖临时下载链接。模型通过DeepSeek官方API远程运行，CPU运行时足够。

为了避免打开交付本时重复付费，`RUN_LIVE_EXPERIMENTS`与`RUN_SUPPLEMENTAL_LIVE`默认均为`False`。已附日志会按SHA校验还原，默认运行全部即可离线复算主实验与补充研究、展示和导出，不会请求Key或调用模型。

只有需要开始/恢复尚未完成的实验时才改成`True`，并在隐藏输入框填写自己的Key。同配置已完成项会跳过。独立重跑必须同时设置新的`PROJECT_DIR`、`RESTORE_EXISTING_RESULTS=False`和`RUN_LIVE_EXPERIMENTS=True`，保留旧结果且不能按答错情况选择重跑。Key不会写进源文件、日志或ZIP。

**交付本是在本地构建的，代码格没有执行记录。**后面的实际成绩（若已附）来自已下载日志的只读汇总。不要把未运行的代码格或开发题成绩当作正式实验结果。
"""), markdown(snapshot_md), markdown("""
## 1. 安装运行依赖

从标准PyPI安装OpenAI兼容SDK和Matplotlib。代码连接的是DeepSeek官方服务，不需要OpenAI Key。此格没有模型调用。
"""), code("%pip -q install --index-url https://pypi.org/simple openai matplotlib"), markdown("""
## 2. 写入冻结源码与数据

代码按原始字节还原并验证SHA-256。真实日志采用压缩包内嵌，验证压缩包及逐文件SHA后，只还原白名单路径。源码和原始日志已有不同内容时停止，避免覆盖旧实验。派生分析若已存在则保留，交给第7步重算，所以可以重复运行。100条原始数据完整保留，模型请求只取题干、题型和非空选项。
"""), code("""
import base64
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from importlib.metadata import version

PROJECT_DIR = Path("/content/task1_assignment_repro")
RESTORE_EXISTING_RESULTS = True  # 默认恢复已附真实日志；独立新实验须新目录+改False。
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDED_FILES = """ + repr(embedded) + """
for relative_name, record in EMBEDDED_FILES.items():
    raw = base64.b64decode(record["base64"])
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise RuntimeError("内嵌文件校验失败：" + relative_name)
    target = PROJECT_DIR / relative_name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != raw:
        raise RuntimeError("已有不同版本文件，请使用新项目目录：" + str(target))
    target.write_bytes(raw)
RESTORED_RESULTS = """ + repr(result_bundle) + """
ALLOWED_RESTORE_PATHS = set(""" + repr(allowed_restore_paths) + """)
compressed = base64.b64decode(RESTORED_RESULTS["base64_zip"])
if hashlib.sha256(compressed).hexdigest() != RESTORED_RESULTS["sha256"]:
    raise RuntimeError("内嵌日志压缩包校验失败。")
expected_files = RESTORED_RESULTS["files_sha256"]
if not set(expected_files) <= ALLOWED_RESTORE_PATHS:
    raise RuntimeError("日志包含白名单之外的路径。")
with zipfile.ZipFile(io.BytesIO(compressed)) as bundle:
    names = bundle.namelist()
    if len(names) != len(set(names)) or set(names) != set(expected_files):
        raise RuntimeError("日志压缩包文件名单不一致。")
    for relative_name in names:
        target = (PROJECT_DIR / relative_name).resolve()
        if not target.is_relative_to(PROJECT_DIR.resolve()):
            raise RuntimeError("拒绝项目目录之外的恢复路径。")
        raw = bundle.read(relative_name)
        if hashlib.sha256(raw).hexdigest() != expected_files[relative_name]:
            raise RuntimeError("日志文件校验失败：" + relative_name)
        if not RESTORE_EXISTING_RESULTS:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != raw:
            if relative_name.startswith(("task1_assignment/analysis/", "task1_assignment/supplemental_analysis/")):
                continue  # 派生分析可由第7步重算，不覆盖用户上次生成的版本。
            raise RuntimeError("已有不同日志，拒绝覆盖；请改用新PROJECT_DIR：" + relative_name)
        target.write_bytes(raw)
os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
environment = {
    "python": sys.version,
    "openai": version("openai"),
    "matplotlib": version("matplotlib"),
    "embedded_sha256": {name: record["sha256"] for name, record in EMBEDDED_FILES.items()},
}
Path("environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
print("冻结源码与数据已校验。项目目录：", PROJECT_DIR)
print("已附真实日志/分析文件数：", len(expected_files), "；本次启用恢复：", RESTORE_EXISTING_RESULTS)
print("本格没有调用模型API。")
"""), markdown("""
## 3. 读取数据并建立实验入口

导入驱动只读取数据，不调用模型。完整数据100题：3条选项全空，主评测97条；另外7条预注册的图示/上下文风险记录用于90题敏感性分析。两套成绩使用同一批预测，不额外选择有利样本。
"""), code("""
from cloud_driver import DATA, DEV_DATA, ROOT, run_development, run_formal
from collections import Counter

print("正式原始记录：", len(DATA))
print("题型：", dict(Counter(q["question_type"] for q in DATA)))
print("独立人工构造开发题：", len(DEV_DATA))
print("请勿把正式题的answer或analysis加入提示或Agent审查。")
"""), markdown("""
## 4. 是否允许开始/恢复付费实验，以及隐藏输入Key

阅读并离线复算已附结果时两个开关都保持`False`。主实验由`RUN_LIVE_EXPERIMENTS`控制，补充研究由`RUN_SUPPLEMENTAL_LIVE`独立控制；任一开启才显示隐藏Key输入。已完成项不会重新调用。主实验独立重跑才需要新的`PROJECT_DIR`并设置`RESTORE_EXISTING_RESULTS=False`；补充重复运行必须保留协议冻结的原formal状态并设置新输出目录，不能混用新跑的主实验。密钥只用于内存中的客户端，不自动重试。

参考：[DeepSeek官方调用说明](https://api-docs.deepseek.com/) · [思考模式](https://api-docs.deepseek.com/guides/thinking_mode/) · [JSON模式](https://api-docs.deepseek.com/guides/json_mode/)
"""), code("""
from getpass import getpass
from openai import OpenAI

RUN_LIVE_EXPERIMENTS = False
RUN_SUPPLEMENTAL_LIVE = False
_assignment2_client = None
if RUN_LIVE_EXPERIMENTS or RUN_SUPPLEMENTAL_LIVE:
    _key_input = getpass("输入DeepSeek官方API Key（隐藏输入）：").strip()
    if not _key_input:
        raise ValueError("没有输入Key；没有启动实验。")
    _assignment2_client = OpenAI(
        api_key=_key_input,
        base_url="https://api.deepseek.com",
        timeout=60.0,
        max_retries=0,
    )
    del _key_input
    print("客户端已配置；接下来先运行独立开发检查。")
else:
    print("只读模式：不会请求API，也不会索取Key。")
"""), markdown("""
## 5. 独立开发检查

三道开发题不来自正式测试题，也不同于few-shot示例。只验证负向问法、单多选及输出格式，不用于报告真实考试准确率。最多24次调用，失败会阻止正式实验启动。
"""), code("""
if RUN_LIVE_EXPERIMENTS:
    dev_engine = run_development(_assignment2_client)
else:
    print("已跳过开发API调用。")
"""), markdown("""
## 6. 正式四组实验

四组为Zero-shot、Few-shot、CoT、反馈驱动Agent。Agent使用与CoT完全相同的初答提示，再由独立审查产生反馈，控制器根据反馈选择提交或修订，最多两轮复核。

统一模型`deepseek-flash`，关闭内置思考模式，温度0，单次输出上限768 token。四组正常最多776次API调用；正式累计上限800留给明确需要的恢复，程序不会隐性重试。每次调用和每题完成都会保存日志、打印进度。

重新运行同一配置会跳过已有完成结果；失败/未确认请求不会自动重发。不要根据测试答案改提示后复用旧实验目录。
"""), code("""
if RUN_LIVE_EXPERIMENTS:
    formal_engine = run_formal(_assignment2_client)
else:
    print("已跳过正式API调用。只读摘要不会被当作本notebook刚执行的结果。")
"""), markdown("""
## 6b. 主实验之后的补充研究（独立开关）

16道人工封闭世界检查的原CoT/v2比较共32次调用；95个固定合法初答的feedback/null控制器比较共190次调用。两组B请求仅反馈字段不同，均没有后续复核轮次。原第14、67条初答无效，在97题口径下两组共同计错。两项新增主比较为A全部16题与B条件95题，单独采用Holm2校正；这是事后补充证据，不替代原主实验。

默认离线。允许补充调用时，`SUPPLEMENTAL_FORMAL_DIR`必须保持原冻结formal日志；同目录中任何已尝试项（包括失败）都不会再次调用。若要做一次新的完整重复运行，保留原formal目录，仅把`SUPPLEMENTAL_OUTPUT_DIR`改为新的空目录，并开启补充开关；不要按答错情况挑题重复。
"""), code("""
from supplemental_runner import run_supplemental

SUPPLEMENTAL_FORMAL_DIR = ROOT / "formal"
SUPPLEMENTAL_OUTPUT_DIR = ROOT / "supplemental"
if RUN_SUPPLEMENTAL_LIVE:
    supplemental_engine = run_supplemental(
        _assignment2_client,
        protocol_path=PROJECT_DIR / "supplemental_protocol.json",
        formal_dir=SUPPLEMENTAL_FORMAL_DIR,
        output_dir=SUPPLEMENTAL_OUTPUT_DIR,
        max_api_calls=240,
    )
else:
    print("已跳过补充研究API调用。")
"""), markdown("""
## 7. 从真实日志离线生成统计与图表

此格不调用API。验证原始数据、指纹、ledger、逐题结果一致后，生成97/90题成绩、95% Wilson区间、配对McNemar检验及Holm校正、API时间/token统计、Agent初答到最终答案变化和真实案例候选。正式记录未完成时停止，不生成假成绩。
"""), code("""
from analyze_results import load_run, write_outputs

formal_dir = ROOT / "formal"
if (formal_dir / "summary.json").exists():
    checked_run = load_run(formal_dir, ROOT / "1.exam.json")
    analysis = write_outputs(checked_run, ROOT / "analysis")
    print("离线分析已保存到：", ROOT / "analysis")
    for row in analysis["main_results"]:
        print(row["subset"], row["method"], f"{row['correct']}/{row['n']}", f"{row['accuracy']:.2%}")
else:
    print("当前运行时没有正式日志。可阅读上面的真实日志摘要，或重新运行实验后再分析。")

from analyze_supplemental import load_supplemental, write_outputs as write_supplemental_outputs

if (SUPPLEMENTAL_OUTPUT_DIR / "summary.json").exists():
    checked_supplemental = load_supplemental(
        SUPPLEMENTAL_OUTPUT_DIR, SUPPLEMENTAL_FORMAL_DIR, PROJECT_DIR / "supplemental_protocol.json")
    supplemental_analysis = write_supplemental_outputs(checked_supplemental, ROOT / "supplemental_analysis")
    for study, conditions in (("A", ("cot_v1", "cot_v2")), ("B", ("feedback", "null_feedback"))):
        subset = "all_16" if study == "A" else "conditional_95"
        for condition in conditions:
            row = supplemental_analysis["study_" + study][condition][subset]
            print("补充", study, condition, subset, f"{row['correct']}/{row['n']}", row["accuracy"])
    print("新增两项主比较（双侧McNemar、Holm2）：", supplemental_analysis["supplementary_primary_comparisons"])
else:
    print("当前目录没有补充日志；不填写补充成绩，也不调用API。")
"""), markdown("""
## 8. 导出源码、数据、日志和分析

此格只打包明确列出的项目文件，不打包客户端、Key或整个运行时。主实验或补充日志尚未完整时，文件名带`partial`，两项状态分开记录。Colab临时运行时可能回收，请下载保存结果。
"""), code("""
import zipfile
from google.colab import files

RESULT_NAMES = """ + repr(RESULT_NAMES) + """
ANALYSIS_NAMES = """ + repr(ANALYSIS_NAMES) + """
SUPPLEMENTAL_NAMES = """ + repr(SUPPLEMENTAL_NAMES) + """
SUPPLEMENTAL_ANALYSIS_NAMES = """ + repr(SUPPLEMENTAL_ANALYSIS_NAMES) + """
summary_path = ROOT / "formal" / "summary.json"
finished = False
if summary_path.exists():
    status = json.loads(summary_path.read_text(encoding="utf-8"))
    finished = all(v.get("all_valid_inputs_finished") for v in status.get("methods", {}).values()) and len(status.get("methods", {})) == 4
supplemental_finished = False
if (SUPPLEMENTAL_OUTPUT_DIR / "summary.json").exists():
    supplemental_finished = json.loads((SUPPLEMENTAL_OUTPUT_DIR / "summary.json").read_text(encoding="utf-8")).get("all_finished") is True
archive = PROJECT_DIR / ("Task1_results_complete.zip" if finished and supplemental_finished else "Task1_results_partial.zip")
export_entries = {}
for name in """ + repr(SOURCE_NAMES + ("environment.json", "recorded_environment.json", "data_quality_review.json")) + """:
    path = PROJECT_DIR / name
    if path.is_file():
        export_entries["assignment2_task1/" + name] = path
export_entries["data/1.exam.json"] = PROJECT_DIR / "task1_smoke/1.exam.json"
for phase in ("development", "formal"):
    for name in RESULT_NAMES:
        path = ROOT / phase / name
        if path.is_file():
            export_entries[f"results/{phase}/{name}"] = path
for name in ANALYSIS_NAMES:
    path = ROOT / "analysis" / name
    if path.is_file():
        export_entries["analysis/" + name] = path
for name in SUPPLEMENTAL_NAMES:
    path = SUPPLEMENTAL_OUTPUT_DIR / name
    if path.is_file():
        export_entries["results/supplemental/" + name] = path
for name in SUPPLEMENTAL_ANALYSIS_NAMES:
    path = ROOT / "supplemental_analysis" / name
    if path.is_file():
        export_entries["supplemental_analysis/" + name] = path
supplemental_formal_archive = "results/formal"
if SUPPLEMENTAL_FORMAL_DIR.resolve() != (ROOT / "formal").resolve():
    supplemental_formal_archive = "results/supplemental_original_formal"
    for name in ("manifest.json", "source_data.json", "ledger.jsonl", "results.jsonl"):
        path = SUPPLEMENTAL_FORMAL_DIR / name
        if path.is_file():
            export_entries[supplemental_formal_archive + "/" + name] = path
runtime_manifest = {
    "formal_run_status": "complete" if finished else "partial",
    "supplemental_run_status": "complete" if supplemental_finished else "partial",
    "supplemental_formal_archive": supplemental_formal_archive,
    "contains_notebook": False,
    "included_tests": [name for name in export_entries if "/test_" in name],
    "files_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in export_entries.items()},
}
runtime_readme = (
    "Task 1 运行时结果导出包\\n\\n"
    + "正式实验状态：" + runtime_manifest["formal_run_status"] + "\\n"
    + "补充研究状态：" + runtime_manifest["supplemental_run_status"] + "\\n"
    + "本ZIP只包含下面列出的源码、数据、日志、测试和已生成分析；不包含notebook。\\n"
    + "API Key和客户端对象没有打包。environment.json为当前复算环境；recorded_environment.json若存在则来自原始运行记录。\\n\\n"
    + "离线复算（无需API Key；仅完整正式日志可生成最终分析）：\\n"
    + "python -m pip install --index-url https://pypi.org/simple matplotlib\\n"
    + "python assignment2_task1/analyze_results.py --run-dir results/formal --data data/1.exam.json --output-dir analysis_reproduced\\n\\n"
    + "python assignment2_task1/analyze_supplemental.py --run-dir results/supplemental --formal-dir " + supplemental_formal_archive + " --output-dir supplemental_analysis_reproduced\\n\\n"
    + "重跑模型需要另外打开完整交付notebook；同配置已完成项会跳过。\\n"
    + "实际文件清单（逐文件SHA见reproduction_manifest.json）：\\n"
    + "\\n".join("- " + name for name in sorted(export_entries))
    + "\\n- README.txt\\n- reproduction_manifest.json\\n"
)
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
    package.writestr("README.txt", runtime_readme)
    package.writestr("reproduction_manifest.json", json.dumps(runtime_manifest, ensure_ascii=False, indent=2))
    for name, path in export_entries.items():
        package.write(path, name)
print("已打包：", archive.name)
print("正式实验完整：", finished)
print("补充研究完整：", supplemental_finished)
files.download(str(archive))
"""), markdown("""
## 报告如何使用这些输出

- `manifest.json`给出实际模型、参数、提示、示例、数据和代码指纹。
- `main_table.csv`和两张比较图用于结果部分；准确率与成本一起报告。
- `agent_changes.csv`区分改对、改错和执行失败；不能只挑成功案例。
- `case_candidates.json`只选实际出现的案例，不存在的改进类型保持空。
- 正式数据缺项、few-shot示例知识难度、样本规模和模型随机性均需讨论。
- `API/parse`失败留在有效输入分母中计错；费用为声明费率下的估计，不是账单。
- `supplemental_analysis`提供事后追加研究的两项对比及Holm2；不能与主实验原比较族混称预先设计，也不能将单步消融描述成完整Agent消融。

本notebook不自动提交Blackboard；LaTeX报告是单独交付物。
""")]
    for index, cell in enumerate(cells):
        cell["id"] = f"task1-{index:03d}"
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, help="实际下载的task1_assignment目录，包含development和formal")
    parser.add_argument("--data", type=Path, default=WORKSPACE / "data/1.exam.json")
    parser.add_argument("--environment", type=Path, help="可选：原始运行的软件版本JSON；只读取版本白名单")
    parser.add_argument("--output-dir", type=Path, default=HERE / "deliverables")
    args = parser.parse_args(argv)
    if args.results_root is not None and not args.results_root.is_dir():
        raise FileNotFoundError("指定的实际日志目录不存在。")
    sources = {name: (HERE / name).read_bytes() for name in SOURCE_NAMES}
    data_raw = args.data.read_bytes()
    data = json.loads(data_raw)
    if not isinstance(data, list) or len(data) != 100:
        raise ValueError("构建使用的正式数据必须包含100条。")
    source_hashes = {name: sha(raw) for name, raw in sources.items()}
    snapshot_md, snapshot = read_snapshot(args.results_root, args.data, source_hashes)
    readme = make_readme(args.results_root is not None)
    quality_path = HERE / "data_quality_review.json"
    quality_raw = quality_path.read_bytes() if quality_path.exists() else None
    restored_files = result_files(args.results_root)
    recorded_environment = version_file(args.environment)
    if recorded_environment is not None:
        restored_files["recorded_environment.json"] = recorded_environment
    notebook = build_notebook(sources, data_raw, snapshot_md, readme, quality_raw, restored_files)
    notebook_raw = json_bytes(notebook)
    package_files = {
        NOTEBOOK_NAME: notebook_raw,
        "README.txt": readme.encode("utf-8"),
        "requirements.txt": b"openai\nmatplotlib\n",
        "data/1.exam.json": data_raw,
        "data/1.exam.json": data_raw,
        "assignment2_task1/build_final_notebook.py": Path(__file__).read_bytes(),
        **{"assignment2_task1/" + name: raw for name, raw in sources.items()},
    }
    if quality_raw is not None:
        package_files["data/data_quality_review.json"] = quality_raw
        package_files["assignment2_task1/data_quality_review.json"] = quality_raw
    for name in ("test_runner.py", "test_analyze_results.py", "test_analyze_supplemental.py", "test_final_notebook.py"):
        path = HERE / name
        if path.exists():
            package_files["assignment2_task1/" + name] = path.read_bytes()
    if args.results_root is not None:
        for phase in ("development", "formal"):
            for name in RESULT_NAMES:
                path = args.results_root / phase / name
                if path.exists():
                    package_files[f"results/{phase}/{name}"] = path.read_bytes()
        for folder in (args.results_root / "analysis", args.results_root / "formal/analysis"):
            if folder.is_dir():
                for name in ANALYSIS_NAMES:
                    path = folder / name
                    if path.exists():
                        package_files["analysis/" + name] = path.read_bytes()
                break
        for name in SUPPLEMENTAL_NAMES:
            path = args.results_root / "supplemental" / name
            if path.is_file():
                package_files[f"results/supplemental/{name}"] = path.read_bytes()
        for folder in (args.results_root / "supplemental_analysis", args.results_root / "supplemental/analysis"):
            if folder.is_dir():
                for name in SUPPLEMENTAL_ANALYSIS_NAMES:
                    path = folder / name
                    if path.is_file():
                        package_files["supplemental_analysis/" + name] = path.read_bytes()
                break
    if recorded_environment is not None:
        package_files["recorded_environment.json"] = recorded_environment
    # 只检查文本文件；不把密钥形式的字符串打包进交付物。
    for name, raw in package_files.items():
        if name.endswith((".py", ".ipynb", ".json", ".jsonl", ".txt", ".csv")):
            text = raw.decode("utf-8-sig")
            if re.search(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}", text):
                raise ValueError("检测到疑似密钥，停止打包：" + name)
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "notebook_execution_state": "unexecuted_code_cells_with_read_only_log_snapshot",
        "source_sha256": source_hashes, "data_file_sha256": sha(data_raw),
        "snapshot": snapshot,
        "notebook_embedded_result_files_sha256": {name: sha(raw) for name, raw in restored_files.items()},
        "files_sha256": {name: sha(raw) for name, raw in sorted(package_files.items())},
    }
    manifest_raw = json_bytes(manifest)
    package_files["build_manifest.json"] = manifest_raw
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / NOTEBOOK_NAME).write_bytes(notebook_raw)
    (args.output_dir / "README.txt").write_text(readme, encoding="utf-8")
    (args.output_dir / "build_manifest.json").write_bytes(manifest_raw)
    archive = args.output_dir / PACKAGE_NAME
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        for name, raw in sorted(package_files.items()):
            package.writestr(name, raw)
    print("已构建：", args.output_dir / NOTEBOOK_NAME)
    print("已打包：", archive)
    print("日志快照状态：", snapshot["status"])
    print("全部代码格保持未执行；没有调用API或修改冻结源码/报告。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
