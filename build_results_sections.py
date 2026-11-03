"""Fill report sections from verified, completed real logs; never calls an API.

python build_results_sections.py --results-root ../actual_run/task1_assignment

Reads formal/, development/, analysis/ (or formal/analysis/) and
software_versions.json. Refuses incomplete or inconsistent snapshots, compares
offline recomputation against all supplied analysis tables, and verifies the
frozen source prompts before changing the report. No synthetic-result mode.
The resulting prose is a factual draft for final human/agent editorial review.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
METHODS = ("zero_shot", "few_shot", "cot", "agent")
LABELS = {"zero_shot": "Zero-shot", "few_shot": "Few-shot", "cot": "CoT", "agent": "Agent"}


class ReportError(ValueError):
    pass


def tex(value):
    chars = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
             "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
             "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(chars.get(c, c) for c in str(value))


def pct(value):
    return f"{100 * value:.2f}\\%"


def num(value, decimals=0):
    return f"{value:,.{decimals}f}"


def compact(text, limit=160):
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= limit else text[:limit] + "……"


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def equal(actual, expected, context):
    """Recursive numeric comparison; generated timestamps are excluded upstream."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ReportError(f"{context}: dictionary keys differ")
        for key in expected:
            equal(actual[key], expected[key], context + "." + str(key))
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ReportError(f"{context}: list lengths differ")
        for i, value in enumerate(expected):
            equal(actual[i], value, context + f"[{i}]")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
            raise ReportError(f"{context}: numeric value differs")
    elif actual != expected or isinstance(expected, bool) != isinstance(actual, bool):
        raise ReportError(f"{context}: value differs")


def verify_csv(path, rows):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        actual = list(reader)
    fields = [key for key in rows[0] if key != "returned_model_counts"]
    if reader.fieldnames != fields or len(actual) != len(rows):
        raise ReportError(f"{path.name}: row count or columns differ")
    for index, (got, expected) in enumerate(zip(actual, rows)):
        for key in fields:
            value = expected[key]
            if isinstance(value, float):
                try:
                    parsed = float(got[key])
                except ValueError as error:
                    raise ReportError(f"{path.name}: invalid numeric field") from error
                equal(parsed, value, f"{path.name}:{index + 2}:{key}")
            elif got[key] != ("" if value is None else str(value)):
                raise ReportError(f"{path.name}:{index + 2}:{key}: differs")


def load_analysis_module():
    spec = importlib.util.spec_from_file_location("_report_offline_analysis", HERE.parent / "analyze_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(args):
    root = args.results_root.resolve()
    formal = root / "formal"
    development = root / "development"
    source = args.data or (root / "1.exam.json" if (root / "1.exam.json").exists()
                           else formal / "source_data.json")
    analysis_dir = args.analysis_dir
    if analysis_dir is None:
        analysis_dir = formal / "analysis" if (formal / "analysis").is_dir() else root / "analysis"
    versions_path = args.versions or root / "software_versions.json"
    analyzer = load_analysis_module()
    run = analyzer.load_run(formal, source)
    dev = analyzer.load_run(development, development / "source_data.json", enforce_course_counts=False)
    if len(dev["data"]) != 3 or any(row["status"] != "completed" for row in dev["rows"].values()):
        raise ReportError("Development check must have three inputs and twelve completed predictions.")
    source_sha = hashlib.sha256((HERE.parent / "runner.py").read_bytes()).hexdigest()
    for name, checked in (("formal", run), ("development", dev)):
        if checked["manifest"]["experiment"]["runner_source_sha256"] != source_sha:
            raise ReportError(f"{name} runner hash differs from frozen source")
        if set(checked["starts"]) != set(checked["finishes"]):
            raise ReportError(f"{name} contains unresolved API requests")
    if dev["manifest"]["experiment"]["prompts"] != run["manifest"]["experiment"]["prompts"]:
        raise ReportError("Development/formal prompts differ; narrative needs a separate design history.")
    computed, main, types, changes, cases = analyzer.analyze(run)
    analysis = json_read(analysis_dir / "analysis.json")
    equal({k: v for k, v in analysis.items() if k != "generated_at_utc"},
          {k: v for k, v in computed.items() if k != "generated_at_utc"}, "analysis")
    verify_csv(analysis_dir / "main_table.csv", main)
    verify_csv(analysis_dir / "question_type_table.csv", types)
    verify_csv(analysis_dir / "agent_changes.csv", changes)
    equal(json_read(analysis_dir / "case_candidates.json"), cases, "case_candidates")
    for name in ("accuracy_comparison.pdf", "efficiency_comparison.pdf"):
        if not (analysis_dir / name).read_bytes().startswith(b"%PDF-"):
            raise ReportError(f"Missing or invalid figure: {name}")
    versions = json_read(versions_path)
    if not isinstance(versions, dict):
        raise ReportError("Software versions must be a dictionary")
    if not all(key in versions for key in ("python", "openai", "matplotlib")):
        raise ReportError("Missing Python/OpenAI/Matplotlib environment versions")
    return run, dev, analysis, cases, changes, versions, analysis_dir


def get_metrics(analysis, subset="primary_97"):
    return {row["method"]: row for row in analysis["main_results"] if row["subset"] == subset}


def times(run):
    values = [datetime.fromisoformat(event["timestamp"]) for event in
              list(run["starts"].values()) + list(run["finishes"].values())]
    zone = timezone(timedelta(hours=8))
    return tuple(value.astimezone(zone).strftime("%Y-%m-%d %H:%M:%S") for value in (min(values), max(values)))


def main_tables(analysis):
    primary, sensitivity = get_metrics(analysis), get_metrics(analysis, "sensitivity_90")
    lines = [
        r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}",
        r"\caption{严格集合匹配结果。主集合与敏感性集合共用预测；原始100条口径将缺选项未答题计错，覆盖率只计算合法预测。}",
        r"\label{tab:results}",
        r"\begin{tabular}{lrrrrr}\toprule",
        r"方法 & 正确/97（\%） & 95\% Wilson区间 & 正确/90（\%） & 正确/100 & 覆盖/100\\\midrule",
    ]
    for method in METHODS:
        row, other = primary[method], sensitivity[method]
        lines.append(
            f"{LABELS[method]} & {row['correct']}/97 ({100*row['accuracy']:.2f}) & "
            f"[{100*row['wilson95_lower']:.1f}, {100*row['wilson95_upper']:.1f}] & "
            f"{other['correct']}/90 ({100*other['accuracy']:.2f}) & "
            f"{row['correct']}/100 & {row['completed_predictions']}/100" + r"\\")
    lines.extend([r"\bottomrule\end{tabular}\end{table}",
                  r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}",
                  r"\caption{97题主集合的全部尝试消耗；失败调用也纳入，token由API返回用量累计。}",
                  r"\label{tab:cost}",
                  r"\begin{tabular}{lrrrrrr}\toprule",
                  r"方法 & 调用数（每题） & 输入token & 输出token & 秒/题 & 元（估计） & API/解析失败\\\midrule"])
    for method in METHODS:
        row = primary[method]
        estimate = row["total_cost_conservative_upper_estimate_cny"]
        cost = "--" if estimate is None else f"{estimate:.3f}"
        lines.append(
            f"{LABELS[method]} & {row['api_attempts']} ({row['mean_api_attempts_per_question']:.2f}) & "
            f"{num(row['known_prompt_tokens'])} & {num(row['known_completion_tokens'])} & "
            f"{row['mean_known_api_elapsed_seconds_per_question']:.2f} & {cost} & "
            f"{row['api_error_outcomes']}/{row['parse_error_outcomes']}" + r"\\")
    lines.extend([r"\bottomrule\end{tabular}\end{table}",
                  r"费用按高峰输入未缓存2元/百万token、输出8元/百万token计算，所有输入视为未命中缓存。",
                  r"该值为给定费率下的保守估计，不是账单；缓存折扣和空闲时段半价可能降低实际收费\citep{deepseekpricing}。"])
    unknown = sum(row["calls_with_unknown_token_usage"] for row in primary.values())
    elapsed_unknown = sum(row["calls_with_unknown_elapsed"] for row in primary.values())
    lines.append(f"缺失用量的调用为 {unknown} 次，缺失耗时的调用为 {elapsed_unknown} 次。")
    if unknown:
        lines.append("未知用量使总费用上界无法确定，已知token小计不能作为总费用上界。")
    lines.extend([
        r"\begin{figure}[t]\centering",
        r"\includegraphics[width=.70\linewidth]{figures/accuracy_comparison.pdf}",
        r"\caption{两种评测口径的准确率与95\% Wilson区间。二者样本重叠，不是独立重复。}\label{fig:accuracy}\end{figure}",
        r"\begin{figure}[t]\centering",
        r"\includegraphics[width=.95\linewidth]{figures/efficiency_comparison.pdf}",
        r"\caption{每题调用、累计API耗时及token消耗。耗时不含人工等待和notebook的其他操作。}\label{fig:efficiency}\end{figure}",
    ])
    return "\n".join(lines) + "\n"


def failure_analysis(run):
    causes = Counter()
    action_examples = []
    for (method, index), row in run["rows"].items():
        if row["status"] != "parse_error":
            continue
        calls = [(call_id, start) for call_id, start in run["starts"].items()
                 if start["method"] == method and start["index"] == index
                 and start["question_attempt_id"] == row["question_attempt_id"]]
        call_id, start = calls[-1]
        response = run["finishes"][call_id].get("response") or {}
        if response.get("finish_reason") != "stop":
            causes["未正常结束"] += 1
            continue
        try:
            output = json.loads(response["content"])
        except (ValueError, KeyError):
            causes["无效JSON"] += 1
            continue
        choices = output.get("answer")
        if (isinstance(choices, list) and len(choices) > 1
                and run["data"][index]["question_type"] != "多项选择题"):
            causes["单选返回多字母"] += 1
            continue
        if start["stage"].startswith("controller"):
            payload = json.loads(start["request"]["messages"][-1]["content"])
            before = payload["candidate"]["answer"]
            if output.get("action") == "finish" and choices != before:
                causes["finish改变答案"] += 1
                action_examples.append((index+1, "".join(before), "".join(choices),
                                        "".join(choices) == run["gold"][index]))
                continue
        causes["其他输出约束"] += 1
    if not causes:
        return "本次没有解析或动作协议失败。"
    text = r"\paragraph{执行失败的具体来源。}" + "解析失败按本地约束归类为" + "、".join(
        f"{tex(name)} {count} 次" for name, count in causes.items()) + "。"
    if not causes.get("无效JSON") and not causes.get("未正常结束"):
        text += "这些输出是合法JSON，失败发生在答案或控制协议验证阶段；JSON输出模式不能保证满足全部任务约束。"
    for row, before, after, correct in action_examples:
        text += (f"位置 {row} 的控制器把 {tex(before)} 改为 {tex(after)}，却标记为 "
                 r"\texttt{finish}；" + ("新答案虽与参考一致，" if correct else "") +
                 "仍按冻结规则计失败。未通过修改规则或重试追补得分。")
    return text


def result_analysis(analysis, run):
    main = get_metrics(analysis)
    best_count = max(row["correct"] for row in main.values())
    best = "、".join(LABELS[m] for m in METHODS if main[m]["correct"] == best_count)
    zero, agent, cot = main["zero_shot"], main["agent"], main["cot"]
    lines = [
        f"主集合最高正确数为 {best_count}/97，由 {best} 获得。"
        f"相对 Zero-shot，Few-shot 的准确率差为 {100*(main['few_shot']['accuracy']-zero['accuracy']):+.2f} 个百分点，"
        f"CoT 为 {100*(cot['accuracy']-zero['accuracy']):+.2f} 个百分点，"
        f"Agent 为 {100*(agent['accuracy']-zero['accuracy']):+.2f} 个百分点。"
        "这描述的是本次固定题集的结果，不直接说明提示技术在其他题目上必然有同样效果。",
        r"\paragraph{配对比较与不确定性。}"
        "同题预测采用双侧精确 McNemar 检验，两个集合上的四项预设比较共8项使用 Holm 校正。"
        "下列为主集合；差异题数比单独比较两组置信区间更能反映配对结构：",
    ]
    for pair in analysis["paired_comparisons"]:
        if pair["subset"] != "primary_97":
            continue
        lines.append(
            f"{LABELS[pair['candidate']]} 对 {LABELS[pair['reference']]}："
            f"仅前者正确 {pair['candidate_only_correct']} 题、仅后者正确 {pair['reference_only_correct']} 题，"
            f"原始 $p={pair['mcnemar_exact_two_sided_p']:.3f}$，"
            f"校正 $p={pair['holm_adjusted_p_all_8_comparisons']:.3f}$。")
    significant = [p for p in analysis["paired_comparisons"] if p["holm_adjusted_p_all_8_comparisons"] < .05]
    lines.append("校正后存在低于0.05的比较，仍需结合差异方向和题集局限解释。"
                 if significant else
                 "八项比较均未在校正后的0.05阈值下达到显著；这不能证明方法等效，也不支持宣称可靠提升。")
    lines.append(
        r"\paragraph{准确率与消耗。}"
        f"Agent 的每题调用为 CoT 的 {agent['mean_api_attempts_per_question']/cot['mean_api_attempts_per_question']:.2f} 倍，"
        f"已知token为 {agent['known_total_tokens']/cot['known_total_tokens']:.2f} 倍，"
        f"二者准确率差为 {100*(agent['accuracy']-cot['accuracy']):+.2f} 个百分点。"
        "增加审查是否值得，应结合纠错/改错数量和这些消耗共同判断。")
    lines.append(failure_analysis(run))
    return "\n".join(lines) + "\n"


def ablation(analysis, changes):
    value = analysis["agent_initial_final"]["primary_97"]
    n = value["initial_available_count"]
    correct = value["initial_correct_count_on_available"]
    final = value["final_correct_count_on_same_available"]
    actions = value["controller_action_counts_in_completed_traces"]
    action_total = sum(actions.values())
    rounds = Counter(c["completed_review_rounds_in_trace"] for c in changes if c["final_status"] == "completed")
    lines = [
        f"在能恢复合法初答的 {n} 题上，Agent 初答正确 {correct} 题（{pct(correct/n)}），"
        f"最终正确 {final} 题（{pct(final/n)}）；"
        f"初答缺失 {value['initial_unavailable_count']} 题。"
        f"其中错转对 {value['wrong_to_correct_count']} 题，对转错且正常返回 {value['correct_to_wrong_completed_count']} 题，"
        f"正确初答随后执行失败 {value['correct_to_execution_failure_count']} 题。"
        f"另有 {value['wrong_to_wrong_completed_count']} 题正常结束但仍错误，"
        f"{value['wrong_to_execution_failure_count']} 题的错误初答随后执行失败。"
        "这是同一次 Agent 运行的内部轨迹比较，能直接识别审查环节的改变；"
        "它不是另一次“关闭复核”的独立实验，不能把它与独立 CoT 组的初答混为一谈。",
        f"完成轨迹中的控制动作共 {action_total} 次："
        f"finish {actions.get('finish', 0)} 次，revise {actions.get('revise', 0)} 次。"
        "完成题目的复核轮次分布为"
        + "、".join(f"{count} 题完成 {round_number} 轮" for round_number, count in sorted(rounds.items())) + "。"
        "失败的部分轨迹可能不含完整控制动作，因此上述动作统计只覆盖已保存的完整轨迹。",
    ]
    primary, sensitive = get_metrics(analysis), get_metrics(analysis, "sensitivity_90")
    lines.append("从97条切换到90条口径后，各组准确率变化（百分点）依次为 "
                 + "，".join(f"{LABELS[m]} {100*(sensitive[m]['accuracy']-primary[m]['accuracy']):+.2f}" for m in METHODS)
                 + "。排除7条风险记录改变的是输入构成；这并非删去模型答错题，也不能保证剩余90条没有其他质量问题。")
    return "\n\n".join(lines) + "\n"


def choose_cases(run, candidates):
    chosen, seen = [], set()
    order = ("agent_correction", "agent_regression", "agent_persistent_error", "method_disagreement")
    for category in order:
        for item in candidates["categories"][category]["selected"]:
            trace = item["methods"]["agent"].get("trace") or []
            if item["row_number"] in seen or not any(e["stage"] == "review" for e in trace):
                continue
            chosen.append((category, item))
            seen.add(item["row_number"])
            break
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        for index in sorted(run["subsets"]["primary_97"]):
            row = run["rows"][("agent", index)]
            trace = row.get("trace") or []
            if index + 1 in seen or not any(e["stage"] == "review" for e in trace):
                continue
            data = run["data"][index]
            chosen.append(("completed_trace", {
                "row_number": index+1, "question_id": row["question_id"],
                "question_type": data["question_type"], "question": data["question"], "options": data["option"],
                "standard_answer_from_explicit_data": run["gold"][index],
                "in_sensitivity_90": index in run["subsets"]["sensitivity_90"],
                "methods": {m: run["rows"][(m, index)] for m in METHODS}}))
            seen.add(index+1)
            if len(chosen) == 2:
                break
    if len(chosen) != 2:
        raise ReportError("Fewer than two actual cases with review actions; cannot fill required cases.")
    return chosen


def cases_section(run, candidates):
    chosen = choose_cases(run, candidates)
    titles = {"agent_correction": "纠错案例", "agent_regression": "改错案例",
              "agent_persistent_error": "持续错误", "method_disagreement": "方法分歧",
              "completed_trace": "完成轨迹"}
    lines = [
        "案例按预先定义类别的最小数据位置选择，优先展示纠错与改错，"
        "不存在的类别不补造；以下为描述性案例，不能代表总体频率。"
        "引号内容为模型输出的节选，不当作经过事实核验的医学解释。",
    ]
    for category, item in chosen:
        row = item["methods"]["agent"]
        trace = row["trace"]
        initial = next(e["draft"] for e in trace if e["stage"] == "initial")
        initial_reason = initial.get("brief_reason", "")
        reason_label = "依据"
        reason_excerpt = compact(initial_reason, 180)
        conclusion_marker = "因此选" + str(row.get("prediction"))
        if category == "agent_correction" and initial_reason.endswith(conclusion_marker + "。"):
            reason_label = "依据末句"
            reason_excerpt = initial_reason[initial_reason.rfind(conclusion_marker):]
        lines.append(
            rf"\paragraph{{{titles[category]}：位置 {item['row_number']}（{tex(item['question_type'])}）。}}"
            + tex(item["question"]) + " 选项：" +
            tex("；".join(k+"："+v for k,v in item["options"].items() if isinstance(v, str) and v.strip())) + "。")
        lines.append("课程参考标签为 " + tex(item["standard_answer_from_explicit_data"]) + "；普通提示输出为 " +
                     "、".join(f"{LABELS[m]} " + tex(item["methods"][m].get("prediction") or
                         item["methods"][m]["status"]) for m in METHODS[:3]) + "。"
                     + "Agent 初答 " + tex("".join(initial["answer"])) + "，" + reason_label + "：“"
                     + tex(reason_excerpt) + "”。")
        for event in trace:
            if event["stage"] != "review":
                continue
            feedback = event["feedback"]
            issues = "；".join(feedback.get("issues", [])) or "未列出具体问题"
            lines.append(
                f"第 {event['round']} 轮：审查建议 {tex(feedback['recommendation'])}，反馈“{tex(compact(issues, 130))}”；"
                f"控制器选择 \\texttt{{{tex(event['action'])}}}，答案 {tex(''.join(event['draft']['answer']))}，"
                f"决策理由“{tex(compact(event['decision_reason'], 100))}”。")
        lines.append("最终答案 " + tex(row.get("prediction") or row["status"]) +
                     (" 与参考一致" if row["correct"] else " 与参考不一致") +
                     "，停止原因为 \\texttt{" + tex(row["stop_reason"]) + "}。"
                     + ("该题保留在90条敏感性集合中。" if item["in_sensitivity_90"]
                        else "该题属于预先标记的缺图/上下文风险记录，不能仅凭反馈推断其事实依据可靠。"))
        if category == "agent_correction":
            reason_choices = re.findall(r"因此选([A-E]+)", initial.get("brief_reason", ""))
            if reason_choices and reason_choices[-1] == row["prediction"]:
                lines.append("这里的可观察纠错信号是答案字段与模型自己的理由不一致。"
                             "审查使二者重新一致，无需引入新的外部知识；"
                             "这说明反馈能够修复表达/选择不一致，不能据此推断审查已独立核实专业知识。")
        elif category == "agent_persistent_error":
            missing = set(item["standard_answer_from_explicit_data"]) - set(row["prediction"])
            extras = set(row["prediction"]) - set(item["standard_answer_from_explicit_data"])
            mismatch = []
            if missing:
                mismatch.append("漏选 " + "、".join(sorted(missing)))
            if extras:
                mismatch.append("多选 " + "、".join(sorted(extras)))
            lines.append("该例的终答" + "，".join(mismatch) + "，严格集合评分因此判错。"
                         "审查建议修订并未强制控制器采纳；控制器自主提交了原答案，"
                         "说明自审能够提出质疑，但共享的知识判断并不保证纠错。")
    return "\n\n".join(lines) + "\n", [item["row_number"] for _, item in chosen]


def macros(run, dev, analysis):
    main = get_metrics(analysis)
    best = max(row["correct"] for row in main.values())
    labels = "、".join(LABELS[m] for m in METHODS if main[m]["correct"] == best)
    counts = "、".join(f"{LABELS[m]} {main[m]['correct']}/97" for m in METHODS)
    development = "、".join(f"{LABELS[m]} {dev['summary']['methods'][m]['correct_count']}/3" for m in METHODS)
    token = sum(v["known_total_tokens"] for v in dev["summary"]["methods"].values())
    significant = any(x["holm_adjusted_p_all_8_comparisons"] < .05 for x in analysis["paired_comparisons"])
    finding = f"四组主集合结果分别为 {counts}。{labels} 获得最高正确数；"
    finding += ("配对检验的具体差异方向与显著性见正文。" if significant
                else "配对比较经多重校正后均未达到0.05显著性阈值。")
    finding += "Agent 的额外审查同时可能纠错和维持错误，须结合日志与消耗评估。"
    conclusion = (
        f"本次数据上 {labels} 获得最高正确数 {best}/97。"
        "本实验支持比较这些具体提示和实现的行为，不支持将它们的单次排名外推为方法的普遍优劣。"
        "后续可在独立开发集上设计更贴近药学的示例、补齐题目材料并进行多次重复；"
        "这些工作没有在本轮结果中冒充已完成实验。")
    development_text = (
        "独立开发检查使用三道不同于正式考试及few-shot示例的人工题，分别覆盖否定问法、完整多选和库存算术。"
        f"正确数为 {development}，12个输出均正常返回和解析；"
        f"共 {dev['summary']['cumulative_api_attempts']} 次调用、{token} token。"
        "开发题成绩仅说明这些输入上的行为，不估计真实考试能力。"
        "单题连通测试仅用于确认API与解析。开发检查后保持提示与参数不变，"
        "保留开发错误，未进行或声称进行多版本迭代调参。")
    return "\n".join([
        "% Generated only after full ledger, manifest, analysis and source validation.",
        r"\newcommand{\ReportStatus}{Task 1 实验报告 · 实际日志已校验}",
        r"\newcommand{\AbstractFinding}{" + finding + "}",
        r"\newcommand{\DevelopmentRecord}{\par\noindent " + development_text + "}",
        r"\newcommand{\FinalConclusion}{" + conclusion + "}", "",
    ])


def settings_section(run, analysis):
    start, end = times(run)
    returned = Counter()
    for row in get_metrics(analysis).values():
        returned.update(row["returned_model_counts"])
    text = (f"正式调用起止时间为 {start} 至 {end}（UTC+8）。"
            "请求使用模型别名，响应中的模型名称与次数为 "
            + "、".join(f"\\texttt{{{tex(k)}}}（{v}次）" for k, v in sorted(returned.items())) + "。"
            "返回名称不必然揭示不可变的模型快照；代码、数据与实验指纹见附录。")
    path = HERE / "generated/settings.tex"
    current = path.read_text(encoding="utf-8")
    pattern = r"(?s)% BEGIN ACTUAL_RUN_SETTINGS.*?% END ACTUAL_RUN_SETTINGS"
    if len(re.findall(pattern, current)) != 1:
        raise ReportError("settings.tex actual-run block marker missing or duplicated")
    return re.sub(pattern, lambda match: "% BEGIN ACTUAL_RUN_SETTINGS\n" + text
                  + "\n% END ACTUAL_RUN_SETTINGS", current)


def record_section(run, dev, analysis, versions):
    main = get_metrics(analysis)
    calls = sum(x["api_attempts"] for x in main.values())
    tokens = sum(x["known_total_tokens"] for x in main.values())
    api_failures = sum(x["api_error_outcomes"] for x in main.values())
    parse_failures = sum(x["parse_error_outcomes"] for x in main.values())
    events = [json.loads(line) for line in
        (Path(run["report_run_dir"]) / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    retries = sum(e["event"] == "question_start" and e.get("retry_of") is not None for e in events)
    version_text = "；".join(tex(k) + " " + tex(str(versions[k]).split()[0])
                            for k in ("python", "openai", "matplotlib", "numpy", "scipy") if k in versions)
    lines = [
        f"正式四组共 {calls} 次API尝试、{tokens:,} 个已知token；"
        f"API失败 {api_failures} 个，协议解析失败 {parse_failures} 个，显式题目重试 {retries} 次。",
        "实际软件环境：" + version_text + "。",
        r"\begin{flushleft}\footnotesize",
        r"实验指纹：\nolinkurl{" + run["manifest"]["fingerprint"] + r"}",
        r"\end{flushleft}",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--versions", type=Path)
    args = parser.parse_args(argv)
    try:
        run, dev, analysis, candidates, changes, versions, analysis_dir = validate(args)
        run["report_run_dir"] = str(args.results_root / "formal")
        case_text, case_rows = cases_section(run, candidates)
        generated = {
            "settings.tex": settings_section(run, analysis),
            "results_table.tex": main_tables(analysis),
            "results_analysis.tex": result_analysis(analysis, run),
            "ablation.tex": ablation(analysis, changes),
            "cases.tex": case_text,
            "results_macros.tex": macros(run, dev, analysis),
            "run_record.tex": record_section(run, dev, analysis, versions),
        }
        if any(r"\Pending" in text for text in generated.values()):
            raise ReportError("A generated result section still contains a pending value")
        subprocess.run([sys.executable, str(HERE / "build_prompt_appendix.py"),
                        str(args.results_root / "formal/manifest.json")], check=True)
        figures = HERE / "figures"
        figures.mkdir(exist_ok=True)
        for name in ("accuracy_comparison.pdf", "efficiency_comparison.pdf"):
            shutil.copyfile(analysis_dir / name, figures / name)
        for filename, text in generated.items():
            (HERE / "generated" / filename).write_text(text, encoding="utf-8")
        provenance = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_formal_directory": str((args.results_root / "formal").resolve()),
            "fingerprint": run["manifest"]["fingerprint"],
            "all_four_methods_complete": True,
            "ledger_and_analysis_recomputed_and_matched": True,
            "prompt_manifest_matched_source": True,
            "selected_case_rows_1_based": case_rows,
            "editorial_review_required": True,
            "generated_files_sha256": {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                                       for name, text in generated.items()},
        }
        (HERE / "report_build_record.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"Report generation stopped: {error}\n")
    print("Validated real formal/development logs and all supplied analysis tables.")
    print("Filled eight generated sections and copied two figures; editorial/layout review remains.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
