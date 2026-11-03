"""Offline analysis of completed Task 1 runs; never calls an API.

Usage (in Colab or a local terminal):
    python analyze_results.py --run-dir task1_results --data course_task1.json

The CLI requires all four methods on the preregistered 97 inputs. Explicit
API/parse failures count as incorrect. Unfinished runs fail closed and create
no analysis outputs. Gold labels are read only for offline scoring and cases.
Prices are supplied analysis assumptions, not a live quote or an invoice.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any

METHODS = ("zero_shot", "few_shot", "cot", "agent")
LABELS = {"zero_shot": "Zero-shot", "few_shot": "Few-shot", "cot": "CoT", "agent": "Agent"}
TYPES = ("最佳选择题", "配伍选择题", "综合分析选择题", "多项选择题")
TERMINAL = {"completed", "api_error", "parse_error"}
EMPTY_ROWS = {21, 26, 66}
SENSITIVITY_ROWS = {11, 27, 28, 56, 58, 67, 91}
INPUT_CNY_PER_MILLION = 2.0
OUTPUT_CNY_PER_MILLION = 8.0


class AnalysisError(ValueError):
    """Missing, inconsistent, incomplete, or unregistered experiment data."""


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise AnalysisError(f"Incomplete last line: {path}")
    rows = []
    for number, line in enumerate(raw.splitlines(), 1):
        try:
            item = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise AnalysisError(f"Invalid JSON at {path}:{number}") from exc
        if not isinstance(item, dict):
            raise AnalysisError(f"Expected object at {path}:{number}")
        rows.append(item)
    return rows


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if not 0 <= successes <= n or n <= 0 or not 0 < confidence < 1:
        raise ValueError("Invalid Wilson interval arguments")
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact(improved: int, worsened: int) -> float:
    """Two-sided conditional binomial test on discordant pairs only."""
    if improved < 0 or worsened < 0:
        raise ValueError("Discordant counts cannot be negative")
    discordant = improved + worsened
    if not discordant:
        return 1.0
    numerator = sum(math.comb(discordant, k) for k in range(min(improved, worsened) + 1))
    return min(1.0, 2 * numerator / (2 ** discordant))


def holm_adjust(pvalues: list[float]) -> list[float]:
    adjusted = [0.0] * len(pvalues)
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        previous = max(previous, min(1.0, pvalues[index] * (len(pvalues) - rank)))
        adjusted[index] = previous
    return adjusted


def normalize_answer(value: Any, item: dict[str, Any]) -> str:
    choices = list(value) if isinstance(value, str) else value
    valid = {k for k, v in item["option"].items() if isinstance(v, str) and v.strip()}
    if (not isinstance(choices, list) or not choices
            or any(not isinstance(x, str) or x not in valid for x in choices)
            or len(choices) != len(set(choices))
            or (item["question_type"] != "多项选择题" and len(choices) != 1)):
        raise AnalysisError("Invalid answer letters for the input question")
    return "".join(sorted(choices))


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def index_ledger(events: list[dict[str, Any]], fingerprint: str) -> tuple[dict, dict, dict]:
    starts, finishes, latest = {}, {}, {}
    for event in events:
        if event.get("fingerprint") != fingerprint:
            raise AnalysisError("Ledger fingerprint differs from manifest")
        kind = event.get("event")
        if kind in {"call_start", "call_finish"}:
            target = starts if kind == "call_start" else finishes
            call_id = event.get("call_id")
            if not call_id or call_id in target:
                raise AnalysisError("Missing or duplicate ledger call ID")
            target[call_id] = event
        elif kind == "question_result":
            latest[(event["method"], event["index"])] = event["result"]
    if set(finishes) - set(starts):
        raise AnalysisError("Ledger finish has no corresponding start")
    return starts, finishes, latest


def call_statistics(starts: dict, finishes: dict, method: str, indices: set[int]) -> dict[str, Any]:
    counts = Counter()
    elapsed = 0.0
    returned_models = Counter()
    for call_id, start in starts.items():
        if start["method"] != method or start["index"] not in indices:
            continue
        counts["api_attempts"] += 1
        finish = finishes.get(call_id)
        if finish is None:
            counts["unresolved_calls"] += 1
            counts["calls_with_unknown_token_usage"] += 1
            counts["calls_with_unknown_elapsed"] += 1
            continue
        duration = finish.get("elapsed_seconds")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool) and math.isfinite(duration) and duration >= 0:
            elapsed += duration
        else:
            counts["calls_with_unknown_elapsed"] += 1
        if finish.get("status") == "api_error":
            counts["api_error_calls"] += 1
        response = finish.get("response") or {}
        if response.get("model_returned"):
            returned_models[str(response["model_returned"])] += 1
        usage = response.get("usage") or {}
        if not all(_nonnegative_int(usage.get(field)) for field in ("prompt_tokens", "completion_tokens")):
            counts["calls_with_unknown_token_usage"] += 1
        for source, target in (("prompt_tokens", "known_prompt_tokens"), ("completion_tokens", "known_completion_tokens")):
            if _nonnegative_int(usage.get(source)):
                counts[target] += usage[source]
    estimate = (counts["known_prompt_tokens"] * INPUT_CNY_PER_MILLION
                + counts["known_completion_tokens"] * OUTPUT_CNY_PER_MILLION) / 1_000_000
    unknown = counts["calls_with_unknown_token_usage"]
    fields = ("api_attempts", "unresolved_calls", "api_error_calls", "calls_with_unknown_token_usage",
              "calls_with_unknown_elapsed", "known_prompt_tokens", "known_completion_tokens")
    return {
        **{field: counts[field] for field in fields},
        "known_total_tokens": counts["known_prompt_tokens"] + counts["known_completion_tokens"],
        "known_api_elapsed_seconds": elapsed,
        "known_tokens_conservative_estimate_cny": estimate,
        "total_cost_conservative_upper_estimate_cny": None if unknown else estimate,
        "cost_usage_complete": not bool(unknown),
        "returned_model_counts": dict(returned_models),
    }


def load_run(run_dir: Path, data_path: Path, *, enforce_course_counts: bool = True) -> dict[str, Any]:
    """Validate snapshots against the manifest, explicit source data, and ledger."""
    data = json.loads(data_path.read_text(encoding="utf-8-sig"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    results = read_jsonl(run_dir / "results.jsonl")
    events = read_jsonl(run_dir / "ledger.jsonl")
    experiment = manifest["experiment"]
    fingerprint = manifest["fingerprint"]
    if canonical_hash(experiment) != fingerprint:
        raise AnalysisError("Manifest payload does not match its fingerprint")
    if canonical_hash(data) != experiment.get("dataset_sha256"):
        raise AnalysisError("Explicit data file differs from preregistered dataset")
    if summary.get("fingerprint") != fingerprint:
        raise AnalysisError("Summary fingerprint differs from manifest")
    if set(experiment.get("methods", [])) != set(METHODS):
        raise AnalysisError("All four preregistered methods are required")
    selected = set(experiment["evaluation_indices_0_based"])
    excluded = {int(i) for i in experiment.get("exclusions", {})}
    main = selected - excluded
    risk_rows = set(experiment["sensitivity_excluded_rows_1_based"])
    sensitivity = {i for i in main if i + 1 not in risk_rows}
    if enforce_course_counts:
        if (len(data) != 100 or selected != set(range(100))
                or {i + 1 for i in excluded} != EMPTY_ROWS
                or risk_rows != SENSITIVITY_ROWS or len(main) != 97 or len(sensitivity) != 90):
            raise AnalysisError("Expected preregistered 100 raw / 97 primary / 90 sensitivity inputs")
        for i in excluded:
            if any(isinstance(v, str) and v.strip() for v in data[i]["option"].values()):
                raise AnalysisError("An excluded course row no longer has all-empty options")
    if not main or not sensitivity:
        raise AnalysisError("Evaluation subsets cannot be empty")
    starts, finishes, latest = index_ledger(events, fingerprint)
    rows = {}
    for row in results:
        key = (row["method"], row["index"])
        if key in rows or row.get("fingerprint") != fingerprint:
            raise AnalysisError("Duplicate result or incompatible fingerprint")
        rows[key] = row
    if set(rows) != {(method, i) for method in METHODS for i in selected}:
        raise AnalysisError("Results must contain exactly one row per selected method/question")
    gold = {i: normalize_answer(data[i]["answer"].strip(), data[i]) for i in main}
    for method in METHODS:
        for i in selected:
            row = rows[(method, i)]
            if i in excluded:
                if row["status"] != "excluded":
                    raise AnalysisError("Excluded input has an unexpected result status")
                continue
            if row["status"] not in TERMINAL:
                raise AnalysisError(f"Run incomplete: {method}, row {i + 1}, status {row['status']}. No final analysis written.")
            recorded = latest.get((method, i))
            if recorded is None:
                raise AnalysisError("Terminal result is missing from the ledger")
            for field in ("status", "prediction", "reference", "correct", "question_attempt_id", "trace"):
                if row.get(field) != recorded.get(field):
                    raise AnalysisError(f"Result/ledger mismatch in {field}; export may be stale")
            if row.get("reference") != gold[i]:
                raise AnalysisError("Recorded reference differs from explicit source data")
            prediction = normalize_answer(row["prediction"], data[i]) if row["status"] == "completed" else None
            correct = row["status"] == "completed" and prediction == gold[i]
            if row.get("correct") is not correct:
                raise AnalysisError("Recorded correctness disagrees with offline rescoring")
            row["correct"] = correct
            if row.get("question_type") != data[i]["question_type"]:
                raise AnalysisError("Question type differs from explicit source data")
        method_summary = summary.get("methods", {}).get(method, {})
        true_correct = sum(rows[(method, i)]["correct"] for i in main)
        if (not method_summary.get("all_valid_inputs_finished")
                or method_summary.get("valid_input_count") != len(main)
                or method_summary.get("correct_count") != true_correct
                or method_summary.get("sensitivity_count") != len(sensitivity)):
            raise AnalysisError("Summary is incomplete or disagrees with results; re-export runner outputs")
    for start in starts.values():
        if start.get("method") not in METHODS or start.get("index") not in main:
            raise AnalysisError("Ledger includes calls outside the registered comparison")
    if summary.get("cumulative_api_attempts") != len(starts):
        raise AnalysisError("Summary call count differs from ledger")
    if not starts:
        raise AnalysisError("No API attempt events; no experimental results to analyze")
    return {"data": data, "manifest": manifest, "summary": summary, "rows": rows,
            "starts": starts, "finishes": finishes, "gold": gold,
            "subsets": {"primary_97": main, "sensitivity_90": sensitivity},
            "source_file_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest()}


def metrics_for(run: dict, method: str, indices: set[int]) -> dict[str, Any]:
    records = [run["rows"][(method, i)] for i in sorted(indices)]
    correct = sum(r["correct"] for r in records)
    lower, upper = wilson_interval(correct, len(records))
    stats = call_statistics(run["starts"], run["finishes"], method, indices)
    counts = Counter(r["status"] for r in records)
    return {
        "method": method, "n": len(records), "correct": correct,
        "accuracy": correct / len(records), "wilson95_lower": lower, "wilson95_upper": upper,
        "completed_predictions": counts["completed"], "api_error_outcomes": counts["api_error"],
        "parse_error_outcomes": counts["parse_error"],
        **stats,
        "mean_api_attempts_per_question": stats["api_attempts"] / len(records),
        "mean_known_api_elapsed_seconds_per_question": stats["known_api_elapsed_seconds"] / len(records),
        "mean_known_tokens_per_question": stats["known_total_tokens"] / len(records),
    }


def paired_comparison(run: dict, candidate: str, reference: str, indices: set[int]) -> dict[str, Any]:
    both_correct = both_wrong = improved = worsened = 0
    for i in indices:
        a, b = run["rows"][(candidate, i)]["correct"], run["rows"][(reference, i)]["correct"]
        both_correct += a and b
        both_wrong += not a and not b
        improved += a and not b
        worsened += b and not a
    return {"candidate": candidate, "reference": reference, "n": len(indices),
            "both_correct": both_correct, "both_wrong": both_wrong,
            "candidate_only_correct": improved, "reference_only_correct": worsened,
            "accuracy_difference_percentage_points": 100 * (improved - worsened) / len(indices),
            "mcnemar_exact_two_sided_p": mcnemar_exact(improved, worsened)}


def initial_agent_answer(run: dict, index: int) -> tuple[str | None, str]:
    row = run["rows"][("agent", index)]
    for event in row.get("trace", []):
        if event.get("stage") == "initial":
            return normalize_answer(event["draft"]["answer"], run["data"][index]), "result_trace"
    # Runner has no completed trace if a later critic/controller call failed.
    # Recover only the initial call from this exact question attempt, never a retry.
    for call_id, start in run["starts"].items():
        if (start.get("method") == "agent" and start.get("index") == index
                and start.get("stage") == "initial"
                and start.get("question_attempt_id") == row.get("question_attempt_id")):
            finish = run["finishes"].get(call_id, {})
            response = finish.get("response") or {}
            if finish.get("status") != "ok" or response.get("finish_reason") != "stop":
                continue
            try:
                content = json.loads(response.get("content", ""))
                return normalize_answer(content["answer"], run["data"][index]), "same_attempt_ledger"
            except (ValueError, KeyError, TypeError):
                continue
    return None, "unavailable"


def analyze(run: dict) -> tuple[dict, list, list, list, dict]:
    main_table, type_table, paired = [], [], []
    for subset, indices in run["subsets"].items():
        for method in METHODS:
            main_table.append({"subset": subset, **metrics_for(run, method, indices)})
            for kind in TYPES:
                typed = {i for i in indices if run["data"][i]["question_type"] == kind}
                if typed:
                    type_table.append({"subset": subset, "question_type": kind, **metrics_for(run, method, typed)})
        for candidate, reference in (("few_shot", "zero_shot"), ("cot", "zero_shot"),
                                     ("agent", "zero_shot"), ("agent", "cot")):
            paired.append({"subset": subset, **paired_comparison(run, candidate, reference, indices)})
    adjusted = holm_adjust([x["mcnemar_exact_two_sided_p"] for x in paired])
    for result, adjusted_p in zip(paired, adjusted):
        result["holm_adjusted_p_all_8_comparisons"] = adjusted_p
    changes = []
    for i in sorted(run["subsets"]["primary_97"]):
        row = run["rows"][("agent", i)]
        initial, provenance = initial_agent_answer(run, i)
        first_correct = None if initial is None else initial == run["gold"][i]
        final = row["prediction"] if row["status"] == "completed" else None
        if initial is None:
            category = "initial_unavailable"
        elif first_correct and row["correct"]:
            category = "both_correct"
        elif not first_correct and row["correct"]:
            category = "wrong_to_correct"
        elif first_correct and not row["correct"]:
            category = "correct_to_wrong_or_failed"
        else:
            category = "both_wrong_or_final_failed"
        reviews = [x for x in row.get("trace", []) if x.get("stage") == "review"]
        changes.append({
            "row_number": i + 1, "question_id": row["question_id"],
            "question_type": run["data"][i]["question_type"],
            "in_sensitivity_90": i in run["subsets"]["sensitivity_90"],
            "initial_prediction": initial, "initial_source": provenance,
            "initial_correct": first_correct, "final_prediction": final,
            "final_correct": row["correct"], "final_status": row["status"],
            "reference": run["gold"][i], "transition": category,
            "answer_unchanged": initial == final if initial is not None and final is not None else None,
            "completed_review_rounds_in_trace": len(reviews),
            "controller_actions": ";".join(x.get("action", "") for x in reviews),
            "stop_reason": row.get("stop_reason"),
        })
    transitions = {}
    for subset, indices in run["subsets"].items():
        selected = [c for c in changes if c["row_number"] - 1 in indices]
        available = [c for c in selected if c["initial_correct"] is not None]
        transitions[subset] = {
            "n": len(selected), "initial_available_count": len(available),
            "initial_unavailable_count": len(selected) - len(available),
            "transition_counts": dict(Counter(c["transition"] for c in selected)),
            "answer_unchanged_count": sum(c["answer_unchanged"] is True for c in selected),
            "answer_changed_count": sum(c["answer_unchanged"] is False for c in selected),
            "wrong_to_correct_count": sum(c["initial_correct"] is False and c["final_correct"] for c in selected),
            "correct_to_wrong_completed_count": sum(c["initial_correct"] is True and not c["final_correct"]
                                                     and c["final_status"] == "completed" for c in selected),
            "correct_to_execution_failure_count": sum(c["initial_correct"] is True
                                                       and c["final_status"] != "completed" for c in selected),
            "wrong_to_wrong_completed_count": sum(c["initial_correct"] is False and not c["final_correct"]
                                                   and c["final_status"] == "completed" for c in selected),
            "wrong_to_execution_failure_count": sum(c["initial_correct"] is False
                                                     and c["final_status"] != "completed" for c in selected),
            "initial_correct_count_on_available": sum(c["initial_correct"] for c in available),
            "final_correct_count_on_same_available": sum(c["final_correct"] for c in available),
            "initial_accuracy_on_available": sum(c["initial_correct"] for c in available) / len(available) if available else None,
            "final_accuracy_on_same_available": sum(c["final_correct"] for c in available) / len(available) if available else None,
            "controller_action_counts_in_completed_traces": dict(Counter(
                action for c in selected for action in c["controller_actions"].split(";") if action)),
            "stop_reason_counts": dict(Counter(str(c["stop_reason"]) for c in selected)),
            "note": "Within-agent initial/final transitions, not the independently sampled CoT baseline. Final API/parse failure is incorrect. Missing initial predictions remain unavailable. Review actions from failed partial traces may be absent.",
        }
    cases = select_cases(run, changes)
    analysis = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fingerprint": run["manifest"]["fingerprint"],
        "source_file_sha256": run["source_file_sha256"],
        "dataset_canonical_sha256": run["manifest"]["experiment"]["dataset_sha256"],
        "model_requested": run["manifest"]["experiment"]["model"],
        "settings": run["manifest"]["experiment"]["settings"],
        "subsets": {key: {"n": len(value), "rows_1based": [i + 1 for i in sorted(value)]} for key, value in run["subsets"].items()},
        "main_results": main_table, "question_type_results": type_table,
        "paired_comparisons": paired, "agent_initial_final": transitions,
        "cost_assumptions": {
            "currency": "CNY", "input_uncached_per_million_tokens": INPUT_CNY_PER_MILLION,
            "output_per_million_tokens": OUTPUT_CNY_PER_MILLION,
            "all_input_treated_as_cache_miss": True,
            "description": "Conservative peak-rate estimate under supplied rates (input 2 CNY/M, output 8 CNY/M), not a bill. Cache discounts are not applied. All attempts and explicit retries are included. Unknown usage prevents reporting a total upper estimate; the known-token subtotal is NOT a bound on total cost.",
        },
        "statistical_notes": [
            "95% Wilson intervals describe accuracy uncertainty on each set; no independence claim between paired methods or overlapping 97/90 sets.",
            "McNemar is the exact two-sided binomial test on discordant pairs, using stdlib math; p=1 when no discordant pairs.",
            "Holm correction uses all eight planned comparisons as one family. Sensitivity results reuse the primary predictions and are not independent replications.",
            "Small convenience sample and related/shared exam stems limit generalization. Non-significant results do not establish equivalence; descriptive gains do not alone establish reliable improvement.",
            "The 97-input set contains known missing-visual/context risks; the 90-input sensitivity set excludes the seven preregistered flagged rows, not errors selected after seeing predictions.",
            "API and parsing failures stay in the effective denominator and count as incorrect. Unfinished runs are rejected, not assigned fabricated final outcomes.",
            "Elapsed values are sums of known API request durations including retries, not end-to-end human or notebook completion time. Unknown durations are flagged.",
        ],
    }
    return analysis, main_table, type_table, changes, cases


def select_cases(run: dict, changes: list[dict], limit: int = 3) -> dict:
    pools = {"agent_correction": [], "agent_regression": [], "agent_persistent_error": [],
             "agent_execution_failure": [], "method_disagreement": []}
    for change in changes:
        i = change["row_number"] - 1
        predictions = {method: run["rows"][(method, i)] for method in METHODS}
        categories = []
        if change["transition"] == "wrong_to_correct":
            categories.append("agent_correction")
        if change["transition"] == "correct_to_wrong_or_failed" and change["final_status"] == "completed":
            categories.append("agent_regression")
        if change["transition"] == "both_wrong_or_final_failed" and change["final_status"] == "completed":
            categories.append("agent_persistent_error")
        if change["final_status"] != "completed":
            categories.append("agent_execution_failure")
        if len({(r["status"], r.get("prediction")) for r in predictions.values()}) > 1:
            categories.append("method_disagreement")
        for category in categories:
            pools[category].append(i)
    def case(i):
        item = run["data"][i]
        agent_initial, initial_source = initial_agent_answer(run, i)
        return {"row_number": i + 1, "question_id": f"q{i + 1:03d}",
                "question_type": item["question_type"], "question": item["question"],
                "options": item["option"], "source": item.get("source"),
                "standard_answer_from_explicit_data": run["gold"][i],
                "agent_initial_prediction": agent_initial, "agent_initial_source": initial_source,
                "in_sensitivity_90": i in run["subsets"]["sensitivity_90"],
                "methods": {method: {key: run["rows"][(method, i)].get(key) for key in
                    ("status", "prediction", "correct", "trace", "stop_reason", "error_code", "question_attempt_id")}
                    for method in METHODS}}
    return {
        "purpose": "Offline report analysis only; standard answers must never be sent back to a model during evaluation.",
        "selection_rule": "Up to three lowest dataset positions in each observed category; categories may overlap. Regression/persistent-error cases require a completed final prediction; execution failures are separate. Empty categories remain empty. These are descriptive candidates, not proof of typical behavior.",
        "categories": {name: {"total_observed": len(indices), "selected": [case(i) for i in indices[:limit]]}
                       for name, indices in pools.items()},
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise AnalysisError("Cannot write an empty analysis table")
    fields = [k for k in rows[0] if k != "returned_model_counts"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_figures(main_table: list[dict], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    lookup = {(row["subset"], row["method"]): row for row in main_table}
    colors = ["#2563A6", "#D87535"]
    fig, ax = plt.subplots(figsize=(7.3, 4.2), layout="constrained")
    for offset, subset, color, label in ((-0.18, "primary_97", colors[0], "Primary text-options set (n=97)"),
                                          (0.18, "sensitivity_90", colors[1], "Sensitivity set (n=90)")):
        selected = [lookup[(subset, method)] for method in METHODS]
        values = [r["accuracy"] * 100 for r in selected]
        errors = [[100 * (r["accuracy"] - r["wilson95_lower"]) for r in selected],
                  [100 * (r["wilson95_upper"] - r["accuracy"]) for r in selected]]
        ax.bar([i + offset for i in range(4)], values, width=0.34, color=color,
               label=label, yerr=errors, capsize=3, error_kw={"linewidth": 1})
    ax.set_xticks(range(4), [LABELS[m] for m in METHODS])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Exact-match accuracy (%)")
    ax.set_title("Accuracy with 95% Wilson intervals")
    ax.legend(loc="lower center", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"accuracy_comparison.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    selected = [lookup[("primary_97", method)] for method in METHODS]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), layout="constrained")
    specs = [("mean_api_attempts_per_question", "API attempts / question"),
             ("mean_known_api_elapsed_seconds_per_question", "Known API time / question (s)"),
             ("mean_known_tokens_per_question", "Known tokens / question")]
    for ax, (field, label) in zip(axes, specs):
        values = [r[field] for r in selected]
        ax.bar(range(4), values, color="#3C7B88", width=0.65)
        ax.set_xticks(range(4), [LABELS[m] for m in METHODS], rotation=20, ha="right")
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    unknown_usage = sum(r["calls_with_unknown_token_usage"] for r in selected)
    unknown_time = sum(r["calls_with_unknown_elapsed"] for r in selected)
    fig.suptitle("Execution cost on 97 inputs per method (all attempts included)", fontsize=11)
    fig.supxlabel(f"Known totals only; calls missing token usage: {unknown_usage}; missing elapsed time: {unknown_time}", fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"efficiency_comparison.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_outputs(run: dict, output_dir: Path, *, figures: bool = True) -> dict:
    analysis, main_table, type_table, changes, cases = analyze(run)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, obj in (("analysis.json", analysis), ("case_candidates.json", cases)):
        (output_dir / filename).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / "main_table.csv", main_table)
    _write_csv(output_dir / "question_type_table.csv", type_table)
    _write_csv(output_dir / "agent_changes.csv", changes)
    if figures:
        make_figures(main_table, output_dir)
    return analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Explicit source JSON, including offline gold labels")
    parser.add_argument("--output-dir", type=Path, help="Default: RUN_DIR/analysis")
    args = parser.parse_args(argv)
    try:
        run = load_run(args.run_dir, args.data)
        output = args.output_dir or args.run_dir / "analysis"
        analysis = write_outputs(run, output)
    except (AnalysisError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Analysis stopped: {exc}\n")
    print(f"Completed offline analysis: {output.resolve()}")
    for row in analysis["main_results"]:
        print(f"{row['subset']} {row['method']}: {row['correct']}/{row['n']} ({row['accuracy']:.2%})")
    print("Cost values are estimates under stated rates, not invoices; inspect unknown usage flags.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
