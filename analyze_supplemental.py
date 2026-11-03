"""补充实验只读复算：不构造API客户端，不改冻结源码或原始日志。"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import supplemental_runner as sr


def load_supplemental(run_dir, formal_dir, protocol_path=None):
    """按已冻结协议和逐调用ledger重新评分，同时验证现有派生结果。"""
    directory = Path(run_dir)
    protocol_path = Path(protocol_path) if protocol_path else directory / "protocol.json"
    protocol, source = sr.load_protocol(protocol_path, formal_dir)
    protocol_copy = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    if protocol_copy != protocol:
        raise ValueError("补充运行目录中的协议不是指定冻结协议。")
    # 仅利用纯评分方法；刻意不调用会创建目录/锁/导出文件的初始化器。
    engine = sr.SupplementalRun.__new__(sr.SupplementalRun)
    engine.protocol, engine.source = protocol, source
    engine.jobs = sr.make_jobs(protocol, source)
    engine.output, engine.ledger = directory, directory / "ledger.jsonl"
    engine.fingerprint = protocol["fingerprint"]
    engine.max_api_calls = protocol["protocol"]["max_api_calls"]
    engine.client = None
    engine.load()
    manifest = json.loads((directory / "job_manifest.json").read_text(encoding="utf-8"))
    expected_manifest = {
        "fingerprint": engine.fingerprint,
        "ordered_job_ids": [job["job_id"] for job in engine.jobs],
        "request_sha256": {job["job_id"]: sr.digest(job["request"]) for job in engine.jobs},
        "planned_calls": 222,
        "common_invalid_initial_rows": [14, 67],
    }
    if manifest != expected_manifest:
        raise ValueError("补充请求manifest与冻结计划不一致。")
    ordered_jobs, seen_ids, seen_attempts = [], set(), set()
    for event in engine.events:
        if event["event"] == "call_start":
            job_id = event["job_id"]
            if job_id in seen_ids or event.get("retry_of") is not None:
                raise ValueError("预注册补充实验不允许任何重复尝试。")
            if sr.digest(event["request"]) != event["request_sha256"] or event["request_sha256"] != manifest["request_sha256"][job_id]:
                raise ValueError("实际请求不等于冻结请求。")
            seen_ids.add(job_id)
            seen_attempts.add(event["attempt_id"])
            ordered_jobs.append(job_id)
        elif event["event"] == "call_finish":
            if event["attempt_id"] not in seen_attempts:
                raise ValueError("调用finish出现在相应start之前。")
            elapsed = event.get("elapsed_seconds")
            if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("调用耗时不合法。")
            if event.get("status") not in {"ok", "api_error"}:
                raise ValueError("未知调用状态。")
        else:
            raise ValueError("补充ledger含未知事件。")
    if ordered_jobs != manifest["ordered_job_ids"][:len(ordered_jobs)]:
        raise ValueError("实际调用顺序不等于冻结计划的前缀。")
    if len(ordered_jobs) > 222:
        raise ValueError("补充调用数量超过一次性计划。")
    actual_rows = sr.read_jsonl(directory / "results.jsonl")
    if actual_rows != engine.rows():
        raise ValueError("补充逐题结果与ledger重新评分不一致。")
    recorded_summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    recorded_budget = recorded_summary.get("max_api_calls")
    if (not isinstance(recorded_budget, int) or isinstance(recorded_budget, bool)
            or not len(ordered_jobs) <= recorded_budget <= protocol["protocol"]["max_api_calls"]):
        raise ValueError("记录的补充预算不合法或实际尝试次数已越界。")
    engine.max_api_calls = recorded_budget
    actual_summary = engine.summary()
    for key in ("generated_at_utc",):
        recorded_summary.pop(key, None)
        actual_summary.pop(key, None)
    if recorded_summary != actual_summary:
        raise ValueError("补充汇总与ledger重新统计不一致。")
    return engine


def write_outputs(engine, output_dir):
    """只写指定派生分析目录；不修改输入目录内已有summary/results/ledger。"""
    output = Path(output_dir).resolve()
    if output == engine.output.resolve() or output == engine.output.parent.resolve():
        raise ValueError("分析目录须独立，不能直接覆盖补充原始输出目录。")
    output.mkdir(parents=True, exist_ok=True)
    summary = engine.summary()
    summary["validation"] = "frozen_protocol_requests_ledger_results_and_summary_passed"
    summary["analysis_mode"] = "offline_no_API"
    sr.atomic_json(output / "analysis.json", summary)
    rows = []
    for study, conditions in (("A", sr.A_CONDITIONS), ("B", sr.B_CONDITIONS)):
        for condition in conditions:
            for subset, metric in summary["study_" + study][condition].items():
                rows.append({"study": study, "condition": condition, "subset": subset,
                             **{key: metric[key] for key in ("n", "correct", "format_valid", "accuracy", "format_valid_rate", "all_finished", "api_attempts", "known_total_tokens", "calls_with_unknown_usage")}})
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sr.atomic_json(output / "paired_comparisons.json", summary["supplementary_primary_comparisons"])
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--formal-dir", required=True, type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    engine = load_supplemental(args.run_dir, args.formal_dir, args.protocol)
    summary = write_outputs(engine, args.output_dir)
    print("补充离线复算完成；全部任务有终结记录：", summary["all_finished"])
    print("API尝试次数：", summary["api_attempts"], "；本次复算API调用：0")
    print("输出：", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
