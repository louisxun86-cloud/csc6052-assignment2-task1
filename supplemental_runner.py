"""Post-hoc supplementary studies; importing this module never calls an API.

Cloud entry: run_supplemental(_assignment2_client)
Offline preregistration:
  python supplemental_runner.py --prepare-protocol --formal-dir actual_run/task1_assignment/formal

Only new supplementary files are written. Frozen main-study sources/logs remain read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import runner as frozen


VERSION = "posthoc-supplement-v1"
MODEL = "deepseek-flash"
MAX_API_CALLS = 240
ORDER_SEED = 605202
SETTINGS = {
    "model": MODEL, "temperature": 0, "max_tokens": 768,
    "extra_body": {"thinking": {"type": "disabled"}},
    "response_format": {"type": "json_object"},
}
COT_V2 = frozen.COT_PROMPT + (
    "\n输出前执行协议核对："
    "（1）确认题目问的是正确项还是错误/不符合项，按题干明确给出的定义和规则作答；"
    "（2）逐项核算，单选只保留一个字母，多选保留全部符合项，禁止重复和空选项；"
    "（3）核对简短依据与最终字母完全一致，若不一致，先修正再输出；"
    "（4）只输出一个合法json对象，不使用Markdown代码块或前后说明。"
    '使用格式{"brief_reason":"不超过120字的可核查依据","answer":["A"]}。'
    "answer必须是大写选项字母组成的非空列表，而不是字符串；不能输出未定义的选项。"
)
A_CONDITIONS = ("cot_v1", "cot_v2")
B_CONDITIONS = ("feedback", "null_feedback")
TERMINAL = {"completed", "api_error", "parse_error", "common_initial_failure"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_jsonl(path):
    raw = Path(path).read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("JSONL末行不完整，不能忽略可能已发出的请求。")
    return [json.loads(line) for line in raw.splitlines()]


def source_material(formal_dir):
    """只按输入schema和来源选入，不按初答正确与否选题。"""
    formal_dir = Path(formal_dir)
    names = ("manifest.json", "ledger.jsonl", "results.jsonl", "source_data.json")
    hashes = {name: file_digest(formal_dir / name) for name in names}
    manifest = json.loads((formal_dir / "manifest.json").read_text(encoding="utf-8"))
    if digest(manifest["experiment"]) != manifest["fingerprint"]:
        raise ValueError("原正式实验manifest指纹不一致。")
    if manifest["experiment"]["runner_source_sha256"] != file_digest(frozen.__file__):
        raise ValueError("导入的runner不是原正式实验冻结版本。")
    data = json.loads((formal_dir / "source_data.json").read_text(encoding="utf-8"))
    if len(data) != 100 or digest(data) != manifest["experiment"]["dataset_sha256"]:
        raise ValueError("原正式数据与指纹不一致。")
    events = read_jsonl(formal_dir / "ledger.jsonl")
    rows = read_jsonl(formal_dir / "results.jsonl")
    if any(x.get("fingerprint") != manifest["fingerprint"] for x in events + rows):
        raise ValueError("原日志或结果混入其他实验。")
    starts, finishes = {}, {}
    for event in events:
        if event["event"] in {"call_start", "call_finish"}:
            target = starts if event["event"] == "call_start" else finishes
            if event["call_id"] in target:
                raise ValueError("原ledger调用ID重复。")
            target[event["call_id"]] = event
    main_indices = set(manifest["experiment"]["evaluation_indices_0_based"]) - {
        int(i) for i in manifest["experiment"]["exclusions"]}
    if len(main_indices) != 97:
        raise ValueError("补充消融要求预注册的97题主集合。")
    original_agent = {x["index"]: x for x in rows if x["method"] == "agent" and x["index"] in main_indices}
    if set(original_agent) != main_indices:
        raise ValueError("原Agent结果缺失。")
    cases, invalid = [], []
    for index in sorted(main_indices):
        row = original_agent[index]
        if row["status"] not in {"completed", "api_error", "parse_error"}:
            raise ValueError("原Agent仍有未完成结果。")
        stage_calls = {x["stage"]: (call_id, finishes.get(call_id)) for call_id, x in starts.items()
                       if x["method"] == "agent" and x["index"] == index
                       and x["question_attempt_id"] == row["question_attempt_id"]}
        payload = frozen.question_input(data[index])
        initial_id, initial_finish = stage_calls.get("initial", (None, None))
        try:
            if not initial_finish or initial_finish["status"] != "ok":
                raise frozen.ResponseFormatError("原初答未成功返回。")
            candidate = frozen._draft(frozen.ExperimentRunner._decode_response(initial_finish["response"]), payload)
        except frozen.ResponseFormatError:
            invalid.append({"index": index, "row_number": index + 1,
                            "initial_call_id": initial_id, "reason": "invalid_original_initial_schema"})
            continue
        critic_id, critic_finish = stage_calls.get("critic_1", (None, None))
        if not critic_finish or critic_finish["status"] != "ok":
            raise ValueError("合法初答缺少已成功返回的第一轮critic，不允许静默改变95题样本。")
        feedback = frozen.ExperimentRunner._decode_response(critic_finish["response"])
        issues, recommendation = feedback.get("issues"), feedback.get("recommendation")
        if not isinstance(issues, list) or any(not isinstance(x, str) for x in issues) or recommendation not in {"accept", "revise"}:
            raise ValueError("第一轮critic schema异常，需先检查预注册方案。")
        feedback = {"issues": issues, "recommendation": recommendation}
        cases.append({"index": index, "row_number": index + 1, "question": payload,
                      "candidate": candidate, "feedback": feedback,
                      "initial_call_id": initial_id, "critic_call_id": critic_id})
    if len(cases) != 95 or [x["row_number"] for x in invalid] != [14, 67]:
        raise ValueError("预期95个合法初答，原第14、67条为两组共同初答失败。")
    return {"hashes": hashes, "manifest": manifest, "data": data, "cases": cases, "invalid": invalid}


def a_messages(condition, item):
    payload = frozen.question_input(item)
    if condition == "cot_v1":
        return frozen.answer_messages("cot", payload)
    if condition == "cot_v2":
        return [{"role": "system", "content": COT_V2}, {"role": "user", "content": canonical(payload)}]
    raise ValueError("未知A条件。")


def b_messages(condition, case):
    if condition not in B_CONDITIONS:
        raise ValueError("未知B条件。")
    payload = {"question": case["question"], "candidate": case["candidate"],
               "feedback": case["feedback"] if condition == "feedback" else None,
               "review_round": 1, "remaining_review_rounds": 0}
    return [{"role": "system", "content": frozen.CONTROLLER_PROMPT},
            {"role": "user", "content": canonical(payload)}]


def prepare_protocol(formal_dir, items_path, protocol_path):
    """只允许首次写入，防止看过补实验结果后悄悄覆盖预注册。"""
    protocol_path = Path(protocol_path)
    if protocol_path.exists():
        raise FileExistsError("协议已存在；不能覆盖预冻结协议。")
    source = source_material(formal_dir)
    items_path = Path(items_path)
    synthetic = json.loads(items_path.read_text(encoding="utf-8"))
    items = synthetic["items"]
    expected_ids = [f"D{i:02d}" for i in range(1, 9)] + [f"P{i:02d}" for i in range(1, 9)]
    if [x["id"] for x in items] != expected_ids:
        raise ValueError("必须固定D01-D08、P01-P08顺序。")
    if Counter(x["split"] for x in items) != {"design_dev": 8, "held_out_probe": 8}:
        raise ValueError("两split必须各8题。")
    original_dev_path = Path(formal_dir).parent / "development/source_data.json"
    original_dev = json.loads(original_dev_path.read_text(encoding="utf-8"))
    old_questions = {re.sub(r"\s+", "", x["question"]) for x in
                     source["data"] + list(frozen.SYNTHETIC_EXAMPLES) + original_dev}
    new_questions = [re.sub(r"\s+", "", x["question"]) for x in items]
    if len(set(new_questions)) != 16 or set(new_questions) & old_questions:
        raise ValueError("新增题重复或与原题逐字重复。")
    for item in items:
        frozen.reference_answer(item)
    order_rng = random.Random(ORDER_SEED)
    a_orders, b_orders = {}, {}
    for item in items:
        order = list(A_CONDITIONS)
        order_rng.shuffle(order)
        a_orders[item["id"]] = order
    for case in source["cases"]:
        order = list(B_CONDITIONS)
        order_rng.shuffle(order)
        b_orders[f"q{case['row_number']:03d}"] = order
    payload = {
        "version": VERSION, "registered_at_utc": now(),
        "timing": "Post-hoc supplementary research after the original formal experiment; never replaces primary results.",
        "supplemental_source_sha256": file_digest(__file__),
        "frozen_runner_sha256": file_digest(frozen.__file__),
        "original_formal_fingerprint": source["manifest"]["fingerprint"],
        "original_formal_files_sha256": source["hashes"],
        "original_development_source_sha256": file_digest(original_dev_path),
        "synthetic_items_file_sha256": file_digest(items_path),
        "synthetic_items": synthetic,
        "source_cases_sha256": digest(source["cases"]),
        "original_invalid_initials": source["invalid"],
        "model_settings": SETTINGS, "max_api_calls": MAX_API_CALLS,
        "condition_order_seed": ORDER_SEED,
        "planned_call_counts": {"protocol_comparison": 32, "one_step_feedback_ablation": 190, "total": 222},
        "study_A": {
            "name": "Closed-world protocol robustness check",
            "conditions": list(A_CONDITIONS), "design_dev_ids": expected_ids[:8], "held_out_probe_ids": expected_ids[8:],
            "frozen_prompts": {"cot_v1": frozen.COT_PROMPT, "cot_v2": COT_V2},
            "design_rationale": "v2 adds a fixed output/negation/answer-rationale alignment checklist motivated by already-observed main-study protocol failures.",
            "order": "design_dev first, then held_out_probe; each item's two conditions are shuffled with one fixed seeded PRNG before any request.",
            "condition_order_by_unit": a_orders,
            "no_adaptation": "Both splits and v2 are frozen before any supplementary API request. No retuning after design or probe responses.",
            "interpretation": "Independent hand-constructed protocol checks; shared arithmetic/unit themes do not establish pharmacy-domain generalization.",
        },
        "study_B": {
            "name": "Controlled one-step feedback ablation",
            "conditions": list(B_CONDITIONS), "overall_n": 97, "conditional_n": 95,
            "controller_prompt": frozen.CONTROLLER_PROMPT,
            "manipulation": "Only the user JSON feedback field changes: original frozen first-critic feedback versus JSON null.",
            "constants": "Same original legal initial candidate, question, controller prompt, review_round=1, remaining_review_rounds=0, model and settings. Both conditions explicitly have no remaining review round.",
            "order": "Each original item's two conditions are shuffled using the same fixed PRNG stream after study A, and explicit orders are frozen below.",
            "condition_order_by_unit": b_orders,
            "stopping": "Exactly one new controller call per condition. No fresh critic, second review, or iterative continuation.",
            "inclusion": "Eligibility uses only original initial-answer schema validity, never correctness. The two invalid initials are shared failures in both overall denominators.",
            "interpretation": "This isolates one controller-stage feedback intervention conditional on fixed original states; it is not a full-agent causal ablation or an independent replication.",
        },
        "supplementary_primary_comparisons": {
            "family": ["A: cot_v2 versus cot_v1 on all 16 constructed checks", "B: feedback versus null_feedback on the 95 schema-valid fixed initial states"],
            "test": "Two-sided exact McNemar tests; Holm correction across exactly these two post-hoc supplementary comparisons.",
            "descriptive_subsets": "Study A design_dev and held_out_probe each have 8 items; B overall97 includes two shared initial failures. These overlapping subsets are descriptive, not additional primary tests.",
            "timing": "Frozen before supplementary requests but after main-study results; distinct from the original main-study multiplicity family.",
        },
        "scoring": "Use original strict JSON/answer and controller action rules in both conditions. Valid incorrect, API failure and parse failure count wrong. Every job is attempted at most once; no retries of failures or incorrect answers.",
        "cost_policy": "Budget counts every started request including unknown/failed attempts. SDK retries disabled. Unknown token use remains unknown. No prices are guessed.",
    }
    protocol = {"fingerprint": digest(payload), "protocol": payload}
    atomic_json(protocol_path, protocol)
    return protocol


def load_protocol(protocol_path, formal_dir):
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    payload = protocol["protocol"]
    if digest(payload) != protocol["fingerprint"]:
        raise ValueError("补充协议指纹不匹配。")
    if payload["supplemental_source_sha256"] != file_digest(__file__) or payload["frozen_runner_sha256"] != file_digest(frozen.__file__):
        raise ValueError("执行代码与预冻结协议不一致。")
    if payload["model_settings"] != SETTINGS or payload["study_A"]["frozen_prompts"] != {"cot_v1": frozen.COT_PROMPT, "cot_v2": COT_V2}:
        raise ValueError("模型设置或提示已改变。")
    if payload["study_B"]["controller_prompt"] != frozen.CONTROLLER_PROMPT:
        raise ValueError("控制器提示已改变。")
    source = source_material(formal_dir)
    if source["hashes"] != payload["original_formal_files_sha256"] or digest(source["cases"]) != payload["source_cases_sha256"]:
        raise ValueError("原正式状态已改变，不能作为相同控制条件。")
    return protocol, source


def make_jobs(protocol, source):
    """所有请求只通过白名单构建；gold/analysis只保留用于本地评分。"""
    jobs = []
    items = protocol["protocol"]["synthetic_items"]["items"]
    for item in items:
        order = protocol["protocol"]["study_A"]["condition_order_by_unit"][item["id"]]
        if sorted(order) != sorted(A_CONDITIONS):
            raise ValueError("A条件顺序异常。")
        for condition in order:
            jobs.append({"job_id": f"A:{item['id']}:{condition}", "study": "A", "condition": condition,
                         "unit_id": item["id"], "split": item["split"], "payload": frozen.question_input(item),
                         "reference": frozen.reference_answer(item), "request": {**SETTINGS, "messages": a_messages(condition, item)}})
    for case in source["cases"]:
        order = protocol["protocol"]["study_B"]["condition_order_by_unit"][f"q{case['row_number']:03d}"]
        if sorted(order) != sorted(B_CONDITIONS):
            raise ValueError("B条件顺序异常。")
        for condition in order:
            jobs.append({"job_id": f"B:q{case['row_number']:03d}:{condition}", "study": "B", "condition": condition,
                         "unit_id": f"q{case['row_number']:03d}", "split": "original_legal_initials",
                         "row_number": case["row_number"], "payload": case["question"], "candidate": case["candidate"],
                         "reference": frozen.reference_answer(source["data"][case["index"]]),
                         "request": {**SETTINGS, "messages": b_messages(condition, case)}})
    if len(jobs) != 222:
        raise ValueError("计划调用数不是222。")
    return jobs


def _evaluate(job, finish):
    if finish["status"] == "api_error":
        return {"status": "api_error", "prediction": None, "action": None, "correct": False, "format_valid": False}
    try:
        obj = frozen.ExperimentRunner._decode_response(finish["response"])
        draft = frozen._draft(obj, job["payload"])
        action = None
        decision_reason = None
        if job["study"] == "B":
            action, decision_reason = obj.get("action"), obj.get("decision_reason")
            if action not in {"finish", "revise"} or not isinstance(decision_reason, str) or not decision_reason.strip():
                raise frozen.ResponseFormatError("控制器action/decision_reason不合法。")
            if action == "finish" and draft["answer"] != job["candidate"]["answer"]:
                raise frozen.ResponseFormatError("finish却改变候选答案。")
        prediction = "".join(draft["answer"])
        return {"status": "completed", "prediction": prediction, "action": action,
                "brief_reason": draft["brief_reason"], "decision_reason": decision_reason,
                "correct": prediction == job["reference"], "format_valid": True}
    except frozen.ResponseFormatError:
        return {"status": "parse_error", "prediction": None, "action": None, "correct": False, "format_valid": False}


class SupplementalRun:
    def __init__(self, client, protocol_path, formal_dir, output_dir, max_api_calls=MAX_API_CALLS):
        self.protocol, self.source = load_protocol(protocol_path, formal_dir)
        self.jobs = make_jobs(self.protocol, self.source)
        self.client = client
        if isinstance(max_api_calls, bool) or not isinstance(max_api_calls, int) or not 0 <= max_api_calls <= MAX_API_CALLS:
            raise ValueError("累计预算必须在0到240之间。")
        self.max_api_calls = max_api_calls
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.ledger = self.output / "ledger.jsonl"
        self.fingerprint = self.protocol["fingerprint"]
        with self.lock():
            protocol_copy = self.output / "protocol.json"
            if protocol_copy.exists():
                if json.loads(protocol_copy.read_text(encoding="utf-8")) != self.protocol:
                    raise ValueError("输出目录已有不同补充协议。")
            else:
                if self.ledger.exists() and self.ledger.stat().st_size:
                    raise ValueError("已有无协议ledger，拒绝继续。")
                atomic_json(protocol_copy, self.protocol)
            atomic_json(self.output / "job_manifest.json", {
                "fingerprint": self.fingerprint, "ordered_job_ids": [x["job_id"] for x in self.jobs],
                "request_sha256": {x["job_id"]: digest(x["request"]) for x in self.jobs},
                "planned_calls": len(self.jobs), "common_invalid_initial_rows": [14, 67],
            })
            self.load()
            self.export()

    @contextmanager
    def lock(self):
        path = self.output / ".supplemental.lock"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError("补充实验已有写入锁；确认无活动进程后再处理遗留锁。") from None
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(canonical({"pid": os.getpid(), "created_at": now()}))
                handle.flush(); os.fsync(handle.fileno())
            yield
        finally:
            path.unlink(missing_ok=True)

    def load(self):
        self.events = read_jsonl(self.ledger) if self.ledger.exists() else []
        self.starts, self.finishes, self.latest = {}, {}, {}
        valid_ids = {job["job_id"] for job in self.jobs}
        for event in self.events:
            if event.get("fingerprint") != self.fingerprint:
                raise ValueError("补充ledger指纹不一致。")
            if event["event"] == "call_start":
                if event["attempt_id"] in self.starts or event["job_id"] not in valid_ids:
                    raise ValueError("重复attempt或未知job。")
                self.starts[event["attempt_id"]] = event
                self.latest[event["job_id"]] = event["attempt_id"]
            elif event["event"] == "call_finish":
                if event["attempt_id"] in self.finishes:
                    raise ValueError("重复finish。")
                self.finishes[event["attempt_id"]] = event
        if set(self.finishes) - set(self.starts):
            raise ValueError("finish缺少对应start。")

    def append(self, event):
        event = {"timestamp": now(), "fingerprint": self.fingerprint, **event}
        with self.ledger.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical(event) + "\n"); handle.flush(); os.fsync(handle.fileno())
        self.load()

    def rows(self):
        result = []
        for job in self.jobs:
            row = {k: job[k] for k in ("job_id", "study", "condition", "unit_id", "split", "reference")}
            row["fingerprint"] = self.fingerprint
            if "row_number" in job:
                row["row_number"] = job["row_number"]
                row["candidate_prediction"] = "".join(job["candidate"]["answer"])
                row["candidate_correct"] = row["candidate_prediction"] == job["reference"]
            attempt_id = self.latest.get(job["job_id"])
            if attempt_id is None:
                row.update(status="not_started", prediction=None, action=None, correct=False, format_valid=False)
            else:
                start = self.starts[attempt_id]
                if start["request_sha256"] != digest(job["request"]):
                    raise ValueError("已记账请求与冻结job不一致。")
                row["attempt_id"] = attempt_id
                finish = self.finishes.get(attempt_id)
                row.update(_evaluate(job, finish) if finish else {
                    "status": "unknown", "prediction": None, "action": None, "correct": False, "format_valid": False})
            result.append(row)
        for invalid in self.source["invalid"]:
            item = self.source["data"][invalid["index"]]
            for condition in B_CONDITIONS:
                result.append({"fingerprint": self.fingerprint,
                    "job_id": f"B:q{invalid['row_number']:03d}:{condition}", "study": "B", "condition": condition,
                    "unit_id": f"q{invalid['row_number']:03d}", "split": "common_invalid_initials",
                    "row_number": invalid["row_number"], "reference": frozen.reference_answer(item),
                    "status": "common_initial_failure", "prediction": None, "action": None,
                    "correct": False, "format_valid": False, "candidate_prediction": None, "candidate_correct": False})
        return result

    def cost(self, job_ids):
        counts = Counter(); elapsed = 0.0
        for attempt_id, start in self.starts.items():
            if start["job_id"] not in job_ids:
                continue
            counts["api_attempts"] += 1
            finish = self.finishes.get(attempt_id)
            if finish is None:
                counts["unresolved_calls"] += 1; counts["calls_with_unknown_usage"] += 1
                continue
            elapsed += finish["elapsed_seconds"]
            usage = finish.get("response", {}).get("usage") or {}
            if any(not isinstance(usage.get(k), int) for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
                counts["calls_with_unknown_usage"] += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                counts["known_" + key] += usage.get(key, 0)
        return {**{key: counts[key] for key in ("api_attempts", "unresolved_calls", "calls_with_unknown_usage",
                    "known_prompt_tokens", "known_completion_tokens", "known_total_tokens")}, "known_api_seconds": elapsed}

    def metric(self, rows):
        n = len(rows); correct = sum(r["correct"] for r in rows)
        valid = sum(r["format_valid"] for r in rows)
        finished = all(r["status"] in TERMINAL for r in rows)
        return {"n": n, "correct": correct, "format_valid": valid, "all_finished": finished,
                "accuracy": correct / n if n and finished else None,
                "format_valid_rate": valid / n if n and finished else None,
                "status_counts": dict(Counter(r["status"] for r in rows)),
                **self.cost({r["job_id"] for r in rows})}

    @staticmethod
    def paired(rows, first, second):
        left = {r["unit_id"]: r for r in rows if r["condition"] == first}
        right = {r["unit_id"]: r for r in rows if r["condition"] == second}
        if set(left) != set(right):
            raise ValueError("配对条件题目集合不同。")
        done = all(r["status"] in TERMINAL for r in list(left.values()) + list(right.values()))
        if not done:
            return {"n": len(left), "all_finished": False, "comparison": [first, second]}
        a = sum(left[k]["correct"] and not right[k]["correct"] for k in left)
        b = sum(right[k]["correct"] and not left[k]["correct"] for k in left)
        d = a + b
        p = 1.0 if not d else min(1.0, 2 * sum(math.comb(d, k) for k in range(min(a, b) + 1)) / (2 ** d))
        return {"n": len(left), "all_finished": True, "comparison": [first, second],
                "first_only_correct": a, "second_only_correct": b,
                "first_minus_second_accuracy_pp": 100 * (a - b) / len(left),
                "mcnemar_exact_two_sided_p": p}

    @staticmethod
    def holm_two(comparisons):
        """新增研究的两个预指定比较独立成族；不与原主研究混用。"""
        output = {key: dict(value) for key, value in comparisons.items()}
        if not all(value["all_finished"] for value in output.values()):
            return output
        ordered = sorted(output, key=lambda key: output[key]["mcnemar_exact_two_sided_p"])
        previous = 0.0
        for rank, key in enumerate(ordered):
            adjusted = min(1.0, max(previous, (2 - rank) * output[key]["mcnemar_exact_two_sided_p"]))
            output[key]["holm_adjusted_p_two_comparison_family"] = adjusted
            previous = adjusted
        return output

    def summary(self):
        rows = self.rows(); a = [r for r in rows if r["study"] == "A"]; b = [r for r in rows if r["study"] == "B"]
        study_a = {condition: {split: self.metric([r for r in a if r["condition"] == condition and (split == "all_16" or r["split"] == split)])
                               for split in ("design_dev", "held_out_probe", "all_16")} for condition in A_CONDITIONS}
        study_b = {condition: {"overall_97": self.metric([r for r in b if r["condition"] == condition]),
                               "conditional_95": self.metric([r for r in b if r["condition"] == condition and r["split"] == "original_legal_initials"])}
                   for condition in B_CONDITIONS}
        eligible = [r for r in b if r["condition"] == "feedback" and r["split"] == "original_legal_initials"]
        primary = self.holm_two({
            "A_all_16": self.paired(a, "cot_v2", "cot_v1"),
            "B_conditional_95": self.paired([r for r in b if r["split"] == "original_legal_initials"], "feedback", "null_feedback"),
        })
        return {"fingerprint": self.fingerprint, "generated_at_utc": now(), "post_hoc": True,
                "all_finished": all(r["status"] in TERMINAL for r in rows),
                "max_api_calls": self.max_api_calls, "planned_api_calls": 222, "api_attempts": len(self.starts),
                "study_A": study_a, "study_B": study_b,
                "original_fixed_initial_correct": sum(r["candidate_correct"] for r in eligible),
                "original_fixed_initial_eligible_n": 95, "original_common_initial_failure_n": 2,
                "supplementary_primary_comparisons": primary,
                "paired_A": {split: self.paired([r for r in a if r["split"] == split], "cot_v2", "cot_v1") for split in ("design_dev", "held_out_probe")},
                "paired_B_overall_97": self.paired(b, "feedback", "null_feedback"),
                "paired_B_conditional_95": self.paired([r for r in b if r["split"] == "original_legal_initials"], "feedback", "null_feedback"),
                "totals": self.cost({j["job_id"] for j in self.jobs}),
                "interpretation": [
                    "All additions are post-hoc and cannot replace the original primary/sensitivity experiment.",
                    "A is an independently constructed closed-world protocol check, not domain-general pharmacy validation.",
                    "B is a controlled one-step feedback ablation on fixed original states, not a complete Agent ablation.",
                    "Both B conditions count the same two invalid original initials as wrong in the 97-question denominator.",
                    "The two pre-specified supplementary comparisons (A all16 and B conditional95) use two-sided exact McNemar and Holm2, and are post-hoc relative to the main study.",
                    "The A 8-item subsets and B overall97 are descriptive and overlapping, not independent replications or additional primary hypotheses.",
                    "Token totals include all attempts; unknown costs are flagged and no currency cost is inferred here.",
                ]}

    def export(self):
        temporary = self.output / ("results." + uuid.uuid4().hex + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in self.rows():
                handle.write(canonical(row) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, self.output / "results.jsonl")
        atomic_json(self.output / "summary.json", self.summary())

    def run(self, *, verbose=True):
        if getattr(self.client, "max_retries", None) != 0 or str(getattr(self.client, "base_url", "")).rstrip("/") not in {
            "https://api.deepseek.com", "https://api.deepseek.com/v1"}:
            raise ValueError("客户端必须使用官方DeepSeek地址且max_retries=0。")
        with self.lock():
            self.load()
            try:
                for ordinal, job in enumerate(self.jobs, 1):
                    prior = self.latest.get(job["job_id"])
                    if prior is not None:
                        continue
                    if len(self.starts) >= self.max_api_calls:
                        if verbose: print("补充实验达到累计请求上限，已保存断点。", flush=True)
                        break
                    attempt_id = uuid.uuid4().hex
                    self.append({"event": "call_start", "attempt_id": attempt_id, "job_id": job["job_id"],
                                 "retry_of": prior, "request_sha256": digest(job["request"]), "request": job["request"]})
                    if verbose: print(f"[补充 {ordinal}/222] {job['job_id']}；累计API {len(self.starts)}/{self.max_api_calls}", flush=True)
                    started = time.perf_counter()
                    try:
                        response = self.client.chat.completions.create(**job["request"])
                    except Exception as exc:
                        status_code = getattr(exc, "status_code", None)
                        self.append({"event": "call_finish", "attempt_id": attempt_id, "status": "api_error",
                                     "error_type": type(exc).__name__, "status_code": status_code if isinstance(status_code, int) else None,
                                     "elapsed_seconds": time.perf_counter() - started})
                    else:
                        self.append({"event": "call_finish", "attempt_id": attempt_id, "status": "ok",
                                     "response": frozen.ExperimentRunner._compact_response(response),
                                     "elapsed_seconds": time.perf_counter() - started})
                    self.export()
                return self.summary()
            finally:
                self.export()


def run_supplemental(client, protocol_path="supplemental_protocol.json",
                     formal_dir="task1_assignment/formal", output_dir="task1_assignment/supplemental",
                     max_api_calls=MAX_API_CALLS, *, verbose=True):
    engine = SupplementalRun(client, protocol_path, formal_dir, output_dir, max_api_calls)
    summary = engine.run(verbose=verbose)
    if verbose:
        print("补充实验已保存：", Path(output_dir), flush=True)
        print("全部完成：", summary["all_finished"], "；累计调用：", summary["api_attempts"], flush=True)
    return engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-protocol", action="store_true")
    parser.add_argument("--formal-dir", type=Path, default=Path("actual_run/task1_assignment/formal"))
    parser.add_argument("--items", type=Path, default=Path("supplemental_items.json"))
    parser.add_argument("--protocol", type=Path, default=Path("supplemental_protocol.json"))
    args = parser.parse_args()
    if not args.prepare_protocol:
        parser.error("命令行仅支持离线--prepare-protocol；真实调用需由notebook明确传入client。")
    result = prepare_protocol(args.formal_dir, args.items, args.protocol)
    print("已离线冻结协议：", args.protocol, "；指纹：", result["fingerprint"])
