"""Task 1 可复现实验核心；导入不会联网，也不会调用模型。

外层 notebook 创建官方 DeepSeek 客户端（max_retries=0），再显式设置调用次数上限。
示例：
    config = ExperimentConfig(model="deepseek-flash", max_api_calls=20)
    runner = ExperimentRunner(data, client, "task1_results", config)
    runner.run(methods=("zero_shot",))

max_api_calls 是同一实验目录历次运行累计的尝试上限，不是本次新增额度。
中断后重放成功响应，不重复收费；未确认请求和失败请求默认不会自动重发。
同一目录只能由一个进程写入。实际断电遗留 .runner.lock 时，先确认无运行进程再移除锁。
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RUNNER_VERSION = "1.0.0"
MAX_API_CALLS = 0
METHODS = ("zero_shot", "few_shot", "cot", "agent")
QUESTION_TYPES = ("最佳选择题", "配伍选择题", "综合分析选择题", "多项选择题")
SENSITIVITY_EXCLUDED_ROWS = (11, 27, 28, 56, 58, 67, 91)
TERMINAL_STATUSES = {"completed", "api_error", "parse_error"}
RETRYABLE_STATUSES = {"api_error", "parse_error", "interrupted"}

COMMON_PROMPT = (
    "你正在完成执业药师考试选择题。只依据给出的题干、选项和你已有的知识作答。"
    "识别题型：多项选择题选择全部正确选项，其余题型只选择一个选项。"
    "留意“不包括”“错误”“不正确”等否定表达、限定条件、单位及题干背景。"
    "不得编造缺失图表内容，不得选择未提供或空白的选项。"
    '只输出 json 对象；答案必须用字母列表，例如 {"answer":["A"]} 或 '
    '{"answer":["A","C"]}。'
)
ZERO_PROMPT = COMMON_PROMPT + "直接给出答案，不提供解释。"
FEW_PROMPT = COMMON_PROMPT + "以下人工构造的独立示例用于演示格式和单多选规则；随后回答新题。直接给出答案。"
COT_PROMPT = COMMON_PROMPT + (
    "先辨认考点和关键条件，检查否定词与各选项是否满足条件，再给出答案。"
    '在 json 的 brief_reason 字段中提供简短的可核查判断依据，不超过120字。'
)
CRITIC_PROMPT = (
    "你是选择题答案审查员。你没有标准答案。审查提供的题目与候选答案："
    "检查单多选规则、否定词、关键条件、单位、选项范围和理由是否一致。"
    "只指出有依据的问题；如果无法核实事实，应注明不确定，不能假称已核对标准答案。"
    '只输出 json：{"issues":["具体问题或不确定点"],"recommendation":"accept或revise"}。'
)
CONTROLLER_PROMPT = (
    COMMON_PROMPT
    + "你会收到候选答案及独立审查反馈。反馈也可能有错，请根据题意判断是否采纳。"
    "自主决定下一步：finish 表示保留当前答案并提交；revise 表示给出修订后的答案和依据。"
    "仅在需要修改或进一步核查时选择 revise，不要机械重复复核。"
    '只输出 json：{"action":"finish或revise","answer":["A"],'
    '"brief_reason":"判断依据","decision_reason":"为何提交或修改"}。'
    "选择 finish 时 answer 必须与候选答案一致；达到轮数上限后将提交本轮答案。"
)

# 示例独立人工构造，仅含可直接核算的单位换算/库存算术，并非真实药师试题。
# 与正式考试的知识难度有差异，应在报告局限部分说明。
SYNTHETIC_EXAMPLES = (
    {
        "provenance": "人工构造的单位换算示例，非正式评测数据",
        "question_type": "最佳选择题",
        "question": "仅作单位换算：1克等于多少毫克？",
        "option": {"A": "10毫克", "B": "100毫克", "C": "1000毫克", "D": "10000毫克"},
        "answer": "C",
    },
    {
        "provenance": "人工构造的库存算术示例，非正式评测数据",
        "question_type": "最佳选择题",
        "question": "某仓库原有240盒物品，入库60盒，出库75盒。未发生其他出入库，现有多少盒？",
        "option": {"A": "165盒", "B": "195盒", "C": "300盒", "D": "225盒"},
        "answer": "D",
    },
    {
        "provenance": "人工构造的多选单位换算示例，非正式评测数据",
        "question_type": "多项选择题",
        "question": "以下哪些质量等于1克？请选择全部正确选项。",
        "option": {"A": "1000毫克", "B": "0.001千克", "C": "100毫克", "D": "1000000微克", "E": "10千克"},
        "answer": "ABD",
    },
)


class ExperimentError(RuntimeError):
    pass


class FingerprintMismatch(ExperimentError):
    pass


class BudgetExceeded(ExperimentError):
    pass


class UnresolvedCall(ExperimentError):
    pass


class APICallError(ExperimentError):
    pass


class ResponseFormatError(ExperimentError):
    pass


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    max_api_calls: int = MAX_API_CALLS
    max_tokens: int = 768
    temperature: float = 0.0
    max_review_rounds: int = 2
    methods: tuple[str, ...] = METHODS
    sensitivity_excluded_rows: tuple[int, ...] = SENSITIVITY_EXCLUDED_ROWS

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("必须明确指定模型名称。")
        if isinstance(self.max_api_calls, bool) or not isinstance(self.max_api_calls, int) or self.max_api_calls < 0:
            raise ValueError("max_api_calls 必须是非负整数，默认0表示禁止调用。")
        if not isinstance(self.max_tokens, int) or self.max_tokens < 128:
            raise ValueError("max_tokens 必须至少为128。")
        if self.temperature != 0:
            raise ValueError("本轮受控实验统一 temperature=0。")
        if isinstance(self.max_review_rounds, bool) or self.max_review_rounds not in (0, 1, 2):
            raise ValueError("max_review_rounds 只能为0、1或2；0用于去除复核的消融。")
        if not self.methods or len(set(self.methods)) != len(self.methods) or not set(self.methods) <= set(METHODS):
            raise ValueError("methods 含未知方法或重复方法。")
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in self.sensitivity_excluded_rows):
            raise ValueError("敏感性排除位置使用从1开始的正整数。")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def question_input(item: Mapping[str, Any]) -> dict[str, Any]:
    """唯一的模型输入入口；显式白名单，不能传入整条原始记录。"""
    question = item.get("question")
    kind = item.get("question_type")
    raw_options = item.get("option")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("missing_question")
    if kind not in QUESTION_TYPES:
        raise ValueError("unknown_question_type")
    if not isinstance(raw_options, dict):
        raise ValueError("invalid_options")
    options = {k: v for k, v in raw_options.items() if isinstance(v, str) and v.strip()}
    if not options:
        raise ValueError("all_options_empty")
    if not set(options) <= set("ABCDE"):
        raise ValueError("invalid_option_letters")
    return {"question_type": kind, "question": question, "options": options}


def parse_choices(choices: Any, payload: Mapping[str, Any]) -> str:
    if not isinstance(choices, list) or not choices:
        raise ResponseFormatError("answer 必须是非空字母列表。")
    if any(not isinstance(x, str) or x not in payload["options"] for x in choices):
        raise ResponseFormatError("答案含无效或空选项。")
    if len(set(choices)) != len(choices):
        raise ResponseFormatError("答案包含重复选项。")
    if payload["question_type"] != "多项选择题" and len(choices) != 1:
        raise ResponseFormatError("单选题必须只选一个选项。")
    return "".join(sorted(choices))


def reference_answer(item: Mapping[str, Any]) -> str:
    raw = item.get("answer")
    if not isinstance(raw, str) or not re.fullmatch(r"[A-E]+", raw.strip()):
        raise ValueError("invalid_reference")
    try:
        return parse_choices(list(raw.strip()), question_input(item))
    except ResponseFormatError as exc:
        raise ValueError("invalid_reference") from exc


def answer_messages(method: str, payload: Mapping[str, Any]) -> list[dict[str, str]]:
    if method not in ("zero_shot", "few_shot", "cot"):
        raise ValueError("未知普通提示方法。")
    prompt = {"zero_shot": ZERO_PROMPT, "few_shot": FEW_PROMPT, "cot": COT_PROMPT}[method]
    messages = [{"role": "system", "content": prompt}]
    if method == "few_shot":
        for example in SYNTHETIC_EXAMPLES:
            messages.append({"role": "user", "content": _json(question_input(example))})
            messages.append({"role": "assistant", "content": _json({"answer": list(example["answer"])})})
    messages.append({"role": "user", "content": _json(dict(payload))})
    return messages


def _draft(result: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    prediction = parse_choices(result.get("answer"), payload)
    reason = result.get("brief_reason", "")
    if not isinstance(reason, str):
        raise ResponseFormatError("brief_reason 必须是字符串。")
    return {"answer": list(prediction), "brief_reason": reason}


class ExperimentRunner:
    """单写入进程、逐请求持久化、可恢复的实验执行器。"""

    def __init__(
        self,
        data: Sequence[Mapping[str, Any]],
        client: Any,
        output_dir: str | Path,
        config: ExperimentConfig,
        *,
        evaluation_indices: Sequence[int] | None = None,
        extra_exclusions: Mapping[int, str] | None = None,
    ):
        self.data = json.loads(_json(list(data)))
        if not self.data:
            raise ValueError("数据不能为空。")
        self.client = client
        self.config = config
        self._verbose = False
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.output_dir / "ledger.jsonl"
        self.manifest_path = self.output_dir / "manifest.json"
        self.indices = tuple(range(len(self.data))) if evaluation_indices is None else tuple(evaluation_indices)
        if not self.indices or len(set(self.indices)) != len(self.indices):
            raise ValueError("评测名单不能为空或包含重复位置。")
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= len(self.data) for i in self.indices):
            raise ValueError("evaluation_indices 使用有效的从0开始的位置。")
        self.extra_exclusions = dict(extra_exclusions or {})
        if any(i not in self.indices or not isinstance(reason, str) or not reason.strip()
               for i, reason in self.extra_exclusions.items()):
            raise ValueError("人工排除项必须属于评测名单，并给出非空理由。")
        self.audit = self._audit_data()
        self.exclusions = {r["index"]: r["exclusion_reason"] for r in self.audit["rows"] if r["excluded"]}
        self.valid_indices = tuple(i for i in self.indices if i not in self.exclusions)
        self.fingerprint_payload = self._fingerprint_payload()
        self.fingerprint = _hash(self.fingerprint_payload)
        with self._lock():
            self._initialize_manifest()
            self._load_ledger()
            self._sync_exports()

    @contextmanager
    def _lock(self):
        lock_path = self.output_dir / ".runner.lock"
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise ExperimentError("实验目录已有写入锁。确认其他进程已停止后，才能处理遗留 .runner.lock。") from exc
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(_json({"pid": os.getpid(), "created_at": _utc()}))
                handle.flush()
                os.fsync(handle.fileno())
            yield
        finally:
            lock_path.unlink(missing_ok=True)

    def _audit_data(self) -> dict[str, Any]:
        rows = []
        for index, item in enumerate(self.data):
            reason = None
            try:
                question_input(item)
                reference_answer(item)
            except (ValueError, TypeError, AttributeError) as exc:
                reason = str(exc) if str(exc) in {
                    "missing_question", "unknown_question_type", "invalid_options", "all_options_empty",
                    "invalid_option_letters", "invalid_reference",
                } else "invalid_record"
            if index in self.extra_exclusions:
                reason = self.extra_exclusions[index]
            rows.append({
                "index": index, "row_number": index + 1,
                "question_id": f"q{index + 1:03d}",
                "question_type": item.get("question_type") if isinstance(item, dict) else None,
                "selected": index in self.indices,
                "excluded": reason is not None,
                "exclusion_reason": reason,
                "sensitivity_excluded": index + 1 in self.config.sensitivity_excluded_rows,
            })
        return {
            "raw_count": len(self.data), "selected_count": len(self.indices),
            "dataset_sha256": _hash(self.data),
            "question_types": dict(Counter(str(x.get("question_type")) for x in self.data if isinstance(x, dict))),
            "rows": rows,
            "sensitivity_excluded_rows_1_based": list(self.config.sensitivity_excluded_rows),
            "note": "非空文本选项不保证图片或共享背景完整。敏感性名单在运行前确定，不按预测表现筛题。",
        }

    def _fingerprint_payload(self) -> dict[str, Any]:
        formal_questions = {re.sub(r"\s+", "", str(x.get("question", ""))) for x in self.data if isinstance(x, dict)}
        for example in SYNTHETIC_EXAMPLES:
            reference_answer(example)
            if re.sub(r"\s+", "", example["question"]) in formal_questions:
                raise ValueError("人工示例与正式题完全重复，不能开始实验。")
        return {
            "runner_version": RUNNER_VERSION,
            "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dataset_sha256": self.audit["dataset_sha256"],
            "model": self.config.model,
            "base_url": "https://api.deepseek.com",
            "settings": {"temperature": self.config.temperature, "max_tokens": self.config.max_tokens,
                         "thinking": "disabled", "response_format": {"type": "json_object"}},
            "methods": list(self.config.methods),
            "evaluation_indices_0_based": list(self.indices),
            "exclusions": self.exclusions,
            "sensitivity_excluded_rows_1_based": list(self.config.sensitivity_excluded_rows),
            "max_review_rounds": self.config.max_review_rounds,
            "agent_round_definition": "一次独立critic调用加一次controller调用；初答另计，最多1+2*rounds次。",
            "prompts": {"zero_shot": ZERO_PROMPT, "few_shot": FEW_PROMPT, "cot": COT_PROMPT,
                        "critic": CRITIC_PROMPT, "controller": CONTROLLER_PROMPT},
            "few_shot_examples": SYNTHETIC_EXAMPLES,
            "scoring": "严格字母集合完全匹配；单选长度1；无效输出/API失败在有效输入分母中计失败。",
        }

    def _initialize_manifest(self) -> None:
        if self.manifest_path.exists():
            previous = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if previous.get("fingerprint") != self.fingerprint:
                raise FingerprintMismatch("数据、代码、提示或实验设置已改变；必须使用新的实验目录，不能混合断点。")
        else:
            if self.ledger_path.exists() and self.ledger_path.stat().st_size:
                raise ExperimentError("存在无manifest的ledger，不能推测其归属。")
            _atomic_json(self.manifest_path, {"fingerprint": self.fingerprint, "created_at": _utc(),
                                              "experiment": self.fingerprint_payload})
            _atomic_json(self.output_dir / "source_data.json", self.data)
            _atomic_json(self.output_dir / "data_audit.json", self.audit)

    def _load_ledger(self) -> None:
        self.events: list[dict[str, Any]] = []
        if self.ledger_path.exists():
            content = self.ledger_path.read_bytes()
            if content and not content.endswith(b"\n"):
                raise ExperimentError("ledger末行不完整，需保留现场并检查；不能忽略可能已收费的调用。")
            for line_number, line in enumerate(content.splitlines(), 1):
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ExperimentError(f"ledger第{line_number}行损坏，停止恢复。") from exc
                if event.get("fingerprint") != self.fingerprint:
                    raise FingerprintMismatch("ledger包含其他实验的事件。")
                self.events.append(event)
        self._reindex()

    def _reindex(self) -> None:
        self.call_starts = {}
        self.call_finishes = {}
        self.stage_calls = {}
        self.question_attempts = {}
        self.latest_results = {}
        for event in self.events:
            kind = event["event"]
            if kind == "question_start":
                self.question_attempts[(event["method"], event["index"])] = event["question_attempt_id"]
            elif kind == "call_start":
                self.call_starts[event["call_id"]] = event
                stage_key = (event["question_attempt_id"], event["stage"])
                if stage_key in self.stage_calls:
                    raise ExperimentError("同一题目尝试阶段出现重复请求，停止以避免重复收费。")
                self.stage_calls[stage_key] = event["call_id"]
            elif kind == "call_finish":
                if event["call_id"] in self.call_finishes:
                    raise ExperimentError("同一调用出现多个finish事件。")
                self.call_finishes[event["call_id"]] = event
            elif kind == "question_result":
                self.latest_results[(event["method"], event["index"])] = event["result"]
        if set(self.call_finishes) - set(self.call_starts):
            raise ExperimentError("发现没有start的finish事件。")

    def _append(self, event: dict[str, Any]) -> None:
        event = {"timestamp": _utc(), "fingerprint": self.fingerprint, **event}
        with self.ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_json(event) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.events.append(event)
        self._reindex()

    def _check_client(self) -> None:
        if self.client is None:
            raise ExperimentError("未提供API客户端。")
        if getattr(self.client, "max_retries", None) != 0:
            raise ExperimentError("外层客户端必须显式设置 max_retries=0，避免隐藏重试。")
        endpoint = str(getattr(self.client, "base_url", "")).rstrip("/")
        if endpoint not in {"https://api.deepseek.com", "https://api.deepseek.com/v1"}:
            raise ExperimentError("仅允许官方 DeepSeek base_url；没有读取或输出客户端密钥。")

    @staticmethod
    def _compact_response(response: Any) -> dict[str, Any]:
        choices = getattr(response, "choices", None)
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        usage_object = getattr(response, "usage", None)
        usage = None
        if usage_object is not None:
            usage = {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(usage_object, key, None)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    usage[key] = value
        return {"content": content if isinstance(content, str) else "",
                "finish_reason": getattr(choice, "finish_reason", None),
                "model_returned": getattr(response, "model", None), "usage": usage or None}

    @staticmethod
    def _decode_response(response: Mapping[str, Any]) -> dict[str, Any]:
        if response.get("finish_reason") != "stop" or not response.get("content", "").strip():
            raise ResponseFormatError("输出为空、不完整或不是正常停止。")
        try:
            result = json.loads(response["content"])
        except json.JSONDecodeError as exc:
            raise ResponseFormatError("输出不是有效JSON。") from exc
        if not isinstance(result, dict):
            raise ResponseFormatError("输出必须是JSON对象。")
        return result

    def _call(self, method: str, index: int, attempt_id: str, stage: str,
              messages: list[dict[str, str]]) -> dict[str, Any]:
        request = {"model": self.config.model, "messages": messages,
                   "temperature": self.config.temperature, "max_tokens": self.config.max_tokens,
                   "extra_body": {"thinking": {"type": "disabled"}},
                   "response_format": {"type": "json_object"}}
        request_hash = _hash(request)
        cached_id = self.stage_calls.get((attempt_id, stage))
        if cached_id is not None:
            if self.call_starts[cached_id]["request_sha256"] != request_hash:
                raise FingerprintMismatch("恢复阶段的请求与原记录不一致。")
            finish = self.call_finishes.get(cached_id)
            if finish is None:
                raise UnresolvedCall("先前请求未确认是否完成；默认不重发，以避免重复费用。")
            if finish["status"] == "api_error":
                raise APICallError("先前API调用失败；默认不重发。")
            return self._decode_response(finish["response"])
        if len(self.call_starts) >= self.config.max_api_calls:
            raise BudgetExceeded("已达到累计API尝试次数上限；尚未发出本阶段请求。")
        self._check_client()
        call_id = uuid.uuid4().hex
        if self._verbose:
            print(f"[{method} 第{index + 1}条] {stage}；累计请求 "
                  f"{len(self.call_starts) + 1}/{self.config.max_api_calls}", flush=True)
        # start先持久化，它立即消耗一次预算；超时/断线也不退还。
        self._append({"event": "call_start", "call_id": call_id, "method": method, "index": index,
                      "question_attempt_id": attempt_id, "stage": stage,
                      "request_sha256": request_hash, "request": request})
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            # 不保存异常原文、请求头、客户端对象或traceback，避免凭据意外出现在日志。
            status_code = getattr(exc, "status_code", None)
            if not isinstance(status_code, int):
                status_code = None
            self._append({"event": "call_finish", "call_id": call_id, "status": "api_error",
                          "elapsed_seconds": time.perf_counter() - started,
                          "error_type": type(exc).__name__, "status_code": status_code})
            raise APICallError("API请求失败，详情仅记录错误类型与状态码；没有自动重试。") from None
        compact = self._compact_response(response)
        self._append({"event": "call_finish", "call_id": call_id, "status": "ok",
                      "elapsed_seconds": time.perf_counter() - started, "response": compact})
        return self._decode_response(compact)

    def _solve(self, method: str, index: int, attempt_id: str, payload: Mapping[str, Any]):
        if method != "agent":
            result = self._call(method, index, attempt_id, "answer", answer_messages(method, payload))
            draft = _draft(result, payload)
            return "".join(draft["answer"]), [{"stage": "answer", "draft": draft}], "single_prompt"
        draft = _draft(self._call(method, index, attempt_id, "initial", answer_messages("cot", payload)), payload)
        trace = [{"stage": "initial", "draft": draft}]
        if self.config.max_review_rounds == 0:
            return "".join(draft["answer"]), trace, "review_disabled_ablation"
        for round_number in range(1, self.config.max_review_rounds + 1):
            feedback = self._call(method, index, attempt_id, f"critic_{round_number}", [
                {"role": "system", "content": CRITIC_PROMPT},
                {"role": "user", "content": _json({"question": payload, "candidate": draft})},
            ])
            issues = feedback.get("issues")
            recommendation = feedback.get("recommendation")
            if (not isinstance(issues, list) or any(not isinstance(x, str) for x in issues)
                    or recommendation not in {"accept", "revise"}):
                raise ResponseFormatError("审查反馈字段不合法。")
            feedback = {"issues": issues, "recommendation": recommendation}
            decision = self._call(method, index, attempt_id, f"controller_{round_number}", [
                {"role": "system", "content": CONTROLLER_PROMPT},
                {"role": "user", "content": _json({"question": payload, "candidate": draft,
                    "feedback": feedback, "review_round": round_number,
                    "remaining_review_rounds": self.config.max_review_rounds - round_number})},
            ])
            action = decision.get("action")
            reason = decision.get("decision_reason")
            if action not in {"finish", "revise"} or not isinstance(reason, str) or not reason.strip():
                raise ResponseFormatError("控制器必须给出finish/revise及非空决策理由。")
            next_draft = _draft(decision, payload)
            if action == "finish" and next_draft["answer"] != draft["answer"]:
                raise ResponseFormatError("finish必须保留候选答案；更改答案应选择revise。")
            trace.append({"stage": "review", "round": round_number, "feedback": feedback,
                          "action": action, "decision_reason": reason, "draft": next_draft})
            draft = next_draft
            if action == "finish":
                return "".join(draft["answer"]), trace, "agent_chose_finish"
        return "".join(draft["answer"]), trace, "max_review_rounds"

    def run(self, methods: Iterable[str] | None = None, *,
            retry_failed: Iterable[tuple[str, int]] = (), verbose: bool = True) -> dict[str, Any]:
        """运行已预注册的方法；retry_failed中的题目位置从1开始。

        默认跳过已有成功或已知失败结果。若显式重试，必须指定失败/未确认项；
        不允许仅因答错而重试成功请求。预算受同一累计上限约束。
        """
        selected_methods = tuple(self.config.methods if methods is None else methods)
        self._verbose = bool(verbose)
        if not selected_methods or len(set(selected_methods)) != len(selected_methods) or not set(selected_methods) <= set(self.config.methods):
            raise ValueError("run方法必须属于预注册方法且不能重复。")
        retry_keys = {(method, row - 1) for method, row in retry_failed}
        if any(method not in selected_methods or index not in self.valid_indices for method, index in retry_keys):
            raise ValueError("重试必须指定本次方法中的有效题目，位置从1开始。")
        with self._lock():
            self._load_ledger()
            for key in retry_keys:
                previous = self.latest_results.get(key)
                if previous is None or previous["status"] not in RETRYABLE_STATUSES:
                    raise ValueError("只允许显式重试已记录的API/解析失败或未确认项；不能挑选答错题重跑。")
            self._append({"event": "run_start", "max_api_calls": self.config.max_api_calls,
                          "methods": selected_methods, "retry_failed": [[m, i + 1] for m, i in sorted(retry_keys)]})
            try:
                for method in selected_methods:
                    for index in self.valid_indices:
                        key = (method, index)
                        previous = self.latest_results.get(key)
                        explicit_retry = key in retry_keys
                        if previous and previous["status"] in TERMINAL_STATUSES | {"interrupted"} and not explicit_retry:
                            continue
                        attempt_id = self.question_attempts.get(key)
                        if attempt_id is None or explicit_retry:
                            retry_of = attempt_id
                            attempt_id = uuid.uuid4().hex
                            self._append({"event": "question_start", "method": method, "index": index,
                                          "question_attempt_id": attempt_id, "retry_of": retry_of})
                        started = time.perf_counter()
                        status = "completed"
                        prediction = None
                        trace = []
                        stop_reason = None
                        error_code = None
                        try:
                            prediction, trace, stop_reason = self._solve(
                                method, index, attempt_id, question_input(self.data[index]))
                        except BudgetExceeded:
                            status, error_code = "budget_exhausted", "cumulative_call_limit"
                        except UnresolvedCall:
                            status, error_code = "interrupted", "unconfirmed_previous_call"
                        except APICallError:
                            status, error_code = "api_error", "api_request_failed"
                        except ResponseFormatError:
                            status, error_code = "parse_error", "invalid_model_response"
                        # 只有求解结束后才访问标准答案，求解函数及反馈从未收到该字段。
                        gold = reference_answer(self.data[index])
                        result = {"method": method, "index": index, "row_number": index + 1,
                                  "question_id": f"q{index + 1:03d}", "status": status,
                                  "question_attempt_id": attempt_id, "prediction": prediction,
                                  "reference": gold, "correct": status == "completed" and prediction == gold,
                                  "trace": trace, "stop_reason": stop_reason, "error_code": error_code,
                                  "invocation_elapsed_seconds": time.perf_counter() - started,
                                  "question_type": self.data[index]["question_type"]}
                        self._append({"event": "question_result", "method": method, "index": index,
                                      "question_attempt_id": attempt_id, "result": result})
                        self._sync_exports()
                        if self._verbose:
                            print(f"[{method} 第{index + 1}条] {status}；"
                                  f"已保存；累计调用 {len(self.call_starts)}", flush=True)
                        if status == "budget_exhausted":
                            return self.summarize()
                return self.summarize()
            finally:
                self._sync_exports()

    def _call_stats(self, method: str, index: int | None = None) -> dict[str, Any]:
        starts = [x for x in self.call_starts.values() if x["method"] == method and (index is None or x["index"] == index)]
        totals = Counter()
        unknown_usage = 0
        unresolved = 0
        elapsed = 0.0
        for start in starts:
            finish = self.call_finishes.get(start["call_id"])
            if finish is None:
                unresolved += 1
                unknown_usage += 1
                continue
            elapsed += finish["elapsed_seconds"]
            usage = finish.get("response", {}).get("usage")
            if not usage or any(key not in usage for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
                unknown_usage += 1
            for field, value in (usage or {}).items():
                totals[field] += value
        return {"api_attempts": len(starts), "unresolved_calls": unresolved,
                "known_prompt_tokens": totals["prompt_tokens"],
                "known_completion_tokens": totals["completion_tokens"],
                "known_total_tokens": totals["total_tokens"],
                "calls_with_unknown_token_usage": unknown_usage,
                "known_api_elapsed_seconds": elapsed}

    def rows(self) -> list[dict[str, Any]]:
        records = []
        for method in self.config.methods:
            for index in self.indices:
                record = dict(self.latest_results.get((method, index), {}))
                if not record:
                    status = "excluded" if index in self.exclusions else (
                        "inflight" if (method, index) in self.question_attempts else "not_started")
                    record = {"method": method, "index": index, "row_number": index + 1,
                              "question_id": f"q{index + 1:03d}", "status": status,
                              "prediction": None, "correct": False,
                              "question_type": self.data[index].get("question_type"),
                              "exclusion_reason": self.exclusions.get(index)}
                record["fingerprint"] = self.fingerprint
                record["sensitivity_excluded"] = index + 1 in self.config.sensitivity_excluded_rows
                record.update(self._call_stats(method, index))
                records.append(record)
        return records

    def summarize(self) -> dict[str, Any]:
        rows = self.rows()
        methods = {}
        for method in self.config.methods:
            selected = [r for r in rows if r["method"] == method]
            valid = [r for r in selected if r["index"] not in self.exclusions]
            sensitivity = [r for r in valid if not r["sensitivity_excluded"]]
            counts = Counter(r["status"] for r in valid)
            correct = sum(r["correct"] for r in valid)
            finished = sum(counts[status] for status in TERMINAL_STATUSES)
            all_finished = finished == len(valid) and bool(valid)
            sensitivity_finished = bool(sensitivity) and all(r["status"] in TERMINAL_STATUSES for r in sensitivity)
            sensitivity_correct = sum(r["correct"] for r in sensitivity)
            stats = self._call_stats(method)
            methods[method] = {
                "raw_dataset_count": len(self.data), "selected_count": len(selected),
                "excluded_input_count": len(selected) - len(valid), "valid_input_count": len(valid),
                "correct_count": correct, "status_counts": dict(counts), "all_valid_inputs_finished": all_finished,
                "accuracy_valid_inputs": correct / len(valid) if all_finished else None,
                "verified_correct_fraction_valid_inputs": correct / len(valid) if valid else None,
                "verified_correct_fraction_raw_dataset": correct / len(self.data),
                "finished_outcome_coverage_valid_inputs": finished / len(valid) if valid else None,
                "prediction_coverage_raw_dataset": counts["completed"] / len(self.data),
                "accuracy_on_finished_outcomes": correct / finished if finished else None,
                "sensitivity_count": len(sensitivity), "sensitivity_correct_count": sensitivity_correct,
                "sensitivity_all_finished": sensitivity_finished,
                "sensitivity_accuracy": sensitivity_correct / len(sensitivity) if sensitivity_finished else None,
                **stats,
            }
        return {"fingerprint": self.fingerprint, "generated_at": _utc(),
                "cumulative_api_attempts": len(self.call_starts),
                "current_max_api_calls": self.config.max_api_calls,
                "remaining_call_allowance": max(0, self.config.max_api_calls - len(self.call_starts)),
                "methods": methods,
                "notes": [
                    "有效输入准确率仅在该方法全部有效输入取得完成/明确失败结果后发布；API/解析失败计错。",
                    "未完成时verified_correct_fraction只是已验证正确比例，不是最终实验准确率。",
                    "accuracy_on_finished_outcomes可能受未完成题选择偏差影响，不能代替共同完整评测集比较。",
                    "调用次数和已知token包含失败及显式重试；未知token没有伪装成0成本。",
                    "主评测有文本选项不保证图表/共享背景完整；敏感性名单已预注册。",
                    "few-shot示例是独立人工构造算术/单位换算，知识难度与真实药师考试不同。",
                ]}

    def _sync_exports(self) -> None:
        rows = self.rows()
        temporary = self.output_dir / ("results." + uuid.uuid4().hex + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in rows:
                handle.write(_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.output_dir / "results.jsonl")
        summary = self.summarize()
        _atomic_json(self.output_dir / "summary.json", summary)
        csv_path = self.output_dir / "summary.csv"
        fields = ["method", "valid_input_count", "correct_count", "accuracy_valid_inputs",
                  "sensitivity_count", "sensitivity_accuracy", "prediction_coverage_raw_dataset",
                  "api_attempts", "known_total_tokens", "calls_with_unknown_token_usage", "known_api_elapsed_seconds"]
        temporary_csv = csv_path.with_name("summary." + uuid.uuid4().hex + ".tmp")
        with temporary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for method, values in summary["methods"].items():
                writer.writerow({"method": method, **{k: values.get(k) for k in fields if k != "method"}})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_csv, csv_path)
