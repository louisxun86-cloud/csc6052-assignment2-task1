"""离线测试：FakeClient不连接网络、不使用任何密钥。"""
import json
import tempfile
import types
import unittest
from pathlib import Path

from runner import (
    APICallError, COMMON_PROMPT, CONTROLLER_PROMPT, COT_PROMPT, CRITIC_PROMPT,
    ExperimentConfig, ExperimentError, ExperimentRunner, FingerprintMismatch,
    ResponseFormatError, parse_choices, question_input, reference_answer,
)


def item(question="示例题", answer="A", multiple=False):
    return {"question": question, "question_type": "多项选择题" if multiple else "最佳选择题",
            "option": {"A": "甲", "B": "乙", "C": "丙", "D": "丁", "E": ""},
            "answer": answer, "analysis": "SECRET_GOLD_ANALYSIS_DO_NOT_SEND", "source": "offline_fixture"}


def response(obj, reason="stop", usage=True):
    content = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return types.SimpleNamespace(
        model="fake-model", choices=[types.SimpleNamespace(
            finish_reason=reason, message=types.SimpleNamespace(content=content))],
        usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15) if usage else None,
    )


class FakeClient:
    base_url = "https://api.deepseek.com"
    max_retries = 0

    def __init__(self, outputs=None):
        self.calls = []
        self.outputs = list(outputs or [])
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.outputs:
            value = self.outputs.pop(0)
            if isinstance(value, BaseException):
                raise value
            return response(value)
        prompt = kwargs["messages"][0]["content"]
        payload = json.loads(kwargs["messages"][-1]["content"])
        if prompt == CRITIC_PROMPT:
            return response({"issues": [], "recommendation": "accept"})
        if prompt == CONTROLLER_PROMPT:
            return response({"action": "finish", **payload["candidate"], "decision_reason": "未发现有依据的错误。"})
        return response({"answer": ["A"], "brief_reason": "离线模拟依据"})


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = Path(__file__).resolve().parent / ".test_tmp"
        self.temp_root.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=self.temp_root)
        self.directory = Path(self.tmp.name) / "experiment"

    def tearDown(self):
        assert Path(self.tmp.name).resolve().is_relative_to(self.temp_root.resolve())
        self.tmp.cleanup()

    def make_runner(self, data=None, client=None, cap=20, methods=("zero_shot",), **kwargs):
        config = ExperimentConfig(model="fake-model", max_api_calls=cap, methods=methods, **kwargs)
        return ExperimentRunner(data or [item()], client or FakeClient(), self.directory, config)

    def test_real_dataset_audit_and_strict_parser(self):
        path = Path(__file__).resolve().parents[1] / "tmp/assignment2_translation/course_task1.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        run = self.make_runner(data=data, cap=0, methods=("zero_shot", "few_shot", "cot", "agent"))
        self.assertEqual(len(run.valid_indices), 97)
        self.assertEqual(set(run.exclusions), {20, 25, 65})
        self.assertEqual(run.summarize()["methods"]["agent"]["sensitivity_count"], 90)
        for i in run.valid_indices:
            payload = question_input(data[i])
            self.assertEqual(set(payload), {"question", "question_type", "options"})
            self.assertEqual(parse_choices(list(reversed(data[i]["answer"])), payload), reference_answer(data[i]))
        for invalid in [[], "A", ["Z"], ["A", "A"], ["A", "B"], ["E"], [1]]:
            with self.assertRaises(ResponseFormatError):
                parse_choices(invalid, question_input(item()))

    def test_default_budget_zero_calls(self):
        client = FakeClient()
        run = self.make_runner(client=client, cap=0)
        summary = run.run(verbose=False)
        self.assertEqual(len(client.calls), 0)
        self.assertEqual(run.rows()[0]["status"], "budget_exhausted")
        self.assertIsNone(summary["methods"]["zero_shot"]["accuracy_valid_inputs"])

    def test_four_methods_no_leak_and_skip_success_on_resume(self):
        client = FakeClient()
        methods = ("zero_shot", "few_shot", "cot", "agent")
        run = self.make_runner(client=client, methods=methods)
        summary = run.run(verbose=False)
        self.assertEqual(len(client.calls), 6)
        self.assertEqual([r["status"] for r in run.rows()], ["completed"] * 4)
        for request in client.calls:
            self.assertNotIn("SECRET_GOLD_ANALYSIS_DO_NOT_SEND", json.dumps(request))
            self.assertEqual(request["extra_body"], {"thinking": {"type": "disabled"}})
            self.assertEqual(request["temperature"], 0)
        # Agent初答与CoT对照请求完全相同。
        self.assertEqual(client.calls[2], client.calls[3])
        self.assertEqual(summary["methods"]["agent"]["api_attempts"], 3)
        resumed = self.make_runner(client=client, methods=methods)
        resumed.run(verbose=False)
        self.assertEqual(len(client.calls), 6)
        with self.assertRaises(ValueError):
            resumed.run(retry_failed=[("zero_shot", 1)], verbose=False)

    def test_agent_two_rounds_maximum_five_calls(self):
        client = FakeClient([
            {"answer": ["B"], "brief_reason": "初答"},
            {"issues": ["重新审题"], "recommendation": "revise"},
            {"action": "revise", "answer": ["A"], "brief_reason": "修订", "decision_reason": "采纳意见"},
            {"issues": [], "recommendation": "accept"},
            {"action": "revise", "answer": ["A"], "brief_reason": "补充依据", "decision_reason": "完善说明"},
        ])
        run = self.make_runner(client=client, methods=("agent",), cap=5)
        run.run(verbose=False)
        row = run.rows()[0]
        self.assertEqual(len(client.calls), 5)
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["stop_reason"], "max_review_rounds")
        self.assertEqual(len(row["trace"]), 3)

    def test_budget_resume_replays_stages_without_duplicate_api(self):
        client = FakeClient()
        run = self.make_runner(client=client, methods=("agent",), cap=1)
        run.run(verbose=False)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(run.rows()[0]["status"], "budget_exhausted")
        resumed = self.make_runner(client=client, methods=("agent",), cap=3)
        resumed.run(verbose=False)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(resumed.rows()[0]["status"], "completed")

    def test_api_failure_requires_explicit_retry_and_counts_cost(self):
        client = FakeClient([RuntimeError("SECRET_KEY_MUST_NOT_BE_LOGGED")])
        run = self.make_runner(client=client, cap=2)
        first = run.run(verbose=False)
        self.assertEqual(first["methods"]["zero_shot"]["accuracy_valid_inputs"], 0)
        self.assertEqual(first["methods"]["zero_shot"]["calls_with_unknown_token_usage"], 1)
        self.assertNotIn("SECRET_KEY_MUST_NOT_BE_LOGGED", run.ledger_path.read_text(encoding="utf-8"))
        resumed = self.make_runner(client=client, cap=2)
        resumed.run(verbose=False)
        self.assertEqual(len(client.calls), 1)
        final = resumed.run(retry_failed=[("zero_shot", 1)], verbose=False)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(final["methods"]["zero_shot"]["api_attempts"], 2)
        self.assertEqual(final["methods"]["zero_shot"]["accuracy_valid_inputs"], 1)

    def test_unconfirmed_call_never_automatically_reissued(self):
        client = FakeClient([KeyboardInterrupt()])
        run = self.make_runner(client=client, cap=2)
        with self.assertRaises(KeyboardInterrupt):
            run.run(verbose=False)
        resumed = self.make_runner(client=client, cap=2)
        resumed.run(verbose=False)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(resumed.rows()[0]["status"], "interrupted")
        self.assertEqual(resumed.summarize()["methods"]["zero_shot"]["unresolved_calls"], 1)
        resumed.run(retry_failed=[("zero_shot", 1)], verbose=False)
        self.assertEqual(len(client.calls), 2)

    def test_success_finish_without_result_rebuilds_without_new_call(self):
        client = FakeClient()
        run = self.make_runner(client=client)
        run.run(verbose=False)
        events = [e for e in run.events if e["event"] != "question_result"]
        run.ledger_path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events), encoding="utf-8")
        resumed = self.make_runner(client=client)
        resumed.run(verbose=False)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(resumed.rows()[0]["status"], "completed")

    def test_parse_errors_count_as_failures(self):
        client = FakeClient(["not json"])
        run = self.make_runner(client=client)
        summary = run.run(verbose=False)
        self.assertEqual(run.rows()[0]["status"], "parse_error")
        self.assertEqual(summary["methods"]["zero_shot"]["accuracy_valid_inputs"], 0)
        self.assertEqual(summary["methods"]["zero_shot"]["known_total_tokens"], 15)

    def test_client_configuration_is_enforced_before_spend(self):
        client = FakeClient()
        client.max_retries = 2
        run = self.make_runner(client=client)
        with self.assertRaises(ExperimentError):
            run.run(verbose=False)
        self.assertEqual(len(client.calls), 0)
        self.assertEqual(len(run.call_starts), 0)

    def test_fingerprint_change_rejected_but_budget_change_allowed(self):
        self.make_runner(cap=0)
        self.make_runner(cap=8)
        with self.assertRaises(FingerprintMismatch):
            self.make_runner(cap=8, max_tokens=512)
        with self.assertRaises(FingerprintMismatch):
            self.make_runner(data=[item(question="另一道题")])

    def test_ledger_truncation_and_lock_fail_closed(self):
        run = self.make_runner(cap=0)
        run.ledger_path.write_text('{"event":', encoding="utf-8")
        with self.assertRaises(ExperimentError):
            self.make_runner(cap=0)
        run.ledger_path.write_text("", encoding="utf-8")
        (self.directory / ".runner.lock").write_text("active", encoding="utf-8")
        with self.assertRaises(ExperimentError):
            self.make_runner(cap=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
