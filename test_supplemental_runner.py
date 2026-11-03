"""补充实验的离线回归；仅使用假的客户端，绝不调用网络。"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import supplemental_runner as sr


ROOT = Path(__file__).resolve().parent
FORMAL = ROOT / "actual_run/task1_assignment/formal"
if not FORMAL.is_dir():
    FORMAL = ROOT.parent / "results/formal"
if not FORMAL.is_dir():
    FORMAL = ROOT / "task1_assignment/formal"
PROTOCOL = ROOT / "supplemental_protocol_candidate.json"
if not PROTOCOL.exists():
    PROTOCOL = ROOT / "supplemental_protocol.json"
TEMP_ROOT = ROOT / ".test_tmp"
TEMP_ROOT.mkdir(exist_ok=True)


class MockClient:
    max_retries = 0
    base_url = "https://api.deepseek.com"

    def __init__(self, jobs, behavior="correct"):
        self.chat = SimpleNamespace(completions=self)
        self.by_request = {sr.digest(job["request"]): job for job in jobs}
        self.calls = []
        self.behavior = behavior

    def create(self, **request):
        self.calls.append(request)
        if self.behavior == "error":
            raise RuntimeError("SECRET_VALUE_SHOULD_NOT_BE_LOGGED")
        job = self.by_request[sr.digest(request)]
        obj = {"answer": list(job["reference"]), "brief_reason": "离线测试假响应。"}
        if job["study"] == "B":
            obj.update(action="revise", decision_reason="离线测试决策。")
        content = json.dumps(obj) if self.behavior != "malformed" else "{broken"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
            model="deepseek-flash",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


class SupplementalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.source = sr.load_protocol(PROTOCOL, FORMAL)
        cls.jobs = sr.make_jobs(cls.protocol, cls.source)
        cls.source_hashes = {p: sr.file_digest(FORMAL / p) for p in cls.source["hashes"]}

    def make_engine(self, directory, client=None, budget=240):
        return sr.SupplementalRun(client or MockClient(self.jobs), PROTOCOL, FORMAL, directory, budget)

    def test_frozen_counts_and_gold_independent_eligibility(self):
        self.assertEqual(len(self.jobs), 222)
        self.assertEqual(sum(j["study"] == "A" for j in self.jobs), 32)
        self.assertEqual(len(self.source["cases"]), 95)
        self.assertEqual([x["row_number"] for x in self.source["invalid"]], [14, 67])
        self.assertEqual(len({j["job_id"] for j in self.jobs}), 222)
        self.assertEqual(self.protocol["protocol"]["condition_order_seed"], 605202)
        self.assertEqual(sr.make_jobs(self.protocol, self.source), self.jobs)
        for study, key in (("A", "study_A"), ("B", "study_B")):
            orders = list(self.protocol["protocol"][key]["condition_order_by_unit"].values())
            self.assertGreater(len({tuple(order) for order in orders}), 1)

    def test_request_isolation_and_only_feedback_intervention(self):
        for item in self.protocol["protocol"]["synthetic_items"]["items"]:
            first, second = sr.a_messages("cot_v1", item), sr.a_messages("cot_v2", item)
            self.assertEqual(first, sr.frozen.answer_messages("cot", sr.frozen.question_input(item)))
            self.assertEqual(first[1:], second[1:])
            self.assertEqual(set(json.loads(first[1]["content"])), {"question", "options", "question_type"})
        for case in self.source["cases"]:
            first, second = sr.b_messages("feedback", case), sr.b_messages("null_feedback", case)
            self.assertEqual(first[0], second[0])
            a, b = json.loads(first[1]["content"]), json.loads(second[1]["content"])
            self.assertEqual(a.pop("feedback"), case["feedback"])
            self.assertIsNone(b.pop("feedback"))
            self.assertEqual(a, b)
            self.assertEqual(a["remaining_review_rounds"], 0)
            self.assertEqual(a["review_round"], 1)
            self.assertEqual(set(a["question"]), {"question", "options", "question_type"})
            self.assertEqual(set(a["candidate"]), {"answer", "brief_reason"})
        for job in self.jobs:
            self.assertEqual(set(job["request"]), set(sr.SETTINGS) | {"messages"})

    def test_complete_mock_run_denominators_accounting_and_resume(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            client = MockClient(self.jobs)
            engine = self.make_engine(directory, client)
            summary = engine.run(verbose=False)
            self.assertTrue(summary["all_finished"])
            self.assertEqual(len(client.calls), 222)
            self.assertEqual(len(engine.rows()), 226)
            self.assertEqual(summary["totals"]["known_total_tokens"], 3330)
            self.assertEqual(summary["totals"]["calls_with_unknown_usage"], 0)
            for condition in sr.A_CONDITIONS:
                self.assertEqual(summary["study_A"][condition]["all_16"]["correct"], 16)
                self.assertEqual(summary["study_A"][condition]["held_out_probe"]["n"], 8)
            for condition in sr.B_CONDITIONS:
                self.assertEqual(summary["study_B"][condition]["conditional_95"]["accuracy"], 1)
                self.assertEqual(summary["study_B"][condition]["overall_97"]["accuracy"], 95 / 97)
                self.assertEqual(summary["study_B"][condition]["overall_97"]["status_counts"]["common_initial_failure"], 2)
            primary = summary["supplementary_primary_comparisons"]
            self.assertEqual(set(primary), {"A_all_16", "B_conditional_95"})
            self.assertEqual(primary["A_all_16"]["n"], 16)
            self.assertEqual(primary["B_conditional_95"]["n"], 95)
            self.assertTrue(all(v["holm_adjusted_p_two_comparison_family"] == 1 for v in primary.values()))
            resumed = self.make_engine(directory, client)
            resumed.run(verbose=False)
            self.assertEqual(len(client.calls), 222)
            self.assertEqual({p: sr.file_digest(FORMAL / p) for p in self.source_hashes}, self.source_hashes)

    def test_zero_budget_and_no_retry_on_error_or_parse_failure(self):
        for behavior, status in (("error", "api_error"), ("malformed", "parse_error")):
            with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
                client = MockClient(self.jobs, behavior)
                engine = self.make_engine(directory, client, budget=0)
                engine.run(verbose=False)
                self.assertEqual(client.calls, [])
                engine.max_api_calls = 1
                engine.run(verbose=False)
                self.assertEqual(engine.rows()[0]["status"], status)
                self.assertFalse(engine.rows()[0]["correct"])
                engine.run(verbose=False)
                self.assertEqual(len(client.calls), 1)
                engine.max_api_calls = 2
                engine.run(verbose=False)
                self.assertEqual(len(client.calls), 2)
                self.assertNotEqual(client.calls[0], client.calls[1])
                self.assertNotIn("SECRET_VALUE_SHOULD_NOT_BE_LOGGED", (Path(directory) / "ledger.jsonl").read_text(encoding="utf-8"))

    def test_unknown_start_is_never_resent_and_consumes_budget(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            client = MockClient(self.jobs)
            engine = self.make_engine(directory, client, budget=1)
            job = engine.jobs[0]
            engine.append({"event": "call_start", "attempt_id": "unknown-test", "job_id": job["job_id"],
                           "request_sha256": sr.digest(job["request"]), "request": job["request"]})
            summary = engine.run(verbose=False)
            self.assertEqual(client.calls, [])
            self.assertEqual(engine.rows()[0]["status"], "unknown")
            self.assertEqual(summary["totals"]["api_attempts"], 1)
            self.assertEqual(summary["totals"]["unresolved_calls"], 1)
            self.assertFalse(summary["all_finished"])

    def test_controller_rules_and_incomplete_outputs(self):
        job = next(j for j in self.jobs if j["study"] == "B")
        changed = next(letter for letter in job["payload"]["options"] if letter not in job["candidate"]["answer"])
        obj = {"answer": [changed], "brief_reason": "x", "action": "finish", "decision_reason": "x"}
        finish = {"status": "ok", "response": {"content": json.dumps(obj), "finish_reason": "stop"}}
        self.assertEqual(sr._evaluate(job, finish)["status"], "parse_error")
        obj["answer"] = job["candidate"]["answer"]
        finish["response"]["content"] = json.dumps(obj)
        self.assertEqual(sr._evaluate(job, finish)["status"], "completed")
        finish["response"]["finish_reason"] = "length"
        self.assertEqual(sr._evaluate(job, finish)["status"], "parse_error")

    def test_two_sided_exact_mcnemar_and_holm(self):
        rows = []
        for index in range(8):
            rows += [{"condition": "v2", "unit_id": str(index), "status": "completed", "correct": True},
                     {"condition": "v1", "unit_id": str(index), "status": "completed", "correct": False}]
        result = sr.SupplementalRun.paired(rows, "v2", "v1")
        self.assertEqual(result["mcnemar_exact_two_sided_p"], 2 / 256)
        adjusted = sr.SupplementalRun.holm_two({"a": result, "b": {"all_finished": True, "mcnemar_exact_two_sided_p": 0.03}})
        self.assertEqual(adjusted["a"]["holm_adjusted_p_two_comparison_family"], 4 / 256)
        self.assertEqual(adjusted["b"]["holm_adjusted_p_two_comparison_family"], 0.03)

    def test_guardrails_protocol_client_and_truncated_ledger(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            path = Path(directory)
            client = MockClient(self.jobs)
            client.max_retries = 2
            engine = self.make_engine(path / "out", client)
            with self.assertRaises(ValueError):
                engine.run(verbose=False)
            self.assertEqual(client.calls, [])
            altered = json.loads(PROTOCOL.read_text(encoding="utf-8"))
            altered["protocol"]["max_api_calls"] = 999
            (path / "bad.json").write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaises(ValueError):
                sr.load_protocol(path / "bad.json", FORMAL)
            (path / "truncated.jsonl").write_bytes(b'{"event":"call_start"}')
            with self.assertRaises(ValueError):
                sr.read_jsonl(path / "truncated.jsonl")
            with self.assertRaises(FileExistsError):
                sr.prepare_protocol(FORMAL, ROOT / "supplemental_items.json", PROTOCOL)


if __name__ == "__main__":
    unittest.main()
