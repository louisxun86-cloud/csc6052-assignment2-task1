"""Synthetic offline tests only; fixtures live in temporary directories.

Run: python -m unittest discover -s assignment2_task1 -p test_analyze_results.py -v
No API clients are created. Synthetic metrics are never saved as real results.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import analyze_results as analysis


def write_fixture(root: Path):
    data = [
        {"question": f"SYNTHETIC OFFLINE TEST ONLY {i + 1}", "option": {"A": "one", "B": "two"},
         "question_type": analysis.TYPES[i % len(analysis.TYPES)], "answer": "A", "analysis": "DO NOT SEND TO MODEL",
         "source": "synthetic test"}
        for i in range(100)
    ]
    for row in analysis.EMPTY_ROWS:
        data[row - 1]["option"] = {letter: "" for letter in "ABCDE"}
    main = [i for i in range(100) if i + 1 not in analysis.EMPTY_ROWS]
    experiment = {
        "dataset_sha256": analysis.canonical_hash(data), "methods": list(analysis.METHODS),
        "evaluation_indices_0_based": list(range(100)),
        "exclusions": {str(i - 1): "all_options_empty" for i in analysis.EMPTY_ROWS},
        "sensitivity_excluded_rows_1_based": sorted(analysis.SENSITIVITY_ROWS),
        "model": "SYNTHETIC_OFFLINE_FIXTURE", "settings": {"temperature": 0},
    }
    fingerprint = analysis.canonical_hash(experiment)
    manifest = {"fingerprint": fingerprint, "experiment": experiment}
    events, rows, method_summaries = [], [], {}
    def append(event):
        events.append({"fingerprint": fingerprint, **event})
    def call(method, i, attempt, stage, content, status="ok"):
        call_id = f"{attempt}-{stage}"
        append({"event": "call_start", "call_id": call_id, "method": method, "index": i,
                "question_attempt_id": attempt, "stage": stage})
        finish = {"event": "call_finish", "call_id": call_id, "status": status, "elapsed_seconds": 1.0}
        if status == "ok":
            finish["response"] = {"content": content, "finish_reason": "stop", "model_returned": "SYNTHETIC_OFFLINE_FIXTURE",
                                  "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
        append(finish)
    for method in analysis.METHODS:
        correct_count = 0
        for i, item in enumerate(data):
            base = {"method": method, "index": i, "row_number": i + 1, "question_id": f"q{i + 1:03d}",
                    "question_type": item["question_type"], "fingerprint": fingerprint}
            if i not in main:
                rows.append({**base, "status": "excluded", "prediction": None, "correct": False})
                continue
            attempt = f"{method}-{i}"
            prediction, initial, status = "A", "A", "completed"
            if method == "few_shot" and i == 0:
                prediction = "B"
            if method == "cot" and i in (0, 1):
                prediction = "B"
            trace = []
            if method == "agent":
                if i == 0:
                    initial = "B"  # wrong -> correct
                elif i == 1:
                    prediction = "B"  # correct -> wrong
                elif i == 2:
                    initial = prediction = "B"  # persistent error
                elif i == 3:
                    prediction, status = None, "parse_error"  # recover initial from ledger
                elif i == 4:
                    prediction, status = None, "api_error"  # initial unavailable
                call(method, i, attempt, "initial", json.dumps({"answer": [initial]}),
                     "api_error" if status == "api_error" else "ok")
                if status == "parse_error":
                    call(method, i, attempt, "controller_1", "not valid json")
                if status == "completed":
                    trace = [{"stage": "initial", "draft": {"answer": [initial]}},
                             {"stage": "review", "round": 1,
                              "action": "revise" if initial != prediction else "finish",
                              "draft": {"answer": [prediction]}, "decision_reason": "synthetic"}]
            else:
                call(method, i, attempt, "answer", json.dumps({"answer": [prediction]}))
                trace = [{"stage": "answer", "draft": {"answer": [prediction]}}]
            correct = status == "completed" and prediction == "A"
            correct_count += correct
            result = {**base, "status": status, "prediction": prediction, "reference": "A", "correct": correct,
                      "question_attempt_id": attempt, "trace": trace, "stop_reason": "synthetic_fixture"}
            append({"event": "question_result", "method": method, "index": i,
                    "question_attempt_id": attempt, "result": {k: v for k, v in result.items() if k != "fingerprint"}})
            rows.append(result)
        method_summaries[method] = {"all_valid_inputs_finished": True, "valid_input_count": 97,
                                    "correct_count": correct_count, "sensitivity_count": 90}
    summary = {"fingerprint": fingerprint, "methods": method_summaries,
               "cumulative_api_attempts": sum(e["event"] == "call_start" for e in events)}
    root.mkdir(parents=True, exist_ok=True)
    for filename, value in (("source.json", data), ("manifest.json", manifest), ("summary.json", summary)):
        (root / filename).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    for filename, values in (("results.jsonl", rows), ("ledger.jsonl", events)):
        (root / filename).write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf-8")
    return root / "source.json"


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(__file__).resolve().parent
        self.temp = tempfile.TemporaryDirectory(prefix=".synthetic_analysis_test_", dir=self.workspace)
        self.root = Path(self.temp.name)
        assert self.root.resolve().is_relative_to(self.workspace)
        self.data_path = write_fixture(self.root)

    def tearDown(self):
        assert self.root.resolve().is_relative_to(self.workspace)
        self.temp.cleanup()

    def load(self):
        return analysis.load_run(self.root, self.data_path)

    def test_statistical_formulas(self):
        lo, hi = analysis.wilson_interval(50, 100)
        self.assertAlmostEqual(lo, 0.4038315304, places=8)
        self.assertAlmostEqual(hi, 0.5961684696, places=8)
        self.assertEqual(analysis.mcnemar_exact(0, 0), 1)
        self.assertEqual(analysis.mcnemar_exact(3, 3), 1)
        self.assertEqual(analysis.mcnemar_exact(6, 0), 0.03125)
        self.assertEqual(analysis.mcnemar_exact(0, 6), 0.03125)
        self.assertEqual(analysis.holm_adjust([0.03, 0.01, 0.2]), [0.06, 0.03, 0.2])

    def test_failures_denominators_transitions_and_cases(self):
        run = self.load()
        report, main, types, changes, cases = analysis.analyze(run)
        agent = next(r for r in main if r["subset"] == "primary_97" and r["method"] == "agent")
        self.assertEqual((agent["correct"], agent["n"]), (93, 97))
        self.assertEqual(agent["api_error_outcomes"], 1)
        self.assertEqual(agent["parse_error_outcomes"], 1)
        self.assertEqual({r["n"] for r in main if r["subset"] == "sensitivity_90"}, {90})
        self.assertEqual(sum(r["n"] for r in types if r["subset"] == "primary_97" and r["method"] == "agent"), 97)
        self.assertEqual(changes[3]["initial_source"], "same_attempt_ledger")
        self.assertEqual(changes[3]["transition"], "correct_to_wrong_or_failed")
        self.assertEqual(changes[4]["transition"], "initial_unavailable")
        transitions = report["agent_initial_final"]["primary_97"]
        self.assertEqual(transitions["transition_counts"]["wrong_to_correct"], 1)
        self.assertEqual(transitions["transition_counts"]["correct_to_wrong_or_failed"], 2)
        self.assertEqual(transitions["correct_to_wrong_completed_count"], 1)
        self.assertEqual(transitions["correct_to_execution_failure_count"], 1)
        self.assertEqual(cases["categories"]["agent_correction"]["selected"][0]["row_number"], 1)
        self.assertEqual(cases["categories"]["agent_regression"]["total_observed"], 1)
        self.assertEqual(cases["categories"]["agent_execution_failure"]["total_observed"], 2)
        self.assertEqual(len(report["paired_comparisons"]), 8)

    def test_unknown_usage_is_not_zero_or_total_upper_bound(self):
        stats = analysis.metrics_for(self.load(), "agent", set(i for i in range(100) if i + 1 not in analysis.EMPTY_ROWS))
        self.assertEqual(stats["api_attempts"], 98)
        self.assertEqual(stats["calls_with_unknown_token_usage"], 1)
        self.assertIsNone(stats["total_cost_conservative_upper_estimate_cny"])
        self.assertGreater(stats["known_tokens_conservative_estimate_cny"], 0)
        zero = analysis.metrics_for(self.load(), "zero_shot", set(i for i in range(100) if i + 1 not in analysis.EMPTY_ROWS))
        self.assertAlmostEqual(zero["total_cost_conservative_upper_estimate_cny"], 97 * (100 * 2 + 20 * 8) / 1e6)

    def test_partial_run_rejected_before_outputs(self):
        rows = analysis.read_jsonl(self.root / "results.jsonl")
        rows[0]["status"] = "budget_exhausted"
        (self.root / "results.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
        with self.assertRaisesRegex(analysis.AnalysisError, "Run incomplete"):
            self.load()
        self.assertFalse((self.root / "analysis").exists())

    def test_source_and_ledger_mismatch_rejected(self):
        data = json.loads(self.data_path.read_text(encoding="utf-8"))
        data[0]["answer"] = "B"
        self.data_path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(analysis.AnalysisError, "Explicit data file differs"):
            self.load()
        write_fixture(self.root)
        rows = analysis.read_jsonl(self.root / "results.jsonl")
        rows[0]["prediction"] = "B"
        (self.root / "results.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
        with self.assertRaisesRegex(analysis.AnalysisError, "Result/ledger mismatch"):
            self.load()

    def test_output_and_plot_smoke(self):
        output = self.root / "SYNTHETIC_TEST_OUTPUT_DO_NOT_REPORT"
        os.environ["MPLCONFIGDIR"] = str(self.root / "matplotlib_cache")
        analysis.write_outputs(self.load(), output)
        expected = {"analysis.json", "main_table.csv", "question_type_table.csv", "agent_changes.csv",
                    "case_candidates.json", "accuracy_comparison.png", "accuracy_comparison.pdf",
                    "efficiency_comparison.png", "efficiency_comparison.pdf"}
        self.assertEqual({p.name for p in output.iterdir()}, expected)
        self.assertTrue(all((output / name).stat().st_size > 100 for name in expected))

    def test_actual_runner_schema_with_offline_client(self):
        # Uses the real runner serialization, but this client never opens a socket.
        from runner import ExperimentConfig, ExperimentRunner
        def fake_create(**kwargs):
            prompt = kwargs["messages"][0]["content"]
            if '"action"' in prompt:
                body = {"action": "finish", "answer": ["A"], "brief_reason": "synthetic",
                        "decision_reason": "synthetic offline test"}
            elif '"recommendation"' in prompt:
                body = {"issues": [], "recommendation": "accept"}
            else:
                body = {"answer": ["A"], "brief_reason": "synthetic"}
            return SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(body)))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20),
                model="SYNTHETIC_OFFLINE_FIXTURE")
        client = SimpleNamespace(base_url="https://api.deepseek.com", max_retries=0,
                                 chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
        data = [{"question": "SYNTHETIC OFFLINE TEST", "question_type": "最佳选择题",
                 "option": {"A": "one", "B": "two"}, "answer": "A", "analysis": "test", "source": "test"}]
        directory = self.root / "actual_runner_synthetic_schema"
        runner = ExperimentRunner(data, client, directory,
                                  ExperimentConfig(model="SYNTHETIC_OFFLINE_FIXTURE", max_api_calls=6, max_review_rounds=1))
        runner.run()
        run = analysis.load_run(directory, directory / "source_data.json", enforce_course_counts=False)
        self.assertEqual(len(run["starts"]), 6)
        self.assertTrue(all(run["rows"][(method, 0)]["correct"] for method in analysis.METHODS))


if __name__ == "__main__":
    unittest.main()
