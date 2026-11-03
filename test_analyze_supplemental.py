"""验证补充分析器只读重放、修改检测与离线派生输出。"""
import json
import tempfile
import unittest
from pathlib import Path

import analyze_supplemental as analysis
import supplemental_runner as sr
from test_supplemental_runner import FORMAL, PROTOCOL, TEMP_ROOT, MockClient


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.directory = Path(self.temp.name)
        self.run = self.directory / "supplemental"
        protocol, source = sr.load_protocol(PROTOCOL, FORMAL)
        self.jobs = sr.make_jobs(protocol, source)
        engine = sr.SupplementalRun(MockClient(self.jobs), PROTOCOL, FORMAL, self.run, max_api_calls=2)
        engine.run(verbose=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_read_only_reproduction_and_separate_outputs(self):
        before = {p.name: sr.file_digest(p) for p in self.run.iterdir() if p.is_file()}
        engine = analysis.load_supplemental(self.run, FORMAL, PROTOCOL)
        self.assertIsNone(engine.client)
        result = analysis.write_outputs(engine, self.directory / "analysis")
        self.assertEqual(result["api_attempts"], 2)
        self.assertFalse(result["all_finished"])
        self.assertEqual(result["analysis_mode"], "offline_no_API")
        self.assertEqual(before, {p.name: sr.file_digest(p) for p in self.run.iterdir() if p.is_file()})
        for name in ("analysis.json", "metrics.csv", "paired_comparisons.json"):
            self.assertTrue((self.directory / "analysis" / name).is_file())
        with self.assertRaises(ValueError):
            analysis.write_outputs(engine, self.run)

    def test_tampered_request_is_rejected(self):
        path = self.run / "ledger.jsonl"
        events = sr.read_jsonl(path)
        events[0]["request"]["messages"][0]["content"] += "CHANGED"
        path.write_text("".join(sr.canonical(event) + "\n" for event in events), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "实际请求"):
            analysis.load_supplemental(self.run, FORMAL, PROTOCOL)

    def test_tampered_results_and_summary_are_rejected(self):
        path = self.run / "results.jsonl"
        original = path.read_bytes()
        rows = sr.read_jsonl(path)
        rows[0]["correct"] = not rows[0]["correct"]
        path.write_text("".join(sr.canonical(row) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "逐题结果"):
            analysis.load_supplemental(self.run, FORMAL, PROTOCOL)
        path.write_bytes(original)
        summary = json.loads((self.run / "summary.json").read_text(encoding="utf-8"))
        summary["api_attempts"] = 999
        sr.atomic_json(self.run / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "补充汇总"):
            analysis.load_supplemental(self.run, FORMAL, PROTOCOL)


if __name__ == "__main__":
    unittest.main()
