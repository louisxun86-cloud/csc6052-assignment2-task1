"""离线执行最终notebook的默认代码链两遍；补充日志只使用mock。"""
import contextlib
import io
import json
import os
import socket
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import openai

import build_final_notebook as builder
import supplemental_runner as sr
from test_supplemental_runner import FORMAL, PROTOCOL, ROOT, TEMP_ROOT, MockClient


class NotebookTests(unittest.TestCase):
    def test_default_run_all_twice_never_constructs_client(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            base = Path(directory)
            protocol, source = sr.load_protocol(PROTOCOL, FORMAL)
            jobs = sr.make_jobs(protocol, source)
            engine = sr.SupplementalRun(MockClient(jobs), PROTOCOL, FORMAL, base / "mock_supplemental")
            engine.run(verbose=False)
            sources = {name: (ROOT / name).read_bytes() for name in builder.SOURCE_NAMES}
            restored = builder.result_files(FORMAL.parent)
            for name in builder.SUPPLEMENTAL_NAMES:
                restored["task1_assignment/supplemental/" + name] = (engine.output / name).read_bytes()
            notebook = builder.build_notebook(sources, (FORMAL / "source_data.json").read_bytes(),
                                              "Mock-only QA snapshot", "test", None, restored)
            codes = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
            self.assertEqual(len(codes), 9)
            self.assertTrue(all(cell["outputs"] == [] and cell["execution_count"] is None
                                for cell in notebook["cells"] if cell["cell_type"] == "code"))
            project = base / "notebook_project"
            downloaded = []
            fake_colab = types.ModuleType("google.colab")
            fake_colab.files = types.SimpleNamespace(download=lambda path: downloaded.append(Path(path)))
            cached_names = ("cloud_driver", "runner", "supplemental_runner", "analyze_results", "analyze_supplemental")
            cached = {name: sys.modules.pop(name, None) for name in cached_names}
            previous_cwd, previous_path = Path.cwd(), list(sys.path)
            namespace = {"__name__": "__main__"}
            def denied(*args, **kwargs):
                raise AssertionError("默认离线notebook不得调用网络、创建API客户端或索取Key")
            try:
                with patch.dict(sys.modules, {"google.colab": fake_colab}), patch("openai.OpenAI", denied), \
                     patch("getpass.getpass", denied), patch.object(socket.socket, "connect", denied), \
                     patch("socket.create_connection", denied), contextlib.redirect_stdout(io.StringIO()):
                    for repeat in range(2):
                        for source_code in codes:
                            if source_code.startswith("%pip"):
                                continue  # 测试环境依赖已装好；测试不得联网安装。
                            source_code = source_code.replace('PROJECT_DIR = Path("/content/task1_assignment_repro")',
                                                              "PROJECT_DIR = Path(" + repr(str(project)) + ")")
                            exec(compile(source_code, f"notebook_default_pass_{repeat}", "exec"), namespace)
                self.assertEqual(len(downloaded), 2)
                self.assertEqual(downloaded[-1].name, "Task1_results_complete.zip")
                with zipfile.ZipFile(downloaded[-1]) as archive:
                    manifest = json.loads(archive.read("reproduction_manifest.json"))
                    self.assertEqual(manifest["formal_run_status"], "complete")
                    self.assertEqual(manifest["supplemental_run_status"], "complete")
                    self.assertFalse(manifest["contains_notebook"])
                    self.assertIn("results/supplemental/ledger.jsonl", archive.namelist())
                    self.assertIn("assignment2_task1/analyze_supplemental.py", archive.namelist())
                    self.assertIn("supplemental_analysis/analysis.json", archive.namelist())
                    for name, expected in manifest["files_sha256"].items():
                        self.assertEqual(sr.file_digest(project / "supplemental_runner.py"), protocol["protocol"]["supplemental_source_sha256"])
                        self.assertEqual(builder.sha(archive.read(name)), expected)
            finally:
                os.chdir(previous_cwd)
                sys.path[:] = previous_path
                for name, original in cached.items():
                    sys.modules.pop(name, None)
                    if original is not None:
                        sys.modules[name] = original


if __name__ == "__main__":
    unittest.main()
