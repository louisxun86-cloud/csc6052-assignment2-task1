"""Package the bilingual final submission after offline and publication checks.

This script performs no API calls and does not submit anything to Blackboard.
Public source archives contain configurable author identifiers; private PDFs
and LaTeX in this submission retain the student's supplied identity.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "deliverables"
PDF_ROOT = ROOT.parent / "output" / "pdf"
REPOSITORY = "https://github.com/louisxun86-cloud/csc6052-assignment2-task1"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--release-commit", required=True)
    parser.add_argument("--english-pdf", type=Path, required=True)
    parser.add_argument("--chinese-pdf", type=Path, required=True)
    args = parser.parse_args()
    for commit in (args.code_commit, args.release_commit):
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("Use the verified full GitHub commit identifier")
    summary = json.loads((ROOT / "actual_run/task1_assignment/supplemental/summary.json").read_text(encoding="utf-8"))
    if not summary["all_finished"] or summary["api_attempts"] != 222:
        raise ValueError("The completed, frozen supplemental run is required")
    latex_archive = OUT / "Task1_Bilingual_LaTeX_Source.zip"
    with zipfile.ZipFile(latex_archive, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in ("report_en", "report_zh_updated"):
            source = ROOT / folder
            for path in sorted(source.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix not in {".tex", ".sty", ".bst", ".bib", ".py", ".png", ".pdf"}:
                    continue
                if path.name == "main.pdf" or "__pycache__" in path.parts:
                    continue
                if path.suffix == ".tex":
                    body = path.read_text(encoding="utf-8")
                    body = re.sub(r"\\newcommand\{\\Pending\}.*", "", body)
                    if "\\Pending{" in body or "PENDING_COMMIT" in body:
                        raise ValueError(f"Unfilled report content: {path}")
                z.write(path, path.relative_to(ROOT).as_posix())
        z.writestr("Compile.txt", "The final generated sections are included. From each report directory, run twice:\nxelatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex\nRequires TeX Live with XeLaTeX, Fandol, and TeX Gyre fonts.\n")
    note = OUT / "Final_Submission_Readme.txt"
    note.write_text(
        "Assignment 2 / Task 1 / CSC 6052\n\n"
        "英文PDF为主提交报告；中文PDF供理解和核对。二者均使用课程提供的LaTeX模板。\n"
        "提交截止：2026年11月6日23:59（UTC+8）。本包没有代为提交Blackboard。\n"
        "完整notebook包含真实日志，默认仅离线复算；新的付费调用必须显式启用。\n"
        "原始正式实验592次API调用，开发检查18次；追加实验222次。\n"
        "追加实验含1次API失败（用量未知），不重试、不删除，不计为正确。\n"
        "97题主集、90题敏感性集与原100题保守分数分别披露；补充实验另行标注为post-hoc。\n\n"
        f"公开源码仓库：{REPOSITORY}\n"
        f"报告引用的实验代码快照：{REPOSITORY}/tree/{args.code_commit}\n"
        f"最终公开源码/报告源码发布：{REPOSITORY}/tree/{args.release_commit}\n\n"
        "canonical_source.zip与scientific_results.zip解压到同一空目录，可按仓库README离线复算。\n"
        "公开LaTeX源码的学号使用占位符；本私有提交包的PDF及LaTeX保留本人信息。\n"
        "两份PDF、双语LaTeX、复现notebook、完整代码和日志均已打包。\n"
        "GitHub_Public_Repository.png提供公开仓库页面的完成记录。\n",
        encoding="utf-8")
    items = [args.english_pdf, args.chinese_pdf, latex_archive, note,
             OUT / "Task1_完整实验与结果.ipynb",
             OUT / "Task1_可复现源码与日志.zip",
             ROOT / "public_release/canonical_source.zip",
             ROOT / "public_release/scientific_results.zip",
             OUT / "GitHub_Public_Repository.png"]
    for optional in (OUT / "云端实际执行记录.txt", OUT / "Colab_完整实验结果.png",
                     OUT / "Colab_Supplement_Completed.png"):
        if optional.is_file():
            items.append(optional)
    for path in items:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    receipt = OUT / "Final_Submission_SHA256.json"
    receipt.write_text(json.dumps({
        "repository": REPOSITORY, "code_commit": args.code_commit,
        "release_commit": args.release_commit,
        "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in items},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    target = OUT / "Assignment2_Task1_Final_Bilingual_Submission.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for path in items + [receipt]:
            z.write(path, path.name)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
    print(json.dumps({"path": str(target), "bytes": target.stat().st_size,
                      "sha256": sha256(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
