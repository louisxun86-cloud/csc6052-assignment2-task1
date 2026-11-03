"""Build public source mirrors and reproducible archives; no upload or API call.

Run from either workspace or assignment directory. The private notebook/report
are never overwritten. Public Python helpers accept ASSIGNMENT_STUDENT_ID.
"""
from __future__ import annotations
import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

HERE = Path(__file__).resolve().parent
REPOSITORY = "https://github.com/louisxun86-cloud/csc6052-assignment2-task1"
EXCLUDE = {"actual_run", "public_release", "deliverables", "__pycache__", ".git",
           ".codex", ".test_tmp", ".public_release_checks", "template_original", "build_check", "tmp"}
TEXT = {".py", ".json", ".jsonl", ".csv", ".txt", ".md", ".tex", ".sty", ".bst", ".bib", ".ipynb"}
CORE = {"runner.py", "cloud_driver.py", "analyze_results.py", "analyze_supplemental.py",
        "Task1_数据跑通_API填空.py"}
PRIVATE_ID = None
CHANGES = []


class ReleaseError(ValueError):
    pass


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def jbytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def archive_bytes(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(files.items()):
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or "\\" in name:
                raise ReleaseError("Unsafe archive path")
            info = zipfile.ZipInfo(name, (2026, 11, 3, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, raw)
    return stream.getvalue()


def core(name):
    name = Path(name).name
    return name in CORE or (name.endswith(".py") and name.startswith(("supplement", "extension_")))


def redact(text):
    if PRIVATE_ID:
        text = text.replace(PRIVATE_ID, "STUDENT_ID_OMITTED")
    text = re.sub(r"https://colab\.research\.google\.com/drive/[A-Za-z0-9_-]+",
                  "PRIVATE_COLAB_LINK_OMITTED", text)
    return re.sub(r"[A-Za-z]:(?:\\{1,2}|/)Users(?:\\{1,2}|/)[^\"\r\n]*",
                  "LOCAL_PATH_OMITTED", text)


def public_python(raw, name):
    text = raw.decode("utf-8-sig")
    if core(name):
        if redact(text) != text:
            raise ReleaseError("Frozen API source requires manual privacy review: " + name)
        return raw
    identity_needed = bool(PRIVATE_ID and PRIVATE_ID in text)
    text = redact(text.replace("data/1.exam.json", "data/1.exam.json"))
    if identity_needed:
        class Identity(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, str) and "STUDENT_ID_OMITTED" in node.value:
                    return ast.copy_location(ast.Call(
                        ast.Attribute(ast.Constant(node.value), "replace", ast.Load()),
                        [ast.Constant("STUDENT_ID_OMITTED"), ast.Name("STUDENT_ID", ast.Load())], []), node)
                return node
        tree = Identity().visit(ast.parse(text))
        pos = int(bool(tree.body and isinstance(tree.body[0], ast.Expr)
                       and isinstance(tree.body[0].value, ast.Constant)
                       and isinstance(tree.body[0].value.value, str)))
        while pos < len(tree.body) and isinstance(tree.body[pos], ast.ImportFrom) and tree.body[pos].module == "__future__":
            pos += 1
        tree.body[pos:pos] = ast.parse(
            'import os\nSTUDENT_ID = os.environ.get("ASSIGNMENT_STUDENT_ID", "YOUR_STUDENT_ID")').body
        text = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    ast.parse(text)
    return text.encode("utf-8")


def sanitize(name, raw):
    suffix = Path(name).suffix
    if suffix == ".py":
        output = public_python(raw, name)
    elif suffix == ".ipynb":
        output = public_notebook(raw)
    elif suffix in TEXT:
        original = raw.decode("utf-8-sig")
        text = redact(original)
        if suffix == ".tex" and PRIVATE_ID and PRIVATE_ID in original:
            marker = r"\documentclass{article}"
            if marker not in text:
                raise ReleaseError("Identity-bearing TeX needs explicit adaptation: " + name)
            text = text.replace("STUDENT_ID_OMITTED", r"\StudentID{}")
            text = text.replace(marker, marker + "\n" + r"\newcommand{\StudentID}{Student ID omitted}", 1)
        output = raw if text == original else text.encode("utf-8")
    else:
        output = raw
    if output != raw:
        CHANGES.append({"file": name, "source_sha256": sha(raw), "public_sha256": sha(output),
                        "reason": "identity/host metadata or portable non-API helper data path"})
    return output


def embedded_archive(raw):
    files = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if archive.testzip() is not None:
            raise ReleaseError("Invalid embedded result archive")
        for name in archive.namelist():
            if not name.endswith("/"):
                files[name] = sanitize(name, archive.read(name))
    return archive_bytes(files), {name: sha(raw) for name, raw in files.items()}


def public_notebook(raw):
    nb = json.loads(raw.decode("utf-8-sig"))
    has_embedded_results = False
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}, "colab": {"provenance": []}}
    for i, cell in enumerate(nb["cells"]):
        cell["metadata"], cell["id"] = {}, f"public-task1-{i:03d}"
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        if cell["cell_type"] == "code":
            cell["execution_count"], cell["outputs"] = None, []
            try:
                tree = ast.parse(source)
            except SyntaxError:
                tree = None
            changed = False
            for node in tree.body if tree is not None else []:
                if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                    continue
                target = getattr(node.targets[0], "id", None)
                if target == "EMBEDDED_FILES":
                    value = ast.literal_eval(node.value)
                    for name, entry in value.items():
                        decoded = base64.b64decode(entry["base64"], validate=True)
                        if sha(decoded) != entry["sha256"]:
                            raise ReleaseError("Embedded source hash mismatch")
                        public = sanitize(name, decoded)
                        if core(name) and public != decoded:
                            raise ReleaseError("Frozen API source changed in notebook")
                        entry.update(base64=base64.b64encode(public).decode("ascii"), sha256=sha(public))
                    node.value = ast.parse(repr(value), mode="eval").body
                    changed = True
                elif target == "RESTORED_RESULTS":
                    value = ast.literal_eval(node.value)
                    if value:
                        has_embedded_results = True
                        decoded = base64.b64decode(value["base64_zip"], validate=True)
                        if sha(decoded) != value["sha256"]:
                            raise ReleaseError("Embedded results hash mismatch")
                        public, hashes = embedded_archive(decoded)
                        value.update(base64_zip=base64.b64encode(public).decode("ascii"),
                                     sha256=sha(public), files_sha256=hashes)
                        node.value = ast.parse(repr(value), mode="eval").body
                        changed = True
            if changed:
                source = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
        cell["source"] = redact(source).splitlines(keepends=True)
    explanation = (
        "This is a public generated reproducibility copy, not an original downloaded execution record. "
        "Code cells are unexecuted; actual observations come from verified embedded logs. "
        "Live paid calls are disabled by default. Private identifiers and metadata are omitted.\n\n"
        if has_embedded_results else
        "This is a historical starter notebook, not the final controlled experiment. "
        "It has no executed output or verified embedded result archive. The API-key field remains empty; "
        "private identifiers and metadata are omitted. Use Task1_public_repro.ipynb for the completed studies.\n\n")
    nb["cells"][0]["source"] = [
        "# Public Task 1 reproducibility notebook\n\n",
        explanation
    ] + nb["cells"][0]["source"]
    return jbytes(nb)


PATTERNS = {
    "api_key_like": re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}"),
    "github_token_like": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "aws_access_key_like": re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "private_colab_link": re.compile(r"colab\.research\.google\.com/drive/[A-Za-z0-9_-]+"),
    "windows_user_path": re.compile(r"[A-Za-z]:(?:\\{1,2}|/)Users(?:\\{1,2}|/)[^\\/\"\r\n]+"),
    "private_key_block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def scan(name, raw, findings):
    suffix = Path(name).suffix
    if suffix == ".zip":
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for child in archive.namelist():
                if not child.endswith("/"):
                    scan(name + "::" + child, archive.read(child), findings)
        return
    if suffix not in TEXT and Path(name).name != ".gitignore":
        return
    text = raw.decode("utf-8-sig")
    counts = {label: len(pattern.findall(text)) for label, pattern in PATTERNS.items()}
    counts["private_student_id"] = text.count(PRIVATE_ID) if PRIVATE_ID else 0
    counts = {key: count for key, count in counts.items() if count}
    if counts:
        findings.append({"file": name, "counts": counts})
    if suffix != ".ipynb":
        return
    for cell in json.loads(text)["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = getattr(node.targets[0], "id", None)
            if target == "EMBEDDED_FILES":
                for child, item in ast.literal_eval(node.value).items():
                    scan(name + "::embedded::" + child, base64.b64decode(item["base64"], validate=True), findings)
            elif target == "RESTORED_RESULTS":
                value = ast.literal_eval(node.value)
                if value:
                    scan(name + "::embedded_results.zip", base64.b64decode(value["base64_zip"], validate=True), findings)


def make_readme(has_supplement):
    status = ("The original experiment and both post-hoc supplementary studies are complete; genuine records are included."
              if has_supplement else "The original experiment is complete; supplementary studies are not yet included.")
    return f"""# CSC 6052 Assignment 2 — Task 1

Repository: {REPOSITORY}

Prompt engineering and feedback-driven agents for pharmacist examination questions.
Executed prompts/questions are Chinese; the submission report is English.
This public version omits the numeric student identifier and private host/Colab metadata.

## Contents

- Root Python files are readable mirrors of all experiment, analysis, test, notebook/report builder, import, packaging, and release scripts. Run the canonical layout below, because relative paths follow it.
- canonical_source.zip contains the runnable directory layout, data, public report source/styles/figures, tests, and public notebook.
- scientific_results.zip contains genuine manifests, request/response ledgers, predictions, summaries, analysis CSV/JSON, figures, environment versions, and source snapshots.
- Task1_public_repro.ipynb is a generated reproducibility copy with genuine embedded observations. Its code cells are unexecuted; it is not represented as a downloaded live notebook.
- public_file_manifest.json maps root mirrors to canonical paths and records hashes, privacy adaptations, frozen-source checks, artifact inventories, and scan outcome.

{status}

## Reproduce saved results without an API key

Extract BOTH ZIPs into the same empty directory. From that directory:

    python -m pip install -r requirements.txt
    python assignment2_task1/analyze_results.py --run-dir assignment2_task1/actual_run/task1_assignment/formal --data assignment2_task1/actual_run/task1_assignment/formal/source_data.json --output-dir reproduced_analysis
    python assignment2_task1/analyze_supplemental.py --run-dir assignment2_task1/actual_run/task1_assignment/supplemental --formal-dir assignment2_task1/actual_run/task1_assignment/formal --output-dir reproduced_supplemental_analysis
    python -m unittest discover -s assignment2_task1 -p "test_*.py" -v
    python assignment2_task1/report_en/build_original_sections.py

These commands analyze saved predictions or run fake-client tests; they do not call a model.
The original result is 79/97 (zero-shot), 80/97 (few-shot), 78/97 (CoT), and 82/97 (agent).
All 100 records are retained: three have empty options; seven additional risks define the fixed 90-record sensitivity set.
The report also gives correct/100 scores and valid-output coverage. The formal run made 592 API attempts,
with six protocol failures, no API request failures, and no explicit retries. Cost estimates are not invoices.

The post-hoc studies made 222 additional attempts:

| Study | Frozen comparison | Observed result |
| --- | --- | --- |
| A: closed-world checks | Original CoT / checklist v2 | 13/16 / 16/16; design 5/8 / 8/8, probe 8/8 / 8/8; Holm p=0.5 |
| B: cached-context single-step control | Original feedback / null feedback | Both 78/95 (78/97 including two shared invalid initials); each wins one paired item; Holm p=1 |

Study A does not show a held-out probe gain; all 32 responses were format-valid.
Study B is a one-step feedback-availability ablation, not a rerun of the full agent.
It includes one API failure and two protocol failures; all count as incorrect.
There are 126,527 known tokens and one request with unknown usage. The CNY 0.365650
known-usage estimate is neither a complete charge nor an upper bound on total cost.

## Optional live rerun

Open Task1_public_repro.ipynb in Colab. Keep both RUN_LIVE_EXPERIMENTS=False and
RUN_SUPPLEMENTAL_LIVE=False to reproduce saved results without a key or network calls.
Only to incur new API charges, enable the intended flag and enter your own DeepSeek key in the hidden prompt.
No credential is bundled. Frozen API source bytes are unchanged; changing prompts/settings requires a fresh directory.
Do not retry valid but wrong answers to improve a score. Post-hoc supplementary studies are separate from the original run.

## Report and public identity

Compile the report source with XeLaTeX twice from its directory. TeX Gyre and Fandol come with full TeX Live.
Public TeX uses a StudentID placeholder; public packaging helpers read ASSIGNMENT_STUDENT_ID or use a placeholder.
The private submission PDF keeps its identity and is not automatically published here.
The English submission source is in report_en/; report_zh_updated/ provides a Chinese counterpart.
The original main-study-only report/ is retained as an earlier version. The completed reports link
the immutable experiment-code snapshot independently of later presentation-only updates.

The course data/template sources are documented in the report; this repository does not claim ownership of them.
Medical examples describe model behavior and benchmark labels, not medical guidance.

## Integrity

The release builder never uploads or calls an API. It scans decoded notebook sources/archives and reports
suspicious matches only as counts and filenames. It excludes the private original notebook/PDF,
.git, .codex, caches, personal paths, private Colab links, and temporary test-output directories.
The formal data is included under data/, not temporary fixture directories.

## Rubric coverage

The English report documents three prompting techniques and task-specific rationale,
an actual frozen refinement comparison, adaptive agent decisions, the common evaluation
data and settings, strict accuracy/cost comparisons, two original cases, supplementary
counterexamples, and a controlled one-step feedback ablation. Development/probe and
post-hoc scopes are explicit; failures, unknown usage, and null findings are retained.
The source includes documented budgets, label isolation, resume ledgers, and offline tests.
Code availability alone is not a claim that bonus marks are guaranteed.
"""


def main():
    global PRIVATE_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "public_release")
    parser.add_argument("--require-supplement", action="store_true")
    args = parser.parse_args()
    PRIVATE_ID = json.loads((HERE/"report/template_provenance.json").read_text(encoding="utf-8"))["student_id"]
    actual = HERE / "actual_run/task1_assignment"
    import analyze_results
    analyze_results.load_run(actual/"formal", actual/"formal/source_data.json")
    paths = sorted(p for p in HERE.rglob("*.py") if not set(p.relative_to(HERE).parts) & EXCLUDE)
    supplemental = [p for p in paths if p.name.startswith(("supplement", "extension"))]
    supplementary_inputs = ("supplemental_items.json", "supplemental_protocol.json")
    if args.require_supplement and (not supplemental or not all((HERE/n).is_file() for n in supplementary_inputs)):
        raise ReleaseError("Final supplementary source/protocol is not yet available")
    supplement_freeze = None
    if (HERE/"supplemental_protocol.json").is_file():
        protocol_raw = (HERE/"supplemental_protocol.json").read_bytes()
        registered = json.loads(protocol_raw)
        protocol = registered["protocol"]
        if analyze_results.canonical_hash(protocol) != registered["fingerprint"]:
            raise ReleaseError("Supplementary protocol fingerprint mismatch")
        if protocol["supplemental_source_sha256"] != sha((HERE/"supplemental_runner.py").read_bytes()):
            raise ReleaseError("Supplementary source differs from frozen protocol")
        if protocol["frozen_runner_sha256"] != sha((HERE/"runner.py").read_bytes()):
            raise ReleaseError("Original runner differs from supplementary protocol")
        if protocol["synthetic_items_file_sha256"] != sha((HERE/"supplemental_items.json").read_bytes()):
            raise ReleaseError("Supplementary items differ from frozen protocol")
        supplement_freeze = {"fingerprint":registered["fingerprint"],
                             "protocol_file_sha256":sha(protocol_raw),
                             "source_sha256":protocol["supplemental_source_sha256"],
                             "planned_call_counts":protocol["planned_call_counts"],
                             "results_present":(actual/"supplemental").is_dir()}
    canonical, mirrors, mapping, checks = {}, {}, {}, []
    for path in paths:
        relative = path.relative_to(HERE).as_posix()
        raw = path.read_bytes()
        public = sanitize(relative, raw)
        target = "assignment2_task1/" + relative
        canonical[target] = public
        mirror = path.name if path.name not in mirrors else relative.replace("/", "__")
        if mirror in mirrors:
            raise ReleaseError("Root mirror collision")
        mirrors[mirror], mapping[mirror] = public, target
        if core(relative):
            assert public == raw
            checks.append({"path": target, "sha256": sha(raw), "byte_identical": True})
    data = (actual/"formal/source_data.json").read_bytes()
    canonical["data/1.exam.json"] = canonical["task1_smoke/1.exam.json"] = data
    canonical["assignment2_task1/data_quality_review.json"] = sanitize("data_quality_review.json",
        (HERE/"data_quality_review.json").read_bytes())
    starter = HERE.parent/"Task1_数据跑通_API填空.py"
    if starter.is_file():
        raw = starter.read_bytes()
        tree = ast.parse(raw.decode("utf-8-sig"))
        assignments = [n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
                       and any(getattr(t,"id","").upper()=="API_KEY" for t in n.targets)]
        if not assignments or not all(isinstance(n,ast.Constant) and n.value=="" for n in assignments):
            raise ReleaseError("Starter API_KEY must be explicitly empty")
        public = sanitize(starter.name,raw)
        assert public==raw
        canonical["starter/"+starter.name]=public
        mirrors[starter.name]=public
        mapping[starter.name]="starter/"+starter.name
        checks.append({"path":"starter/"+starter.name,"sha256":sha(raw),"byte_identical":True})
    starter_nb = HERE.parent/"Task1_数据跑通_API填空.ipynb"
    if starter_nb.is_file():
        public = public_notebook(starter_nb.read_bytes())
        canonical["starter/Task1_starter_public.ipynb"]=public
        mirrors["Task1_starter_public.ipynb"]=public
        mapping["Task1_starter_public.ipynb"]="starter/Task1_starter_public.ipynb"
    mirrors["data_quality_review.json"] = canonical["assignment2_task1/data_quality_review.json"]
    mapping["data_quality_review.json"] = "assignment2_task1/data_quality_review.json"
    for name in supplementary_inputs:
        if (HERE/name).is_file():
            raw = (HERE/name).read_bytes()
            public = sanitize(name, raw)
            if public != raw:
                raise ReleaseError("Final supplementary inputs need privacy review: "+name)
            canonical["assignment2_task1/"+name] = public
            mirrors[name] = public
            mapping[name] = "assignment2_task1/"+name
    report_states = {}
    for report in ("report", "report_en", "report_zh_updated"):
        directory = HERE / report
        if not directory.is_dir():
            continue
        selected = []
        for pattern in ("*.tex", "*.sty", "*.bst", "*.bib"):
            selected.extend(directory.glob(pattern))
        selected.extend((directory/"generated").glob("*.tex"))
        selected.extend((directory/"figures").glob("*.pdf"))
        selected.extend((directory/"figures").glob("*.png"))
        pending = []
        for path in sorted(selected):
            relative = path.relative_to(HERE).as_posix()
            raw = path.read_bytes()
            canonical["assignment2_task1/"+relative] = sanitize(relative, raw)
            if path.suffix == ".tex" and b"\\Pending{" in raw:
                pending.append(relative)
        provenance = directory/"template_provenance.json"
        if provenance.exists():
            canonical["assignment2_task1/"+report+"/template_provenance.json"] = sanitize(
                report+"/template_provenance.json", provenance.read_bytes())
        report_states[report] = {"pending_sections": pending}
    notebook = public_notebook((HERE/"deliverables/Task1_完整实验与结果.ipynb").read_bytes())
    canonical["assignment2_task1/notebooks/Task1_public_repro.ipynb"] = notebook
    results = {}
    folders = (HERE/"actual_run", HERE/"supplemental_run", HERE/"supplemental_results", HERE/"supplementary_run")
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix in {".json",".jsonl",".csv",".pdf",".png",".py",".txt"}:
                if (path.name == "supplemental_protocol_candidate.json"
                        or any(word in path.name.lower() for word in ("mock", "fixture"))):
                    continue
                if set(path.relative_to(folder).parts) & EXCLUDE:
                    continue
                relative = path.relative_to(HERE).as_posix()
                results["assignment2_task1/"+relative] = sanitize(relative, path.read_bytes())
    requirements = b"openai==2.54.0\nmatplotlib==3.10.0\nnumpy==2.4.6\nscipy==1.16.3\n"
    ignore = b"__pycache__/\n*.py[cod]\n.test_tmp/\n.env\n.env.*\n*.key\n*.pem\n*.aux\n*.log\n*.out\n*.xdv\nbuild_check/\npublic_release/\n"
    readme = make_readme(bool(supplemental)).encode("utf-8")
    canonical.update({"requirements.txt":requirements,"README.md":readme,".gitignore":ignore})
    outputs = {**mirrors, "README.md":readme, "requirements.txt":requirements, ".gitignore":ignore,
               "Task1_public_repro.ipynb":notebook, "canonical_source.zip":archive_bytes(canonical),
               "scientific_results.zip":archive_bytes(results)}
    findings = []
    for name, raw in outputs.items():
        scan(name, raw, findings)
    if findings:
        print(json.dumps({"scan_findings":findings},ensure_ascii=False))
        raise ReleaseError("Content scan rejected release; values are not printed")
    manifest = {"built_at_utc":datetime.now(timezone.utc).isoformat(),
        "intended_repository":REPOSITORY, "uploaded_by_this_script":False,
        "root_mirror_to_canonical_path":mapping, "frozen_api_sources":checks,
        "supplementary_freeze":supplement_freeze,
        "privacy_adaptations":CHANGES, "report_source_status":report_states,
        "scan":{"credential_like_match_count":0,"nested_notebook_payloads_scanned":True,"values_printed":False},
        "canonical_files":{n:{"sha256":sha(r),"bytes":len(r)} for n,r in canonical.items()},
        "scientific_files":{n:{"sha256":sha(r),"bytes":len(r)} for n,r in results.items()},
        "public_root_files":{n:{"sha256":sha(r),"bytes":len(r)} for n,r in outputs.items()}}
    manifest_raw = jbytes(manifest)
    scan("public_file_manifest.json", manifest_raw, findings)
    if findings:
        print(json.dumps({"scan_findings":findings},ensure_ascii=False))
        raise ReleaseError("Content scan rejected public manifest")
    outputs["public_file_manifest.json"] = manifest_raw
    out = args.output_dir.resolve()
    if not out.is_relative_to(HERE.resolve()):
        raise ReleaseError("Output must stay inside this assignment project")
    out.mkdir(parents=True, exist_ok=True)
    unexpected = {p.name for p in out.iterdir()} - set(outputs)
    if unexpected:
        raise ReleaseError("Unknown release leftovers: "+", ".join(sorted(unexpected)))
    for name, raw in outputs.items():
        (out/name).write_bytes(raw)
    print(json.dumps({"public_root_file_count":len(outputs),"canonical_file_count":len(canonical),
        "scientific_file_count":len(results),"frozen_api_source_count":len(checks),
        "privacy_adaptation_count":len(CHANGES),"credential_like_match_count":0,"uploaded":False}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError,OSError,KeyError,zipfile.BadZipFile) as error:
        raise SystemExit("Release build stopped: "+str(error))
