# CSC 6052 Assignment 2 — Task 1

Repository: https://github.com/louisxun86-cloud/csc6052-assignment2-task1

Prompt engineering and feedback-driven agents for pharmacist examination questions.
Executed prompts/questions are Chinese; the submission report is English.
This public version omits the numeric student identifier and private host/Colab metadata.

## Contents

- Root Python files are readable mirrors of all experiment, analysis, test, notebook/report builder, import, packaging, and release scripts. Run the canonical layout below, because relative paths follow it.
- canonical_source.zip contains the runnable directory layout, data, public report source/styles/figures, tests, and public notebook.
- scientific_results.zip contains genuine manifests, request/response ledgers, predictions, summaries, analysis CSV/JSON, figures, environment versions, and source snapshots.
- Task1_public_repro.ipynb is a generated reproducibility copy with genuine embedded observations. Its code cells are unexecuted; it is not represented as a downloaded live notebook.
- public_file_manifest.json maps root mirrors to canonical paths and records hashes, privacy adaptations, frozen-source checks, artifact inventories, and scan outcome.

Supplementary files are included where listed in the manifest; inspect run summaries for completion.

## Reproduce saved results without an API key

Extract BOTH ZIPs into the same empty directory. From that directory:

    python -m pip install -r requirements.txt
    python assignment2_task1/analyze_results.py --run-dir assignment2_task1/actual_run/task1_assignment/formal --data assignment2_task1/actual_run/task1_assignment/formal/source_data.json --output-dir reproduced_analysis
    python -m unittest discover -s assignment2_task1 -p "test_*.py" -v
    python assignment2_task1/report_en/build_original_sections.py

These commands analyze saved predictions or run fake-client tests; they do not call a model.
The original result is 79/97 (zero-shot), 80/97 (few-shot), 78/97 (CoT), and 82/97 (agent).
All 100 records are retained: three have empty options; seven additional risks define the fixed 90-record sensitivity set.
The report also gives correct/100 scores and valid-output coverage. The formal run made 592 API attempts,
with six protocol failures, no API request failures, and no explicit retries. Cost estimates are not invoices.

## Optional live rerun

Open Task1_public_repro.ipynb in Colab. Keep RUN_LIVE_EXPERIMENTS=False for existing results.
Only to incur new API charges, change it to True and enter your own DeepSeek key in the hidden prompt.
No credential is bundled. Frozen API source bytes are unchanged; changing prompts/settings requires a fresh directory.
Do not retry valid but wrong answers to improve a score. Post-hoc supplementary studies are separate from the original run.

## Report and public identity

Compile the report source with XeLaTeX twice from its directory. TeX Gyre and Fandol come with full TeX Live.
Public TeX uses a StudentID placeholder; public packaging helpers read ASSIGNMENT_STUDENT_ID or use a placeholder.
The private submission PDF keeps its identity and is not automatically published here.
No final report is claimed while its TeX still contains Pending markers.

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
