"""Build English report sections from the unchanged original verified run.

This script never calls an API and does not edit the Chinese report.
The supplementary sections and GitHub provenance are managed separately.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def original_validator():
    spec = importlib.util.spec_from_file_location("_original_report_validator", PROJECT / "report/build_results_sections.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_prompts(manifest, tex):
    names = {"COMMON_PROMPT", "ZERO_PROMPT", "FEW_PROMPT", "COT_PROMPT",
             "CRITIC_PROMPT", "CONTROLLER_PROMPT", "SYNTHETIC_EXAMPLES"}
    values = {}
    def value(node):
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return value(node.left) + value(node.right)
        return ast.literal_eval(node)
    raw = (PROJECT / "runner.py").read_bytes()
    for node in ast.parse(raw.decode("utf-8-sig")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in names:
                values[name] = value(node.value)
    prompts = manifest["experiment"]["prompts"]
    for key, symbol in (("zero_shot","ZERO_PROMPT"), ("few_shot","FEW_PROMPT"),
                        ("cot","COT_PROMPT"), ("critic","CRITIC_PROMPT"), ("controller","CONTROLLER_PROMPT")):
        assert prompts[key] == values[symbol], f"Prompt mismatch: {key}"
    assert manifest["experiment"]["few_shot_examples"] == list(values["SYNTHETIC_EXAMPLES"])
    def box(title, content):
        return r"\begin{promptbox}{" + title + "}\n\\small\n" + tex(content) + "\n\\end{promptbox}\n"
    common = values["COMMON_PROMPT"]
    parts = [
        "% Extracted from frozen runner; SHA256 " + hashlib.sha256(raw).hexdigest() + "\n",
        "These are the exact Chinese prompts used in the original run, not English replacements. "
        "Each answering/controller system message directly concatenates the common instruction "
        "with its suffix; the critic uses its own complete message. Layout wrapping is for display. "
        "The manifest preserves the complete strings and demonstrations.\n",
        box("Common instruction", common),
    ]
    for title, symbol in (("Zero-shot suffix","ZERO_PROMPT"), ("Few-shot suffix","FEW_PROMPT"),
                          ("CoT suffix (also the agent initial call)","COT_PROMPT")):
        assert values[symbol].startswith(common)
        parts.append(box(title, values[symbol][len(common):]))
    parts.append(box("Critic: complete system message", values["CRITIC_PROMPT"]))
    parts.append(box("Controller suffix", values["CONTROLLER_PROMPT"][len(common):]))
    parts.append(
        r"\paragraph{All three few-shot demonstrations.}"
        "Each is a user question followed by an assistant answer, then the current evaluation question. "
        r"Inputs use keys \texttt{question\_type}, \texttt{question}, and \texttt{options}, "
        "serialized as compact, key-sorted JSON. Provenance and explanations are not sent.\n")
    for i, example in enumerate(values["SYNTHETIC_EXAMPLES"], 1):
        answer = '{"answer":[' + ",".join('"' + c + '"' for c in example["answer"]) + "]}"
        content = ("题型：" + example["question_type"] + "。题干：" + example["question"] + "\n\n" +
                   "；".join(k+"："+v for k,v in example["option"].items()) + "。\n\n" +
                   "assistant: " + answer)
        parts.append(box(f"Synthetic demonstration {i}", content))
    parts.append(
        r"\paragraph{Intermediate messages.}"
        r"The critic receives \texttt{question} and \texttt{candidate}; the controller also receives "
        r"\texttt{feedback}, \texttt{review\_round}, and \texttt{remaining\_review\_rounds}. "
        r"A candidate contains only \texttt{answer} and \texttt{brief\_reason}. "
        "The ledger records the actual complete messages, feedback, and decisions.\n")
    return "".join(parts)


def result_tables(a):
    methods = ("zero_shot","few_shot","cot","agent")
    labels = {"zero_shot":"Zero-shot","few_shot":"Few-shot","cot":"CoT","agent":"Agent"}
    lookup = {(r["subset"],r["method"]):r for r in a["main_results"]}
    lines = [
        r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{3pt}",
        r"\caption{Original strict accuracy. Original-100 accuracy counts the three missing-option records as failures; coverage counts valid predictions only.}",
        r"\label{tab:accuracy}",
        r"\begin{tabular}{lrrrrr}\toprule",
        r"Method & Correct/97 (\%) & 95\% Wilson CI & Correct/90 (\%) & Correct/100 & Coverage/100\\\midrule",
    ]
    for m in methods:
        p,s=lookup[("primary_97",m)],lookup[("sensitivity_90",m)]
        lines.append(
            f"{labels[m]} & {p['correct']}/97 ({100*p['accuracy']:.2f}) & "
            f"[{100*p['wilson95_lower']:.1f}, {100*p['wilson95_upper']:.1f}] & "
            f"{s['correct']}/90 ({100*s['accuracy']:.2f}) & {p['correct']}/100 & "
            f"{p['completed_predictions']}/100" + r"\\")
    lines.extend([
        r"\bottomrule\end{tabular}\end{table}",
        r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{3pt}",
        r"\caption{Consumption on the original 97 inputs, including failed attempts. API failures were zero; protocol failures are shown separately.}",
        r"\label{tab:cost}",
        r"\begin{tabular}{lrrrrrr}\toprule",
        r"Method & Calls (per item) & Input tokens & Output tokens & Seconds/item & CNY est. & Protocol failures\\\midrule"])
    for m in methods:
        p=lookup[("primary_97",m)]
        assert p["calls_with_unknown_token_usage"] == p["calls_with_unknown_elapsed"] == 0
        lines.append(
            f"{labels[m]} & {p['api_attempts']} ({p['mean_api_attempts_per_question']:.2f}) & "
            f"{p['known_prompt_tokens']:,} & {p['known_completion_tokens']:,} & "
            f"{p['mean_known_api_elapsed_seconds_per_question']:.2f} & "
            f"{p['total_cost_conservative_upper_estimate_cny']:.3f} & {p['parse_error_outcomes']}" + r"\\")
    lines.extend([
        r"\bottomrule\end{tabular}\end{table}",
        "Costs use the stated peak rates of CNY 2/million input and CNY 8/million output tokens, "
        "treating all input as a cache miss. This conservative estimate is not an invoice; "
        r"cache discounts and off-peak rates can reduce the charge\citep{deepseekpricing}.",
        r"\begin{figure}[t]\centering",
        r"\includegraphics[width=.68\linewidth]{figures/accuracy_comparison.pdf}",
        r"\caption{Original accuracy and Wilson intervals. The overlapping 97/90 scopes are not independent replications.}\label{fig:accuracy}\end{figure}",
        r"\begin{figure}[t]\centering",
        r"\includegraphics[width=.92\linewidth]{figures/efficiency_comparison.pdf}",
        r"\caption{Original per-item attempts, API time, and tokens. These include all attempts and exclude human waiting time.}\label{fig:cost}\end{figure}",
    ])
    return "\n".join(lines)+"\n"


def original_analysis(a):
    p={r["method"]:r for r in a["main_results"] if r["subset"]=="primary_97"}
    transition=a["agent_initial_final"]["primary_97"]
    assert [p[m]["correct"] for m in ("zero_shot","few_shot","cot","agent")]==[79,80,78,82]
    assert transition["wrong_to_correct_count"]==5
    lines=[
        r"\paragraph{Accuracy and uncertainty.}"
        "The agent scores 82/97 (84.54\\%), ahead of zero-shot by 3.09 percentage points and CoT "
        "by 4.12 points. Few-shot gains one net correct answer over zero-shot, while CoT loses one. "
        "The small differences do not establish a reliable ordering. We use exact two-sided McNemar "
        "tests on paired correctness, with Holm correction over four comparisons on each of the "
        "two scopes (eight tests total).",
        r"\begin{table}[t]\centering\small",
        r"\caption{Paired comparisons on the original 97 inputs. A/B counts denote items correct only for that method. All eight Holm-adjusted $p$ values are 1.000.}",
        r"\begin{tabular}{lrrr}\toprule",
        r"Comparison (A vs.\ B) & A only & B only & Exact $p$\\\midrule",
    ]
    labels={"zero_shot":"Zero-shot","few_shot":"Few-shot","cot":"CoT","agent":"Agent"}
    for pair in a["paired_comparisons"]:
        if pair["subset"]=="primary_97":
            lines.append(f"{labels[pair['candidate']]} vs.\\ {labels[pair['reference']]} & "
                         f"{pair['candidate_only_correct']} & {pair['reference_only_correct']} & "
                         f"{pair['mcnemar_exact_two_sided_p']:.3f}" + r"\\")
    lines.extend([
        r"\bottomrule\end{tabular}\end{table}",
        "No adjusted comparison reaches 0.05; non-significance does not establish equivalence. "
        "Wilson intervals describe binomial uncertainty, but shared stems and a convenience sample "
        "also limit generalization. The ranking is unchanged on the 90-record subset: agent, "
        "few-shot, zero-shot, then CoT. Accuracy drops by 0.09, 1.36, 1.44, and 1.52 points, "
        "respectively; removing input risks does not necessarily increase accuracy.",
        r"\paragraph{What additional reasoning buys.}"
        "Few-shot achieves valid output on all 97 items versus 96 for zero-shot, but the demonstrations "
        "do not produce a significant accuracy gain. A visible rationale also does not guarantee "
        "answer consistency (record 6). The agent uses 3.10 times the calls and 4.54 times the tokens "
        "of CoT. Its additional cost should be assessed against both corrections and failures.",
        r"\paragraph{Within-agent observations.}"
        "A legal initial answer is recoverable for 95 records: 77 are initially correct "
        "(81.05\\%), versus 82 final correct answers on those same 95 (86.32\\%). "
        "Five wrong answers become correct; no correct answer becomes wrong or fails execution. "
        "Eleven finish incorrectly, two wrong initial answers later fail protocol validation, "
        "and two of the 97 initial answers are invalid. Complete traces contain 92 "
        r"\texttt{finish} and 8 \texttt{revise} actions; 86 successful cases use one round and seven use two. "
        "These are descriptive observations, not a separately randomized ablation; they also differ "
        "from the independently sampled CoT baseline.",
        r"\paragraph{JSON validity is not protocol validity.}"
        "All six reported parsing failures contain syntactically valid JSON. Five return multiple "
        "letters for a single-choice record (zero-shot/CoT/agent record 14, agent 67 and 71). "
        "At record 76 the controller changes ABCE to the correct ACE while declaring "
        r"\texttt{finish}; the frozen protocol requires \texttt{revise} for a change. "
        "It therefore counts as failure. We neither relax the rule after seeing the label nor retry "
        "the item to recover credit. This illustrates the difference between semantic correctness "
        "and an executable action contract.",
    ])
    return "\n\n".join(lines)+"\n"


def main():
    args=argparse.Namespace(results_root=PROJECT/"actual_run/task1_assignment",
                            data=PROJECT/"actual_run/task1_assignment/formal/source_data.json",
                            analysis_dir=None,versions=None)
    validator=original_validator()
    run,dev,analysis,cases,changes,versions,directory=validator.validate(args)
    outputs={
        "prompts.tex":exact_prompts(run["manifest"],validator.tex),
        "original_results.tex":result_tables(analysis),
        "original_analysis.tex":original_analysis(analysis),
    }
    for name,text in outputs.items():
        (HERE/"generated"/name).write_text(text,encoding="utf-8")
    print("Original logs and all analysis artifacts validated; three English sections generated.")
    print("The Chinese report, original prompts, experiments, and supplementary placeholders were not changed.")


if __name__=="__main__":
    main()
