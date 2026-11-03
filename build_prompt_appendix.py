"""Render exact frozen prompts without importing the experiment or calling APIs.

Usage: python build_prompt_appendix.py [path/to/formal/manifest.json]
With a manifest, all prompts and examples must agree before any output is written.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "runner.py"
NAMES = {
    "COMMON_PROMPT", "ZERO_PROMPT", "FEW_PROMPT", "COT_PROMPT",
    "CRITIC_PROMPT", "CONTROLLER_PROMPT", "SYNTHETIC_EXAMPLES",
}


def static_value(node, values):
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return static_value(node.left, values) + static_value(node.right, values)
    return ast.literal_eval(node)


values = {}
source_bytes = SOURCE.read_bytes()
for node in ast.parse(source_bytes.decode("utf-8-sig")).body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        name = getattr(node.targets[0], "id", None)
        if name in NAMES:
            values[name] = static_value(node.value, values)
assert set(values) == NAMES

prompts = {name: values[symbol] for name, symbol in {
    "zero_shot": "ZERO_PROMPT", "few_shot": "FEW_PROMPT", "cot": "COT_PROMPT",
    "critic": "CRITIC_PROMPT", "controller": "CONTROLLER_PROMPT",
}.items()}
if len(sys.argv) > 1:
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    experiment = manifest["experiment"]
    assert experiment["prompts"] == prompts, "Manifest prompts differ from frozen source."
    assert experiment["few_shot_examples"] == list(values["SYNTHETIC_EXAMPLES"]), "Examples differ."


def tex(text):
    substitutions = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                     "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                     "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(substitutions.get(char, char) for char in text)


def box(title, text):
    return "\\begin{promptbox}{" + title + "}\n\\small\n" + tex(text) + "\n\\end{promptbox}\n"


common = values["COMMON_PROMPT"]
parts = [
    "% Automatically extracted from runner.py; do not manually edit prompt text.\n",
    "% runner.py SHA256: " + hashlib.sha256(source_bytes).hexdigest() + "\n",
    "以下文本从冻结的 \\texttt{runner.py} 自动提取；为节省篇幅，共同指令只列一次。\n",
    "Zero-shot、Few-shot、CoT 和 Controller 的完整 system 消息分别是共同指令与其增补的直接拼接，",
    "Critic 使用独立的完整消息。\\texttt{manifest.json} 中保存未删节字符串及示例。\n",
    box("所有答题方法的共同指令", common),
]
for title, symbol in (("Zero-shot 增补", "ZERO_PROMPT"), ("Few-shot 增补", "FEW_PROMPT"),
                      ("CoT 增补（亦用于 Agent 初答）", "COT_PROMPT")):
    assert values[symbol].startswith(common)
    parts.append(box(title, values[symbol][len(common):]))
parts.append(box("Critic 完整 system 消息", values["CRITIC_PROMPT"]))
assert values["CONTROLLER_PROMPT"].startswith(common)
parts.append(box("Controller 增补", values["CONTROLLER_PROMPT"][len(common):]))
parts.append(
    "\\paragraph{Few-shot 的全部示例。}以下每个示例以 user 题目、assistant 答案的顺序进入消息；"
    "它们不含解析。三个示例之后再附上当前评测题。输入字段为"
    " \\texttt{question\\_type}、\\texttt{question}、\\texttt{options}，"
    "用无额外空格、键名排序的 JSON 序列化。\n"
)
for number, example in enumerate(values["SYNTHETIC_EXAMPLES"], 1):
    answer = json.dumps({"answer": list(example["answer"])}, ensure_ascii=False, separators=(",", ":"))
    body = ("题型：" + example["question_type"] + "。题干：" + example["question"] + "\n\n"
            + "；".join(k + "：" + v for k, v in example["option"].items()) + "。\n\n"
            + "assistant 输出：" + answer)
    parts.append(box(f"人工示例 {number}", body))
parts.append(
    "\\paragraph{Agent 中间消息。}Critic 的 user JSON 包含 \\texttt{question} 与"
    " \\texttt{candidate}；Controller 另接收 \\texttt{feedback}、"
    "\\texttt{review\\_round} 与 \\texttt{remaining\\_review\\_rounds}。候选答案仅包含"
    " \\texttt{answer} 和 \\texttt{brief\\_reason}，不包含评分标签。"
    "每轮实际消息、反馈与动作可由 ledger 和逐题轨迹复核。\n"
)
(ROOT / "generated" / "prompts.tex").write_text("".join(parts), encoding="utf-8")
print("Extracted five exact prompts and three examples from frozen runner.py.")
if len(sys.argv) > 1:
    print("All prompt strings and examples match the supplied manifest.")
