"""Build bilingual report sections from the completed real supplement; no API."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import analyze_supplemental as analysis

HERE = Path(__file__).resolve().parent
RUN = HERE / "actual_run/task1_assignment"
EN = HERE / "report_en"
ZH = HERE / "report_zh_updated"


def replace_once(path, old, new):
    text = path.read_text(encoding="utf-8")
    if old not in text:
        if new in text:
            return
        raise ValueError("Report marker missing: " + path.name)
    assert text.count(old) == 1
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def table(summary, chinese=False):
    values = [("A: CoT v1", summary["study_A"]["cot_v1"]["all_16"]),
              ("A: CoT v2", summary["study_A"]["cot_v2"]["all_16"]),
              ("B: Feedback", summary["study_B"]["feedback"]["conditional_95"]),
              ("B: Null", summary["study_B"]["null_feedback"]["conditional_95"])]
    caption = ("追加研究的结果与消耗。B按95条合法初答计；星号表示另有1次调用用量未知。"
               if chinese else "Supplementary outcomes and consumption. B uses 95 valid initial states; the asterisk flags one call with unknown token usage.")
    heading = (r"条件 & 正确 & 合法输出 & 调用 & 已知token & API秒\\\midrule" if chinese else
               r"Condition & Correct & Valid output & Calls & Known tokens & API seconds\\\midrule")
    lines = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}",
             r"\caption{" + caption + r"}\label{tab:supplement}",
             r"\begin{tabular}{lrrrrr}\toprule", heading]
    for name, value in values:
        marker = r"$^\ast$" if value["calls_with_unknown_usage"] else ""
        lines.append(f"{name} & {value['correct']}/{value['n']} & {value['format_valid']}/{value['n']} & "
                     f"{value['api_attempts']} & {value['known_total_tokens']:,}{marker} & "
                     f"{value['known_api_seconds']:.2f}" + r"\\")
    return "\n".join(lines+[r"\bottomrule\end{tabular}\end{table}"])+"\n"


def main():
    engine = analysis.load_supplemental(RUN/"supplemental",RUN/"formal",HERE/"supplemental_protocol.json")
    summary = engine.summary()
    assert summary["all_finished"] and summary["api_attempts"] == summary["planned_api_calls"] == 222
    rows = {r["job_id"]:r for r in engine.rows()}
    for key, prediction, correct in [
        ("A:D06:cot_v1","ACE",False),("A:D06:cot_v2","AC",True),
        ("B:q018:feedback","ACE",True),("B:q018:null_feedback","AE",False),
        ("B:q028:feedback","C",False),("B:q028:null_feedback","B",True)]:
        assert rows[key]["prediction"] == prediction and rows[key]["correct"] == correct
    assert "C" in rows["B:q028:null_feedback"]["decision_reason"]
    status = Counter(r["status"] for r in rows.values())
    assert status["api_error"] == 1 and status["parse_error"] == 2 and status["common_initial_failure"] == 4
    a,b=summary["study_A"],summary["study_B"]
    assert (a["cot_v1"]["all_16"]["correct"],a["cot_v2"]["all_16"]["correct"]) == (13,16)
    assert (b["feedback"]["conditional_95"]["correct"],b["null_feedback"]["conditional_95"]["correct"]) == (78,78)
    tests=summary["supplementary_primary_comparisons"]
    assert tests["A_all_16"]["holm_adjusted_p_two_comparison_family"] == .5
    assert tests["B_conditional_95"]["holm_adjusted_p_two_comparison_family"] == 1
    total=summary["totals"]
    assert total["calls_with_unknown_usage"] == 1
    known_cost=(total["known_prompt_tokens"]*2+total["known_completion_tokens"]*8)/1_000_000
    en=table(summary)+r"""
\paragraph{A: a narrow improvement, not probe generalization.}
CoT v2 improves 13/16 (81.25\%) to 16/16, with all three corrections in the development
split (5/8 to 8/8); both prompts score 8/8 on the probe split.
All 32 outputs satisfy the schema, so this is not an observed increase in format-validity rate.
The corrections address option omission (D03), an answer--rationale contradiction (D06),
and confusion between record count and item quantity (D08).
For D06, v1 explicitly says E fails the stated upper bound but still returns ACE;
v2 returns the reference AC.
The primary paired test is $p=.25$, Holm-adjusted $p=.50$.
The checklist helps these cases, but a small synthetic set and a probe ceiling do not
demonstrate improved unseen pharmacy performance.

\paragraph{B: equal scores hide different decisions.}
Both arms score 78/95 (82.11\%), or 78/97 (80.41\%) with the two shared initial failures.
Each arm wins on one record; exact and Holm-adjusted $p$ values are both 1.00.
Relative to the same 77/95 correct initial candidates, feedback corrects two
(6 and 18) and breaks one (28); null feedback corrects one (6) and breaks none.
This neither establishes equivalence nor shows that feedback is universally ineffective.
The original agent's higher multi-round score is not evidence that merely supplying
its first critique causes a reliable gain.

Record 18 changes AE to the reference ACE with feedback and \texttt{revise};
without feedback it retains AE with \texttt{finish}.
Conversely, record 28 changes a correct B to C after feedback repeats a formula lacking
the elimination-rate factor present in the course explanation.
The null arm retains B, but its text also argues for C: label agreement is not correct reasoning.
Record 28 is already in the original missing-structure risk list, although numerical inputs
are present. Neither arm used external verification; generated claims of verification
are not evidence of a lookup.

\paragraph{Failures and additional cost.}
Feedback has one API error (record 54) and one protocol failure; null feedback has one
protocol failure. At record 76 both change ABCE to the reference ACE while declaring
\texttt{finish}, violating the unchanged-candidate rule.
All failures remain in the denominator and none is retried.
"""
    en+=(f"The supplement uses {total['api_attempts']} attempts, {total['known_total_tokens']:,} known tokens, "
         f"and {total['known_api_seconds']:.2f} API seconds. The known-token subtotal is CNY {known_cost:.6f} "
         "at the stated rates. One API-error call has unknown usage, so this is neither the complete charge "
         "nor an upper bound on total cost.\n")
    zh=table(summary,True)+r"""
\paragraph{A：有限的改进证据，不代表探查集泛化提升。}
v2将13/16（81.25\%）提高到16/16，三道改对题全部来自设计开发集合（5/8变为8/8）；
探查集合两组均为8/8。32个输出全部满足schema，因此并未观察到格式合法率提升。
三处修正分别是D03遗漏选项、D06答案与依据矛盾、D08混淆记录笔数与物品件数。
D06中v1已说明E超过题面上限、不符合条件，却仍返回ACE；v2返回参考AC。
主要配对检验$p=.25$，Holm校正$p=.50$。
清单在这些案例上有帮助，但小型人工集合与探查题天花板不支持宣称新药学题泛化改善。

\paragraph{B：相同总体分数掩盖不同决策。}
两组均为78/95（82.11\%）；加上两道共同无效初答后均为78/97（80.41\%）。
各有1题仅本组正确，原始及Holm校正$p$均为1.00。
相对相同的77/95正确初答，有反馈组改对2题（6、18）并改错1题（28），
空反馈组改对1题（6）且未改错。
这既不证明等效，也不证明反馈普遍无效；原Agent的较高多轮分数不能被直接解释为
提供第一次反馈就能可靠提高准确率。

位置18中，有反馈组选择\texttt{revise}，把AE改为参考ACE；空反馈组选择\texttt{finish}，保留AE。
位置28则相反：反馈重复了缺少课程解析中消除速率因子的公式，诱使控制器把正确B改为C。
空反馈组虽保留B，但文字同样主张C，因此不能称为空反馈作出了正确推理。
该条已在原缺结构图风险名单中，虽然计算所需数值条件在文本中，这一数据限制须同时披露。
两组均未使用外部查证工具，模型声称“经核实”只是生成文本。

\paragraph{执行失败与追加消耗。}
有反馈组有1次API错误（位置54）与1次协议失败，空反馈组有1次协议失败。
位置76两组均把ABCE改为参考ACE，却声明\texttt{finish}，违反保持候选答案的规则；
所有失败留在分母中且没有重试。
"""
    zh+=(f"补充研究共{total['api_attempts']}次调用、{total['known_total_tokens']:,}个已知token、"
         f"{total['known_api_seconds']:.2f}秒累计API耗时。按声明费率，已知token部分估计为{known_cost:.6f}元；"
         "1次API错误调用缺少用量，因此这不是完整费用、账单或总费用上界。\n")
    (EN/"generated/supplementary_results.tex").write_text(en,encoding="utf-8")
    (ZH/"generated/supplementary_results.tex").write_text(zh,encoding="utf-8")
    replacements=[
      (EN/"main.tex",r"\Pending{Summarize the completed supplementary refinement and controlled feedback study.}",
       "Post-hoc checks improve from 13/16 to 16/16 under a fixed refinement, with both probe scores 8/8; "
       "a controlled single-step feedback ablation ties at 78/95. These narrower results qualify the original ranking."),
      (EN/"main.tex",r"\Pending{Integrate the completed supplementary findings and their distinct denominators.}",
       "The supplementary checklist repairs three constructed development cases without an observed probe gain. "
       "Fixed-context feedback changes individual decisions in both directions but has no aggregate advantage in this run."),
      (ZH/"main.tex",r"\Pending{补充真实追加研究的摘要结论；原四组结果保持不变。}",
       "事后追加的封闭世界检查由13/16提高到16/16，但两组探查题均为8/8；"
       "控制单步反馈消融两组均为78/95，其结论不能替代原多轮Agent比较。"),
      (ZH/"main.tex",r"\Pending{结合真实追加结果补充结论，区分原正式比较与事后研究。}",
       "追加清单修复了3道人工开发题，但未观察到探查题提升；固定上下文中的反馈会向两个方向改变决策，本次没有总体分数优势。")]
    for path,old,new in replacements:
        replace_once(path,old,new)
    receipt={"fingerprint":summary["fingerprint"],"all_222_verified":True,"API_calls_by_builder":0,
             "known_cost_subtotal_cny":known_cost,"unknown_usage_calls":1,
             "output_sha256":{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [EN/"generated/supplementary_results.tex",ZH/"generated/supplementary_results.tex"]}}
    (HERE/"supplemental_report_record.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print("Verified supplemental sections generated for both reports; no API calls.")


if __name__=="__main__":
    main()
