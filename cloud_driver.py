"""Colab 实验入口；导入只读数据，API 调用由两个明确的函数触发。"""
import json
from pathlib import Path
import urllib.request

from runner import ExperimentConfig, ExperimentRunner, METHODS

DATA_URL = "https://raw.githubusercontent.com/NLP-Course-CUHKSZ/NLP-course-cuhksz.github.io/main/Assignments/Assignment2/task1/data/1.exam.json"
ROOT = Path("task1_assignment")
ROOT.mkdir(exist_ok=True)
DATA_PATH = Path("task1_smoke/1.exam.json")
if not DATA_PATH.exists():
    DATA_PATH = ROOT / "1.exam.json"
    if not DATA_PATH.exists():
        with urllib.request.urlopen(DATA_URL, timeout=30) as response:
            DATA_PATH.write_bytes(response.read())
DATA = json.loads(DATA_PATH.read_text(encoding="utf-8"))
assert len(DATA) == 100, "正式数据不是100条，请先检查。"
(ROOT / "1.exam.json").write_text(json.dumps(DATA, ensure_ascii=False, indent=2), encoding="utf-8")

# 独立人工构造的开发题，仅用于检查负向问法、单多选和格式；不估计真实考试性能。
# 它们不是 few-shot 演示题，也不从正式100题及其标准解析生成。
DEV_DATA = [
    {"question_type": "最佳选择题", "question": "仅作单位换算检查：以下哪项不等于2克？",
     "option": {"A": "2000毫克", "B": "0.002千克", "C": "2000000微克", "D": "200毫克"},
     "answer": "D", "analysis": "仅供开发评分：200毫克等于0.2克。", "source": "独立人工构造开发题dev01"},
    {"question_type": "多项选择题", "question": "仅作体积换算检查：下列哪些体积等于3升？请选择全部正确选项。",
     "option": {"A": "3000毫升", "B": "300毫升", "C": "0.003立方米", "D": "3000000微升", "E": "0.3升"},
     "answer": "ACD", "analysis": "仅供开发评分：1升=1000毫升=0.001立方米=1000000微升。", "source": "独立人工构造开发题dev02"},
    {"question_type": "配伍选择题", "question": "仅作库存算术检查：原有180件物品，收到35件，发出60件，没有其他出入库。剩余数量对应哪个选项？",
     "option": {"A": "215件", "B": "155件", "C": "145件", "D": "275件"},
     "answer": "B", "analysis": "仅供开发评分：180+35-60=155。", "source": "独立人工构造开发题dev03"},
]


def compact_summary(summary):
    """不展示凭据或客户端，只展示各方法的已完成状态。"""
    for method, values in summary["methods"].items():
        print(method, "状态=", values["status_counts"],
              "正确数=", values["correct_count"],
              "主集合准确率=", values["accuracy_valid_inputs"],
              "90题敏感性准确率=", values["sensitivity_accuracy"],
              "调用=", values["api_attempts"],
              "token=", values["known_total_tokens"], flush=True)
    print("累计调用次数：", summary["cumulative_api_attempts"], flush=True)


def run_development(client):
    """最多24次调用，验证3个独立开发题的四种流程。"""
    config = ExperimentConfig(model="deepseek-flash", max_api_calls=24,
                              max_tokens=768, max_review_rounds=2,
                              sensitivity_excluded_rows=())
    engine = ExperimentRunner(DEV_DATA, client, ROOT / "development", config)
    print("开始独立开发检查：3道人工构造题 × 4种方法。", flush=True)
    result = engine.run()
    compact_summary(result)
    for method, stats in result["methods"].items():
        if not stats["all_valid_inputs_finished"] or set(stats["status_counts"]) != {"completed"}:
            raise RuntimeError(f"{method} 存在调用/格式问题，先修复再启动正式实验。")
    print("开发检查完成。请检查以上结果，再运行正式实验单元格。", flush=True)
    return engine


def run_formal(client):
    """对同一100条原始记录预注册：97条主集合、90条敏感性集合。"""
    development_summary = ROOT / "development/summary.json"
    if not development_summary.exists():
        raise RuntimeError("请先完成独立开发检查。")
    development = json.loads(development_summary.read_text(encoding="utf-8"))
    if not all(s["all_valid_inputs_finished"] and set(s["status_counts"]) == {"completed"}
               for s in development["methods"].values()):
        raise RuntimeError("开发检查仍有失败或未完成项。")
    config = ExperimentConfig(model="deepseek-flash", max_api_calls=800,
                              max_tokens=768, max_review_rounds=2)
    engine = ExperimentRunner(DATA, client, ROOT / "formal", config)
    print("正式设置已冻结：", engine.fingerprint, flush=True)
    print("原始100条；97条有文本选项；90条预注册敏感性集合。", flush=True)
    print("四组无重试时最多776次API调用；预算800只用于明确需要的恢复，程序不隐性重试。", flush=True)
    for method in METHODS:
        print("\n开始方法：", method, flush=True)
        result = engine.run(methods=(method,))
        compact_summary(result)
        if not result["methods"][method]["all_valid_inputs_finished"]:
            print("有未完成项，已保存断点。本次停止，不伪造完整结果。", flush=True)
            return engine
    print("\n四组正式评测已结束。下一步从真实日志生成图表、案例与LaTeX报告。", flush=True)
    return engine


print("完整实验模块已加载；此时没有调用模型 API。")
print("运行顺序：dev_engine = run_development(_assignment2_client)")
print("检查通过后：formal_engine = run_formal(_assignment2_client)")
