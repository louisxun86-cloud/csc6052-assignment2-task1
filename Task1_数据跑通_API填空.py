# Task 1 数据与 API 连通性检查：可以整段粘贴到一个 Colab 代码单元运行。
# 建议保持 API_KEY 为空，运行后在隐藏输入框粘贴密钥；直接回车则只检查数据。
# 如果填写下面的空字符串，密钥会保存在 notebook 源码中，请勿分享带密钥的副本。
API_KEY = ""  # 唯一填空位置；也可以留空，在运行后的隐藏输入框填写。

import getpass
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"
DATA_URL = "https://raw.githubusercontent.com/NLP-Course-CUHKSZ/NLP-course-cuhksz.github.io/main/Assignments/Assignment2/task1/data/1.exam.json"
OUTPUT_DIR = Path("task1_smoke")


def question_input(item):
    # 只挑出允许发给模型的字段，不发送标准答案和标准解析。
    options = {k: v for k, v in item["option"].items() if isinstance(v, str) and v.strip()}
    if not options or not set(options) <= set("ABCDE"):
        raise ValueError("题目缺少有效选项，不能发送给模型。")
    return {"question_type": item["question_type"], "question": item["question"], "options": options}


def parse_answer(raw_text, item):
    result = json.loads(raw_text)
    if not isinstance(result, dict):
        raise ValueError("模型必须返回 JSON 对象。")
    choices = result.get("answer")
    options = question_input(item)["options"]
    if not isinstance(choices, list) or not choices:
        raise ValueError("answer 必须是非空字母列表。")
    if any(not isinstance(a, str) or a not in options for a in choices):
        raise ValueError("答案包含无效或空选项。")
    if len(choices) != len(set(choices)):
        raise ValueError("答案包含重复选项。")
    if item["question_type"] != "多项选择题" and len(choices) != 1:
        raise ValueError("单选题必须只选一个选项。")
    return "".join(sorted(choices))


def reference_answer(item):
    # 此函数只在本地评分和数据检查时使用。
    raw = item["answer"].strip()
    if not re.fullmatch(r"[A-E]+", raw):
        raise ValueError("标准答案格式异常。")
    return parse_answer(json.dumps({"answer": list(raw)}), item)


def main():
    print("第 1 步：下载 Assignment 2 正式数据……")
    with urllib.request.urlopen(DATA_URL, timeout=30) as response:
        raw_data = response.read()
    data = json.loads(raw_data)
    required = {"question", "option", "question_type", "answer", "analysis", "source"}
    if not isinstance(data, list) or len(data) != 100:
        raise ValueError("预期为 100 道题，请检查数据地址或课程更新。")
    if not all(isinstance(q, dict) and required <= q.keys() for q in data):
        raise ValueError("数据字段不符合预期。")
    counts = dict(Counter(q["question_type"] for q in data))
    expected = {"最佳选择题": 35, "配伍选择题": 45, "综合分析选择题": 12, "多项选择题": 8}
    if counts != expected:
        raise ValueError("题型分布发生变化，请先检查数据。")
    missing = [i for i, q in enumerate(data) if not any(isinstance(v, str) and v.strip() for v in q["option"].values())]
    usable = [i for i in range(len(data)) if i not in missing]
    for i in usable:
        reference_answer(data[i])
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "1.exam.json").write_bytes(raw_data)
    audit = {
        "dataset_url": DATA_URL,
        "sha256": hashlib.sha256(raw_data).hexdigest(),
        "total": len(data),
        "question_types": counts,
        "missing_options_row_numbers_1_based": [i + 1 for i in missing],
        "rows_with_text_options": len(usable),
        "note": "保留全部原始记录；有文本选项不保证题干中的图片或背景完整。",
    }
    (OUTPUT_DIR / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("数据读取成功：", len(data), "题")
    print("题型统计：", counts)
    print("缺少全部选项的记录位置（从 1 开始）：", [i + 1 for i in missing])
    print("有文本选项且通过本地解析检查：", len(usable), "条")
    print("原始数据和检查记录已保存到 task1_smoke；没有删除缺项记录。")

    index = usable[0]
    item = data[index]
    payload = question_input(item)
    print("\n第 2 步：将只测试下面这一题，不跑完整实验。")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    key = API_KEY.strip() or getpass.getpass("粘贴 DeepSeek 官方 API Key（隐藏输入；直接回车跳过）：").strip()
    if not key:
        print("\n数据已跑通。未填写 Key，本次没有调用模型 API。")
        return

    # 只有填写密钥后才加载 SDK；缺少依赖时从标准包仓库安装。
    if importlib.util.find_spec("openai") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "openai"])
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=BASE_URL, timeout=60.0, max_retries=0)
    prompt = (
        "请回答药师考试选择题。多项选择题选择全部正确选项，其他题型只选择一个选项。"
        '只输出 json，例如 {"answer":["A"]}；多选例如 {"answer":["A","C"]}。'
    )
    print("\n第 3 步：向 DeepSeek 官方 API 发送一次单题请求……")
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": prompt},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            extra_body={"thinking": {"type": "disabled"}},
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=256,
        )
    except Exception as exc:
        # 不输出服务器原始报文或客户端对象，避免意外展示凭据。
        status = getattr(exc, "status_code", None)
        print("API 调用未成功，错误类型：", type(exc).__name__, "状态码：", status)
        print("请检查官方 Key、账户余额、网络及模型名称。没有自动重试；数据检查结果仍然保留。")
        return
    finally:
        client.close()
        key = ""
    elapsed = time.perf_counter() - started
    choice = response.choices[0]
    raw_answer = (choice.message.content or "").strip()
    if choice.finish_reason != "stop" or not raw_answer:
        raise ValueError("API 已返回，但答案为空或被截断。先检查，不自动重试。")
    prediction = parse_answer(raw_answer, item)
    gold = reference_answer(item)
    result = {
        "purpose": "connectivity_check_not_final_experiment",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_requested": MODEL, "model_returned": response.model,
        "base_url": BASE_URL, "dataset_sha256": audit["sha256"],
        "question_row_number_1_based": index + 1, "input": payload,
        "system_prompt": prompt,
        "settings": {"thinking": "disabled", "temperature": 0, "max_tokens": 256},
        "prediction": prediction, "reference": gold, "correct": prediction == gold,
        "elapsed_seconds": elapsed,
        "usage": response.usage.model_dump() if response.usage else None,
        "raw_response": raw_answer,
    }
    (OUTPUT_DIR / "single_question_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("模型答案：", prediction, "；参考答案：", gold)
    print("是否正确：", prediction == gold, "；耗时：", round(elapsed, 2), "秒")
    print("Token 使用：", result["usage"])
    print("结果已保存到 task1_smoke/single_question_result.json。")
    print("单题连通检查结束。这不是正式实验；再次运行并填 Key 会再次调用一次 API。")


main()
