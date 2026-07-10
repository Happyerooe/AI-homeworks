"""
grader.py
基于 DeepSeek API 的 AI 作业批改模块

使用 openai 库调用 DeepSeek 接口，实现自动评分与评语生成。
"""

import json
from openai import OpenAI


# ---------------------------------------------------------------------------
# DeepSeek API 配置
# ---------------------------------------------------------------------------
BASE_URL = "https://api.deepseek.com/v1"
API_KEY = "请在此填入你的API Key"
MODEL = "deepseek-chat"


# ---------------------------------------------------------------------------
# 全局客户端（懒加载）
# ---------------------------------------------------------------------------
_client = None


def _get_client():
    """获取（必要时初始化）全局 OpenAI 兼容客户端实例。"""
    global _client
    if _client is None:
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    return _client


def grade_homework(student_answer: str, ref_answer: str) -> dict:
    """调用 DeepSeek 大模型批改学生作业。

    参数:
        student_answer (str): 学生的答案。
        ref_answer (str): 参考答案。

    返回:
        dict: 包含 score(分数)、comment(评语)、weakness(薄弱点分析) 的字典。

    异常:
        ValueError: API Key 未配置时抛出。
        RuntimeError: API 调用失败或返回结果无法解析为 JSON 时抛出。
    """
    # 检查 API Key 是否已配置
    if not API_KEY or API_KEY == "你的API Key":
        raise ValueError(
            "API Key 尚未配置。请在 grader.py 文件中将 API_KEY 替换为真实的 DeepSeek API Key。"
        )

    # 构建 Prompt（增强版：容忍 OCR 识别噪声）
    prompt = (
        "你是一位严格的老师，正在批改学生作业。\n\n"
        "重要提示：学生答案是通过 OCR 从手写作业照片中识别出来的，可能存在识别错误，\n"
        "例如：数学符号丢失、数字粘连、上下标丢失、矩阵格式混乱等。\n"
        "请你结合参考答案，尽可能从 OCR 结果中还原学生的真实解题意图进行批改。\n"
        "如果 OCR 结果完全无法辨认，请在评语中说明，并给较低分数。\n"
        "如果 OCR 结果部分可辨认，请根据可辨认的部分进行评分。\n\n"
        f"参考答案：\n{ref_answer}\n\n"
        f"学生答案（OCR识别结果，可能含噪声）：\n{student_answer}\n\n"
        "请返回JSON格式：\n"
        '{"score": 分数(0-100的整数), "comment": "评语", "weakness": "薄弱点分析"}'
    )

    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个严格的教师，擅长客观批改作业。"
                        "学生答案来自OCR识别，可能存在格式混乱或字符错误，"
                        "你需要尽量还原学生意图并公平评分。"
                        "输出必须是合法的 JSON 格式。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # 低温度使评分更稳定、客观
            response_format={"type": "json_object"},
        )
    except Exception as e:
        raise RuntimeError(f"DeepSeek API 调用失败: {e}") from e

    raw_content = response.choices[0].message.content.strip()

    # 解析 JSON
    try:
        result = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"模型返回内容无法解析为 JSON: {e}\n原始内容: {raw_content}"
        ) from e

    # 检查必要字段
    required_keys = {"score", "comment", "weakness"}
    missing = required_keys - set(result.keys())
    if missing:
        raise RuntimeError(
            f"返回的 JSON 缺少必要字段: {missing}\n原始内容: {raw_content}"
        )

    return result


# ---------------------------------------------------------------------------
# 测试代码：直接运行本文件时执行
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("AI 批改功能测试")
    print("=" * 50)

    # 检查 API Key
    if not API_KEY or API_KEY == "你的API Key":
        print()
        print("⚠️ 警告：API Key 尚未配置！")
        print("   请编辑 grader.py，将 API_KEY 替换为真实的 DeepSeek API Key。")
        print("   获取方式：访问 https://platform.deepseek.com/api_keys")
        print()
        print("以下为 Prompt 预览（供调试参考）：")
        student = "水是生命之源，我们需要节约用水。"
        ref = "水（H₂O）是维持生命活动所必需的物质。节约用水是每个公民的责任。"
        prompt = (
            "你是一位严格的老师。请批改以下作业。\n"
            f"参考答案：{ref}\n"
            f"学生答案：{student}\n"
            '请返回JSON格式：{"score": 分数(0-100), "comment": "评语", "weakness": "薄弱点分析"}'
        )
        print(f"\n{prompt}")
    else:
        # API Key 已配置，执行实际调用
        test_student = "水是生命之源，我们需要节约用水。"
        test_ref = "水（H₂O）是维持生命活动所必需的物质。节约用水是每个公民的责任。"

        print(f"\n学生答案: {test_student}")
        print(f"参考答案: {test_ref}")
        print()

        try:
            result = grade_homework(test_student, test_ref)
            print("✅ 批改成功！")
            print(f"   得分: {result['score']}")
            print(f"   评语: {result['comment']}")
            print(f"   薄弱点: {result['weakness']}")
        except Exception as e:
            print(f"❌ 批改失败: {e}")

    print()
    print("=" * 50)
