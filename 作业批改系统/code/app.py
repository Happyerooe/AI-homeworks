"""
app.py
AI 智能作业批改系统 —— Gradio Web 界面

功能：
1. 上传作业照片 → OCR 识别学生答案
2. 输入参考答案
3. 调用 DeepSeek AI 批改
4. 展示识别文字、得分、评语与薄弱点分析
"""

import gradio as gr
from ocr_utils import ocr_image, _get_ocr_engine
from grader import grade_homework, _get_client


# ---------------------------------------------------------------------------
# 启动时预加载模型，避免首次点击时超时
# ---------------------------------------------------------------------------
print("正在预加载 OCR 模型，请稍候...")
_get_ocr_engine()
print("OCR 模型加载完成。")

print("正在初始化 DeepSeek API 客户端...")
_get_client()
print("API 客户端初始化完成。")
print("所有服务就绪，可以开始批改！\n")


# ---------------------------------------------------------------------------
# 核心处理函数
# ---------------------------------------------------------------------------
def process_homework(image_path, ref_answer):
    """完整的作业批改流程。

    参数:
        image_path (str): 用户上传的作业图片路径。
        ref_answer (str): 老师输入的参考答案。

    返回:
        tuple: (识别文字, 得分, 评语, 薄弱点) 或错误提示。
    """
    # 检查输入
    if image_path is None:
        return "请先上传作业图片", "", "", ""
    if not ref_answer or not ref_answer.strip():
        return "", "", "", "请输入参考答案"

    # 步骤 1: OCR 识别
    try:
        recognized_text = ocr_image(image_path)
    except FileNotFoundError as e:
        return f"图片文件不存在: {e}", "", "", ""
    except RuntimeError as e:
        return f"OCR 识别失败: {e}", "", "", ""
    except Exception as e:
        return f"OCR 发生未知错误: {e}", "", "", ""

    if not recognized_text.strip():
        return "未能从图片中识别到任何文字，请检查图片清晰度。", "", "", ""

    # 步骤 2: AI 批改
    try:
        result = grade_homework(recognized_text, ref_answer.strip())
    except ValueError as e:
        return recognized_text, "", "", f"批改配置错误: {e}"
    except RuntimeError as e:
        return recognized_text, "", "", f"AI 批改失败: {e}"
    except Exception as e:
        return recognized_text, "", "", f"批改发生未知错误: {e}"

    score = result.get("score", "")
    comment = result.get("comment", "")
    weakness = result.get("weakness", "")

    return recognized_text, str(score), comment, weakness


# ---------------------------------------------------------------------------
# Gradio 界面构建
# ---------------------------------------------------------------------------
with gr.Blocks(title="AI智能作业批改系统") as demo:
    # 页面标题
    gr.Markdown("# 🤖 AI智能作业批改系统")
    gr.Markdown("上传学生作业照片，输入参考答案，AI 自动完成 OCR 识别与智能批改。")

    with gr.Row():
        # ---------- 左侧：输入区 ----------
        with gr.Column(scale=1):
            gr.Markdown("### 📤 上传作业")
            image_input = gr.Image(
                type="filepath",
                label="作业照片",
                sources=["upload"],
            )

            gr.Markdown("### ✏️ 参考答案")
            ref_input = gr.Textbox(
                label="参考答案",
                placeholder="请在此输入参考答案...",
                lines=6,
            )

            submit_btn = gr.Button("🚀 开始批改", variant="primary", size="lg")

        # ---------- 右侧：输出区 ----------
        with gr.Column(scale=1):
            gr.Markdown("### 📝 识别结果")
            ocr_output = gr.Textbox(
                label="OCR 识别出的文字",
                lines=8,
                interactive=False,
            )

            gr.Markdown("### 📊 批改结果")
            score_output = gr.Textbox(
                label="得分",
                interactive=False,
            )
            comment_output = gr.Textbox(
                label="评语",
                lines=3,
                interactive=False,
            )
            weakness_output = gr.Textbox(
                label="薄弱点分析",
                lines=3,
                interactive=False,
            )

    # 按钮点击事件绑定
    submit_btn.click(
        fn=process_homework,
        inputs=[image_input, ref_input],
        outputs=[ocr_output, score_output, comment_output, weakness_output],
    )

    gr.Markdown("---")
    gr.Markdown(
        "<center>Powered by PaddleOCR + DeepSeek | 本地部署，数据安全</center>"
    )


# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
