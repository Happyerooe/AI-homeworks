"""
ocr_utils.py
基于 PaddleOCR 的图片文字识别工具模块

使用 paddleocr 3.x API（predict 方法），支持中英文识别。
"""

from paddleocr import PaddleOCR


# ---------------------------------------------------------------------------
# 全局 OCR 引擎（懒加载，避免在 import 时就下载模型）
# ---------------------------------------------------------------------------
_ocr_engine = None


def _get_ocr_engine():
    """获取（必要时初始化）全局 OCR 引擎实例。

    使用 lang='ch' 初始化，支持中文 + 英文识别。
    enable_mkldnn=False 用于规避部分环境下 OneDNN 推理报错的问题。
    """
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(lang="ch", enable_mkldnn=False)
    return _ocr_engine


def ocr_image(image_path):
    """对指定图片进行 OCR 文字识别。

    参数:
        image_path (str): 图片文件路径（支持 png / jpg / jpeg / bmp 等）。

    返回:
        str: 识别出的所有文字，按行合并成一个字符串（每行文本用换行符 \\n 分隔）。
             如果未识别到任何文字，返回空字符串。

    异常:
        FileNotFoundError: 图片文件不存在时抛出。
        RuntimeError: OCR 推理过程中发生错误时抛出。
    """
    import os

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    engine = _get_ocr_engine()

    try:
        # paddleocr 3.x 使用 predict() 方法（旧版 ocr() 已废弃）
        results = engine.predict(image_path)
    except Exception as e:
        raise RuntimeError(f"OCR 识别失败: {e}") from e

    # results 是一个列表，每个元素对应一张图片的识别结果（OCRResult 对象）
    all_lines = []
    for result in results:
        # rec_texts 字段是当前图片中所有识别到的文本行列表
        rec_texts = result.get("rec_texts", [])
        all_lines.extend(rec_texts)

    return "\n".join(all_lines)


# ---------------------------------------------------------------------------
# 测试代码：直接运行本文件时执行
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from PIL import Image, ImageDraw, ImageFont

    # 创建一张测试图片，写入中英文文字
    test_image_path = "/data/user/work/test_sample.png"
    img = Image.new("RGB", (500, 250), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "Hello World", fill="black")
    draw.text((50, 100), "12345", fill="black")
    draw.text((50, 150), "PaddleOCR Test", fill="black")
    img.save(test_image_path)
    print(f"测试图片已生成: {test_image_path}")

    # 调用 ocr_image 进行识别
    print("\n========== OCR 识别结果 ==========")
    text = ocr_image(test_image_path)
    print(text)
    print("==================================")

    # 清理测试图片
    os.remove(test_image_path)
    print(f"\n测试图片已清理: {test_image_path}")
