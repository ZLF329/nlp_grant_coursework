"""
文件类型检测器
"""
from enum import Enum
from pathlib import Path
import pdfplumber
from collections import defaultdict


class FileType(Enum):
    """文件类型枚举"""
    PDF_TEXT = "pdf_text"
    PDF_SCAN = "pdf_scan"
    DOCX = "docx"
    PPTX = "pptx"
    TXT = "txt"
    UNKNOWN = "unknown"


def detect_file_type(file_path: str) -> FileType:
    """检测文件类型"""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return _detect_pdf_type(file_path)
    elif ext == ".docx":
        return FileType.DOCX
    elif ext == ".pptx":
        return FileType.PPTX
    elif ext == ".txt":
        return FileType.TXT
    else:
        return FileType.UNKNOWN


def _detect_pdf_type(file_path: str) -> FileType:
    """
    判断 PDF 是纯文本还是扫描版

    原理：
    - 纯文本 PDF: 可以用 pdfplumber 提取大量文本
    - 扫描版 PDF: 主要包含图像，文本很少
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            if not pdf.pages:
                return FileType.UNKNOWN

            # 采样策略：前3页 + 中间 + 最后3页（避免只看开头判断失误）
            total_pages = len(pdf.pages)
            if total_pages <= 6:
                sample_indices = list(range(total_pages))
            else:
                sample_indices = (
                    list(range(3)) +  # 前3页
                    [total_pages // 2] +  # 中间1页
                    list(range(total_pages - 3, total_pages))  # 最后3页
                )

            sample_pages = [pdf.pages[i] for i in sample_indices if i < total_pages]

            # 统计文字和图片
            total_chars = 0
            total_images = 0

            for page in sample_pages:
                # 提取文本
                try:
                    text = page.extract_text()
                    if text:
                        total_chars += len(text)
                except:
                    pass

                # 计算图片
                try:
                    if hasattr(page, "images"):
                        total_images += len(page.images)
                except:
                    pass

            # 判断
            # 如果字符数 > 100，可能是纯文本
            # 如果图片比较多且字符少，可能是扫描版
            if total_chars > 100:
                return FileType.PDF_TEXT
            elif total_images > 0:
                return FileType.PDF_SCAN
            else:
                # 无法确定，假设是扫描版（需要OCR）
                return FileType.PDF_SCAN

    except Exception as e:
        print(f"❌ 检测 PDF 类型失败: {e}")
        return FileType.PDF_SCAN  # 降级为扫描版，触发OCR


def is_scanned_pdf(file_path: str) -> bool:
    """判断是否为扫描版 PDF"""
    return detect_file_type(file_path) == FileType.PDF_SCAN
