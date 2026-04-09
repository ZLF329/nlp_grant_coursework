"""
PDF 扫描版 OCR 解析器
"""
from typing import List, Dict, Any
import pdfplumber
import io
from PIL import Image

from .base import DocumentParser, ParsedDocument, ParsedSection


class PDFOCRParser(DocumentParser):
    """
    PDF 扫描版解析器 (OCR)

    支持两种 OCR 引擎：
    1. Tesseract (需要 pytesseract, 快速)
    2. PaddleOCR (需要 paddleocr, 准确，支持中文)
    """

    def __init__(self, ocr_engine: str = "auto", debug: bool = False):
        """
        初始化

        参数：
        - ocr_engine: "tesseract", "paddleocr" 或 "auto"
        - debug: 调试模式
        """
        self.ocr_engine = ocr_engine
        self.debug = debug
        self.ocr = None

        self._init_ocr()

    def _init_ocr(self):
        """初始化 OCR 引擎"""
        if self.ocr_engine == "auto":
            # 自动选择可用的引擎
            try:
                from paddleocr import PaddleOCR

                self.ocr = PaddleOCR(use_textline_orientation=True, lang="en")
                self.ocr_engine = "paddleocr"
                if self.debug:
                    print("[OCR] 使用 PaddleOCR")
            except ImportError:
                try:
                    import pytesseract

                    self.ocr_engine = "tesseract"
                    if self.debug:
                        print("[OCR] 使用 Tesseract")
                except ImportError:
                    print("⚠️  未安装任何 OCR 引擎")
                    self.ocr_engine = None

        elif self.ocr_engine == "paddleocr":
            try:
                from paddleocr import PaddleOCR

                self.ocr = PaddleOCR(use_textline_orientation=True, lang="en")
                if self.debug:
                    print("[OCR] 初始化 PaddleOCR")
            except ImportError:
                print("❌ PaddleOCR 未安装")
                self.ocr_engine = None

        elif self.ocr_engine == "tesseract":
            try:
                import pytesseract

                if self.debug:
                    print("[OCR] 初始化 Tesseract")
            except ImportError:
                print("❌ Tesseract 未安装")
                self.ocr_engine = None

    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        """解析扫描版 PDF"""
        if not self.ocr_engine:
            return self._no_ocr_result(file_path)

        sections = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # 转换页面为图像
                    im = page.to_image()
                    pil_image = im.original

                    # OCR 识别
                    text = self._ocr_image(pil_image)

                    if text:
                        sections.append(
                            ParsedSection(
                                title=f"Page {page_num}",
                                content=text,
                                type="text",
                            )
                        )

        except Exception as e:
            if self.debug:
                print(f"❌ OCR 处理失败: {e}")

        return ParsedDocument(
            file_name=file_path.split("/")[-1],
            file_type="pdf_scan",
            sections=sections,
            raw_params={"ocr_engine": self.ocr_engine},
        )

    def _ocr_image(self, image) -> str:
        """对图像进行 OCR"""
        try:
            if self.ocr_engine == "paddleocr":
                return self._paddleocr_extract(image)
            elif self.ocr_engine == "tesseract":
                return self._tesseract_extract(image)
        except Exception as e:
            if self.debug:
                print(f"⚠️  OCR 失败: {e}")
            return ""

        return ""

    def _paddleocr_extract(self, image) -> str:
        """使用 PaddleOCR 提取文本"""
        try:
            result = self.ocr.ocr(image)

            # 提取文本
            texts = []
            for line in result:
                if line:
                    for word_info in line:
                        text = word_info[1][0]
                        conf = word_info[1][1]

                        # 只保留置信度较高的文本
                        if conf > 0.5:
                            texts.append(text)

            return "\n".join(texts)
        except Exception as e:
            if self.debug:
                print(f"❌ PaddleOCR 失败: {e}")
            return ""

    def _tesseract_extract(self, image) -> str:
        """使用 Tesseract 提取文本"""
        try:
            import pytesseract

            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            if self.debug:
                print(f"❌ Tesseract 失败: {e}")
            return ""

    def auto_detect_params(self, file_path: str) -> Dict[str, Any]:
        """检测参数"""
        return {"ocr_engine": self.ocr_engine}

    def _no_ocr_result(self, file_path: str) -> ParsedDocument:
        """未安装 OCR 的结果"""
        return ParsedDocument(
            file_name=file_path.split("/")[-1],
            file_type="pdf_scan",
            sections=[],
            raw_params={
                "error": "未安装 OCR 引擎",
                "hint": "安装: pip install paddleocr 或 pip install pytesseract",
            },
        )
