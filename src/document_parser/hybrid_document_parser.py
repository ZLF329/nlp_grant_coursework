"""
混合文档解析器 - 统一格式输出
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import pdfplumber

from .base import DocumentParser, ParsedDocument, ParsedSection


@dataclass
class ContentBlock:
    """统一的内容块"""
    block_type: str  # "title" / "text" / "table"
    content: Any  # 内容
    metadata: Dict[str, Any] = None


class HybridDocumentParser(DocumentParser):
    """
    混合文档解析器

    策略：
    1. PDF纯文本 → pdfplumber
    2. PDF有表格 → pdfplumber.extract_tables()
    3. PDF有图片/扫描版 → OCR降级
    4. DOCX → python-docx
    5. 输出统一格式 blocks数组
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.ocr = None
        self.title_library = self._load_title_library()
        self._init_ocr()

    def _load_title_library(self) -> set:
        """加载标题库"""
        try:
            lib_path = Path(__file__).parent / 'title_library.json'
            with open(lib_path, 'r', encoding='utf-8') as f:
                titles = json.load(f)
                return set(titles)
        except Exception as e:
            if self.debug:
                print(f"⚠️ 加载标题库失败: {e}")
            return set()

    def _normalize_title(self, title: str) -> str:
        """规范化标题：去掉数字前缀（如 "1. " 或 "10. "）"""
        import re
        # 匹配开头的数字和点（如 "1. " 或 "10. "）
        normalized = re.sub(r'^\d+\.\s+', '', title.strip())
        return normalized

    def _is_title(self, text: str) -> bool:
        """检查文本是否是标题"""
        normalized = self._normalize_title(text)
        return normalized in self.title_library

    def _init_ocr(self):
        """初始化OCR"""
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_textline_orientation=True, lang="en")
            if self.debug:
                print("[OCR] 初始化成功")
        except Exception as e:
            if self.debug:
                print(f"⚠️ OCR初始化失败: {e}")

    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        """解析文件"""
        file_path = str(file_path)
        ext = Path(file_path).suffix.lower()

        blocks = []

        if ext == '.pdf':
            blocks = self._parse_pdf(file_path)
        elif ext == '.docx':
            blocks = self._parse_docx(file_path)
        elif ext == '.pptx':
            blocks = self._parse_pptx(file_path)
        else:
            return self._unknown_result(file_path)

        # 转换成ParsedSection
        sections = self._blocks_to_sections(blocks)

        return ParsedDocument(
            file_name=Path(file_path).name,
            file_type=ext.lstrip('.'),
            sections=sections,
            raw_params={"blocks_count": len(blocks)}
        )

    def _parse_pdf(self, file_path: str) -> List[ContentBlock]:
        """PDF混合解析"""
        blocks = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # 步骤1：尝试提取表格
                    tables = page.extract_tables()
                    if tables:
                        if self.debug:
                            print(f"  Page {page_num}: 检测到{len(tables)}个表格")
                        for table in tables:
                            blocks.append(ContentBlock(
                                block_type="table",
                                content=table,
                                metadata={"page": page_num, "method": "pdfplumber"}
                            ))

                    # 步骤2：提取文本
                    text = page.extract_text()
                    if text:
                        if self.debug:
                            print(f"  Page {page_num}: 提取纯文本")
                        page_blocks = self._process_pdf_text(text, page_num)
                        blocks.extend(page_blocks)
                    else:
                        # 纯文本提取失败，尝试OCR
                        if self.ocr:
                            if self.debug:
                                print(f"  Page {page_num}: 纯文本为空，使用OCR")
                            page_blocks = self._parse_pdf_page_with_ocr(page, page_num)
                            blocks.extend(page_blocks)

        except Exception as e:
            if self.debug:
                print(f"❌ PDF解析失败: {e}")

        return blocks

    def _process_pdf_text(self, text: str, page_num: int) -> List[ContentBlock]:
        """处理PDF文本：识别标题和文本块"""
        blocks = []
        lines = text.split('\n')

        current_title = None
        current_text = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检查是否是标题
            if self._is_title(line):
                # 保存前一个文本块
                if current_text and current_title:
                    blocks.append(ContentBlock(
                        block_type="text",
                        content="\n".join(current_text),
                        metadata={"page": page_num, "title": current_title}
                    ))
                current_text = []

                # 添加标题
                blocks.append(ContentBlock(
                    block_type="title",
                    content=line,
                    metadata={"page": page_num}
                ))
                current_title = line
            else:
                # 普通文本
                current_text.append(line)

        # 保存最后的文本块
        if current_text and current_title:
            blocks.append(ContentBlock(
                block_type="text",
                content="\n".join(current_text),
                metadata={"page": page_num, "title": current_title}
            ))

        return blocks

    def _parse_pdf_page_with_ocr(self, page, page_num: int) -> List[ContentBlock]:
        """用OCR解析PDF页面"""
        blocks = []

        try:
            im = page.to_image()
            pil_image = im.original

            ocr_result = self.ocr.ocr(pil_image, cls=True)

            if ocr_result:
                blocks = self._process_ocr_result(ocr_result, page_num)

        except Exception as e:
            if self.debug:
                print(f"⚠️ OCR处理失败: {e}")

        return blocks

    def _process_ocr_result(self, ocr_result: List, page_num: int) -> List[ContentBlock]:
        """处理OCR结果"""
        blocks = []
        current_title = None
        current_text = []

        for line in ocr_result:
            if not line:
                continue

            line_text = " ".join([word[0] for word in line])

            # 检查是否是标题
            if self._is_title(line_text):
                if current_text and current_title:
                    blocks.append(ContentBlock(
                        block_type="text",
                        content="\n".join(current_text),
                        metadata={"page": page_num, "title": current_title, "method": "ocr"}
                    ))
                current_text = []

                blocks.append(ContentBlock(
                    block_type="title",
                    content=line_text,
                    metadata={"page": page_num, "method": "ocr"}
                ))
                current_title = line_text
            else:
                if line_text.strip():
                    current_text.append(line_text)

        if current_text and current_title:
            blocks.append(ContentBlock(
                block_type="text",
                content="\n".join(current_text),
                metadata={"page": page_num, "title": current_title, "method": "ocr"}
            ))

        return blocks

    def _parse_docx(self, file_path: str) -> List[ContentBlock]:
        """解析DOCX"""
        from docx import Document

        blocks = []

        try:
            doc = Document(file_path)

            current_title = None
            current_text = []

            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue

                # 启发式识别标题：优先检查段落样式
                is_title = False

                # 1. 检查段落样式是否是Heading
                if para.style and para.style.name.startswith('Heading'):
                    is_title = True

                # 2. 再检查标题库
                elif self._is_title(text):
                    is_title = True

                if is_title:
                    # 保存前一个文本块
                    if current_text and current_title:
                        blocks.append(ContentBlock(
                            block_type="text",
                            content="\n".join(current_text),
                            metadata={"title": current_title}
                        ))
                    current_text = []

                    # 添加标题
                    blocks.append(ContentBlock(
                        block_type="title",
                        content=text,
                        metadata={}
                    ))
                    current_title = text
                else:
                    # 文本
                    current_text.append(text)

            # 保存最后的文本块
            if current_text and current_title:
                blocks.append(ContentBlock(
                    block_type="text",
                    content="\n".join(current_text),
                    metadata={"title": current_title}
                ))

            # 处理表格
            for table in doc.tables:
                table_data = []
                for row in table.rows:
                    row_data = [cell.text for cell in row.cells]
                    table_data.append(row_data)

                blocks.append(ContentBlock(
                    block_type="table",
                    content=table_data,
                    metadata={"method": "docx"}
                ))

        except Exception as e:
            if self.debug:
                print(f"❌ DOCX解析失败: {e}")

        return blocks

    def _parse_pptx(self, file_path: str) -> List[ContentBlock]:
        """解析PPT"""
        # TODO: 实现PPT解析
        return []

    def _blocks_to_sections(self, blocks: List[ContentBlock]) -> List[ParsedSection]:
        """将ContentBlock转换为ParsedSection"""
        sections = []

        for block in blocks:
            title = block.block_type

            sections.append(ParsedSection(
                title=title,
                content=block.content,
                type=block.block_type
            ))

        return sections

    def auto_detect_params(self, file_path: str) -> Dict[str, Any]:
        """检测参数"""
        return {"method": "hybrid"}

    def _unknown_result(self, file_path: str) -> ParsedDocument:
        """未知文件类型"""
        return ParsedDocument(
            file_name=Path(file_path).name,
            file_type="unknown",
            sections=[],
            raw_params={"error": "不支持的文件类型"}
        )
