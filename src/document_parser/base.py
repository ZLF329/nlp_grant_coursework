"""
通用文档解析器基类
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class ParsedSection:
    """解析后的一个章节"""
    title: str                          # 章节标题
    content: Any                        # 内容（可以是 str, dict, list 等）
    type: str                           # 内容类型：text, dict, list, table 等
    metadata: Dict[str, Any] = None     # 元数据


@dataclass
class ParsedDocument:
    """解析后的完整文档"""
    file_name: str
    file_type: str                      # pdf_text, docx, pptx 等
    sections: List[ParsedSection]       # 所有章节
    raw_params: Dict[str, Any]          # 自动检测到的参数（用于调试）
    metadata: Dict[str, Any] = None


class DocumentParser(ABC):
    """所有解析器的基类"""

    @abstractmethod
    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        """解析文件，返回标准化结果"""
        pass

    @abstractmethod
    def auto_detect_params(self, file_path: str) -> Dict[str, Any]:
        """自动检测这个文件的参数"""
        pass
