import fitz  # PyMuPDF
import re
import logging
import os

# 配置日志输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AcademicPDFParser:
    def __init__(self, header_ratio=0.08, footer_ratio=0.08):
        """
        初始化 PDF 解析器
        :param header_ratio: 页面顶部 8% 区域判定为页眉
        :param footer_ratio: 页面底部 8% 区域判定为页脚
        """
        self.header_ratio = header_ratio
        self.footer_ratio = footer_ratio
        
        # 论文结构识别 (支持数字编号如 "1. Introduction" 以及无编号如 "Abstract")
        self.section_pattern = re.compile(
            r'^(?:'
            r'(?:(?:I{1,3}|IV|V|VI{1,3}|IX|X|\d+(?:\.\d+)*)[\.\s]+[A-Z][a-zA-Z\s]+)|'  
            r'(?:Abstract|Introduction|Background|Methodology|Methods|'               
            r'Experiments?|Results?|Discussion|Conclusion)'
            r')$', 
            re.IGNORECASE
        )
        
        # 匹配参考文献标识
        self.ref_pattern = re.compile(
            r'^(?:(?:\d+(?:\.\d+)*|I{1,3}|IV|V|VI{1,3}|IX|X)[\.\s]+)?'
            r'(?:References|Bibliography|REFERENCES|BIBLIOGRAPHY)\s*$', 
            re.IGNORECASE
        )

    def parse(self, pdf_path: str, external_metadata: dict = None) -> dict:
        """
        解析单篇学术 PDF 并提取结构化文本
        :return: 包含 metadata 和 sections 的字典，若解析失败返回 None
        """
        if not os.path.exists(pdf_path):
            logging.error(f"❌ 文件不存在: {pdf_path}")
            return None

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logging.error(f"❌ 无法打开 PDF {pdf_path}: {e}")
            return None

        # 异常 PDF 检测（跳过扫描版或无文本层）
        check_pages = min(2, len(doc))
        if check_pages > 0:
            text_length = sum([len(page.get_text().strip()) for page in doc[:check_pages]])
            if text_length < 100:
                logging.warning(f"⚠️ [跳过扫描版/无文本层 PDF]: {pdf_path}")
                doc.close()
                return None
        else:
            doc.close()
            return None

        # 基础元数据对齐
        metadata = external_metadata or {}
        if not metadata:
            pdf_meta = doc.metadata
            metadata = {
                "title": pdf_meta.get("title") or os.path.basename(pdf_path),
                "authors": pdf_meta.get("author", "Unknown"),
                "year": int(pdf_meta.get("creationDate", "D:2024")[2:6]) if pdf_meta.get("creationDate") else 2024,
                "arxiv_id": "Unknown",
                "local_path": pdf_path
            }

        sections = []
        current_section = "Title_and_Abstract"
        current_content = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_height = page.rect.height if page.rect.height > 0 else 842
            
            blocks = page.get_text("blocks", sort=True)
            
            for block in blocks:
                x0, y0, x1, y1, text, block_no, block_type = block
                if block_type != 0:
                    continue
                
                text = text.strip()
                if not text:
                    continue

                # 启发式规则过滤页眉页脚
                if y0 < (page_height * self.header_ratio) or y1 > (page_height * (1 - self.footer_ratio)):
                    if len(text) < 10 and text.isdigit():
                        continue 
                    continue

                # 核心逻辑: 剔除参考文献及之后的内容
                first_line = text.split('\n')[0].strip()
                if self.ref_pattern.match(first_line):
                    logging.info(f"🛑 在 {os.path.basename(pdf_path)} 第 {page_num+1} 页检测到参考文献，安全截断。")
                    if current_content:
                        sections.append({"section_name": current_section, "content": " ".join(current_content)})
                    doc.close()
                    return {"metadata": metadata, "sections": sections}

                # 章节标题识别
                if len(first_line) < 150 and self.section_pattern.match(first_line):
                    if current_content:
                        sections.append({"section_name": current_section, "content": " ".join(current_content)})
                    
                    current_section = first_line.replace('\n', ' ')
                    current_content = []
                    
                    remaining_text = text[len(first_line):].strip()
                    if remaining_text:
                        remaining_text = remaining_text.replace('-\n', '').replace('\n', ' ')
                        current_content.append(remaining_text)
                else:
                    text = text.replace('-\n', '').replace('\n', ' ')
                    current_content.append(text)

        if current_content:
            sections.append({"section_name": current_section, "content": " ".join(current_content)})

        doc.close()
        logging.info(f"✅ 成功解析 PDF: {os.path.basename(pdf_path)}, 共提取 {len(sections)} 个结构化章节。")
        return {"metadata": metadata, "sections": sections}
