# pdf_parser.py
import fitz  # PyMuPDF
import re
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AcademicPDFParser:
    def __init__(self, header_margin=50, footer_margin=50):
        """
        初始化解析器
        :param header_margin: 页面顶部多少像素内被认为是页眉
        :param footer_margin: 页面底部多少像素内被认为是页脚
        """
        self.header_margin = header_margin
        self.footer_margin = footer_margin
        
        # 匹配学术论文常见的章节标题，如 "1. Introduction", "II. Related Work", "3 METHODOLOGY"
        self.section_pattern = re.compile(r'^(I{1,3}|IV|V|VI{1,3}|IX|X|\d+(\.\d+)*)[\.\s]+[A-Z][a-zA-Z\s]+$')
        
        # 匹配参考文献标识
        self.ref_pattern = re.compile(r'^(References|Bibliography|REFERENCES|BIBLIOGRAPHY)\s*$')

    def parse(self, pdf_path, external_metadata=None):
        """
        解析单篇PDF
        :param pdf_path: PDF文件路径
        :param external_metadata: 外部传入的元数据(来自成员A的下载器，如标题、作者、年份)
        :return: {"metadata": dict, "sections": [{"section_name": str, "content": str}]}
        """
        if not os.path.exists(pdf_path):
            logging.error(f"文件不存在: {pdf_path}")
            return None

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logging.error(f"无法打开PDF {pdf_path}: {e}")
            return None

        # 1. 异常PDF检测：检查是否为扫描版（前两页文本极少）
        text_length = sum([len(page.get_text()) for page in doc[:2]])
        if text_length < 100:
            logging.warning(f"跳过扫描版或无文本层PDF: {pdf_path}")
            doc.close()
            return None

        # 提取基础元数据
        metadata = external_metadata or {}
        if not metadata:
            pdf_meta = doc.metadata
            metadata = {
                "title": pdf_meta.get("title", os.path.basename(pdf_path)),
                "author": pdf_meta.get("author", "Unknown"),
                "year": pdf_meta.get("creationDate", "")[2:6] if pdf_meta.get("creationDate") else "Unknown",
                "source_file": os.path.basename(pdf_path)
            }

        sections = []
        current_section = "Abstract/Introduction"  # 默认起始章节
        current_content = []
        
        page_height = doc[0].rect.height if len(doc) > 0 else 842 # 默认A4高度
        
        # 2. 逐页读取并清洗
        for page_num in range(len(doc)):
            page = doc[page_num]
            # 获取文本块，包含坐标等信息，按照阅读顺序排序
            blocks = page.get_text("blocks", sort=True)
            
            for block in blocks:
                x0, y0, x1, y1, text, block_no, block_type = block
                
                # 过滤非文本块(图像等)
                if block_type != 0:
                    continue
                
                text = text.strip()
                if not text:
                    continue

                # 3. 启发式规则过滤页眉页脚
                if y0 < self.header_margin or y1 > (page_height - self.footer_margin):
                    # 也有可能是单行数字(页码)
                    if len(text) < 10 and text.isdigit():
                        continue 
                    continue # 跳过页眉页脚块

                # 4. 剔除参考文献及之后内容
                if self.ref_pattern.match(text):
                    logging.info(f"在 {pdf_path} 第 {page_num+1} 页检测到参考文献，停止提取后续内容。")
                    # 保存当前收集的最后一个章节
                    if current_content:
                        sections.append({"section_name": current_section, "content": "\n".join(current_content)})
                    doc.close()
                    return {"metadata": metadata, "sections": sections}

                # 5. 章节标题识别
                # 满足正则，且通常标题比较短（例如小于100个字符）
                if len(text) < 100 and self.section_pattern.match(text.split('\n')[0]):
                    # 保存上一个章节
                    if current_content:
                        sections.append({"section_name": current_section, "content": "\n".join(current_content)})
                    # 开启新章节
                    current_section = text.replace('\n', ' ')
                    current_content = []
                else:
                    # 去除论文中常见的换行连字符
                    text = text.replace('-\n', '')
                    current_content.append(text)

        # 保存最后一节
        if current_content:
            sections.append({"section_name": current_section, "content": "\n".join(current_content)})

        doc.close()
        logging.info(f"成功解析PDF: {pdf_path}, 共提取 {len(sections)} 个主要章节。")
        return {"metadata": metadata, "sections": sections}