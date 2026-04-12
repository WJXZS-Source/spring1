# text_chunker.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

class SemanticChunker:
    def __init__(self, chunk_size=1024, chunk_overlap=102):
        """
        初始化分块器
        中文字符1024约等于512 token。Overlap 10% 为 102。
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 递归字符分割器：优先按段落切分，然后按句子，最后按字符，最大程度保证语义不断裂
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""]
        )

    def chunk_documents(self, parsed_data):
        """
        将解析后的文档结构化并分块
        :param parsed_data: pdf_parser.py 的输出字典
        :return: list of dict, 每个 dict 包含 text 和 metadata
        """
        if not parsed_data:
            return []

        doc_metadata = parsed_data['metadata']
        sections = parsed_data['sections']
        
        all_chunks = []

        for sec in sections:
            section_name = sec['section_name']
            content = sec['content']
            
            # 对当前章节内容进行分块
            text_chunks = self.splitter.split_text(content)
            
            for i, chunk_text in enumerate(text_chunks):
                # 合并文档级元数据与块级元数据（所在章节）
                chunk_meta = doc_metadata.copy()
                chunk_meta.update({
                    "section": section_name,
                    "chunk_index": i
                })
                
                all_chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_meta
                })
                
        logging.info(f"为文档《{doc_metadata.get('title')}》生成了 {len(all_chunks)} 个Text Chunk。")
        return all_chunks