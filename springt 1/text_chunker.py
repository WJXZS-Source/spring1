import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SemanticChunker:
    def __init__(self, chunk_size=1024, chunk_overlap=102):
        """
        采用 RecursiveCharacterTextSplitter 进行高级语义分块
        1024 字符（约 512 token），保留 10% overlap
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""]
        )

    def chunk_documents(self, parsed_data: dict) -> list:
        """
        分块并严格按照 Sprint 2 全局规范构造 Metadata
        """
        if not parsed_data:
            return []

        doc_metadata = parsed_data.get('metadata', {})
        sections = parsed_data.get('sections', [])
        
        all_chunks = []

        for sec in sections:
            section_name = sec.get('section_name', 'Unknown')
            content = sec.get('content', '')
            
            if not content.strip():
                continue
                
            text_chunks = self.splitter.split_text(content)
            
            for chunk_text in text_chunks:
                # 构造符合 Sprint 2 规范的 Metadata
                chunk_meta = {
                    "paper_title": doc_metadata.get("title", ""),    
                    "authors": doc_metadata.get("authors", ""),      
                    "year": int(doc_metadata.get("year", 0)),
                    "arxiv_id": doc_metadata.get("arxiv_id", ""),
                    "section": section_name                          
                }
                
                all_chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_meta
                })
                
        logging.info(f"为文档《{doc_metadata.get('title')}》生成了 {len(all_chunks)} 个结构化 Chunk。")
        return all_chunks
