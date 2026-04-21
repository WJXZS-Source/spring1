import logging
from pdf_parser import AcademicPDFParser
from text_chunker import SemanticChunker

# ==========================================
# 模拟成员B的入库接口 (供联调使用，后续由成员B真实实现替换)
# ==========================================
class MemberB_VectorDB:
    @staticmethod
    def add_documents(documents):
        """
        接收符合全局规范的 Chunk 列表并入库
        """
        logging.info(f"🟢 [调用成员B接口] 成功接收并入库 {len(documents)} 个 Chunk！")

# ==========================================
# 向量化处理器
# ==========================================
class VectorProcessor:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        logging.info("加载本地 Embedding 模型: all-MiniLM-L6-v2 (384维)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_and_store(self, chunks):
        if not chunks:
            return

        texts = [chunk['text'] for chunk in chunks]
        logging.info(f"正在向量化 {len(texts)} 个文本块...")
        
        # 生成向量
        embeddings = self.model.encode(texts, show_progress_bar=False).tolist()

        # 构造符合 Sprint 2 全局接口规范的数据结构
        documents_to_insert = []
        for i, chunk in enumerate(chunks):
            doc = {
                "text": chunk['text'],
                "embedding": embeddings[i],
                "metadata": chunk['metadata']
            }
            documents_to_insert.append(doc)

        # 传递给成员B
        MemberB_VectorDB.add_documents(documents_to_insert)

# ==========================================
# 成员D 对外暴露的核心主干函数
# ==========================================
def parse_and_embed(pdf_info_list: list):
    """
    成员D的入口函数，由成员A在图中调用。
    :param pdf_info_list: 成员C返回的列表 
           [{"arxiv_id": str, "local_path": str, "title": str, "authors": str, "year": int}]
    """
    parser = AcademicPDFParser()
    chunker = SemanticChunker(chunk_size=1024, chunk_overlap=102)
    embedder = VectorProcessor() 

    for pdf_info in pdf_info_list:
        local_path = pdf_info.get("local_path")
        logging.info(f"\n🚀 --- 开始处理论文: {pdf_info.get('title')} ---")
        
        # 1. 解析与清洗
        parsed_data = parser.parse(local_path, external_metadata=pdf_info)
        if not parsed_data or not parsed_data.get('sections'):
            continue  # 异常拦截已在 parse() 中打印日志
            
        # 2. 语义分块
        chunks = chunker.chunk_documents(parsed_data)
        if not chunks:
            logging.warning(f"⚠️ [跳过] 未能生成任何有效的文本块: {local_path}")
            continue
            
        # 3. 向量化并入库 (对接成员B)
        embedder.embed_and_store(chunks)

    logging.info("\n🎉 === 所有的 PDF 处理与入库流程执行完毕 ===")
