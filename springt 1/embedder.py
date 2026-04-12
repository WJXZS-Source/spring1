# embedder.py
import logging

# 导入成员B提供的假想入库接口 (实际开发中替换为真正的 import)
class MemberB_VectorDB:
    @staticmethod
    def add_documents(documents):
        """
        模拟成员B的入库接口
        documents: List[Dict], keys: 'text', 'vector', 'metadata'
        """
        logging.info(f"[成员B接口] 成功接收并入库 {len(documents)} 条记录！")
        # print(f"Sample data: {documents[0]['metadata']} | Vector dim: {len(documents[0]['vector'])}")

class VectorProcessor:
    def __init__(self, model_type="local"):
        """
        初始化 Embedding 模型
        :param model_type: "local" (all-MiniLM-L6-v2) 或 "openai" (text-embedding-3-small)
        """
        self.model_type = model_type
        if self.model_type == "local":
            from sentence_transformers import SentenceTransformer
            # 使用本地轻量级模型
            logging.info("加载本地模型 all-MiniLM-L6-v2...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        elif self.model_type == "openai":
            from langchain_openai import OpenAIEmbeddings
            logging.info("初始化 OpenAI Embedding...")
            self.model = OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            raise ValueError("不支持的模型类型")

    def embed_and_store(self, chunks):
        """
        批量向量化并调用成员B的入库接口
        :param chunks: text_chunker.py 输出的块列表
        """
        if not chunks:
            logging.warning("没有需要向量化的块。")
            return

        texts = [chunk['text'] for chunk in chunks]
        
        logging.info(f"正在向量化 {len(texts)} 个块...")
        if self.model_type == "local":
            vectors = self.model.encode(texts, show_progress_bar=True).tolist()
        else:
            vectors = self.model.embed_documents(texts)

        # 构造最终的数据结构
        documents_to_insert = []
        for i, chunk in enumerate(chunks):
            doc = {
                "text": chunk['text'],
                "vector": vectors[i],
                "metadata": chunk['metadata']
            }
            documents_to_insert.append(doc)

        # 调用成员B的入库接口
        MemberB_VectorDB.add_documents(documents_to_insert)


# ==========================================
# 核心流水线：集成 C 成员的所有工作
# ==========================================
def process_pdfs_pipeline(pdf_paths, external_metadata_list=None):
    """
    成员C的主干调度函数
    :param pdf_paths: 从成员A/文件系统获取的PDF路径列表
    """
    parser = AcademicPDFParser()
    chunker = SemanticChunker(chunk_size=1024, chunk_overlap=102)
    # 默认使用本地模型方便测试，生产环境可改为 "openai"
    embedder = VectorProcessor(model_type="local") 

    for idx, pdf_path in enumerate(pdf_paths):
        logging.info(f"--- 开始处理: {pdf_path} ---")
        
        meta = external_metadata_list[idx] if external_metadata_list else None
        
        # 1. 解析与清洗
        parsed_data = parser.parse(pdf_path, external_metadata=meta)
        if not parsed_data:
            continue
            
        # 2. 分块
        chunks = chunker.chunk_documents(parsed_data)
        if not chunks:
            continue
            
        # 3. 向量化并入库 (直接对接成员B)
        embedder.embed_and_store(chunks)

# 测试代码
if __name__ == "__main__":
    # 假设这里是成员A下载好的文件列表
    sample_pdfs = ["sample_paper_1.pdf", "scanned_paper_2.pdf"] 
    
    # 为了测试跑通，你可以随便找个科研PDF重命名为 sample_paper_1.pdf 放在同级目录
    # process_pdfs_pipeline(sample_pdfs)