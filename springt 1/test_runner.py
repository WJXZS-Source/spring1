# test_runner.py
import json
from pdf_parser import AcademicPDFParser
from text_chunker import SemanticChunker
from embedder import VectorProcessor

def test_pipeline(pdf_file_path):
    print(f"========== 开始测试 PDF: {pdf_file_path} ==========\n")
    
    # ---------------------------------------------------------
    # 阶段 1: 测试 PDF 解析与清洗
    # ---------------------------------------------------------
    print(">>> [阶段1] 执行 PDF 解析...")
    parser = AcademicPDFParser(header_margin=50, footer_margin=50)
    # 模拟成员A传来的外部元数据
    mock_external_metadata = {
        "title": "Attention Is All You Need",
        "author": "Vaswani et al.",
        "year": "2017",
        "url": "https://arxiv.org/abs/1706.03762"
    }
    
    parsed_data = parser.parse(pdf_file_path, external_metadata=mock_external_metadata)
    
    if not parsed_data:
        print("❌ 解析失败或跳过（如：扫描版 PDF）。测试提前结束。")
        return
        
    print(f"✅ 解析成功！")
    print(f"提取到的元数据: {json.dumps(parsed_data['metadata'], ensure_ascii=False, indent=2)}")
    
    sections = parsed_data['sections']
    print(f"成功提取 {len(sections)} 个章节:")
    for i, sec in enumerate(sections):
        title = sec['section_name']
        content_len = len(sec['content'])
        print(f"  [{i+1}] 章节名: {title:<25} | 字符数: {content_len}")
    print("\n")

    # ---------------------------------------------------------
    # 阶段 2: 测试语义分块
    # ---------------------------------------------------------
    print(">>> [阶段2] 执行语义分块 (Chunking)...")
    chunker = SemanticChunker(chunk_size=1024, chunk_overlap=102)
    chunks = chunker.chunk_documents(parsed_data)
    
    print(f"✅ 分块完成！共生成 {len(chunks)} 个 Chunk。")
    if chunks:
        print("展示第一个 Chunk 的详情:")
        print(f"  - Metadata: {chunks[0]['metadata']}")
        print(f"  - Text (前150字): {chunks[0]['text'][:150]}...\n")
        
        print("展示最后一个 Chunk 的详情:")
        print(f"  - Metadata: {chunks[-1]['metadata']}")
        print(f"  - Text (前150字): {chunks[-1]['text'][:150]}...\n")
        
    # ---------------------------------------------------------
    # 阶段 3: 测试向量化 (Embedding)
    # ---------------------------------------------------------
    print(">>> [阶段3] 执行向量化并模拟入库...")
    # 使用本地模型测试（不需要消耗 OpenAI API 额度）
    embedder = VectorProcessor(model_type="local")
    
    # 为了防止本地测试过慢，我们只取前 3 个 Chunk 测试向量化
    test_chunks = chunks[:3]
    print(f"选取前 {len(test_chunks)} 个 Chunk 进行向量化测试...")
    
    # 临时覆盖 embedder 中调用成员B的方法，改为打印输出
    from embedder import MemberB_VectorDB
    def mock_add_documents(documents):
        print(f"✅ [Mock成员B接口] 成功接收 {len(documents)} 条向量数据！")
        print(f"检查第一条数据结构:")
        print(f"  - Text: 存在 ({len(documents[0]['text'])} 字符)")
        print(f"  - Metadata: {documents[0]['metadata']}")
        print(f"  - Vector 维度: {len(documents[0]['vector'])} 维 (正常应为 384 或 1536)")
        print(f"  - Vector 预览: {documents[0]['vector'][:5]} ...")
        
    MemberB_VectorDB.add_documents = mock_add_documents
    
    embedder.embed_and_store(test_chunks)
    print("\n========== 测试流程结束 ==========\n")

if __name__ == "__main__":
    # 请确保同级目录下有一个名为 sample.pdf 的学术论文
    # 你可以随便找一篇英文或中文的学术论文重命名为 sample.pdf
    test_pipeline("sample.pdf")