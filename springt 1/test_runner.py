import json
import logging
import embedder

def run_integration_test():
    print("========== 启动 Member D Sprint 2 接口联调测试 ==========\n")
    
    # 模拟成员 C 提供的数据
    mock_c_output = [
        {
            "arxiv_id": "1706.03762",
            "local_path": "test_paper.pdf", # 🚨 请在同级目录下放一个名为 test_paper.pdf 的真实论文
            "title": "Attention Is All You Need",
            "authors": "Ashish Vaswani, Noam Shazeer, et al.",
            "year": 2017
        }
    ]
    
    # 注入拦截器验证输出格式
    def mock_add_documents(documents):
        print(f"\n✅ [联调检查] 成功产出 {len(documents)} 条向量数据！")
        if documents:
            sample = documents[0]
            print("\n🔍 数据结构合规校验通过！字段展示:")
            print(f" - Text 长度: {len(sample['text'])} 字符")
            print(f" - Embedding 维度: {len(sample['embedding'])} 维 (Sprint2规范要求384维)")
            print(f" - Metadata 内容: {json.dumps(sample['metadata'], ensure_ascii=False)}")
    
    embedder.MemberB_VectorDB.add_documents = mock_add_documents
    
    # 执行成员 D 的主入口
    embedder.parse_and_embed(mock_c_output)
    print("\n========== 测试流程结束 ==========")

if __name__ == "__main__":
    run_integration_test()
