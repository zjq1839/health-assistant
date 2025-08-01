import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

# RAG知识库路径
KNOWLEDGE_BASE_PATH = "/home/zjq/document/langchain_learn/rag_knowledge_base"

# 加载知识库
print("正在加载RAG知识库...")
embeddings = OllamaEmbeddings(model="nn200433/text2vec-bge-large-chinese:latest")

# 加载FAISS索引
# 注意：faiss-cpu目前不支持`FAISS.load_local`中的`allow_dangerous_deserialization`参数
# 如果遇到pickle相关的安全错误，请确保你的加载环境是安全的
# 详情请见LangChain官方文档：https://python.langchain.com/docs/integrations/vectorstores/faiss
try:
    db = FAISS.load_local(KNOWLEDGE_BASE_PATH, embeddings, allow_dangerous_deserialization=True)
    print("RAG知识库加载成功！")
except Exception as e:
    print(f"加载知识库时出错: {e}")
    exit()

# 定义一个测试查询
query = "苹果的热量为"
print(f"\n正在执行相似性搜索，查询: '{query}'")

# 执行相似性搜索
try:
    # 使用similarity_search_with_score来获取相似度分数
    results_with_scores = db.similarity_search_with_score(query, k=2)

    # 检查是否有结果
    if not results_with_scores:
        print("未找到相关文档。")
    else:
        print("\n搜索结果:")
        for doc, score in results_with_scores:
            print("-" * 50)
            print(f"相似度分数: {score}")
            print(f"文档内容: \n{doc.page_content}")
            # 如果有元数据，也可以打印出来
            if doc.metadata:
                print(f"元数据: {doc.metadata}")
            print("-" * 50)

except Exception as e:
    print(f"执行相似性搜索时出错: {e}")