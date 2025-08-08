from langchain_community.vectorstores import FAISS
import json
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# 步骤1: 自定义加载JSONL文件
def load_jsonl(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            description = data.get('description', '')
            nutrients = data.get('nutrients')
            
            content = description
            if nutrients:
                # Format the nutrients dictionary into a readable string and append it
                nutrients_str = json.dumps(nutrients, ensure_ascii=False, separators=(',', ':'))
                content += nutrients_str

            documents.append(Document(page_content=content, metadata={"source": file_path}))
    return documents

# 步骤2: 加载文档
file_path = "/home/zjq/document/langchain_learn/combined_train.jsonl"
docs = load_jsonl(file_path)

# 步骤3: 由于每个文档都是独立的JSONL行，我们不需要分割文本
split_docs = docs


import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore

embeddings = OllamaEmbeddings(model="nn200433/text2vec-bge-large-chinese:latest")
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vectorstore = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)
vectorstore.add_documents(documents=docs)
vectorstore.save_local("./rag_knowledge_base")
print("RAG知识库已创建！")