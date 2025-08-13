"""RAG 服务 - 检索增强生成

支持：
1. 知识库构建与索引
2. 语义检索
3. 上下文增强
4. Agentic RAG工作流
"""

import os
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

from config import config as cfg
from utils.logger import logger


@dataclass
class RetrievalResult:
    """检索结果"""
    documents: List[Document]
    scores: List[float]
    query: str
    retrieval_method: str
    metadata: Dict[str, Any]


@dataclass
class RAGContext:
    """RAG上下文"""
    query: str
    retrieved_docs: List[Document]
    user_profile: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    domain_context: str  # nutrition, exercise, health_advice


class KnowledgeRetriever(ABC):
    """知识检索器抽象接口"""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """检索相关文档"""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """添加文档到知识库"""
        pass


class VectorKnowledgeRetriever(KnowledgeRetriever):
    """基于向量的知识检索器"""
    
    def __init__(self, embedding_model: str, knowledge_base_path: str):
        self.embedding_model = embedding_model
        self.knowledge_base_path = Path(knowledge_base_path)
        self.vector_store_path = self.knowledge_base_path / "vector_store"
        
        # 初始化嵌入模型
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        
        # 加载或创建向量存储
        self.vector_store = self._load_or_create_vector_store()
        
    def _load_or_create_vector_store(self) -> FAISS:
        """加载或创建向量存储"""
        if self.vector_store_path.exists():
            try:
                return FAISS.load_local(
                    str(self.vector_store_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                logger.warning(f"Failed to load existing vector store: {e}")
        
        # 创建空的向量存储
        dummy_texts = ["初始化文档"]
        dummy_metadatas = [{"source": "init", "type": "dummy"}]
        vector_store = FAISS.from_texts(
            dummy_texts, 
            self.embeddings, 
            metadatas=dummy_metadatas
        )
        return vector_store
    
    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """检索相关文档"""
        try:
            # 使用相似性搜索
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            
            documents = [doc for doc, score in docs_with_scores]
            scores = [float(score) for doc, score in docs_with_scores]
            
            return RetrievalResult(
                documents=documents,
                scores=scores,
                query=query,
                retrieval_method="vector_similarity",
                metadata={
                    "embedding_model": self.embedding_model,
                    "total_docs": self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return RetrievalResult(
                documents=[],
                scores=[],
                query=query,
                retrieval_method="vector_similarity",
                metadata={"error": str(e)}
            )
    
    def add_documents(self, documents: List[Document]) -> None:
        """添加文档到向量库"""
        try:
            if not documents:
                return
                
            # 添加文档到向量存储
            self.vector_store.add_documents(documents)
            
            # 保存向量存储
            self.vector_store.save_local(str(self.vector_store_path))
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")


class HybridKnowledgeRetriever(KnowledgeRetriever):
    """混合检索器 - 结合向量检索和BM25"""
    
    def __init__(self, embedding_model: str, knowledge_base_path: str):
        self.knowledge_base_path = Path(knowledge_base_path)
        
        # 向量检索器
        self.vector_retriever = VectorKnowledgeRetriever(embedding_model, knowledge_base_path)
        
        # BM25检索器存储路径
        self.bm25_path = self.knowledge_base_path / "bm25_retriever.pkl"
        
        # 加载或创建BM25检索器
        self.bm25_retriever = self._load_or_create_bm25()
        
    def _load_or_create_bm25(self) -> Optional[BM25Retriever]:
        """加载或创建BM25检索器"""
        if self.bm25_path.exists():
            try:
                with open(self.bm25_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load BM25 retriever: {e}")
        
        return None
    
    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """混合检索"""
        try:
            # 向量检索
            vector_result = self.vector_retriever.retrieve(query, k=k)
            
            # BM25检索（如果可用）
            bm25_docs = []
            if self.bm25_retriever:
                try:
                    bm25_docs = self.bm25_retriever.get_relevant_documents(query)[:k]
                except Exception as e:
                    logger.warning(f"BM25 retrieval failed: {e}")
            
            # 合并结果并去重
            all_docs = vector_result.documents + bm25_docs
            unique_docs = []
            seen_content = set()
            
            for doc in all_docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(content_hash)
            
            # 限制返回数量
            final_docs = unique_docs[:k]
            
            return RetrievalResult(
                documents=final_docs,
                scores=vector_result.scores[:len(final_docs)],
                query=query,
                retrieval_method="hybrid_vector_bm25",
                metadata={
                    "vector_docs": len(vector_result.documents),
                    "bm25_docs": len(bm25_docs),
                    "final_docs": len(final_docs)
                }
            )
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return self.vector_retriever.retrieve(query, k)
    
    def add_documents(self, documents: List[Document]) -> None:
        """添加文档到混合检索器"""
        # 添加到向量检索器
        self.vector_retriever.add_documents(documents)
        
        # 重建BM25检索器
        try:
            if documents:
                self.bm25_retriever = BM25Retriever.from_documents(documents)
                # 保存BM25检索器
                with open(self.bm25_path, 'wb') as f:
                    pickle.dump(self.bm25_retriever, f)
                logger.info(f"Updated BM25 retriever with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to update BM25 retriever: {e}")


class RAGService:
    """RAG服务 - 检索增强生成"""
    
    def __init__(self, 
                 knowledge_retriever: KnowledgeRetriever,
                 llm_service: Any):
        self.retriever = knowledge_retriever
        self.llm_service = llm_service
        
        # 文本分块器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        
    def load_knowledge_base(self, knowledge_path: str) -> None:
        """加载知识库"""
        knowledge_dir = Path(knowledge_path)
        if not knowledge_dir.exists():
            logger.warning(f"Knowledge directory not found: {knowledge_path}")
            return
        
        try:
            # 加载文档
            loader = DirectoryLoader(
                str(knowledge_dir),
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            documents = loader.load()
            
            if not documents:
                logger.warning("No documents found in knowledge directory")
                return
            
            # 分块处理
            chunks = self.text_splitter.split_documents(documents)
            
            # 添加到检索器
            self.retriever.add_documents(chunks)
            
            logger.info(f"Loaded {len(documents)} documents, {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
    
    def retrieve_context(self, query: str, 
                        user_profile: Dict[str, Any] = None,
                        conversation_history: List[Dict[str, str]] = None,
                        domain_context: str = "general",
                        k: int = 5) -> RAGContext:
        """检索相关上下文"""
        
        print("\n" + "="*80)
        print("🔍 RAG 检索过程开始")
        print("="*80)
        
        # 增强查询
        enhanced_query = self._enhance_query(
            query, user_profile, conversation_history, domain_context
        )
        
        print(f"📝 原始查询: {query}")
        print(f"🔧 增强查询: {enhanced_query}")
        print(f"🎯 领域上下文: {domain_context}")
        print(f"👤 用户画像: {user_profile}")
        
        # 检索文档
        print(f"\n🔎 开始检索相关文档 (k={k})...")
        retrieval_result = self.retriever.retrieve(enhanced_query, k=k)
        
        print(f"✅ 检索完成，找到 {len(retrieval_result.documents)} 个相关文档")
        print(f"🔧 检索方法: {retrieval_result.retrieval_method}")
        print(f"📊 检索元数据: {retrieval_result.metadata}")
        
        # 打印检索到的文档段落
        print("\n📖 检索到的文档段落:")
        print("-"*60)
        for i, doc in enumerate(retrieval_result.documents):
            score = retrieval_result.scores[i] if i < len(retrieval_result.scores) else "N/A"
            print(f"📄 文档 {i+1} (相似度分数: {score})")
            print(f"📂 来源: {doc.metadata.get('source', '未知')}")
            print(f"📝 内容预览: {doc.page_content[:200]}...")
            if len(doc.page_content) > 200:
                print(f"   (完整内容长度: {len(doc.page_content)} 字符)")
            print("-"*60)
        
        context = RAGContext(
            query=enhanced_query,
            retrieved_docs=retrieval_result.documents,
            user_profile=user_profile or {},
            conversation_history=conversation_history or [],
            domain_context=domain_context
        )
        
        print("🎯 RAG 上下文构建完成")
        print("="*80)
        
        return context
    
    def generate_with_context(self, context: RAGContext) -> str:
        """基于上下文生成回答"""
        
        print("\n" + "="*80)
        print("🤖 RAG 生成过程开始")
        print("="*80)
        
        # 构建增强提示词
        print("🔧 正在构建增强提示词...")
        enhanced_prompt = self._build_enhanced_prompt(context)
        
        print(f"📊 使用的文档数量: {len(context.retrieved_docs)}")
        print(f"🎯 目标领域: {context.domain_context}")
        
        # 显示构建的提示词（部分内容）
        print("\n📝 构建的提示词预览:")
        print("-"*60)
        prompt_preview = enhanced_prompt[:500] + "..." if len(enhanced_prompt) > 500 else enhanced_prompt
        print(prompt_preview)
        print(f"(完整提示词长度: {len(enhanced_prompt)} 字符)")
        print("-"*60)
        
        print("\n🧠 模型思考过程:")
        print("💭 正在分析检索到的知识...")
        print("💭 结合用户画像进行个性化...")
        print("💭 整合多个信息源...")
        print("💭 生成专业建议...")
        
        try:
            # 使用LLM生成回答
            print("\n⚙️ 正在调用LLM生成回答...")
            response = self.llm_service.generate_response(enhanced_prompt, "")
            
            print(f"✅ 生成完成，回答长度: {len(response)} 字符")
            
            # 记录RAG使用情况
            logger.info(
                "RAG generation completed",
                extra={
                    "query": context.query,
                    "retrieved_docs": len(context.retrieved_docs),
                    "domain": context.domain_context,
                    "response_length": len(response)
                }
            )
            
            print("🎯 RAG 生成过程完成")
            print("="*80)
            
            return response
            
        except Exception as e:
            error_msg = f"抱歉，生成回答时遇到问题：{str(e)}"
            print(f"❌ 生成失败: {error_msg}")
            print("="*80)
            logger.error(f"RAG generation failed: {e}")
            return error_msg
    
    def _enhance_query(self, query: str,
                      user_profile: Dict[str, Any],
                      conversation_history: List[Dict[str, str]],
                      domain_context: str) -> str:
        """增强查询"""
        enhanced_parts = [query]
        
        # 添加领域上下文
        if domain_context != "general":
            enhanced_parts.append(f"领域：{domain_context}")
        
        # 添加用户画像信息
        if user_profile:
            profile_info = []
            if user_profile.get('age'):
                profile_info.append(f"年龄{user_profile['age']}岁")
            if user_profile.get('gender'):
                profile_info.append(f"{user_profile['gender']}")
            if user_profile.get('health_goal'):
                profile_info.append(f"健康目标：{user_profile['health_goal']}")
            
            if profile_info:
                enhanced_parts.append("用户特征：" + " ".join(profile_info))
        
        # 添加对话历史中的关键信息
        if conversation_history:
            recent_topics = []
            for msg in conversation_history[-3:]:
                content = msg.get('content', '')
                if any(keyword in content for keyword in ['饮食', '运动', '健康', '减肥', '增重']):
                    recent_topics.append(content[:50])
            
            if recent_topics:
                enhanced_parts.append("最近讨论：" + " ".join(recent_topics))
        
        return " ".join(enhanced_parts)
    
    def _build_enhanced_prompt(self, context: RAGContext) -> str:
        """构建增强提示词"""
        
        # 整理检索到的知识
        knowledge_sections = []
        for i, doc in enumerate(context.retrieved_docs[:3]):  # 只使用前3个最相关的文档
            knowledge_sections.append(f"参考资料{i+1}：\n{doc.page_content}")
        
        knowledge_context = "\n\n".join(knowledge_sections) if knowledge_sections else "暂无相关参考资料"
        
        # 用户画像信息
        profile_info = ""
        if context.user_profile:
            profile_parts = []
            for key, value in context.user_profile.items():
                if value:
                    profile_parts.append(f"{key}: {value}")
            if profile_parts:
                profile_info = f"\n用户信息：{', '.join(profile_parts)}"
        
        # 构建完整提示词
        prompt = f"""你是一名专业的健康顾问。请基于以下信息为用户提供准确、实用的建议。

用户问题：{context.query}
{profile_info}

相关知识：
{knowledge_context}

请注意：
0. 你的建议必须指出信息来源
1. 优先使用提供的参考资料中的信息
2. 结合用户的个人情况给出个性化建议
3. 如果参考资料不足，请说明


请用简体中文回答："""

        return prompt
    
    def add_knowledge_document(self, content: str, metadata: Dict[str, Any]) -> None:
        """添加单个知识文档"""
        doc = Document(page_content=content, metadata=metadata)
        chunks = self.text_splitter.split_documents([doc])
        self.retriever.add_documents(chunks)
        
        logger.info(f"Added knowledge document: {metadata.get('title', 'Unknown')}")


class AgenticRAGService(RAGService):
    """智能体RAG服务 - 支持多步骤推理和决策"""
    
    def __init__(self, knowledge_retriever: KnowledgeRetriever, llm_service: Any):
        super().__init__(knowledge_retriever, llm_service)
        
    def multi_step_retrieve(self, query: str, 
                          user_profile: Dict[str, Any] = None,
                          max_iterations: int = 3) -> List[RAGContext]:
        """多步骤检索 - 根据初始结果决定是否需要进一步检索"""
        
        contexts = []
        current_query = query
        
        for iteration in range(max_iterations):
            # 检索当前查询
            context = self.retrieve_context(
                current_query, 
                user_profile=user_profile,
                domain_context=self._detect_domain(current_query)
            )
            contexts.append(context)
            
            # 如果检索到足够的相关文档，停止
            if len(context.retrieved_docs) >= 3:
                break
                
            # 分析当前结果，生成更精确的查询
            refined_query = self._refine_query_based_on_results(current_query, context)
            if refined_query == current_query:  # 没有改进，停止
                break
                
            current_query = refined_query
            logger.info(f"Refined query for iteration {iteration + 1}: {refined_query}")
        
        return contexts
    
    def _detect_domain(self, query: str) -> str:
        """检测查询所属领域"""
        nutrition_keywords = ['饮食', '营养', '食物', '卡路里', '蛋白质', '维生素']
        exercise_keywords = ['运动', '锻炼', '健身', '跑步', '力量训练']
        health_keywords = ['健康', '疾病', '症状', '预防', '治疗']
        
        if any(keyword in query for keyword in nutrition_keywords):
            return "nutrition"
        elif any(keyword in query for keyword in exercise_keywords):
            return "exercise"
        elif any(keyword in query for keyword in health_keywords):
            return "health_advice"
        else:
            return "general"
    
    def _refine_query_based_on_results(self, original_query: str, context: RAGContext) -> str:
        """基于检索结果优化查询"""
        if not context.retrieved_docs:
            # 如果没有检索到文档，尝试使用更一般的术语
            return self._generalize_query(original_query)
        
        # 如果检索到的文档质量不高，尝试使用更具体的术语
        if all(len(doc.page_content) < 100 for doc in context.retrieved_docs):
            return self._specialize_query(original_query)
        
        return original_query
    
    def _generalize_query(self, query: str) -> str:
        """泛化查询"""
        generalizations = {
            '减肥方法': '健康减重',
            '增肌训练': '力量训练',
            '营养搭配': '饮食指导'
        }
        
        for specific, general in generalizations.items():
            if specific in query:
                return query.replace(specific, general)
        
        return query
    
    def _specialize_query(self, query: str) -> str:
        """特化查询"""
        specializations = {
            '健康': '健康管理方法',
            '运动': '运动训练计划',
            '饮食': '营养饮食指导'
        }
        
        for general, specific in specializations.items():
            if general in query and specific not in query:
                return query.replace(general, specific)
        
        return query


def create_rag_service(use_hybrid: bool = True) -> RAGService:
    """创建RAG服务实例"""
    
    # 根据配置选择检索器
    if use_hybrid:
        retriever = HybridKnowledgeRetriever(
            embedding_model=cfg.knowledge_base.embedding_model,
            knowledge_base_path=cfg.knowledge_base.path
        )
    else:
        retriever = VectorKnowledgeRetriever(
            embedding_model=cfg.knowledge_base.embedding_model,
            knowledge_base_path=cfg.knowledge_base.path
        )
    
    # 创建RAG服务（这里需要注入LLM服务）
    from core.service_container import get_container, LLMService as _LLMInterface
    container = get_container()
    llm_service = container.get(_LLMInterface)
    
    return AgenticRAGService(retriever, llm_service)


# 使用示例
if __name__ == "__main__":
    # 创建RAG服务
    rag_service = create_rag_service()
    
    # 加载知识库
    rag_service.load_knowledge_base("/home/zjq/document/langchain_learn/rag_knowledge_base")
    
    # 测试检索和生成
    context = rag_service.retrieve_context(
        "如何制定健康的减肥计划？",
        user_profile={"age": 25, "gender": "女", "health_goal": "减肥"},
        domain_context="health_advice"
    )
    
    response = rag_service.generate_with_context(context)
    print(response)