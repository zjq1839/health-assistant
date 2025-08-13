"""增强的 RAG 服务 - 支持多种文档格式和 LlamaIndex 集成

新功能：
1. 支持 PDF、TXT、MD、JSON 等多种格式
2. 可选的 LlamaIndex 集成
3. 智能文档解析和预处理
4. 批量导入知识库功能
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import shutil

# LangChain imports
from langchain_community.document_loaders import (
    TextLoader, DirectoryLoader, PyPDFLoader, 
    JSONLoader, UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 尝试导入 LlamaIndex (可选)
try:
    from llama_index.core import (
        VectorStoreIndex, Document as LlamaDocument, 
        StorageContext, load_index_from_storage
    )
    from llama_index.core import Settings
    from llama_index.vector_stores.faiss import FaissVectorStore
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.readers.file import PyMuPDFReader
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False

from .rag_service import RAGService, VectorKnowledgeRetriever, HybridKnowledgeRetriever
from config import config as cfg
from utils.logger import logger


class EnhancedDocumentLoader:
    """增强的文档加载器 - 支持多种格式"""
    
    def __init__(self):
        self.supported_extensions = {
            '.txt': self._load_text,
            '.pdf': self._load_pdf,
            '.md': self._load_markdown,
            '.json': self._load_json
        }
    
    def load_documents(self, directory_path: str) -> List[Document]:
        """从目录加载所有支持的文档"""
        documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.warning(f"Directory not found: {directory_path}")
            return documents
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    docs = self._load_file(file_path)
                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents from {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
        
        return documents
    
    def _load_file(self, file_path: Path) -> List[Document]:
        """加载单个文件"""
        extension = file_path.suffix.lower()
        loader_func = self.supported_extensions.get(extension)
        
        if loader_func:
            return loader_func(file_path)
        else:
            logger.warning(f"Unsupported file type: {extension}")
            return []
    
    def _load_text(self, file_path: Path) -> List[Document]:
        """加载文本文件"""
        try:
            loader = TextLoader(str(file_path), encoding='utf-8')
            return loader.load()
        except UnicodeDecodeError:
            # 尝试其他编码
            for encoding in ['gbk', 'gb2312', 'latin1']:
                try:
                    loader = TextLoader(str(file_path), encoding=encoding)
                    docs = loader.load()
                    logger.info(f"Loaded {file_path.name} with {encoding} encoding")
                    return docs
                except UnicodeDecodeError:
                    continue
            
            logger.error(f"Failed to decode {file_path} with any encoding")
            return []
    
    def _load_pdf(self, file_path: Path) -> List[Document]:
        """加载 PDF 文件"""
        try:
            loader = PyPDFLoader(str(file_path))
            return loader.load()
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")
            return []
    
    def _load_markdown(self, file_path: Path) -> List[Document]:
        """加载 Markdown 文件"""
        try:
            loader = UnstructuredMarkdownLoader(str(file_path))
            return loader.load()
        except Exception as e:
            logger.error(f"Failed to load Markdown {file_path}: {e}")
            return []
    
    def _load_json(self, file_path: Path) -> List[Document]:
        """加载 JSON 文件"""
        try:
            loader = JSONLoader(
                file_path=str(file_path),
                jq_schema='.[]',
                text_content=False
            )
            return loader.load()
        except Exception as e:
            logger.error(f"Failed to load JSON {file_path}: {e}")
            return []


class LlamaIndexRAGService:
    """基于 LlamaIndex 的 RAG 服务"""
    
    def __init__(self, knowledge_base_path: str, embedding_model: str):
        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError("LlamaIndex not available. Please install: pip install llama-index")
        
        self.knowledge_base_path = Path(knowledge_base_path)
        self.storage_path = self.knowledge_base_path / "llama_index_storage"
        self.embedding_model = embedding_model
        
        # 初始化嵌入模型
        self.embed_model = OllamaEmbedding(
            model_name=embedding_model,
            base_url="http://localhost:11434"
        )
        
        # 配置 LlamaIndex 全局设置：关闭 LLM，仅使用本地嵌入
        Settings.llm = None
        Settings.embed_model = self.embed_model
        
        # 加载或创建索引
        self.index = self._load_or_create_index()
    
    def _load_or_create_index(self):
        """加载或创建索引"""
        if self.storage_path.exists():
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(self.storage_path)
                )
                index = load_index_from_storage(storage_context)
                logger.info("Loaded existing LlamaIndex")
                return index
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
        
        # 创建新的空索引
        return VectorStoreIndex([], embed_model=self.embed_model)
    
    def load_documents_from_directory(self, directory_path: str):
        """从目录加载文档到 LlamaIndex"""
        loader = EnhancedDocumentLoader()
        langchain_docs = loader.load_documents(directory_path)
        
        # 转换为 LlamaIndex 文档格式
        llama_docs = []
        for doc in langchain_docs:
            llama_doc = LlamaDocument(
                text=doc.page_content,
                metadata=doc.metadata
            )
            llama_docs.append(llama_doc)
        
        if llama_docs:
            # 添加文档到索引
            self.index = VectorStoreIndex.from_documents(
                llama_docs, 
                embed_model=self.embed_model
            )
            
            # 持久化索引
            self.index.storage_context.persist(persist_dir=str(self.storage_path))
            
            logger.info(f"Added {len(llama_docs)} documents to LlamaIndex")
    
    def query(self, query: str, top_k: int = 5) -> str:
        """查询索引"""
        query_engine = self.index.as_query_engine(similarity_top_k=top_k, llm=None)
        response = query_engine.query(query)
        return str(response)


class EnhancedRAGService(RAGService):
    """增强的 RAG 服务 - 支持多种文档格式"""
    
    def __init__(self, knowledge_retriever, llm_service):
        super().__init__(knowledge_retriever, llm_service)
        self.document_loader = EnhancedDocumentLoader()
        # Initialize text_splitter from parent class
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
    
    def load_knowledge_base(self, knowledge_path: str) -> None:
        """加载知识库 - 支持多种格式"""
        knowledge_dir = Path(knowledge_path)
        if not knowledge_dir.exists():
            logger.warning(f"Knowledge directory not found: {knowledge_path}")
            return
        
        try:
            # 使用增强的文档加载器
            documents = self.document_loader.load_documents(str(knowledge_dir))
            
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
    
    def import_from_directory(self, source_dir: str, target_dir: str = None) -> bool:
        """从源目录导入文档到知识库"""
        if target_dir is None:
            target_dir = cfg.knowledge_base.path
        
        source_path = Path(source_dir)
        target_path = Path(target_dir)
        
        if not source_path.exists():
            logger.error(f"Source directory not found: {source_dir}")
            return False
        
        # 确保目标目录存在
        target_path.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        total_count = 0
        
        # 复制文件到目标目录
        for file_path in source_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.pdf', '.md', '.json']:
                total_count += 1
                try:
                    target_file = target_path / file_path.name
                    
                    # 避免覆盖同名文件
                    counter = 1
                    while target_file.exists():
                        stem = file_path.stem
                        suffix = file_path.suffix
                        target_file = target_path / f"{stem}_{counter}{suffix}"
                        counter += 1
                    
                    shutil.copy2(file_path, target_file)
                    success_count += 1
                    logger.info(f"Copied {file_path.name} -> {target_file.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to copy {file_path}: {e}")
        
        if success_count > 0:
            # 重新加载知识库
            self.load_knowledge_base(target_dir)
            logger.info(f"Successfully imported {success_count}/{total_count} files")
            return True
        else:
            logger.warning("No files were imported")
            return False


def create_enhanced_rag_service(use_llama_index: bool = False) -> Union[EnhancedRAGService, LlamaIndexRAGService]:
    """创建增强的 RAG 服务"""
    
    if use_llama_index and LLAMA_INDEX_AVAILABLE:
        # 使用 LlamaIndex
        return LlamaIndexRAGService(
            knowledge_base_path=cfg.knowledge_base.path,
            embedding_model=cfg.knowledge_base.embedding_model
        )
    else:
        # 使用增强的 LangChain RAG
        from core.service_container import get_container, LLMService as _LLMInterface
        
        # 选择检索器类型
        retriever = HybridKnowledgeRetriever(
            embedding_model=cfg.knowledge_base.embedding_model,
            knowledge_base_path=cfg.knowledge_base.path
        )
        
        container = get_container()
        llm_service = container.get(_LLMInterface)
        
        return EnhancedRAGService(retriever, llm_service)


def import_knowledge_from_directory(source_dir: str, use_llama_index: bool = False) -> bool:
    """便捷函数：从目录导入知识库"""
    try:
        rag_service = create_enhanced_rag_service(use_llama_index=use_llama_index)
        
        if isinstance(rag_service, LlamaIndexRAGService):
            # LlamaIndex 方式
            rag_service.load_documents_from_directory(source_dir)
            return True
        else:
            # LangChain 方式
            return rag_service.import_from_directory(source_dir)
            
    except Exception as e:
        logger.error(f"Failed to import knowledge from {source_dir}: {e}")
        return False


# 使用示例
if __name__ == "__main__":
    # 测试文档加载
    loader = EnhancedDocumentLoader()
    docs = loader.load_documents("/home/zjq/document/langchain_learn/知识库")
    print(f"Loaded {len(docs)} documents")
    
    # 测试导入功能
    success = import_knowledge_from_directory(
        "/home/zjq/document/langchain_learn/知识库",
        use_llama_index=False  # 使用 LangChain 方式
    )
    print(f"Import success: {success}")