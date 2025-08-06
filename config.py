import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    # LLM 配置
    LLM_MODEL = os.getenv('LLM_MODEL', 'qwen3:4b')
    LLM_LITE_MODEL = os.getenv('LLM_LITE_MODEL', 'qwen3:1.7b')
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0'))
    
    # 数据库配置
    DB_PATH = os.getenv('DB_PATH', '/home/zjq/document/langchain_learn/diet.db')
    
    # 知识库配置
    KNOWLEDGE_BASE_PATH = os.getenv('KNOWLEDGE_BASE_PATH', '/home/zjq/document/langchain_learn/rag_knowledge_base')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'nn200433/text2vec-bge-large-chinese:latest')
    
    # 向量检索配置
    VECTOR_SEARCH_K = int(os.getenv('VECTOR_SEARCH_K', '2'))
    
    # 对话配置
    MAX_MESSAGES = int(os.getenv('MAX_MESSAGES', '20'))
    
    # 缓存配置
    ENABLE_CACHE = os.getenv('ENABLE_CACHE', 'true').lower() == 'true'
    CACHE_SIZE = int(os.getenv('CACHE_SIZE', '100'))
    CACHE_TYPE = os.getenv('CACHE_TYPE', 'sqlite')
    
    # 日志配置
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'app.log')

config = Config()