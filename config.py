import os
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# Define Pydantic models for structured configuration
class LLMConfig(BaseModel):
    model: str = Field(default='gpt-oss:20b', alias='LLM_MODEL')
    lite_model: str = Field(default='qwen3:4b', alias='LLM_LITE_MODEL')
    temperature: float = Field(default=0.0, alias='LLM_TEMPERATURE')

class DatabaseConfig(BaseModel):
    path: str = Field(default='/home/zjq/document/langchain_learn/diet.db', alias='DB_PATH')

class KnowledgeBaseConfig(BaseModel):
    path: str = Field(default='/home/zjq/document/langchain_learn/rag_knowledge_base', alias='KNOWLEDGE_BASE_PATH')
    embedding_model: str = Field(default='nn200433/text2vec-bge-large-chinese:latest', alias='EMBEDDING_MODEL')

class VectorSearchConfig(BaseModel):
    k: int = Field(default=2, alias='VECTOR_SEARCH_K')

class DialogueConfig(BaseModel):
    max_messages: int = Field(default=20, alias='MAX_MESSAGES')
    immediate_context_size: int = Field(default=3, alias='IMMEDIATE_CONTEXT_SIZE')
    enable_intelligent_truncation: bool = Field(default=True, alias='ENABLE_INTELLIGENT_TRUNCATION')
    enable_context_compression: bool = Field(default=False, alias='ENABLE_CONTEXT_COMPRESSION')

class CacheConfig(BaseModel):
    enable: bool = Field(default=True, alias='ENABLE_CACHE')
    size: int = Field(default=100, alias='CACHE_SIZE')
    type: str = Field(default='sqlite', alias='CACHE_TYPE')
    intent_cache_size: int = Field(default=200, alias='INTENT_CACHE_SIZE')
    intent_cache_ttl_hours: int = Field(default=24, alias='INTENT_CACHE_TTL_HOURS')

class IntentConfig(BaseModel):
    enable_enhanced_recognition: bool = Field(default=True, alias='ENABLE_ENHANCED_RECOGNITION')
    confidence_threshold: float = Field(default=0.7, alias='INTENT_CONFIDENCE_THRESHOLD')
    enable_context_enhancement: bool = Field(default=True, alias='ENABLE_CONTEXT_ENHANCEMENT')
    keyword_weight: float = Field(default=0.6, alias='KEYWORD_WEIGHT')
    llm_weight: float = Field(default=0.4, alias='LLM_WEIGHT')
    keyword_direct_return_threshold: float = Field(default=0.95, alias='KEYWORD_DIRECT_RETURN_THRESHOLD')
    llm_temperature: float = Field(default=0.1, alias='INTENT_LLM_TEMPERATURE')
    llm_timeout_seconds: int = Field(default=5, alias='INTENT_LLM_TIMEOUT_SECONDS')
    llm_max_response_length: int = Field(default=1024, alias='LLM_MAX_RESPONSE_LENGTH')

class LoggingConfig(BaseModel):
    level: str = Field(default='INFO', alias='LOG_LEVEL')
    file: str = Field(default='app.log', alias='LOG_FILE')

class AppConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    database: DatabaseConfig = DatabaseConfig()
    knowledge_base: KnowledgeBaseConfig = KnowledgeBaseConfig()
    vector_search: VectorSearchConfig = VectorSearchConfig()
    dialogue: DialogueConfig = DialogueConfig()
    cache: CacheConfig = CacheConfig()
    intent: IntentConfig = IntentConfig()
    logging: LoggingConfig = LoggingConfig()

    model_config = {
        'case_sensitive': False,
        'env_prefix': '',
        'env_nested_delimiter': '__'
    }

def load_config_from_yaml(path: str) -> dict:
    """Load configuration from a YAML file."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def merge_configs(yaml_config: dict, env_config: dict) -> dict:
    """Merge YAML config with environment variables, giving priority to env vars."""
    def _merge(a, b):
        for key, value in b.items():
            if isinstance(value, dict) and key in a and isinstance(a[key], dict):
                a[key] = _merge(a[key], value)
            else:
                a[key] = value
        return a
    return _merge(yaml_config, env_config)

# Load configurations
yaml_config = load_config_from_yaml('config.yml')

# Pydantic will automatically read environment variables
# We can pass the YAML config as initial data
config = AppConfig.model_validate(yaml_config)