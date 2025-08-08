from langchain_ollama import ChatOllama
from config import config
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
import os

# Set up caching
if config.cache.enable:
    cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    if config.cache.type == 'sqlite':
        db_path = os.path.join(cache_dir, 'llm_cache.db')
        set_llm_cache(SQLiteCache(database_path=db_path))

# Initialize LLMs with the new configuration structure
llm = ChatOllama(model=config.llm.model, temperature=config.llm.temperature)
llm_lite = ChatOllama(model=config.llm.lite_model, temperature=config.llm.temperature)

# ---- Dynamic LLM factory ----
from functools import lru_cache

_TASK_TEMPERATURES = {
    'extraction': 0.0,      # 结构化信息抽取
    'classification': 0.2,  # 分类/路由判断
    'generation': 0.7,      # 文本生成或报告
    'default': config.llm.temperature
}

@lru_cache(maxsize=16)
def get_llm(task_type: str = 'default', lite: bool = False) -> ChatOllama:
    """根据任务类型动态返回 LLM 实例，并做简单缓存。"""
    temperature = _TASK_TEMPERATURES.get(task_type, _TASK_TEMPERATURES['default'])
    model_name = config.llm.lite_model if lite else config.llm.model
    return ChatOllama(model=model_name, temperature=temperature)