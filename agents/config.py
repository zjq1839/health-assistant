from langchain_ollama import ChatOllama
from config import config
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
import os

# 设置缓存
if config.ENABLE_CACHE:
    cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=os.path.join(cache_dir, 'llm_cache.db')))

llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
llm_lite = ChatOllama(model=config.LLM_LITE_MODEL, temperature=config.LLM_TEMPERATURE, reasoning=True)