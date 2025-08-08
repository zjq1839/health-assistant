import time
import functools
from typing import Any, Callable, Dict, List, Optional
from collections import OrderedDict
import threading
from config import config
from utils.logger import logger, struct_logger

class LRUCache:
    """线程安全的 LRU 缓存实现"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                # 移动到末尾（最近使用）
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def put(self, key: str, value: Any) -> None:
        with self.lock:
            if key in self.cache:
                # 更新现有键
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # 移除最久未使用的项
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        with self.lock:
            return len(self.cache)

# 全局缓存实例
llm_cache = LRUCache(max_size=config.cache.size)
vector_cache = LRUCache(max_size=50)

def cache_llm_response(func: Callable) -> Callable:
    """LLM 响应缓存装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not config.ENABLE_CACHE:
            return func(*args, **kwargs)
        
        # 生成缓存键
        cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
        
        # 尝试从缓存获取
        cached_result = llm_cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for {func.__name__}")
            return cached_result
        
        # 执行函数并缓存结果
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        
        llm_cache.put(cache_key, result)
        logger.debug(f"Cache miss for {func.__name__}, execution time: {duration:.2f}s")
        
        return result
    
    return wrapper

def performance_monitor(func: Callable) -> Callable:
    """性能监控装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = get_memory_usage()
        result = None
        success = False
        error = None
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            error = str(e)
            raise
        finally:
            end_time = time.time()
            end_memory = get_memory_usage()
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # 记录性能指标
            struct_logger.info(
                "performance_metrics",
                function=func.__name__,
                duration=duration,
                memory_delta=memory_delta,
                success=success,
                error=error
            )
            
            # 如果执行时间过长，记录警告
            if duration > 5.0:  # 5秒阈值
                logger.warning(f"Slow function detected: {func.__name__} took {duration:.2f}s")
        
        return result
    
    return wrapper

def get_memory_usage() -> float:
    """获取当前内存使用量（MB）"""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # 转换为 MB
    except ImportError:
        return 0.0

class BatchProcessor:
    """批处理器，用于批量处理数据库操作"""
    
    def __init__(self, batch_size: int = 10, flush_interval: float = 5.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.pending_operations = []
        self.last_flush = time.time()
        self.lock = threading.Lock()
    
    def add_operation(self, operation: Dict[str, Any]) -> None:
        """添加操作到批处理队列"""
        with self.lock:
            self.pending_operations.append(operation)
            
            # 检查是否需要刷新
            if (len(self.pending_operations) >= self.batch_size or 
                time.time() - self.last_flush >= self.flush_interval):
                self._flush()
    
    def _flush(self) -> None:
        """执行批处理操作"""
        if not self.pending_operations:
            return
        
        operations = self.pending_operations.copy()
        self.pending_operations.clear()
        self.last_flush = time.time()
        
        try:
            self._execute_batch(operations)
            logger.info(f"Batch processed {len(operations)} operations")
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # 重新添加失败的操作
            self.pending_operations.extend(operations)
    
    def _execute_batch(self, operations: List[Dict[str, Any]]) -> None:
        """执行批量操作（需要子类实现）"""
        raise NotImplementedError("Subclasses must implement _execute_batch")
    
    def force_flush(self) -> None:
        """强制刷新所有待处理操作"""
        with self.lock:
            self._flush()

class DatabaseBatchProcessor(BatchProcessor):
    """数据库批处理器"""
    
    def _execute_batch(self, operations: List[Dict[str, Any]]) -> None:
        """执行批量数据库操作"""
        import sqlite3
        from database import DB_PATH
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            for op in operations:
                if op['type'] == 'insert_meal':
                    cursor.execute(
                        "INSERT INTO meals (date, meal_type, description, calories, nutrients) VALUES (?, ?, ?, ?, ?)",
                        (op['date'], op['meal_type'], op['description'], op.get('calories', 0), op.get('nutrients'))
                    )
                elif op['type'] == 'insert_exercise':
                    cursor.execute(
                        "INSERT INTO exercises (date, exercise_type, duration, description, calories_burned, intensity) VALUES (?, ?, ?, ?, ?, ?)",
                        (op['date'], op['exercise_type'], op['duration'], op.get('description'), op.get('calories_burned', 0), op.get('intensity', 'medium'))
                    )
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

# 全局批处理器实例
db_batch_processor = DatabaseBatchProcessor()

def optimize_vector_search(query_embedding, embeddings, top_k: int = 5):
    """优化向量搜索性能"""
    import numpy as np
    
    # 使用缓存
    cache_key = f"vector_search_{hash(str(query_embedding.tolist()))}_{top_k}"
    cached_result = vector_cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    # 计算相似度
    similarities = np.dot(embeddings, query_embedding)
    
    # 获取 top-k 结果
    top_indices = np.argpartition(similarities, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
    
    result = [(idx, similarities[idx]) for idx in top_indices]
    
    # 缓存结果
    vector_cache.put(cache_key, result)
    
    return result