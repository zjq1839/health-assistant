import logging
import structlog
from loguru import logger
from config import config
import sys
import os

# 配置 loguru
logger.remove()  # 移除默认处理器

# 添加控制台输出
logger.add(
    sys.stderr,
    level=config.LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# 添加文件输出
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
logger.add(
    os.path.join(log_dir, config.LOG_FILE),
    level=config.LOG_LEVEL,
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

# 配置 structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# 创建结构化日志记录器
struct_logger = structlog.get_logger()

def log_llm_call(agent_name: str, prompt: str, response: str, duration: float):
    """记录 LLM 调用信息"""
    struct_logger.info(
        "llm_call",
        agent=agent_name,
        prompt_length=len(prompt),
        response_length=len(response),
        duration=duration,
        prompt_preview=prompt[:100] + "..." if len(prompt) > 100 else prompt
    )

def log_user_input(user_input: str, intent: str = "unknown", next_agent: str = "supervisor"):
    """记录用户输入和路由信息"""
    struct_logger.info(
        "user_input",
        input=user_input,
        intent=intent,
        next_agent=next_agent
    )

def log_error(error_type: str, error_message: str, context: dict = None):
    """记录错误信息"""
    struct_logger.error(
        "error_occurred",
        error_type=error_type,
        error_message=error_message,
        context=context or {}
    )

def log_database_operation(operation: str, table: str, affected_rows: int = 0):
    """记录数据库操作"""
    struct_logger.info(
        "database_operation",
        operation=operation,
        table=table,
        affected_rows=affected_rows
    )