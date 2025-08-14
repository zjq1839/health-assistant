# 移除未使用的 structlog 依赖
# from structlog import get_logger  # 已不再使用
from loguru import logger
from config import config
import sys
import os

# 配置 loguru
logger.remove()  # 移除默认处理器

# 控制台输出
logger.add(
    sys.stderr,
    level=config.logging.level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# 文件输出
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
logger.add(
    os.path.join(log_dir, config.logging.file),
    level=config.logging.level,
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

# 导出 logger 供其他模块使用
__all__ = ['logger']