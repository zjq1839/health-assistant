"""Agent 统一协议实现

解决问题：Agent 无统一协议，耦合 I/O，难以测试
方案：定义统一协议，依赖注入，纯函数设计
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .enhanced_state import IntentType, EnhancedState


class AgentResult(Enum):
    """Agent 执行结果状态"""
    SUCCESS = "success"
    ERROR = "error"
    REDIRECT = "redirect"  # 需要重定向到其他 Agent
    PARTIAL = "partial"   # 部分成功，需要用户补充信息


@dataclass
class AgentResponse:
    """Agent 响应结果"""
    status: AgentResult
    message: str
    data: Dict[str, Any] = None
    next_agent: Optional[str] = None
    evidence: List[Dict] = None  # 证据链
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}
        if self.evidence is None:
            self.evidence = []
    
    def add_evidence(self, source: str, content: str, confidence: float = 1.0):
        """添加证据"""
        self.evidence.append({
            'source': source,
            'content': content,
            'confidence': confidence,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        })


@runtime_checkable
class AgentProtocol(Protocol):
    """Agent 统一协议
    
    所有 Agent 必须实现此协议，确保：
    1. 统一的接口
    2. 可测试性
    3. 可组合性
    """
    
    name: str
    intents: List[IntentType]
    
    def can_handle(self, intent: IntentType) -> bool:
        """判断是否能处理指定意图"""
        ...
    
    def run(self, state: EnhancedState) -> AgentResponse:
        """执行 Agent 逻辑（纯函数）"""
        ...
    
    def validate_input(self, state: EnhancedState) -> bool:
        """验证输入是否有效"""
        ...


class DatabaseService(ABC):
    """数据库服务抽象接口"""
    
    @abstractmethod
    def save_meal(self, meal_data: Dict) -> Dict:
        """保存饮食记录"""
        pass
    
    @abstractmethod
    def save_exercise(self, exercise_data: Dict) -> Dict:
        """保存运动记录"""
        pass
    
    @abstractmethod
    def query_meals(self, date: str = None, limit: int = 10) -> List[Dict]:
        """查询饮食记录"""
        pass
    
    @abstractmethod
    def query_exercises(self, date: str = None, limit: int = 10) -> List[Dict]:
        """查询运动记录"""
        pass


class LLMService(ABC):
    """LLM 服务抽象接口"""
    
    @abstractmethod
    def extract_entities(self, text: str, schema: Dict) -> Dict:
        """提取实体信息"""
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str, context: str = "") -> str:
        """生成响应"""
        pass
    
    @abstractmethod
    def classify_intent(self, text: str, context: str = "") -> Dict:
        """意图分类"""
        pass


class NutritionService(ABC):
    """营养计算服务抽象接口"""
    
    @abstractmethod
    def calculate_nutrition(self, food_description: str) -> Dict:
        """计算营养信息"""
        pass
    
    @abstractmethod
    def get_food_database(self) -> Dict:
        """获取食物数据库信息"""
        pass


class BaseAgent(ABC):
    """Agent 基类
    
    提供通用功能：
    1. 依赖注入
    2. 输入验证
    3. 错误处理
    4. 日志记录
    """
    
    def __init__(self, 
                 name: str, 
                 intents: List[IntentType],
                 db_service: DatabaseService = None,
                 llm_service: LLMService = None,
                 nutrition_service: NutritionService = None):
        self.name = name
        self.intents = intents
        self.db_service = db_service
        self.llm_service = llm_service
        self.nutrition_service = nutrition_service
        
        # 导入日志模块
        from utils.logger import logger
        self.logger = logger
    
    def can_handle(self, intent: IntentType) -> bool:
        """判断是否能处理指定意图"""
        return intent in self.intents
    
    def validate_input(self, state: EnhancedState) -> bool:
        """基础输入验证"""
        if not state or not state.get('messages'):
            return False
        
        last_message = state['messages'][-1]
        if not last_message:
            return False
        
        # 提取用户输入
        user_input = self._extract_user_input(last_message)
        return bool(user_input and user_input.strip())
    
    def _extract_user_input(self, message) -> str:
        """提取用户输入文本"""
        if isinstance(message, tuple):
            return message[1] if len(message) > 1 else ""
        elif hasattr(message, 'content'):
            return message.content
        else:
            return str(message)
    
    def _create_success_response(self, message: str, data: Dict = None) -> AgentResponse:
        """创建成功响应"""
        return AgentResponse(
            status=AgentResult.SUCCESS,
            message=message,
            data=data or {}
        )
    
    def _create_error_response(self, message: str, error_code: str = None) -> AgentResponse:
        """创建错误响应"""
        data = {'error_code': error_code} if error_code else {}
        return AgentResponse(
            status=AgentResult.ERROR,
            message=message,
            data=data
        )
    
    def _create_redirect_response(self, message: str, next_agent: str, data: Dict = None) -> AgentResponse:
        """创建重定向响应"""
        return AgentResponse(
            status=AgentResult.REDIRECT,
            message=message,
            next_agent=next_agent,
            data=data or {}
        )
    
    @abstractmethod
    def run(self, state: EnhancedState) -> AgentResponse:
        """执行 Agent 逻辑（子类必须实现）"""
        pass
    
    def _log_execution(self, state: EnhancedState, response: AgentResponse):
        """记录执行日志"""
        user_input = self._extract_user_input(state['messages'][-1])
        self.logger.info(
            f"Agent {self.name} executed",
            extra={
                'agent': self.name,
                'user_input': user_input[:100],  # 截断长输入
                'status': response.status.value,
                'has_evidence': len(response.evidence) > 0
            }
        )


class AgentFactory:
    """Agent 工厂类
    
    负责创建和管理 Agent 实例
    实现依赖注入
    """
    
    def __init__(self, container):
        self.container = container
        self._agents = {}
    
    def create_agent(self, agent_name: str) -> BaseAgent:
        """创建 Agent 实例"""
        if agent_name in self._agents:
            return self._agents[agent_name]
        
        # 根据名称创建对应的 Agent
        if agent_name == "dietary":
            from agents.dietary_agent_v2 import DietaryAgentV2
            agent = DietaryAgentV2(
                db_service=self.container.get(DatabaseService),
                llm_service=self.container.get(LLMService),
                nutrition_service=self.container.get(NutritionService)
            )
        elif agent_name == "exercise":
            from agents.exercise_agent_v2 import ExerciseAgentV2
            agent = ExerciseAgentV2(
                db_service=self.container.get(DatabaseService),
                llm_service=self.container.get(LLMService)
            )
        elif agent_name == "query":
            from agents.query_agent_v2 import QueryAgentV2
            agent = QueryAgentV2(
                db_service=self.container.get(DatabaseService),
                llm_service=self.container.get(LLMService)
            )
        elif agent_name == "report":
            from agents.report_agent_v2 import ReportAgentV2
            agent = ReportAgentV2(
                db_service=self.container.get(DatabaseService),
                llm_service=self.container.get(LLMService)
            )
        elif agent_name == "advice":
            from agents.advice_agent_v2 import AdviceAgentV2
            agent = AdviceAgentV2(
                llm_service=self.container.get(LLMService)
            )
        else:
            from agents.general_agent_v2 import GeneralAgentV2
            agent = GeneralAgentV2(
                llm_service=self.container.get(LLMService)
            )
        
        self._agents[agent_name] = agent
        return agent
    
    def get_available_agents(self) -> List[str]:
        """获取可用的 Agent 列表"""
        return ["dietary", "exercise", "query", "report", "advice", "general"]
    
    def clear_cache(self):
        """清理 Agent 缓存"""
        self._agents.clear()


class ServiceContainer:
    """依赖注入容器
    
    管理服务的生命周期和依赖关系
    """
    
    def __init__(self):
        self._services = {}
        self._singletons = {}
        self._factories = {}
    
    def register_singleton(self, interface: type, implementation: type):
        """注册单例服务"""
        self._services[interface] = {
            'implementation': implementation,
            'singleton': True
        }
    
    def register_transient(self, interface: type, implementation: type):
        """注册瞬态服务"""
        self._services[interface] = {
            'implementation': implementation,
            'singleton': False
        }
    
    def register_factory(self, interface: type, factory_func):
        """注册工厂方法"""
        self._factories[interface] = factory_func
    
    def get(self, interface: type):
        """获取服务实例"""
        # 优先使用工厂方法
        if interface in self._factories:
            return self._factories[interface]()
        
        if interface not in self._services:
            raise ValueError(f"Service {interface.__name__} not registered")
        
        service_config = self._services[interface]
        
        if service_config['singleton']:
            if interface not in self._singletons:
                self._singletons[interface] = self._create_instance(service_config['implementation'])
            return self._singletons[interface]
        else:
            return self._create_instance(service_config['implementation'])
    
    def _create_instance(self, implementation_class):
        """创建服务实例"""
        # 简单的构造函数注入
        import inspect
        
        sig = inspect.signature(implementation_class.__init__)
        params = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            if param.annotation != inspect.Parameter.empty:
                # 递归解析依赖
                params[param_name] = self.get(param.annotation)
        
        return implementation_class(**params)
    
    def clear(self):
        """清理容器"""
        self._singletons.clear()


# 具体实现示例
class SQLiteDatabaseService(DatabaseService):
    """SQLite 数据库服务实现"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            from config import config
            db_path = config.database.path
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建 meals 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                meal_type TEXT,
                description TEXT,
                calories REAL,
                nutrients TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建 exercises 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exercises (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                exercise_type TEXT,
                duration INTEGER,
                description TEXT,
                calories_burned REAL,
                intensity TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_meals_date ON meals(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exercises_date ON exercises(date)")
        
        conn.commit()
        conn.close()
    
    def save_meal(self, meal_data: Dict) -> Dict:
        """保存饮食记录"""
        import sqlite3
        import datetime
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO meals (date, meal_type, description, calories, nutrients)
                VALUES (?, ?, ?, ?, ?)
            """, (
                meal_data.get('date', datetime.date.today().isoformat()),
                meal_data.get('meal_type', ''),
                meal_data.get('description', ''),
                meal_data.get('calories', 0),
                meal_data.get('nutrients', '')
            ))
            
            meal_id = cursor.lastrowid
            conn.commit()
            
            return {
                'id': meal_id,
                'status': 'success',
                'message': '饮食记录保存成功'
            }
            
        except Exception as e:
            conn.rollback()
            return {
                'status': 'error',
                'message': f'保存失败: {str(e)}'
            }
        finally:
            conn.close()
    
    def save_exercise(self, exercise_data: Dict) -> Dict:
        """保存运动记录"""
        import sqlite3
        import datetime
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO exercises (date, exercise_type, duration, description, calories_burned, intensity)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                exercise_data.get('date', datetime.date.today().isoformat()),
                exercise_data.get('exercise_type', ''),
                exercise_data.get('duration', 0),
                exercise_data.get('description', ''),
                exercise_data.get('calories_burned', 0),
                exercise_data.get('intensity', 'medium')
            ))
            
            exercise_id = cursor.lastrowid
            conn.commit()
            
            return {
                'id': exercise_id,
                'status': 'success',
                'message': '运动记录保存成功'
            }
            
        except Exception as e:
            conn.rollback()
            return {
                'status': 'error',
                'message': f'保存失败: {str(e)}'
            }
        finally:
            conn.close()
    
    def query_meals(self, date: str = None, limit: int = 10) -> List[Dict]:
        """查询饮食记录"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if date:
                cursor.execute(
                    "SELECT * FROM meals WHERE date = ? ORDER BY created_at DESC LIMIT ?",
                    (date, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM meals ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                )
            
            rows = cursor.fetchall()
            
            # 转换为字典格式
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
            
        finally:
            conn.close()
    
    def query_exercises(self, date: str = None, limit: int = 10) -> List[Dict]:
        """查询运动记录"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if date:
                cursor.execute(
                    "SELECT * FROM exercises WHERE date = ? ORDER BY created_at DESC LIMIT ?",
                    (date, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM exercises ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                )
            
            rows = cursor.fetchall()
            
            # 转换为字典格式
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
            
        finally:
            conn.close()


# 注意：setup_container 函数已移至 core/service_container.py
# 请使用 from core.service_container import get_container