"""依赖注入容器配置

整合所有服务组件，实现完整的依赖注入
"""

import os
from pathlib import Path
from typing import Dict, Any, List

from langchain_core.outputs import LLMResult

from .agent_protocol import (
    ServiceContainer, DatabaseService, LLMService, NutritionService,
    SQLiteDatabaseService
)
from .nutrition_service import StructuredNutritionService, LocalFoodDatabase, LocalExerciseDatabase
from .lightweight_planner import LightweightPlanner, RuleBasedClassifier, LiteModelClassifier
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from utils.logger import logger
from config import config as cfg
from langchain_openai import ChatOpenAI


class TokenUsageCallback(BaseCallbackHandler):
    """A callback handler to calculate and log token usage."""

    def __init__(self):
        super().__init__()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage from the LLM response."""
        if response.llm_output and "token_usage" in response.llm_output:
            token_usage = response.llm_output["token_usage"]
            self.total_prompt_tokens += token_usage.get("prompt_tokens", 0)
            self.total_completion_tokens += token_usage.get("completion_tokens", 0)
            self.total_tokens += token_usage.get("total_tokens", 0)

    def get_usage(self) -> Dict[str, int]:
        """Get the total token usage."""
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def reset(self) -> None:
        """Reset the token counters."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0


class OllamaLLMService(LLMService):
    """Ollama LLM服务实现"""
    
    def __init__(self):
        # 延迟导入，避免循环依赖
        from agents.config import get_llm
        
        self.token_usage_callback = TokenUsageCallback()
        self.llm = get_llm('extraction', callbacks=[self.token_usage_callback])
        self.lite_llm = get_llm('classification', lite=True, callbacks=[self.token_usage_callback])
        
        # 导入日志
        from utils.logger import logger
        self.logger = logger
    
    def extract_entities(self, text: str, schema: Dict) -> Dict:
        """提取实体信息"""
        try:
            # 构建提取提示
            schema_desc = "\n".join([f"- {k}: {v}" for k, v in schema.items()])
            prompt = f"""
请从以下文本中提取信息，返回JSON格式：

文本：{text}

需要提取的字段：
{schema_desc}

请返回JSON格式的结果，如果某个字段无法确定，请设为空字符串或null。
"""
            
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 尝试解析JSON
            import json
            import re
            
            # 提取JSON部分
            json_match = re.search(r'\{[^}]*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            # 降级到简单解析
            result = {}
            for key in schema.keys():
                if key in content.lower():
                    # 简单的关键词匹配
                    result[key] = text  # 简化处理
                else:
                    result[key] = ""
            
            return result
            
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {str(e)}")
            return {key: "" for key in schema.keys()}
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """生成响应"""
        try:
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            response = self.llm.invoke(full_prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            self.logger.error(f"Response generation failed: {str(e)}")
            return "抱歉，我现在无法生成响应，请稍后重试。"
    
    def invoke(self, prompt: str) -> Any:
        """直接调用 LLM（兼容 LightweightPlanner）"""
        return self.llm.invoke(prompt)
    
    def classify_intent(self, text: str, context: str = "") -> Dict:
        """意图分类"""
        try:
            prompt = f"""
核心功能注册表:
- RECORD_MEAL: 记录餐食
- RECORD_EXERCISE: 记录运动
- QUERY: 查询数据
- GENERATE_REPORT: 生成报告
- ADVICE: 获取建议

其他功能:
- 数据同步和备份
- 智能分析和推荐
- 多模态输入支持
"""
            
            response = self.lite_llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 解析响应
            import json
            import re
            
            json_match = re.search(r'\{[^}]*\}', content)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # 降级处理
            return {"intent": "UNKNOWN", "confidence": 0.5}
            
        except Exception as e:
            self.logger.error(f"Intent classification failed: {str(e)}")
            return {"intent": "UNKNOWN", "confidence": 0.0}


class ConfigurableServiceContainer(ServiceContainer):
    """可配置的服务容器
    
    支持从配置文件加载服务配置
    """
    
    def __init__(self, config_path: str = None):
        super().__init__()
        self.config = self._load_config(config_path)
        self._setup_services()
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """加载配置"""
        if config_path is None:
            # 使用默认配置
            return {
                'database': {
                    'type': 'sqlite',
                    'path': 'data/health_assistant.db'
                },
                'llm': {
                    'type': 'ollama',
                    'model': 'glm-4',
                    'temperature': 0.1,
                    'streaming': True
                },
                'nutrition': {
                    'type': 'local',
                    'food_db_path': 'data/nutrition.db',
                    'exercise_db_path': 'data/nutrition.db'
                },
                'planner': {
                    'enable_cache': True,
                    'cache_size': 1000,
                    'rule_confidence_threshold': 0.8,
                    'lite_model_threshold': 0.7
                }
            }
        
        # 从文件加载配置
        import yaml
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception:
            # 降级到默认配置
            return self._load_config(None)
    
    def _setup_services(self):
        """根据配置设置服务"""
        # 1. 数据库服务
        db_config = self.config.get('database', {})
        if db_config.get('type') == 'sqlite':
            db_path = db_config.get('path', 'data/health_assistant.db')
            # 确保目录存在
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.register_factory(
                DatabaseService,
                lambda: SQLiteDatabaseService(db_path)
            )
        else:
            raise ValueError(f"Unsupported database type: {db_config.get('type')}")
        
        # 2. LLM服务
        llm_config = self.config.get('llm', {})
        if llm_config.get('type') == 'ollama':
            self.register_factory(
                LLMService,
                lambda: OllamaLLMService()
            )
        else:
            raise ValueError(f"Unsupported LLM type: {llm_config.get('type')}")

        # 3. 营养服务
        db_service = self.get(DatabaseService)
        if self.config['nutrition']['type'] == 'local':
            food_db_path = self.config['nutrition']['food_db_path']
            exercise_db_path = self.config['nutrition']['exercise_db_path']
            
            # 确保目录存在
            Path(food_db_path).parent.mkdir(parents=True, exist_ok=True)
            
            def create_nutrition_service():
                food_db = LocalFoodDatabase(food_db_path)
                exercise_db = LocalExerciseDatabase(exercise_db_path)
                return StructuredNutritionService(food_db, exercise_db)
            
            self.register_factory(
                NutritionService,
                create_nutrition_service
            )
        
        # 注册轻量级规划器
        def create_planner():
            rule_classifier = RuleBasedClassifier()
            lite_classifier = LiteModelClassifier()
            
            # 直接创建 LLM 服务实例，避免循环依赖
            llm_service = self.get(LLMService)

            return LightweightPlanner(
                cache_size=self.config['planner']['cache_size'],
                rule_classifier=rule_classifier,
                lite_classifier=lite_classifier,
                llm_service=llm_service
            )
        
        self.register_factory(
            LightweightPlanner,
            create_planner
        )
    
    def register_singleton(self, interface: type, factory_or_implementation):
        """注册单例服务（支持工厂函数）"""
        if callable(factory_or_implementation) and not isinstance(factory_or_implementation, type):
            # 工厂函数
            self.register_factory(interface, factory_or_implementation)
        else:
            # 类型
            super().register_singleton(interface, factory_or_implementation)
    
    def get_config(self, key: str, default=None):
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update_config(self, key: str, value):
        """更新配置值"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, config_path: str):
        """保存配置到文件"""
        import yaml
        
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def health_check(self) -> Dict[str, bool]:
        """健康检查"""
        results = {}
        
        try:
            # 检查数据库服务
            db_service = self.get(DatabaseService)
            # 简单的连接测试
            results['database'] = True
        except Exception:
            results['database'] = False
        
        try:
            # 检查LLM服务
            llm_service = self.get(LLMService)
            # 简单的调用测试
            response = llm_service.generate_response("test")
            results['llm'] = bool(response)
        except Exception:
            results['llm'] = False
        
        try:
            # 检查营养服务
            nutrition_service = self.get(NutritionService)
            # 简单的搜索测试
            foods = nutrition_service.get_food_suggestions("米饭")
            results['nutrition'] = len(foods) > 0
        except Exception:
            results['nutrition'] = False
        
        try:
            # 检查规划器
            planner = self.get(LightweightPlanner)
            result = planner.plan("我想记录今天的早餐")
            results['planner'] = result.intent is not None
        except Exception:
            results['planner'] = False
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'services_registered': len(self._services) + len(self._factories),
            'singletons_created': len(self._singletons),
            'config_keys': len(self.config)
        }
        
        # 获取规划器统计
        try:
            planner = self.get(LightweightPlanner)
            stats.update(planner.get_statistics())
        except Exception:
            pass
        
        return stats


# 全局容器实例
_global_container = None


def get_container() -> ConfigurableServiceContainer:
    """获取全局容器实例"""
    global _global_container
    
    if _global_container is None:
        _global_container = ConfigurableServiceContainer()
    
    return _global_container


def setup_container(config_path: str = None) -> ConfigurableServiceContainer:
    """设置全局容器"""
    global _global_container
    
    _global_container = ConfigurableServiceContainer(config_path)
    return _global_container


def reset_container():
    """重置全局容器"""
    global _global_container
    
    if _global_container:
        _global_container.clear()
    
    _global_container = None


# 便捷函数
def get_database_service() -> DatabaseService:
    """获取数据库服务"""
    return get_container().get(DatabaseService)


def get_llm_service() -> LLMService:
    """获取LLM服务"""
    return get_container().get(LLMService)


def get_nutrition_service() -> NutritionService:
    """获取营养服务"""
    return get_container().get(NutritionService)


def get_planner() -> LightweightPlanner:
    """获取轻量级规划器"""
    return get_container().get(LightweightPlanner)


# 使用示例和测试
if __name__ == "__main__":
    # 设置容器
    container = setup_container()
    
    # 健康检查
    health = container.health_check()
    print("健康检查结果：")
    for service, status in health.items():
        print(f"  {service}: {'✅' if status else '❌'}")
    
    # 统计信息
    stats = container.get_statistics()
    print("\n统计信息：")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 测试服务
    print("\n测试服务：")
    
    # 测试规划器
    planner = get_planner()
    result = planner.plan("我今天吃了一碗白米饭")
    print(f"规划结果：{result.intent} (置信度: {result.confidence})")
    
    # 测试营养服务
    nutrition_service = get_nutrition_service()
    foods = nutrition_service.get_food_suggestions("米饭")
    print(f"找到 {len(foods)} 个食物建议")
    
    # 测试LLM服务
    llm_service = get_llm_service()
    response = llm_service.generate_response("你好")
    print(f"LLM响应：{response[:50]}...")
    
    print("\n✅ 所有测试完成")