from typing import Annotated, Literal, Dict, List, Optional, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class IntentType(Enum):
    """意图类型枚举"""
    RECORD_MEAL = "record_meal"
    RECORD_EXERCISE = "record_exercise"
    GENERATE_REPORT = "generate_report"
    QUERY = "query"
    ADVICE = "advice"  
    UNKNOWN = "unknown"

@dataclass
class DialogTurn:
    """对话轮次信息"""
    turn_id: int
    timestamp: datetime
    user_input: str
    intent: IntentType
    confidence: float
    entities: Dict[str, Any] = field(default_factory=dict)
    context_used: List[str] = field(default_factory=list)

@dataclass
class DialogState:
    """对话状态跟踪"""
    current_intent: Optional[IntentType] = None
    intent_confidence: float = 0.0
    entities: Dict[str, Any] = field(default_factory=dict)
    context_summary: str = ""
    turn_history: List[DialogTurn] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)
    
    def update_intent(self, intent: IntentType, confidence: float, entities: Dict[str, Any] = None):
        """更新意图状态"""
        self.current_intent = intent
        self.intent_confidence = confidence
        if entities:
            self.entities.update(entities)
        self.last_update = datetime.now()
    
    def add_turn(self, turn: DialogTurn):
        """添加对话轮次"""
        self.turn_history.append(turn)
        # 保持最近10轮对话
        if len(self.turn_history) > 10:
            self.turn_history = self.turn_history[-10:]
    
    def get_recent_intents(self, n: int = 3) -> List[IntentType]:
        """获取最近n轮的意图"""
        return [turn.intent for turn in self.turn_history[-n:]]

    def get_recent_turns(self, n: int = 3) -> List[DialogTurn]:
        """获取最近n轮的对话"""
        return self.turn_history[-n:]
    
    def get_context_entities(self) -> Dict[str, Any]:
        """获取上下文中的实体信息"""
        context_entities = {}
        for turn in self.turn_history[-3:]:  # 最近3轮
            context_entities.update(turn.entities)
        return context_entities

class EnhancedState(TypedDict):
    """增强的状态定义（精简版）"""
    messages: Annotated[list, add_messages]
    intent: Literal["record_meal", "record_exercise", "generate_report", "query", "advice", "unknown"]
    # 核心字段保留
    dialog_state: DialogState
    turn_id: int

class ContextManager:
    """上下文管理器"""
    
    def __init__(self, max_messages: int = 20, immediate_context_size: int = 3):
        self.max_messages = max_messages
        self.immediate_context_size = immediate_context_size
    
    def intelligent_truncate(self, messages: List, dialog_state: DialogState) -> List:
        """智能截断消息历史"""
        if len(messages) <= self.max_messages:
            return messages
        
        # 保留最近的immediate_context_size条消息
        recent_messages = messages[-self.immediate_context_size:]
        
        # 保留重要的历史消息（包含关键实体或高置信度意图的消息）
        important_messages = []
        for i, msg in enumerate(messages[:-self.immediate_context_size]):
            if self._is_important_message(msg, dialog_state, i):
                important_messages.append(msg)
        
        # 计算可保留的历史消息数量
        available_slots = self.max_messages - len(recent_messages)
        if len(important_messages) > available_slots:
            # 按重要性排序，保留最重要的消息
            important_messages = self._rank_messages_by_importance(
                important_messages, dialog_state
            )[:available_slots]
        
        return important_messages + recent_messages
    
    def _is_important_message(self, message, dialog_state: DialogState, index: int) -> bool:
        """判断消息是否重要"""
        if isinstance(message, tuple):
            content = message[1]
        else:
            content = getattr(message, 'content', '')
        
        # 检查是否包含关键实体
        entities = dialog_state.get_context_entities()
        for entity_value in entities.values():
            if str(entity_value).lower() in content.lower():
                return True
        
        # 检查是否包含关键词
        keywords = ['饮食', '运动', '报告', '查询', '昨天', '今天', '前天']
        for keyword in keywords:
            if keyword in content:
                return True
        
        return False
    
    def _rank_messages_by_importance(self, messages: List, dialog_state: DialogState) -> List:
        """按重要性排序消息"""
        def importance_score(msg):
            if isinstance(msg, tuple):
                content = msg[1]
            else:
                content = getattr(msg, 'content', '')
            
            score = 0
            # 实体匹配得分
            entities = dialog_state.get_context_entities()
            for entity_value in entities.values():
                if str(entity_value).lower() in content.lower():
                    score += 2
            
            # 关键词得分
            keywords = ['饮食', '运动', '报告', '查询']
            for keyword in keywords:
                if keyword in content:
                    score += 1
            
            return score
        
        return sorted(messages, key=importance_score, reverse=True)
    
    def generate_context_summary(self, messages: List, dialog_state: DialogState) -> str:
        """生成上下文摘要"""
        if not messages:
            return ""
        
        # 提取关键信息
        recent_intents = [i for i in dialog_state.get_recent_intents() if i != IntentType.UNKNOWN]
        entities = {k: v for k, v in dialog_state.get_context_entities().items() if v}

        summary_parts = []

        # 仅保留最近 3 个非 UNKNOWN 意图
        if recent_intents:
            intent_summary = ", ".join([intent.value for intent in recent_intents[-3:]])
            summary_parts.append(f"最近意图: {intent_summary}")

        # 仅保留最多 3 个实体，按键名排序保持稳定
        if entities:
            # 取前 3 个实体
            top_entities = list(entities.items())[:3]
            entity_summary = ", ".join([f"{k}: {v}" for k, v in top_entities])
            summary_parts.append(f"关键信息: {entity_summary}")

        return "; ".join(summary_parts)

def create_enhanced_state() -> EnhancedState:
    """创建增强状态实例"""
    return {
        'messages': [],
        'intent': 'unknown',
        'dialog_state': DialogState(),
        'turn_id': 0
    }