from typing import Annotated, Dict, List, Optional, Any
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

@dataclass
class DialogState:
    """对话状态跟踪"""
    current_intent: Optional[IntentType] = None
    intent_confidence: float = 0.0
    entities: Dict[str, Any] = field(default_factory=dict)
    turn_history: List[DialogTurn] = field(default_factory=list)
    
    def get_recent_intents(self, n: int = 3) -> List[IntentType]:
        """获取最近n轮的意图"""
        return [turn.intent for turn in self.turn_history[-n:]]
    
    def get_context_entities(self) -> Dict[str, Any]:
        """获取上下文中的实体信息"""
        context_entities = {}
        for turn in self.turn_history[-3:]:  # 最近3轮
            context_entities.update(turn.entities)
        return context_entities

class EnhancedState(TypedDict):
    """增强的状态定义（精简版）"""
    messages: Annotated[list, add_messages]
    dialog_state: DialogState
    turn_id: int

def create_enhanced_state() -> EnhancedState:
    """创建增强状态实例"""
    return {
        'messages': [],
        'dialog_state': DialogState(),
        'turn_id': 0
    }