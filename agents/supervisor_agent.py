import re
from core.state import State
from core.enhanced_state import EnhancedState, IntentType, DialogState, DialogTurn, ContextManager
from core.intent_recognizer import get_intent_recognizer, get_user_input
from datetime import datetime
from .config import llm
from utils.logger import logger

# 初始化上下文管理器
context_manager = ContextManager()

def supervisor(state):
    """增强的主管代理，集成意图识别和上下文管理"""
    # 兼容性处理：如果是旧的State，转换为EnhancedState
    if not isinstance(state, dict) or 'dialog_state' not in state:
        state = _upgrade_state_to_enhanced(state)
    
    # 获取意图识别器
    intent_recognizer = get_intent_recognizer(llm)
    
    # 智能截断消息历史
    state["messages"] = context_manager.intelligent_truncate(
        state["messages"], state["dialog_state"]
    )
    
    # 生成上下文摘要
    state["context_summary"] = context_manager.generate_context_summary(
        state["messages"], state["dialog_state"]
    )
    
    # 识别意图
    intent, confidence, entities = intent_recognizer.recognize_intent(state)
    
    # 更新对话状态
    state["turn_id"] += 1
    state["intent_confidence"] = confidence
    
    # 创建对话轮次记录
    user_input = get_user_input(state)
    dialog_turn = DialogTurn(
        turn_id=state["turn_id"],
        timestamp=datetime.now(),
        user_input=user_input,
        intent=intent,
        confidence=confidence,
        entities=entities
    )
    
    # 更新对话状态
    state["dialog_state"].update_intent(intent, confidence, entities)
    state["dialog_state"].add_turn(dialog_turn)
    
    # 映射意图到代理
    agent_mapping = {
        IntentType.RECORD_MEAL: "dietary",
        IntentType.RECORD_EXERCISE: "exercise", 
        IntentType.GENERATE_REPORT: "report",
        IntentType.QUERY: "query",
        IntentType.ADVICE: "advice",
        IntentType.UNKNOWN: "general"
    }
    
    selected_agent = agent_mapping.get(intent, "general")
    
    logger.info(f"Intent: {intent.value}, Confidence: {confidence:.2f}, Agent: {selected_agent}")
    
    return {"next_agent": selected_agent}

def _upgrade_state_to_enhanced(old_state) -> EnhancedState:
    """将旧状态升级为增强状态"""
    if isinstance(old_state, dict):
        enhanced_state = old_state.copy()
    else:
        # 如果是State对象，转换为字典
        enhanced_state = dict(old_state)
    
    # 添加缺失的字段
    if 'dialog_state' not in enhanced_state:
        enhanced_state['dialog_state'] = DialogState()
    if 'context_summary' not in enhanced_state:
        enhanced_state['context_summary'] = ""
    if 'intent_confidence' not in enhanced_state:
        enhanced_state['intent_confidence'] = 0.0
    if 'turn_id' not in enhanced_state:
        enhanced_state['turn_id'] = 0
    
    return enhanced_state

def legacy_supervisor(state: State):
    """保留的传统supervisor实现，用于向后兼容"""
    content = state["messages"][-1].content
    
    # 获取对话历史上下文
    recent_messages = state["messages"][-3:] if len(state["messages"]) >= 3 else state["messages"]
    context = ""
    for msg in recent_messages[:-1]:  # 排除当前消息
        if isinstance(msg, tuple):
            context += f"用户: {msg[1]}\n" if msg[0] == "user" else f"助手: {msg[1]}\n"
        else:
            # 处理HumanMessage和AIMessage对象
            if hasattr(msg, 'content'):
                if msg.__class__.__name__ == "HumanMessage":
                    context += f"用户: {msg.content}\n"
                elif msg.__class__.__name__ == "AIMessage":
                    context += f"助手: {msg.content}\n"
    
    prompt = f"""你是一个主管代理，负责协调任务。基于用户输入和对话上下文，选择合适的代理：

对话上下文：
{context}

当前用户输入: {content}

代理选择规则：
- 如果用户提到吃了什么、饮食记录，如'我早餐吃了...'或'今天午餐是...'：dietary
- 如果用户提到运动、锻炼记录、图片、截图、OCR等与图像提取相关的内容：exercise
- 如果用户请求生成报告、查看分析，如'生成今天的报告'：report
- 如果用户查询已有信息，如查询哪一天吃了什么，或者查询哪一天做了什么运动：query
- 如果用户只说"昨天呢"、"前天呢"、"那天呢"等，需要结合上下文判断：
  * 如果上下文中有饮食相关查询，则选择：query
  * 如果上下文中有运动相关查询，则选择：query
  * 如果上下文中有报告相关内容，则选择：report
- 其他不确定或一般查询：general

只返回代理名称，不要添加额外解释。"""
    response = llm.invoke(prompt)
    # 使用正则表达式移除<think>标签
    cleaned_response = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
    agent = cleaned_response
    print(agent)
    print(context)
    return {"next_agent": agent}