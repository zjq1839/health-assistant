from core.state import State
from .config import llm

def unknown_intent(state: State):
    # 获取对话历史
    history = "\n".join([msg[1] if isinstance(msg, tuple) else msg.content for msg in state["messages"]])
    
    # 获取用户的最后一条消息
    last_user_msg = state["messages"][-1]
    user_content = last_user_msg[1] if isinstance(last_user_msg, tuple) else last_user_msg.content
    
    # 获取助手的上一条回复（如果存在）
    for i in range(len(state["messages"]) - 2, -1, -1):
        msg = state["messages"][i]
        if isinstance(msg, tuple) and msg[0] == "ai":
            msg[1]
            break
        elif hasattr(msg, 'content') and msg.__class__.__name__ == "AIMessage":
            msg.content
            break
    
    # 增强的提示词，提供更好的情绪价值
    prompt = f"""你是一个温暖友好的健康助手，专门帮助用户管理饮食和运动。你的目标是：
1. 理解用户的真实需求和情感
2. 提供有用的信息和建议
3. 保持积极、支持和鼓励的态度
4. 让用户感到被理解和关心

特别注意：
- 如果用户询问模糊，尝试理解其背后的需求
- 始终保持友好和支持的语气

对话历史：
{history}

请根据用户的最新消息，给出温暖、有帮助的回复。"""
    
    response = llm.invoke([("system", prompt), ("user", user_content)])
    cleaned_content = response.content.strip()
    return {"messages": [("ai", cleaned_content)]}