from core.state import State
from .config import llm

def unknown_intent(state: State):
    history = "\n".join([msg[1] if isinstance(msg, tuple) else msg.content for msg in state["messages"]])
    prompt = f"你是一个饮食分析助手，可以帮助用户分析饮食和运动数据，并给出相应的建议和分析。基于以下对话历史，适当响应用户的最后消息，保持友好。历史：{history}"
    response = llm.invoke([("system", prompt), ("user", state["messages"][-1].content)])
    cleaned_content = response.content.strip()
    return {"messages": [("ai", cleaned_content)]}