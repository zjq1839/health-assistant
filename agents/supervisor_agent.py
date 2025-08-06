import re
from langchain_ollama import ChatOllama
from core.state import State
from .config import llm

def supervisor(state: State):
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
    return {"next_agent": agent}