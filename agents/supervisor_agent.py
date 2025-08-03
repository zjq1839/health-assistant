import re
from langchain_ollama import ChatOllama
from core.state import State
from .config import llm

def supervisor(state: State):
    content = state["messages"][-1].content
    prompt = f"""你是一个主管代理，负责协调任务。基于用户输入，选择合适的代理：
- 如果用户提到吃了什么、饮食记录，如'我早餐吃了...'或'今天午餐是...'：dietary
- 如果用户提到运动、锻炼记录、图片、截图、OCR等与图像提取相关的内容：exercise
- 如果用户请求生成报告、查看分析，如'生成今天的报告'：report
- 如果用户查询已有信息，如查询哪一天吃了什么，或者查询哪一天做了什么运动：query
- 其他不确定或一般查询：general

用户输入: {content}

只返回代理名称，不要添加额外解释。"""
    response = llm.invoke(prompt)
    # 使用正则表达式移除<think>标签
    cleaned_response = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
    agent = cleaned_response
    return {"next_agent": agent}