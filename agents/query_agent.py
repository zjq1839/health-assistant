import json
import re
import datetime
from langchain_core import tools
from langchain_ollama import ChatOllama
from core.state import State
from database import get_meals_by_date, get_exercises_by_date
from .config import llm, llm_lite
from langchain_core.tools import tool
def parse_llm_response(response_text, content, context=""):
    """解析LLM返回的响应，提取查询参数或使用备选方法"""
    cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    
    try:
        params = json.loads(cleaned_response)
        query_date = params.get("date", datetime.date.today().isoformat())
        query_type = params.get("query_type", "unknown")
    except (json.JSONDecodeError, AttributeError):
        # 备选解析方法
        today = datetime.date.today()
        
        # 日期解析
        if "昨天" in content:
            query_date = (today - datetime.timedelta(days=1)).isoformat()
        elif "前天" in content:
            query_date = (today - datetime.timedelta(days=2)).isoformat()
        elif re.search(r'(\d+)月(\d+)[号日]', content):
            match = re.search(r'(\d+)月(\d+)[号日]', content)
            month, day = int(match.group(1)), int(match.group(2))
            query_date = f"2025-{month:02d}-{day:02d}"
        else:
            query_date = today.isoformat()
            
        # 查询类型解析 - 增强上下文处理
        if any(kw in content for kw in ['吃', '喝', '饮食', '早餐', '午餐', '晚餐']):
            query_type = 'dietary'
        elif any(kw in content for kw in ['运动', '跑步', '锻炼']):
            query_type = 'exercise'
        elif content in ['昨天呢', '前天呢', '那天呢'] and context:
            # 基于上下文推断查询类型
            if any(kw in context for kw in ['饮食记录', '吃了什么', '早餐', '午餐', '晚餐']):
                query_type = 'dietary'
            elif any(kw in context for kw in ['运动', '跑步', '锻炼']):
                query_type = 'exercise'
            else:
                query_type = 'unknown'
        else:
            query_type = 'unknown'
            
    return query_date, query_type

def extract_query_params(state: State):
    """Extracts date and query type (dietary or exercise) from user input."""
    last_message = state["messages"][-1]
    # 处理不同的消息格式
    if isinstance(last_message, tuple):
        content = last_message[1]
    else:
        content = last_message.content

    # 获取当前日期信息
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    day_before_yesterday = today - datetime.timedelta(days=2)
    
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
    
    # 简化的提示词，包含必要规则和上下文
    prompt = f"""分析用户输入，结合对话上下文，提取查询日期和类型。

当前日期信息：
- 今天: {today.isoformat()}
- 昨天: {yesterday.isoformat()}
- 前天: {day_before_yesterday.isoformat()}

对话上下文：
{context}

当前用户输入: {content}

规则:
1. 日期提取：
   - '今天'为{today.isoformat()}
   - '昨天'为{yesterday.isoformat()}
   - '前天'为{day_before_yesterday.isoformat()}
   - 具体日期如'8月1号'、'8月1日'等，转换为YYYY-MM-DD格式（当前年份2025年）
   - 如果用户只说"昨天呢"、"前天呢"等，需要结合上下文推断查询类型
   - 如果未提及日期，默认为今天
2. 类型推断：
   - 如果上下文中提到饮食查询，且当前输入是相对日期（如"昨天呢"），则为'dietary'
   - 如果上下文中提到运动查询，且当前输入是相对日期，则为'exercise'
   - 直接提及饮食相关（如'吃'、'饮食'、'早餐'、'午餐'、'晚餐'）为'dietary'
   - 直接提及运动相关（如'运动'、'跑步'）为'exercise'
   - 否则'unknown'

返回格式: {{"date": "YYYY-MM-DD", "query_type": "dietary" | "exercise" | "unknown"}}

示例：
- "我8月1号吃了什么" -> {{"date": "2025-08-01", "query_type": "dietary"}}
- 上下文有饮食查询，当前输入"昨天呢" -> {{"date": "{yesterday.isoformat()}", "query_type": "dietary"}}
- "前天吃了什么" -> {{"date": "{day_before_yesterday.isoformat()}", "query_type": "dietary"}}
"""

    response = llm_lite.bind_tools([gettime]).invoke(prompt)
    query_date, query_type = parse_llm_response(response.content, content, context)
    
    return {"query_date": query_date, "query_type": query_type}

@tool
def gettime():
    """获取当前日期"""
    return {datetime.date.today().isoformat()}

def query_database(state: State):
    """Queries the database based on the extracted parameters and returns a formatted response."""
    query_date = state.get("query_date")
    query_type = state.get("query_type")

    if query_type == 'dietary':
        records = get_meals_by_date(query_date)
        if not records:
            response_message = f"{query_date}没有饮食记录。"
        else:
            response_message = f"{query_date}的饮食记录：\n"
            for r in records:
                response_message += f"- {r[2]}: {r[3]}\n"
    elif query_type == 'exercise':
        records = get_exercises_by_date(query_date)
        if not records:
            response_message = f"{query_date}没有运动记录。"
        else:
            response_message = f"{query_date}的运动记录：\n"
            for r in records:
                response_message += f"- {r[2]}: {r[3]} ({r[4]}分钟)\n"
    else:
        response_message = "抱歉，我无法确定您想查询什么。"

    return {"messages": [("ai", response_message.strip())]}