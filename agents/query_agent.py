import json
import re
import datetime
from langchain_core import tools
from langchain_ollama import ChatOllama
from core.state import State
from database import get_meals_by_date, get_exercises_by_date
from .config import llm, llm_lite
from langchain_core.tools import tool
def parse_llm_response(response_text, content):
    """解析LLM返回的响应，提取查询参数或使用备选方法"""
    cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    
    try:
        params = json.loads(cleaned_response)
        query_date = params.get("date", datetime.date.today().isoformat())
        query_type = params.get("query_type", "unknown")
    except (json.JSONDecodeError, AttributeError):
        # 备选解析方法
        query_date = datetime.date.today().isoformat()
        if any(kw in content for kw in ['吃', '喝', '饮食', '早餐', '午餐', '晚餐']):
            query_type = 'dietary'
        elif any(kw in content for kw in ['运动', '跑步', '锻炼']):
            query_type = 'exercise'
        else:
            query_type = 'unknown'
            
    return query_date, query_type

def extract_query_params(state: State):
    """Extracts date and query type (dietary or exercise) from user input."""
    content = state["messages"][-1].content

    # 简化的提示词，包含必要规则
    prompt = f"""分析用户输入，提取查询日期和类型。

用户输入: {content}

规则:
1. 日期：'今天'为当前日期，'昨天'为前一天，具体日期用YYYY-MM-DD格式。如果未提及，默认为今天。
2. 类型：饮食相关（如'吃'、'饮食'）为'dietary'，运动相关（如'运动'、'跑步'）为'exercise'，否则'unknown'。

返回格式: {{"date": "YYYY-MM-DD", "query_type": "dietary" | "exercise" | "unknown"}}
"""

    response = llm_lite.bind_tools([gettime]).invoke(prompt)
    print(response.content)
    query_date, query_type = parse_llm_response(response.content, content)
    
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