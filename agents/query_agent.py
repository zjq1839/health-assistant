import json
import re
import datetime
from langchain_ollama import ChatOllama
from core.state import State
from database import get_meals_by_date, get_exercises_by_date
from .config import llm

def extract_query_params(state: State):
    """Extracts date and query type (dietary or exercise) from user input."""
    content = state["messages"][-1].content

    prompt = f"""从用户输入中提取查询的日期和类型（饮食或运动）。

用户输入: {content}

规则:
1. 日期：识别'今天'、'昨天'或具体日期（YYYY-MM-DD）。如果未提及，默认为'今天'。
2. 类型：如果提及'吃'、'喝'、'饮食'、'早餐'、'午餐'、'晚餐'，则为'dietary'。如果提及'运动'、'跑步'、'锻炼'，则为'exercise'。如果不确定，则为'unknown'。

返回JSON格式: {{"date": "YYYY-MM-DD", "query_type": "dietary" | "exercise" | "unknown"}}

示例:
- "我今天早上吃了什么" -> {{"date": "{datetime.date.today().isoformat()}", "query_type": "dietary"}}
- "昨天的运动记录" -> {{"date": "{(datetime.date.today() - datetime.timedelta(days=1)).isoformat()}", "query_type": "exercise"}}

请只返回JSON："""

    response = llm.invoke(prompt)
    cleaned_response = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()

    try:
        params = json.loads(cleaned_response)
        query_date = params.get("date", datetime.date.today().isoformat())
        query_type = params.get("query_type", "unknown")
    except (json.JSONDecodeError, AttributeError):
        query_date = datetime.date.today().isoformat()
        if any(kw in content for kw in ['吃', '喝', '饮食', '早餐', '午餐', '晚餐']):
            query_type = 'dietary'
        elif any(kw in content for kw in ['运动', '跑步', '锻炼']):
            query_type = 'exercise'
        else:
            query_type = 'unknown'

    return {"query_date": query_date, "query_type": query_type}

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