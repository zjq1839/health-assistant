from core.state import State
from database import add_meal
import datetime
from .utils import extract_info_with_llm, parse_date_with_llm
from .prompts import DIETARY_PROMPT

def extract_meal_info(state: State):
    last_message = state["messages"][-1]
    content = last_message[1] if isinstance(last_message, tuple) else last_message.content

    extracted_data = extract_info_with_llm(content, DIETARY_PROMPT, {"meal_type": "午餐", "description": "", "date": datetime.date.today().isoformat()})

    if not extracted_data.get("description") or extracted_data.get("description", "").lower() in ["什么", "啥"]:
        return {"next_agent": "query"}

    # 进一步处理和验证提取的数据
    meal_date = parse_date_with_llm(extracted_data.get("date", content))

    return {
        "meal_type": extracted_data.get("meal_type"),
        "meal_description": extracted_data.get("description"),
        "meal_date": meal_date
    }

def record_meal(state: State):
    meal_date = state.get('meal_date', datetime.date.today().isoformat())
    add_meal(meal_date, state['meal_type'], state['meal_description'])
    return {"messages": [("ai", f"{state['meal_type']}已记录")]}