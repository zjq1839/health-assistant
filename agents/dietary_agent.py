import re
import json
from langchain_ollama import ChatOllama
from core.state import State
from database import add_meal
import datetime
from .config import llm


def parse_intent(state: State):
    last_message = state["messages"][-1]
    content = last_message[1] if isinstance(last_message, tuple) else last_message.content

    # 使用LLM来判断意图
    prompt = f"""请分析用户输入的意图，只返回以下四个选项之一：record_meal、record_exercise、generate_report、unknown

用户输入: {content}

判断规则：
1. 如果用户提到了食物、饮料、吃、喝等与饮食相关的内容，返回：record_meal
2. 如果用户提到了运动、跑步、游泳、锻炼等与运动相关的内容，返回：record_exercise
3. 如果用户要求生成报告、分析、总结等，返回：generate_report  
4. 其他情况返回：unknown

示例：
- "晚上吃了200g牛排" -> record_meal
- "跑了30分钟" -> record_exercise
- "生成今天的报告" -> generate_report
- "你好" -> unknown

请只返回一个词："""

    response = llm.invoke(prompt)
    print(response)
    # 清理LLM响应中的think标签
    cleaned_response = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
    intent = cleaned_response.replace("'", "").replace('"', '').strip()
    

    if intent in ["record_meal", "record_exercise", "generate_report"]:
        return {"intent": intent}
    else:
        return {"intent": "unknown"}

def extract_meal_info(state: State):
    last_message = state["messages"][-1]
    content = last_message[1] if isinstance(last_message, tuple) else last_message.content

    # 使用LLM来提取膳食类型和描述
    prompt = f"""从用户输入中提取膳食类型和食物描述，返回JSON格式。

用户输入: {content}

提取规则：
1. 膳食类型判断：
   - 包含"早上"、"早餐"、"早晨" -> "早餐"
   - 包含"中午"、"午餐"、"午饭" -> "午餐" 
   - 包含"晚上"、"晚餐"、"晚饭"、"夜宵" -> "晚餐"
   - 如果没有明确时间，默认为"午餐"

2. 食物描述：提取具体的食物和数量信息

返回格式：{{"meal_type": "膳食类型", "description": "食物描述"}}

示例：
- "晚上吃了200g牛排" -> {{"meal_type": "晚餐", "description": "200g牛排"}}
- "早餐喝了一杯咖啡" -> {{"meal_type": "早餐", "description": "一杯咖啡"}}

请只返回JSON："""

    response = llm.invoke(prompt)
    # 清理think标签
    cleaned_response = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
    
    try:
        # 尝试从LLM的输出中解析JSON
        result = json.loads(cleaned_response)
        meal_type = result.get("meal_type", "午餐")
        description = result.get("description", "")
    except (json.JSONDecodeError, AttributeError) as e:
        # 如果解析失败，则使用简单的规则
        meal_type_match = re.search(r'(早餐|午餐|晚餐)', content)
        meal_type = meal_type_match.group(0) if meal_type_match else "午餐"
        description = re.sub(r'(?i)(早餐|午餐|晚餐|早上|中午|晚上|吃了|喝了)', '', content).strip()

    # 如果没有提取到具体的食物描述，则判断为查询意图
    if not description or description.lower() in ["什么", "啥"]:
        return {"next_agent": "query"}

    return {"meal_type": meal_type, "meal_description": description}

def record_meal(state: State):
    today = datetime.date.today().isoformat()
    add_meal(today, state['meal_type'], state['meal_description'])
    return {"messages": [("ai", f"{state['meal_type']}已记录")]}