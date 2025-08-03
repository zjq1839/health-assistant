import re
import json
from langchain_ollama import ChatOllama
from core.state import State
from database import add_exercise
import datetime
from .config import llm


import easyocr

def extract_exercise_info(state: State):
    last_message = state["messages"][-1].content
    path_match = re.search(r'/[\w/\.]+\.(jpg|jpeg|png|gif)', last_message, re.IGNORECASE)
    if path_match:
        return extract_exercise_info_from_ocr(state)
    else:
        content = last_message
        # 原文本提取逻辑
        prompt = f"""从用户输入中提取运动类型、持续时间（分钟）和描述，返回JSON格式。

用户输入: {content}

提取规则：
1. 运动类型判断：如跑步、游泳、瑜伽等，如果没有指定，默认为\"其他\"
2. 持续时间：提取数字，如果没有，默认为0
3. 描述：提取额外信息

返回格式：{{\"exercise_type\": \"运动类型\", \"duration\": 数字, \"description\": \"描述\"}}

示例：
- \"跑了30分钟\" -> {{\"exercise_type\": \"跑步\", \"duration\": 30, \"description\": \"\"}}
- \"游泳1小时，很累\" -> {{\"exercise_type\": \"游泳\", \"duration\": 60, \"description\": \"很累\"}}

请只返回JSON："""
        response = llm.invoke(prompt)
        cleaned_response = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
        try:
            result = json.loads(cleaned_response)
            exercise_type = result.get("exercise_type", "其他")
            duration = result.get("duration", 0)
            description = result.get("description", "")
        except (json.JSONDecodeError, AttributeError):
            exercise_type = "其他"
            duration_match = re.search(r'(\d+)分钟|(\d+)小时', content)
            duration = int(duration_match.group(1)) if duration_match else 0
            if '小时' in content: duration *= 60
            description = content
        return {"exercise_type": exercise_type, "exercise_duration": duration, "exercise_description": description}

def extract_exercise_info_from_ocr(state: State):
    last_message = state["messages"][-1].content
    path_match = re.search(r'/[\w/\.]+\.(jpg|jpeg|png|gif)', last_message, re.IGNORECASE)
    if not path_match:
        return {"messages": [("ai", "请提供有效的图片路径。")]}
    image_path = path_match.group(0)
    reader = easyocr.Reader(['ch_sim', 'en'])
    result = reader.readtext(image_path, detail=0)
    extracted_text = ' '.join(result)
    prompt = f"""从以下OCR提取的文本中提取运动类型、持续时间（分钟）和描述，返回JSON格式。

文本: {extracted_text}

返回格式：{{'exercise_type': '运动类型', 'duration': 数字, 'description': '描述'}}

请只返回JSON："""
    response = llm.invoke(prompt)
    cleaned_response = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
    try:
        result = json.loads(cleaned_response)
        exercise_type = result.get('exercise_type', '其他')
        duration = result.get('duration', 0)
        description = result.get('description', '')
    except:
        exercise_type = '其他'
        duration = 0
        description = extracted_text
    return {"exercise_type": exercise_type, "exercise_duration": duration, "exercise_description": description}

def record_exercise(state: State):
    today = datetime.date.today().isoformat()
    add_exercise(today, state['exercise_type'], state['exercise_duration'], state['exercise_description'])
    source = "从图片提取的" if 'ocr' in state.get('next_agent', '') else ""
    return {"messages": [("ai", f"{source}{state['exercise_type']}运动已记录，持续 {state['exercise_duration']} 分钟")]}