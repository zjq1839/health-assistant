import re
import json
import datetime
from langchain_ollama import ChatOllama
from core.state import State
from database import add_exercise
from .config import llm
import easyocr

def extract_ocr_exercise_info(state: State):
    last_message = state["messages"][-1].content
    # 假设用户输入包含图片路径，如"从图片 /path/to/image.png 提取"
    path_match = re.search(r'/[\w/\.]+\.png', last_message)
    if not path_match:
        return {"messages": [("ai", "请提供有效的图片路径。")]}
    image_path = path_match.group(0)
    reader = easyocr.Reader(['ch_sim', 'en'])  # 支持中文和英文
    result = reader.readtext(image_path, detail=0)
    extracted_text = ' '.join(result)
    
    # 使用LLM解析提取的文本
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

def record_ocr_exercise(state: State):
    today = datetime.date.today().isoformat()
    add_exercise(today, state['exercise_type'], state['exercise_duration'], state['exercise_description'])
    return {"messages": [("ai", f"从图片提取的{state['exercise_type']}运动已记录，持续 {state['exercise_duration']} 分钟")]}