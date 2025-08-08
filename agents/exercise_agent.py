import re
import datetime
from core.state import State
from database import add_exercise
from .utils import extract_info_with_llm
from .prompts import EXERCISE_PROMPT, OCR_EXERCISE_PROMPT
from utils.performance import cache_llm_response, performance_monitor
from utils.logger import log_error
import easyocr

def extract_exercise_info(state: State):
    last_message = state["messages"][-1].content
    path_match = re.search(r'/[\w/\.]+\.(jpg|jpeg|png|gif)', last_message, re.IGNORECASE)
    if path_match:
        # If an image path is found, delegate to the OCR extraction function
        ocr_result = extract_exercise_info_from_ocr(state)
        # The next agent should be 'record_exercise' to process the OCR result
        ocr_result['next_agent'] = 'record_exercise'
        return ocr_result
    else:
        content = last_message
        
        # Check if the user's intent is to query past exercises
        query_keywords = ['做了什么运动', '运动了什么', '锻炼了什么', '今天运动', '昨天运动', '运动记录', '查看运动', '运动情况']
        if any(keyword in content for keyword in query_keywords):
            return {"next_agent": "query"}
        
        # Extract exercise info from text using the generalized LLM utility
        extracted_data = extract_info_with_llm(
            content, 
            EXERCISE_PROMPT, 
            default_values={"exercise_type": "其他", "duration": 0, "description": content}
        )
        return {
            "exercise_type": extracted_data.get("exercise_type"),
            "exercise_duration": extracted_data.get("duration"),
            "exercise_description": extracted_data.get("description"),
            "next_agent": "record_exercise"
        }

@performance_monitor
@cache_llm_response
def extract_exercise_info_from_ocr(state: State):
    last_message = state["messages"][-1].content
    path_match = re.search(r'/[\w/\.]+\.(jpg|jpeg|png|gif)', last_message, re.IGNORECASE)
    if not path_match:
        return {"messages": [("ai", "请提供有效的图片路径。 ")]}
    
    image_path = path_match.group(0)
    try:
        # Initialize the OCR reader and extract text from the image
        reader = easyocr.Reader(['ch_sim', 'en'])
        result = reader.readtext(image_path, detail=0)
        extracted_text = ' '.join(result)
    except Exception as e:
        log_error("ocr_error", f"OCR processing failed: {str(e)}", {"image_path": image_path})
        return {"messages": [("ai", f"图片读取失败: {e}")]}

    # Extract exercise info from OCR text using the generalized LLM utility
    extracted_data = extract_info_with_llm(
        extracted_text, 
        OCR_EXERCISE_PROMPT, 
        default_values={"exercise_type": "其他", "duration": 0, "description": extracted_text}
    )
    return {
        "exercise_type": extracted_data.get("exercise_type"),
        "exercise_duration": extracted_data.get("duration"),
        "exercise_description": extracted_data.get("description")
    }

def record_exercise(state: State):
    today = datetime.date.today().isoformat()
    # Add the extracted exercise record to the database
    add_exercise(today, state['exercise_type'], state['exercise_duration'], state['exercise_description'])
    # Determine the source of the information for the confirmation message
    source_match = re.search(r'/[\w/\.]+\.(jpg|jpeg|png|gif)', state["messages"][-1].content, re.IGNORECASE)
    source_info = "从图片提取的" if source_match else ""
    return {"messages": [("ai", f"{source_info}{state['exercise_type']}运动已记录，持续 {state['exercise_duration']} 分钟")]}