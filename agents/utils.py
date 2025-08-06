import re
import json
import datetime
from .config import llm

def extract_info_with_llm(content: str, prompt_template: str, default_values: dict) -> dict:
    """使用LLM提取信息并处理响应。"""
    prompt = prompt_template.format(content=content)
    response = llm.invoke(prompt)
    cleaned_response = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
    
    try:
        return json.loads(cleaned_response)
    except (json.JSONDecodeError, AttributeError):
        return default_values

def parse_date(date_str: str) -> str:
    """解析多种格式的日期字符串。"""
    today = datetime.date.today()
    if not date_str or "今天" in date_str:
        return today.isoformat()
    if "昨天" in date_str:
        return (today - datetime.timedelta(days=1)).isoformat()
    if "前天" in date_str:
        return (today - datetime.timedelta(days=2)).isoformat()
    
    match = re.search(r'(\d+)月(\d+)[号日]', date_str)
    if match:
        month, day = int(match.group(1)), int(match.group(2))
        # 假设是当前年份，除非提供了年份信息
        year = today.year
        return f"{year}-{month:02d}-{day:02d}"
    
    # 尝试直接解析
    try:
        # 支持 YYYY-MM-DD, MM-DD等格式
        if len(date_str.split('-')) == 2:
            date_str = f"{today.year}-{date_str}"
        return datetime.datetime.strptime(date_str, '%Y-%m-%d').date().isoformat()
    except ValueError:
        return today.isoformat()