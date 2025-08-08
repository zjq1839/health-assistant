import re
import json
import datetime
from .config import llm, get_llm
from langchain_core.tools import tool

def _build_tool_schema(name: str, default_values: dict) -> dict:
    """根据默认值动态生成 function calling 的 Tool Schema"""
    properties = {}
    required = []
    for k, v in default_values.items():
        # 仅根据类型简单推断，复杂类型统一用 string
        json_type = "number" if isinstance(v, (int, float)) else "string"
        properties[k] = {"type": json_type, "description": k}
        required.append(k)
    return {
        "name": name,
        "description": "从文本中抽取结构化信息",  # 简短描述即可
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def extract_info_with_llm(content: str, prompt_template: str, default_values: dict) -> dict:
    """使用LLM提取信息并处理响应，优先通过 Function Calling 获取结构化结果。"""
    prompt = prompt_template.format(content=content)

    # 动态构造工具 schema，并绑定到 LLM
    tool_schema = _build_tool_schema("extract_info", default_values)
    llm_instance = get_llm("extraction", lite=True).bind_tools([tool_schema])

    try:
        response = llm_instance.invoke(prompt)

        # LangChain Function Calling 的结果通常在 additional_kwargs 中
        tool_calls = response.additional_kwargs.get("tool_calls") if hasattr(response, "additional_kwargs") else None
        if tool_calls:
            # 取第一条调用结果
            args = tool_calls[0].get("args", {})
            if isinstance(args, dict) and args:
                return args

        # 如果没有 tool_calls，则尝试直接解析 content
        cleaned_response = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()
        return json.loads(cleaned_response)
    except Exception:
        # 任意异常均回退到旧逻辑
        try:
            cleaned = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip() if "response" in locals() else ""
            return json.loads(cleaned)
        except Exception:
            return default_values

@tool
def get_current_date():
    """获取当前日期，格式为YYYY-MM-DD"""
    return datetime.date.today().isoformat()
@tool
def parse_date_with_llm(user_input: str) -> str:
    """使用LLM智能解析用户输入中的日期表达"""
    today = datetime.date.today()
    # 直接交由 LLM 解析日期，不再进行本地规则优先处理
    prompt = f"""你是一个日期解析助手。请根据用户输入解析出具体的日期，返回YYYY-MM-DD格式。

当前日期：{today.isoformat()}
当前年份：{today.year}

用户输入：{user_input}

如果用户使用相对日期（例如“今天”、“昨天”），请换算为具体日期。
如果用户输入无法解析，返回今天的日期。

只返回日期，格式：YYYY-MM-DD"""
    

    
    try:
        response = llm.bind_tools([get_current_date]).invoke(prompt)
        parsed_date = response.content.strip()
        
        # 验证返回的日期格式
        datetime.datetime.strptime(parsed_date, '%Y-%m-%d')
        print(parsed_date)
        return parsed_date
    except (ValueError, AttributeError):
        # 如果LLM解析失败，回退到简单解析
        return parse_date(user_input)

def parse_date(date_str: str) -> str:
    """解析多种格式的日期字符串（备用方法）。"""
    today = datetime.date.today()
    if not date_str or "今天" in date_str:
        return today.isoformat()
    if "昨天" in date_str:
        return (today - datetime.timedelta(days=1)).isoformat()
    if "前天" in date_str:
        return (today - datetime.timedelta(days=2)).isoformat()
    if "大前天" in date_str:
        return (today - datetime.timedelta(days=3)).isoformat()
    
    # 更严格的月日解析，避免漏掉某些格式
    match = re.search(r'(\d+)月(\d+)[号日]?', date_str)
    if match:
        month, day = int(match.group(1)), int(match.group(2))
        # 假设是当前年份，除非提供了年份信息
        year = today.year
        try:
            return datetime.date(year, month, day).isoformat()
        except ValueError:
            # 如果日期无效（如2月30日），返回今天
            return today.isoformat()
    
    # 尝试直接解析
    try:
        # 支持 YYYY-MM-DD, MM-DD等格式
        if len(date_str.split('-')) == 2:
            date_str = f"{today.year}-{date_str}"
        return datetime.datetime.strptime(date_str, '%Y-%m-%d').date().isoformat()
    except ValueError:
        return today.isoformat()