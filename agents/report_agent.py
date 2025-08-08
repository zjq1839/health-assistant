import re
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from core.state import State
from database import get_meals_by_date
from .config import llm
from .utils import parse_date_with_llm
from config import config

embeddings = OllamaEmbeddings(model=config.knowledge_base.embedding_model)
vector_store = FAISS.load_local(config.knowledge_base.path, embeddings, allow_dangerous_deserialization=True)

def extract_report_date(state: State):
    """使用LLM智能解析用户输入中的报告日期"""
    last_message = state["messages"][-1].content
    
    # 使用LLM进行智能日期解析
    report_date = parse_date_with_llm(last_message)
    
    return {"report_date": report_date}

def prepare_report_data(state: State):
    from database import get_exercises_by_date
    meals = get_meals_by_date(state['report_date'])
    exercises = get_exercises_by_date(state['report_date'])
    if not meals and not exercises:
        return {
            "messages": [("ai", f"{state['report_date']}没有任何饮食或运动记录。")]
        }

    full_meal_desc = " ".join([f"{meal[2]}: {meal[3]}" for meal in meals]) if meals else "无饮食记录"
    full_exercise_desc = " ".join([f"{exercise[2]} 持续 {exercise[3]} 分钟: {exercise[4]}" for exercise in exercises]) if exercises else "无运动记录"
    full_description = f"饮食: {full_meal_desc} 运动: {full_exercise_desc}"
    return {
        "full_meal_description": full_meal_desc,
        "full_exercise_description": full_exercise_desc,
        "messages": [("user", full_description)]
    }

def prompt_divide(state: State):
    last_message = state["messages"][-1]
    content = last_message[1] if isinstance(last_message, tuple) else last_message.content
    prompt = "请你按照用户的描述，将用户的描述划分为以一种食品为一段的描述，不同食品的描述中间以空格隔开，下面是一个例子'我吃了一个100g的椰子冻，还有一碗绿豆沙，大概200g，里面有50g绿豆和150g水'可以划分成'100g椰子冻 200g绿豆沙，里面有50g绿豆和150g水'，请你严格按照这个格式输出，不要添加任何额外的内容"
    messages = [("system", prompt), ("user", content)]
    response = llm.invoke(messages)
    cleaned_content = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
    print(cleaned_content)
    return {"messages": [("user", cleaned_content)]}

def retriever(state: State):
    last_message = state["messages"][-1]
    content = last_message[1] if isinstance(last_message, tuple) else last_message.content
    queries = content.split()
    docs = []
    for q in queries:
        if q.strip():
            samples = vector_store.similarity_search(q, k=config.vector_search.k)
            for doc in samples:
                docs.append(doc.page_content)
    # 去重
    docs = list(set(docs))
    return {"docs": docs}

def generator(state: State):
    last_message_content = state["messages"][-1][1] if isinstance(state["messages"][-1], tuple) else state["messages"][-1].content
    docs = state["docs"]
    system_message = "你是一个专业的健康分析助手，你的任务是根据用户的饮食和运动描述，查询相关的营养和健康信息，给出综合分析和建议，包括热量摄入、消耗、营养平衡等，对于运动信息，基于你的知识进行总结并提出建议。你可以参照以下信息：{docs}，确保不要捏造信息，如果你已知的信息不足以完成任务，请说明。".format(docs="\n".join(docs))
    new_messages = [("system", system_message), ("user", last_message_content)]
    response = llm.invoke(new_messages)
    print(response)
    cleaned_content = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
    return {"messages": [("ai", cleaned_content)]}