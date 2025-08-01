from langgraph.graph import StateGraph, START, END
from core.state import State
from agents.dietary_agent import extract_meal_info, record_meal
from agents.exercise_agent import extract_exercise_info, record_exercise
from agents.report_agent import extract_report_date, prepare_report_data, prompt_divide, retriever, generator
from agents.general_agent import unknown_intent
from agents.supervisor_agent import supervisor
from agents.query_agent import extract_query_params, query_database
from database import init_db

# 初始化数据库
init_db()

graph_builder = StateGraph(State)

graph_builder.add_node("extract_meal_info", extract_meal_info)
graph_builder.add_node("record_meal", record_meal)
graph_builder.add_node("extract_exercise_info", extract_exercise_info)
graph_builder.add_node("record_exercise", record_exercise)
graph_builder.add_node("extract_report_date", extract_report_date)
graph_builder.add_node("prepare_report_data", prepare_report_data)
graph_builder.add_node("prompt_divide_node", prompt_divide)
graph_builder.add_node("retriever_node", retriever)
graph_builder.add_node("generator_node", generator)
graph_builder.add_node("extract_query_params", extract_query_params)
graph_builder.add_node("query_database", query_database)
graph_builder.add_node("unknown_intent", unknown_intent)

graph_builder.add_node("supervisor", supervisor)
graph_builder.add_edge(START, "supervisor")

def route_agent(state: State):
    next_agent = state.get("next_agent", "general")
    if next_agent == "dietary":
        return "extract_meal_info"
    elif next_agent == "exercise":
        return "extract_exercise_info"
    elif next_agent == "report":
        return "extract_report_date"
    elif next_agent == "query":
        return "extract_query_params"
    else:
        return "unknown_intent"

graph_builder.add_conditional_edges("supervisor", route_agent, {
    "extract_meal_info": "extract_meal_info",
    "extract_exercise_info": "extract_exercise_info",
    "extract_report_date": "extract_report_date",
    "extract_query_params": "extract_query_params",
    "unknown_intent": "unknown_intent"
})

def route_after_meal_extraction(state: State):
    if state.get("next_agent") == "query":
        return "extract_query_params"
    return "record_meal"

graph_builder.add_conditional_edges(
    "extract_meal_info",
    route_after_meal_extraction,
    {
        "record_meal": "record_meal",
        "extract_query_params": "extract_query_params",
    }
)
graph_builder.add_edge("record_meal", END)
graph_builder.add_edge("extract_exercise_info", "record_exercise")
graph_builder.add_edge("record_exercise", END)
graph_builder.add_edge("extract_report_date", "prepare_report_data")

def should_generate_report(state: State):
    if not state.get('full_meal_description'):
        return END
    return "prompt_divide_node"

graph_builder.add_conditional_edges("prepare_report_data", should_generate_report, {
    "prompt_divide_node": "prompt_divide_node",
    END: END
})

graph_builder.add_edge("prompt_divide_node", "retriever_node")
graph_builder.add_edge("retriever_node", "generator_node")
graph_builder.add_edge("generator_node", END)
graph_builder.add_edge("extract_query_params", "query_database")
graph_builder.add_edge("query_database", END)
graph_builder.add_edge("unknown_intent", END)

graph = graph_builder.compile()

def main():
    from IPython.display import Image, display
    import os
    # 确保存在保存图片的目录
    os.makedirs('graph_images', exist_ok=True)
    
    try:
        # 生成图片并保存到文件
        graph_png = graph.get_graph().draw_mermaid_png()
        image_path = os.path.join('graph_images', 'workflow_graph.png')
        with open(image_path, 'wb') as f:
            f.write(graph_png)
        print(f"图片已保存到: {image_path}")
        
        # 如果在notebook环境中，同时显示图片
        try:
            display(Image(graph_png))
        except:
            pass
    except Exception as e:
        print(f"保存图片时发生错误: {str(e)}")
    print("你好！我是你的饮食分析助手。你可以告诉我你吃了什么，或者让我为你生成一份饮食报告。输入 '退出' 来结束对话。")
    while True:
        user_input = input("> ")
        if user_input.lower() == '退出':
            break
        response = graph.invoke({"messages": [("user", user_input)]})
        last_message = response['messages'][-1]
        if isinstance(last_message, tuple):
            print(last_message[1])
        else:
            print(last_message.content)
    


if __name__ == "__main__":
    main()
