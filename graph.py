import argparse
from langgraph.graph import StateGraph, START, END
from core.state import State
from agents.dietary_agent import extract_meal_info, record_meal
from agents.exercise_agent import extract_exercise_info, record_exercise
from agents.report_agent import extract_report_date, prepare_report_data, prompt_divide, retriever, generator
from agents.general_agent import unknown_intent
from agents.supervisor_agent import supervisor
from agents.query_agent import extract_query_params, query_database
from database import init_db
from config import config
from utils.logger import logger, log_user_input, log_error
from utils.user_experience import formatter, guidance
from utils.performance import performance_monitor
import time

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

def route_after_exercise_extraction(state: State):
    if state.get("next_agent") == "query":
        return "extract_query_params"
    return "record_exercise"

graph_builder.add_conditional_edges(
    "extract_exercise_info",
    route_after_exercise_extraction,
    {
        "record_exercise": "record_exercise",
        "extract_query_params": "extract_query_params",
    }
)
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

@performance_monitor
def main():
    parser = argparse.ArgumentParser(description='健康助手')
    parser.add_argument('--no-graph', action='store_true', help='不生成流程图')
    parser.add_argument('--help-mode', action='store_true', help='显示帮助信息')
    args = parser.parse_args()
    
    # 显示帮助信息
    if args.help_mode:
        print(guidance.get_help_message())
        return
    
    # 初始化状态
    state = {"messages": []}
    
    print("🤖 健康助手已启动！")
    print("💡 输入 'help' 查看使用指南，输入 'quit' 退出。")
    logger.info("Health assistant started")
    
    while True:
        try:
            user_input = input("\n👤 用户: ")
            
            if user_input.lower() == 'quit':
                print("👋 再见！")
                logger.info("Health assistant stopped by user")
                break
            
            if user_input.lower() == 'help':
                print(guidance.get_help_message())
                continue
            
            if not user_input.strip():
                print("⚠️ 请输入有效的内容")
                continue
            
            # 记录用户输入
            log_user_input(user_input)
            start_time = time.time()
            
            # 添加用户消息到状态
            state["messages"].append(("user", user_input))
            
            # 限制消息历史长度
            state["messages"] = state["messages"][-config.MAX_MESSAGES:]
            
            # 运行图
            try:
                response = graph.invoke(state)
                # 更新状态
                state = response
                
                # 显示最后的AI回复
                if state["messages"]:
                    last_message = state["messages"][-1]
                    if isinstance(last_message, tuple) and last_message[0] == "ai":
                        formatted_response = formatter.format_response(last_message[1])
                        print(f"\n🤖 助手: {formatted_response}")
                    elif hasattr(last_message, 'content'):
                        formatted_response = formatter.format_response(last_message.content)
                        print(f"\n🤖 助手: {formatted_response}")
                
                # 记录处理时间
                processing_time = time.time() - start_time
                if processing_time > 3.0:
                    print(f"\n⏱️ 处理时间：{processing_time:.1f}秒")
                
            except Exception as e:
                error_msg = formatter.format_error_message("unknown_error", str(e))
                print(f"\n{error_msg}")
                log_error("graph_execution_error", str(e), {"user_input": user_input})
                
        except KeyboardInterrupt:
            print("\n\n👋 程序被中断，再见！")
            logger.info("Health assistant interrupted by user")
            break
        except EOFError:
            print("\n\n👋 输入结束，再见！")
            logger.info("Health assistant stopped due to EOF")
            break
        except Exception as e:
            error_msg = formatter.format_error_message("unknown_error", str(e))
            print(f"\n{error_msg}")
            log_error("main_loop_error", str(e))
            # 对于严重错误，退出程序
            if "Connection refused" in str(e) or "Recursion limit" in str(e):
                print("\n❌ 遇到严重错误，程序退出")
                break
    
    # 可选：生成并显示流程图
    if not args.no_graph:
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
            print(f"\n📊 流程图已保存到: {image_path}")
            logger.info("Graph saved successfully")
            
            # 如果在notebook环境中，同时显示图片
            try:
                display(Image(graph_png))
            except:
                pass
        except Exception as e:
            print(f"\n⚠️ 生成流程图失败: {e}")
            log_error("graph_generation_error", str(e))
    


if __name__ == "__main__":
    main()
