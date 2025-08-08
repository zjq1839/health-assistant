import argparse
import time
from graph import graph
from core.enhanced_state import create_enhanced_state
from config import config
from utils.logger import logger, log_user_input, log_error
from utils.user_experience import formatter, guidance
from utils.performance import performance_monitor

@performance_monitor
def main():
    parser = argparse.ArgumentParser(description='健康助手')
    parser.add_argument('--no-graph', action='store_true', help='不生成流程图')
    parser.add_argument('--help-mode', action='store_true', help='显示帮助信息')
    args = parser.parse_args()
    
    if args.help_mode:
        print(guidance.get_help_message())
        return
    
    # 使用增强状态管理
    try:
        state = create_enhanced_state()
        logger.info("Using enhanced state management")
    except Exception as e:
        # 降级到传统状态
        state = {"messages": []}
        logger.warning(f"Fallback to legacy state: {e}")
    
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
            
            log_user_input(user_input)
            start_time = time.time()
            
            # 添加用户消息
            state["messages"].append(("user", user_input))
            
            # 使用增强状态管理（如果可用）
            if hasattr(state, 'get') and 'dialog_state' in state:
                # 增强状态已经在supervisor中处理智能截断
                pass
            else:
                # 传统的固定窗口截断
                state["messages"] = state["messages"][-config.dialogue.max_messages:]
            
            try:
                response = graph.invoke(state)
                state = response
                
                if state["messages"]:
                    last_message = state["messages"][-1]
                    if isinstance(last_message, tuple) and last_message[0] == "ai":
                        formatted_response = formatter.format_response(last_message[1])
                        print(f"\n🤖 助手: {formatted_response}")
                    elif hasattr(last_message, 'content'):
                        formatted_response = formatter.format_response(last_message.content)
                        print(f"\n🤖 助手: {formatted_response}")
                
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
            if "Connection refused" in str(e) or "Recursion limit" in str(e):
                print("\n❌ 遇到严重错误，程序退出")
                break

if __name__ == "__main__":
    main()