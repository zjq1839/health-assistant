"""健康助手主程序 V2

基于重构后的架构：
1. 轻量级意图分类前移
2. Agent协议统一化
3. 结构化工具调用
4. 依赖注入容器
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from core.service_container import setup_container, get_container
from core.agent_protocol import AgentFactory, AgentResponse, AgentResult
from core.lightweight_planner import LightweightPlanner, PlanResult
from core.enhanced_state import EnhancedState, DialogState, IntentType
from utils.logger import logger
from utils.user_experience import UserGuidance


class HealthAssistantV2:
    """健康助手 V2
    
    特点：
    1. 三级决策机制（规则 -> 小模型 -> 大模型）
    2. 统一的Agent协议
    3. 完整的依赖注入
    4. 结构化的证据链
    """
    
    def __init__(self, config_path: str = None):
        # 设置依赖注入容器
        self.container = setup_container(config_path)
        
        # 获取核心服务
        self.planner = self.container.get(LightweightPlanner)
        self.agent_factory = AgentFactory(self.container)
        
        # 初始化状态
        self.state = self._init_state()
        
        # 用户指南
        self.user_guidance = UserGuidance()
        
        logger.info("HealthAssistantV2 initialized successfully")
    
    def _init_state(self) -> EnhancedState:
        """初始化增强状态"""
        return {
            'messages': [],
            'docs': [],
            'intent': None,
            'dialog_state': DialogState(
                current_intent=None,
                intent_confidence=0.0,
                entities={},
                context_summary="",
                turn_history=[]
            ),
            'next_agent': None,
            # 餐食相关字段
            'meal_description': '',
            'meal_type': '',
            'meal_date': '',
            'meal_calories': 0,
            'meal_nutrients': '',
            # 运动相关字段
            'exercise_description': '',
            'exercise_type': '',
            'exercise_duration': 0,
            'exercise_date': '',
            'exercise_calories_burned': 0,
            'exercise_intensity': '',
            # 查询相关字段
            'query_date': '',
            'query_type': '',
            # 报告相关字段
            'report_date': '',
            'report_type': ''
        }
    
    def run(self):
        """运行主循环"""
        print("🏥 健康助手 V2 启动成功！")
        print("💡 输入 'help' 查看使用指南，输入 'quit' 退出")
        print("📊 输入 'stats' 查看系统统计信息")
        print("-" * 50)
        
        # 系统健康检查
        self._perform_health_check()
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n👤 您：").strip()
                
                if not user_input:
                    continue
                
                # 处理特殊命令
                if user_input.lower() == 'quit':
                    print("👋 再见！祝您身体健康！")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'stats':
                    self._show_statistics()
                    continue
                elif user_input.lower() == 'health':
                    self._perform_health_check()
                    continue
                elif user_input.lower() == 'reset':
                    self._reset_conversation()
                    continue
                
                # 处理用户请求
                response = self._process_user_input(user_input)
                
                # 显示响应
                self._display_response(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 程序被用户中断，再见！")
                break
            except EOFError:
                print("\n\n👋 输入结束，再见！")
                break
            except Exception as e:
                logger.error(f"Main loop error: {str(e)}", exc_info=True)
                print(f"❌ 发生错误：{str(e)}")
                print("🔄 请重试或输入 'reset' 重置对话")
                # 如果是 EOF 错误，退出循环
                if "EOF when reading a line" in str(e):
                    break
    
    def _process_user_input(self, user_input: str) -> AgentResponse:
        """处理用户输入"""
        # 1. 添加用户消息到状态
        self.state['messages'].append({"role": "user", "content": user_input})
        
        # 2. 使用轻量级规划器进行意图识别
        plan_result = self.planner.plan(user_input, self._get_context())
        
        logger.info(
            f"Intent planning completed",
            extra={
                'user_input': user_input[:50],
                'intent': plan_result.intent.value if plan_result.intent else None,
                'confidence': plan_result.confidence,
                'method': plan_result.method
            }
        )
        
        # 3. 更新状态
        self.state['intent'] = plan_result.intent
        self.state['dialog_state'].current_intent = plan_result.intent
        self.state['dialog_state'].intent_confidence = plan_result.confidence
        
        # 安全地更新实体信息
        if plan_result.entities and isinstance(plan_result.entities, dict):
            self.state['dialog_state'].entities.update(plan_result.entities)
        else:
            logger.warning(f"Invalid entities in plan_result: {plan_result.entities}")
        
        # 4. 选择合适的Agent
        agent_name = self._select_agent(plan_result)
        
        # 5. 创建Agent并执行
        try:
            logger.info(f"Creating agent: {agent_name}")
            agent = self.agent_factory.create_agent(agent_name)
            logger.info(f"Agent created successfully: {agent.name if agent else None}")
        except Exception as e:
            logger.error(f"Agent creation failed: {str(e)}", exc_info=True)
            return self._create_error_response(f"Agent创建失败: {str(e)}")
        
        # 6. 验证Agent是否能处理该意图
        if plan_result.intent and not agent.can_handle(plan_result.intent):
            logger.warning(f"Agent {agent_name} cannot handle intent {plan_result.intent}")
            # 降级到通用Agent
            agent = self.agent_factory.create_agent("general")
        
        # 7. 执行Agent
        try:
            logger.info(f"About to run agent: {agent.name}")
            response = agent.run(self.state)
            logger.info(f"Agent response: status={response.status if response else None}, message={response.message[:50] if response and response.message else None}")
        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}", exc_info=True)
            response = self._create_error_response(f"Agent执行失败: {str(e)}")
        
        # 8. 更新对话历史
        self._update_dialog_history(user_input, response)
        
        # 9. 处理重定向
        if response.status == AgentResult.REDIRECT and response.next_agent:
            logger.info(f"Redirecting to agent: {response.next_agent}")
            redirect_agent = self.agent_factory.create_agent(response.next_agent)
            response = redirect_agent.run(self.state)
        
        return response
    
    def _get_context(self) -> str:
        """获取对话上下文"""
        # 获取最近的对话历史
        recent_messages = self.state['messages'][-5:]  # 最近5轮对话
        
        context_parts = []
        for msg in recent_messages:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
            elif isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
            else:
                # Skip malformed messages
                continue

            if role == "user":
                context_parts.append(f"用户：{content}")
            elif role == "assistant":
                context_parts.append(f"助手：{content[:100]}...")
        
        return "\n".join(context_parts)
    
    def _create_error_response(self, message: str) -> 'AgentResponse':
        """创建错误响应"""
        from core.agent_protocol import AgentResponse, AgentResult
        return AgentResponse(
            status=AgentResult.ERROR,
            message=message,
            data={}
        )
    
    def _select_agent(self, plan_result: PlanResult) -> str:
        """根据规划结果选择Agent"""
        if not plan_result.intent:
            return "general"
        
        # 意图到Agent的映射
        intent_to_agent = {
            IntentType.RECORD_MEAL: "dietary",
            IntentType.RECORD_EXERCISE: "exercise",
            IntentType.QUERY_DATA: "query",
            IntentType.GENERATE_REPORT: "report",
            IntentType.ADVICE: "advice",
            IntentType.UNKNOWN: "general"
        }
        
        return intent_to_agent.get(plan_result.intent, "general")
    
    def _update_dialog_history(self, user_input: str, response: AgentResponse):
        """更新对话历史"""
        # 添加助手回复到消息历史
        self.state['messages'].append({"role": "assistant", "content": response.message})
        
        # 更新对话状态
        turn_info = {
            'user_input': user_input,
            'intent': self.state['dialog_state'].current_intent.value if self.state['dialog_state'].current_intent else None,
            'agent_used': response.data.get('agent_name', 'unknown') if response.data else 'unknown',
            'status': response.status.value,
            'evidence_count': len(response.evidence) if response.evidence else 0
        }
        
        self.state['dialog_state'].turn_history.append(turn_info)
        
        # 保持历史长度
        if len(self.state['dialog_state'].turn_history) > 10:
            self.state['dialog_state'].turn_history = self.state['dialog_state'].turn_history[-10:]
    
    def _display_response(self, response: AgentResponse):
        """显示Agent响应"""
        if response is None:
            print("\n🤖 助手：❌ 处理请求时发生内部错误")
            return
            
        # 状态图标
        status_icons = {
            AgentResult.SUCCESS: "✅",
            AgentResult.ERROR: "❌",
            AgentResult.REDIRECT: "🔄",
            AgentResult.PARTIAL: "⚠️"
        }
        
        icon = status_icons.get(response.status, "ℹ️")
        
        print(f"\n🤖 助手：{icon} {response.message}")
        
        # 显示证据信息（如果有）
        if response.evidence and len(response.evidence) > 0:
            print(f"\n📋 数据来源：")
            for i, evidence in enumerate(response.evidence[:3], 1):  # 最多显示3个证据
                confidence_bar = "█" * int(evidence['confidence'] * 10)
                print(f"  {i}. {evidence['source']}: {evidence['content'][:50]}... (置信度: {confidence_bar})")
            
            if len(response.evidence) > 3:
                print(f"  ... 还有 {len(response.evidence) - 3} 个数据源")
        
        # 显示警告信息（如果有）
        if response.data and 'warnings' in response.data:
            warnings = response.data['warnings']
            if warnings and isinstance(warnings, dict):
                print(f"\n⚠️ 注意事项：")
                for field, warning in warnings.items():
                    print(f"  • {warning}")
    
    def _show_help(self):
        """显示帮助信息"""
        print("\n" + "="*60)
        print("🏥 健康助手 V2 使用指南")
        print("="*60)
        
        print("\n📝 基本命令：")
        print("  • help  - 显示此帮助信息")
        print("  • quit  - 退出程序")
        print("  • stats - 显示系统统计信息")
        print("  • health - 执行系统健康检查")
        print("  • reset - 重置对话历史")
        
        print("\n🍽️ 饮食记录示例：")
        examples = self.user_guidance.get_examples_by_intent('add_meal')
        for example in examples[:3]:
            print(f"  • {example}")
        
        print("\n🏃 运动记录示例：")
        examples = self.user_guidance.get_examples_by_intent('add_exercise')
        for example in examples[:3]:
            print(f"  • {example}")
        
        print("\n📊 查询数据示例：")
        examples = self.user_guidance.get_examples_by_intent('query_data')
        for example in examples[:3]:
            print(f"  • {example}")
        
        print("\n📈 生成报告示例：")
        examples = self.user_guidance.get_examples_by_intent('generate_report')
        for example in examples[:2]:
            print(f"  • {example}")
        
        print("\n💡 健康咨询示例：")
        examples = self.user_guidance.get_examples_by_intent('advice')
        for example in examples[:2]:
            print(f"  • {example}")
        
        print("\n" + "="*60)
    
    def _show_statistics(self):
        """显示系统统计信息"""
        print("\n" + "="*50)
        print("📊 系统统计信息")
        print("="*50)
        
        # 容器统计
        container_stats = self.container.get_statistics()
        print("\n🔧 服务容器：")
        for key, value in container_stats.items():
            print(f"  • {key}: {value}")
        
        # 对话统计
        dialog_state = self.state['dialog_state']
        print("\n💬 对话状态：")
        print(f"  • 当前意图: {dialog_state.current_intent.value if dialog_state.current_intent else 'None'}")
        print(f"  • 意图置信度: {dialog_state.intent_confidence:.2f}")
        print(f"  • 对话轮数: {len(dialog_state.turn_history)}")
        print(f"  • 消息总数: {len(self.state['messages'])}")
        
        # 意图分布
        intent_counts = {}
        for turn in dialog_state.turn_history:
            intent = turn.get('intent', 'unknown')
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        if intent_counts:
            print("\n🎯 意图分布：")
            for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {intent}: {count}")
        
        print("\n" + "="*50)
    
    def _perform_health_check(self):
        """执行系统健康检查"""
        print("\n🔍 执行系统健康检查...")
        
        health_results = self.container.health_check()
        
        print("\n📋 检查结果：")
        all_healthy = True
        for service, status in health_results.items():
            icon = "✅" if status else "❌"
            print(f"  {icon} {service}: {'正常' if status else '异常'}")
            if not status:
                all_healthy = False
        
        if all_healthy:
            print("\n🎉 所有服务运行正常！")
        else:
            print("\n⚠️ 部分服务存在问题，可能影响功能使用")
    
    def _reset_conversation(self):
        """重置对话"""
        self.state = self._init_state()
        print("\n🔄 对话历史已重置")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="健康助手 V2 - 基于重构架构的智能健康管理系统"
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='启用调试模式'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置日志级别
    import logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    if args.debug:
        print("🐛 调试模式已启用")
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 创建健康助手实例
        assistant = HealthAssistantV2(config_path=args.config)
        
        # 运行主循环
        assistant.run()
        
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}", exc_info=True)
        print(f"❌ 程序启动失败：{str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()