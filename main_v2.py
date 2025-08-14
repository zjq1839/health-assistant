"""健康助手主程序 V2

基于重构后的架构：
1. 轻量级意图分类前移
2. Agent协议统一化
3. 结构化工具调用
4. 依赖注入容器
"""

import sys
import re
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from core.service_container import setup_container, LLMService
from core.agent_protocol import AgentFactory, AgentResponse, AgentResult
from core.lightweight_planner import LightweightPlanner, PlanResult
from core.enhanced_state import EnhancedState, DialogState, IntentType
from utils.logger import logger
from utils.user_experience import UserGuidance
from utils.common_parsers import intent_to_agent_mapping
from core.multi_requirement_parser import MultiRequirementParser


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
        
        # 多需求解析器
        try:
            llm = self.container.get(LLMService)
        except Exception as e:
            logger.warning(f"LLMService unavailable for multi-requirement parser: {e}, using fallback parser without LLM")
            llm = None
        self.multi_req_parser = MultiRequirementParser(llm)

        
        logger.info("HealthAssistantV2 initialized successfully")
    
    def _init_state(self) -> EnhancedState:
        """初始化增强状态（精简版）"""
        return {
            'messages': [],
            'dialog_state': DialogState(
                current_intent=None,
                intent_confidence=0.0,
                entities={},
                turn_history=[]
            ),
            'turn_id': 0
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
        plan_result = self.planner.plan(user_input, self._get_context(), self.state)
        
        logger.info(
            "Intent planning completed",
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
        agent_name = self._select_agent(plan_result, user_input)
        
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
            # 降级到通用Agent -> 使用建议Agent替代已移除的general
            agent = self.agent_factory.create_agent("advice")
        
        # 7. 执行Agent
        try:
            logger.info(f"About to run agent: {agent.name}")
            response = agent.run(self.state)
            # Sanitize hidden thinking tags from logs
            if response and getattr(response, 'message', None):
                _preview = response.message
                if isinstance(_preview, str):
                    _preview = re.sub(r'<think>.*?</think>', '', _preview, flags=re.DOTALL).strip()
                    _preview = _preview[:50]
            else:
                _preview = None
            logger.info(f"Agent response: status={response.status if response else None}, message={_preview}")
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
        return AgentResponse(
            status=AgentResult.ERROR,
            message=message,
            data={}
        )
    
    def _select_agent(self, plan_result: PlanResult, user_input: str) -> str:
        """根据规划结果选择Agent"""
        if not plan_result.intent:
            return "advice"
        
        # 优先检测是否为多需求查询：当识别出多个需求时，统一路由到 multi_requirement_advice
        # 该 Agent 支持 ADVICE/QUERY/GENERATE_REPORT 三类复合咨询场景
        if self.multi_req_parser is not None:
            try:
                parse_result = self.multi_req_parser.parse(user_input or "")
                req_count = len(getattr(parse_result, 'requirements', []) or [])
                if req_count >= 2:
                    allowed_for_multi = {IntentType.ADVICE, IntentType.QUERY, IntentType.GENERATE_REPORT}
                    if plan_result.intent in allowed_for_multi:
                        return "multi_requirement_advice"
            except Exception:
                # 解析失败时，回退到默认映射
                pass

        # 使用统一的意图到Agent映射
        return intent_to_agent_mapping(plan_result.intent)
    
    def _update_dialog_history(self, user_input: str, response: AgentResponse):
        """更新对话历史"""
        # 添加助手回复到消息历史
        cleaned_msg_for_history = re.sub(r'<think>.*?</think>', '', response.message, flags=re.DOTALL).strip() if isinstance(response.message, str) else response.message
        self.state['messages'].append({"role": "assistant", "content": cleaned_msg_for_history})
        
        # 将最近一轮识别的意图写入 DialogState.turn_history（使用 DialogTurn 对象）
        from datetime import datetime
        from core.enhanced_state import DialogTurn
        intent = self.state['dialog_state'].current_intent or IntentType.UNKNOWN
        confidence = self.state['dialog_state'].intent_confidence or 0.0
        turn_obj = DialogTurn(
            turn_id=(self.state.get('turn_id', 0) or 0) + 1,
            timestamp=datetime.now(),
            user_input=user_input,
            intent=intent,
            confidence=confidence,
            entities=self.state['dialog_state'].entities.copy()
        )
        # 更新 turn_id
        self.state['turn_id'] = turn_obj.turn_id
        
        # 追加到状态的 turn_history
        self.state['dialog_state'].turn_history.append(turn_obj)
        
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
        
        cleaned_message = re.sub(r'<think>.*?</think>', '', response.message, flags=re.DOTALL).strip() if isinstance(response.message, str) else response.message
        print(f"\n🤖 助手：{icon} {cleaned_message}")
        
        # 显示证据信息（如果有）
        if response.evidence and len(response.evidence) > 0:
            print("\n📋 数据来源：")
            for i, evidence in enumerate(response.evidence[:3], 1):  # 最多显示3个证据
                confidence_bar = "█" * int(evidence['confidence'] * 10)
                print(f"  {i}. {evidence['source']}: {evidence['content'][:50]}... (置信度: {confidence_bar})")
            
            if len(response.evidence) > 3:
                print(f"  ... 还有 {len(response.evidence) - 3} 个数据源")
        
        # 显示警告信息（如果有）
        if response.data and 'warnings' in response.data:
            warnings = response.data['warnings']
            if warnings and isinstance(warnings, dict):
                print("\n⚠️ 注意事项：")
                for field, warning in warnings.items():
                    print(f"  • {warning}")

        # 显示Token使用情况
        llm_service = self.container.get(LLMService)


    def _show_help(self):
        """显示帮助信息"""
        print("\n📋 健康助手使用指南：")
        print("1. 📝 记录饮食：'我早餐吃了鸡蛋和牛奶'")
        
        # 系统健康检查
        self._perform_health_check()
        
        # 获取示例
        try:
            meal_examples = self.user_guidance.get_examples_by_intent("record_meal")
            exercise_examples = self.user_guidance.get_examples_by_intent("record_exercise") 
            query_examples = self.user_guidance.get_examples_by_intent("query")
            report_examples = self.user_guidance.get_examples_by_intent("generate_report")
            advice_examples = self.user_guidance.get_examples_by_intent("advice")
            
            print(f"   示例：{meal_examples[0] if meal_examples else '我早餐吃了鸡蛋和牛奶'}")
            print("2. 🏃 记录运动：'我跑步30分钟'")
            print(f"   示例：{exercise_examples[0] if exercise_examples else '我跑步30分钟'}")
            print("3. 📊 查询记录：'查询我昨天的饮食记录'")
            print(f"   示例：{query_examples[0] if query_examples else '查询我昨天的饮食记录'}")
            print("4. 📈 生成报告：'生成本周健康报告'")
            print(f"   示例：{report_examples[0] if report_examples else '生成本周健康报告'}")
            print("5. 💡 获取建议：'推荐一些健康食谱'")
            print(f"   示例：{advice_examples[0] if advice_examples else '推荐一些健康食谱'}")
        except Exception as e:
            logger.warning(f"获取示例失败: {e}")
            # 提供默认示例
            print("   示例：我早餐吃了鸡蛋和牛奶")
            print("2. 🏃 记录运动：'我跑步30分钟'")
            print("   示例：我跑步30分钟")
            print("3. 📊 查询记录：'查询我昨天的饮食记录'")
            print("   示例：查询我昨天的饮食记录")
            print("4. 📈 生成报告：'生成本周健康报告'")
            print("   示例：生成本周健康报告")
            print("5. 💡 获取建议：'推荐一些健康食谱'")
            print("   示例：推荐一些健康食谱")
        
        print("\n🔧 系统命令：")
        print("- help: 显示此帮助")
        print("- stats: 显示统计信息")
        print("- health: 健康检查")
        print("- reset: 重置对话")
        print("- quit: 退出程序")
    
    def _perform_health_check(self):
        """执行系统健康检查"""
        print("\n🔍 系统健康检查中...")
        
        # 检查数据库连接
        try:
            from core.agent_protocol import DatabaseService
            db_service = self.container.get(DatabaseService)
            # 测试数据库连接
            db_service.query_meals(limit=1)
            print("  ✅ 数据库连接正常")
        except Exception as e:
            print(f"  ❌ 数据库连接失败: {str(e)}")
        
        # 检查LLM服务
        try:
            llm_service = self.container.get(LLMService)
            # 测试LLM调用
            test_response = llm_service.generate_response("测试", "")
            if test_response:
                print("  ✅ LLM服务正常")
            else:
                print("  ⚠️ LLM服务响应为空")
        except Exception as e:
            print(f"  ❌ LLM服务异常: {str(e)}")
        
        # 检查轻量级规划器
        try:
            test_plan = self.planner.plan("测试输入", "")
            if test_plan and test_plan.intent:
                print("  ✅ 意图规划器正常")
            else:
                print("  ⚠️ 意图规划器响应异常")
        except Exception as e:
            print(f"  ❌ 意图规划器异常: {str(e)}")
        
        # 检查Agent工厂
        try:
            test_agent = self.agent_factory.create_agent("general")
            if test_agent:
                print("  ✅ Agent工厂正常")
            else:
                print("  ❌ Agent工厂创建失败")
        except Exception as e:
            print(f"  ❌ Agent工厂异常: {str(e)}")
        
        print("✅ 健康检查完成")
    
    def _show_statistics(self):
        """显示系统统计信息"""
        print("\n📊 系统统计信息：")
        
        # 对话统计
        total_turns = len(self.state['dialog_state'].turn_history)
        print(f"  💬 对话轮次：{total_turns}")
        
        if total_turns > 0:
            # 意图分布统计
            intent_counts = {}
            agent_counts = {}
            
            for turn in self.state['dialog_state'].turn_history:
                intent = turn.get('intent', 'unknown')
                agent = turn.get('agent_used', 'unknown')
                
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
            
            print("  🎯 意图分布：")
            for intent, count in intent_counts.items():
                print(f"    - {intent}: {count}")
            
            print("  🤖 Agent使用情况：")
            for agent, count in agent_counts.items():
                print(f"    - {agent}: {count}")
        
        # 数据库统计
        try:
            from core.agent_protocol import DatabaseService
            db_service = self.container.get(DatabaseService)
            
            meals = db_service.query_meals(limit=1000)  # 获取最近1000条记录
            exercises = db_service.query_exercises(limit=1000)
            
            print(f"  🍽️ 饮食记录数：{len(meals)}")
            print(f"  🏃 运动记录数：{len(exercises)}")
            
        except Exception as e:
            print(f"  ❌ 数据库统计失败: {str(e)}")
        
    
    def _reset_conversation(self):
        """重置对话状态"""
        print("\n🔄 重置对话状态...")
        
        # 重置状态
        self.state = self._init_state()
        
        
        print("✅ 对话已重置，可以开始新的会话")


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


# 内联意图到 Agent 的映射，替代已删除的 utils.common_parsers.intent_to_agent_mapping
def intent_to_agent_mapping(intent: IntentType) -> str:
    mapping = {
        IntentType.RECORD_MEAL: "dietary",
        IntentType.RECORD_EXERCISE: "exercise",
        IntentType.GENERATE_REPORT: "report",
        IntentType.QUERY: "query",
        IntentType.ADVICE: "advice",
    }
    return mapping.get(intent, "advice")