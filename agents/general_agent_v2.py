from core.agent_protocol import BaseAgent, AgentResponse
from core.enhanced_state import EnhancedState, IntentType
from core.agent_protocol import LLMService

class GeneralAgentV2(BaseAgent):
    """通用智能体 V2 - 处理未知意图和一般性对话"""
    
    def __init__(self, llm_service: LLMService):
        super().__init__(
            name="general",
            intents=[IntentType.UNKNOWN],
            llm_service=llm_service
        )
    
    def run(self, state: EnhancedState) -> AgentResponse:
        """处理通用对话"""
        try:
            # 获取对话历史
            messages = state.get('messages', [])
            history = "\n".join([msg.get('content', '') for msg in messages])
            
            # 获取用户的最后一条消息
            last_user_msg = messages[-1] if messages else {}
            user_content = last_user_msg.get('content', '')
            
            # 增强的提示词
            prompt = f"""你是一个温暖友好的健康助手，专门帮助用户管理饮食和运动。你的目标是：
1. 理解用户的真实需求和情感
2. 提供有用的信息和建议
3. 保持积极、支持和鼓励的态度
4. 让用户感到被理解和关心

特别注意：
- 如果用户询问模糊，尝试理解其背后的需求
- 始终保持友好和支持的语气

对话历史：
{history}

请根据用户的最新消息，给出温暖、有帮助的回复。"""
            
            # 调用 LLM 服务
            response = self.llm_service.generate_response(prompt, "")
            
            # 添加助手回复到状态
            if 'messages' not in state:
                state['messages'] = []
            state['messages'].append({"role": "assistant", "content": response})
            
            return self._create_success_response(response)
            
        except Exception as e:
            # 错误处理
            error_msg = f"处理请求时发生错误：{str(e)}"
            return self._create_error_response(error_msg)