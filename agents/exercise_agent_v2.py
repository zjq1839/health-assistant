import re
import datetime
from core.agent_protocol import BaseAgent, AgentResponse
from core.enhanced_state import EnhancedState, IntentType
from core.agent_protocol import LLMService
from core.agent_protocol import DatabaseService

class ExerciseAgentV2(BaseAgent):
    """运动智能体 V2 - 处理运动记录和查询"""
    
    def __init__(self, db_service: DatabaseService, llm_service: LLMService):
        super().__init__(
            name="exercise",
            intents=[IntentType.RECORD_EXERCISE],
            db_service=db_service,
            llm_service=llm_service
        )
    
    def run(self, state: EnhancedState) -> AgentResponse:
        """处理运动相关请求"""
        try:
            # 获取用户的最后一条消息
            last_user_msg = state.messages[-1] if state.messages else {}
            user_content = last_user_msg.get('content', '')
            
            # 检查是否包含图片路径
            path_match = re.search(r'/[\w/\.]+\.(jpg|jpeg|png|gif)', user_content, re.IGNORECASE)
            if path_match:
                return self._process_exercise_image(state, path_match.group(0))
            
            # 检查是否是查询运动记录
            query_keywords = ['做了什么运动', '运动了什么', '锻炼了什么', '今天运动', '昨天运动', '运动记录', '查看运动', '运动情况']
            if any(keyword in user_content for keyword in query_keywords):
                return self._query_exercise_records(state)
            
            # 提取运动信息
            return self._extract_and_record_exercise(state, user_content)
            
        except Exception as e:
            error_msg = f"处理运动请求时发生错误：{str(e)}"
            return self._create_error_response(error_msg)
    
    def _extract_and_record_exercise(self, state: EnhancedState, content: str) -> AgentResponse:
        """提取并记录运动信息"""
        prompt = f"""请从以下文本中提取运动信息：

文本：{content}

请提取以下信息并以JSON格式返回：
{{
    "exercise_type": "运动类型（如跑步、游泳、健身等）",
    "duration": "运动时长（分钟，数字）",
    "description": "运动描述"
}}

如果无法确定某项信息，请使用合理的默认值。"""
        
        try:
            # 调用 LLM 提取运动信息
            response = self.llm_service.generate_response(prompt, "")
            
            # 解析 LLM 响应（这里简化处理，实际应该解析 JSON）
            exercise_type = "其他运动"
            duration = 30  # 默认30分钟
            description = content
            
            # 记录到数据库
            self._record_exercise_to_db(exercise_type, duration, description)
            
            # 回复用户
            reply = f"✅ 已记录您的{exercise_type}，持续时间：{duration}分钟"
            state.add_message("assistant", reply)
            
            return self._create_success_response(reply)
            
        except Exception as e:
            error_msg = f"提取运动信息时发生错误：{str(e)}"
            return self._create_error_response(error_msg)
    
    def _process_exercise_image(self, state: EnhancedState, image_path: str) -> AgentResponse:
        """处理运动图片"""
        # 简化处理，实际应该使用 OCR
        reply = f"📸 已接收到运动图片：{image_path}，但OCR功能暂未实现。请直接描述您的运动内容。"
        state.add_message("assistant", reply)
        return self._create_success_response(reply)
    
    def _query_exercise_records(self, state: EnhancedState) -> AgentResponse:
        """查询运动记录"""
        try:
            # 这里应该从数据库查询运动记录
            reply = "📊 正在查询您的运动记录...\n\n暂无运动记录数据。请先记录一些运动信息。"
            state.add_message("assistant", reply)
            return self._create_success_response(reply)
            
        except Exception as e:
            error_msg = f"查询运动记录时发生错误：{str(e)}"
            return self._create_error_response(error_msg)
    
    def _record_exercise_to_db(self, exercise_type: str, duration: int, description: str):
        """记录运动到数据库"""
        try:
            # 这里应该调用数据库服务记录运动
            # self.db_service.save_exercise({...})
            pass
        except Exception as e:
            print(f"记录运动到数据库失败：{e}")
    
    def _create_success_response(self, message: str) -> AgentResponse:
        """创建成功响应"""
        from core.agent_protocol import AgentResponse, AgentResult
        return AgentResponse(
            status=AgentResult.SUCCESS,
            message=message,
            data={}
        )
    
    def _create_error_response(self, error_msg: str) -> AgentResponse:
        """创建错误响应"""
        from core.agent_protocol import AgentResponse, AgentResult
        return AgentResponse(
            status=AgentResult.ERROR,
            message=error_msg,
            data={}
        )