import re
import datetime
from core.agent_protocol import BaseAgent, AgentResponse
from core.enhanced_state import EnhancedState, IntentType
from core.agent_protocol import LLMService
from core.agent_protocol import DatabaseService
from utils.common_parsers import parse_duration, parse_exercise_type, parse_date_from_text

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
            messages = state.get('messages', [])
            last_user_msg = messages[-1] if messages else {}
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
        try:
            # 解析运动时长
            duration = parse_duration(content)
            
            # 解析运动类型
            exercise_type = parse_exercise_type(content)
            
            # 记录到数据库
            self._record_exercise_to_db(exercise_type, duration, content)
            
            # 回复用户
            reply = f"✅ 已记录您的{exercise_type}，持续时间：{duration}分钟"
            
            return self._create_success_response(reply)
            
        except Exception as e:
            error_msg = f"提取运动信息时发生错误：{str(e)}"
            return self._create_error_response(error_msg)

    def _query_exercise_records(self, state: EnhancedState) -> AgentResponse:
        """查询运动记录"""
        try:
            # 从用户内容解析日期（支持 今天/昨天/前天），默认为今天
            messages = state.get('messages', [])
            last_user_msg = messages[-1] if messages else {}
            content = last_user_msg.get('content', '')
            today = datetime.date.today()
            d = parse_date_from_text(content, base_date=today)
            date = (d or today).isoformat()

            records = self.db_service.query_exercises(date, limit=50)
            
            if not records:
                reply = f"📊 {date} 暂无运动记录。您可以说：'我跑步30分钟' 来记录。"
            else:
                lines = [f"📅 日期：{date}", "🏃 运动记录："]
                for r in records:
                    lines.append(f"- {r.get('exercise_type','未知')}，时长{r.get('duration',0)}分钟：{r.get('description','')}")
                reply = "\n".join(lines)
            return self._create_success_response(reply)
            
        except Exception as e:
            error_msg = f"查询运动记录时发生错误：{str(e)}"
            return self._create_error_response(error_msg)

    def _record_exercise_to_db(self, exercise_type: str, duration: int, description: str):
        """记录运动到数据库"""
        try:
            self.db_service.save_exercise({
                'exercise_type': exercise_type,
                'duration': duration,
                'description': description,
            })
        except Exception as e:
            print(f"记录运动到数据库失败：{e}")
    
    def _process_exercise_image(self, state: EnhancedState, image_path: str) -> AgentResponse:
        """处理运动图片"""
        # 简化处理，实际应该使用 OCR
        reply = f"📸 已接收到运动图片：{image_path}，但OCR功能暂未实现。请直接描述您的运动内容。"
        return self._create_success_response(reply)