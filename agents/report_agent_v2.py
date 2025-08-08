import re
import datetime
from core.agent_protocol import BaseAgent
from core.enhanced_state import EnhancedState
from core.agent_protocol import LLMService
from core.agent_protocol import DatabaseService

class ReportAgentV2(BaseAgent):
    """报告智能体 V2 - 生成健康报告和分析"""
    
    def __init__(self, db_service: DatabaseService, llm_service: LLMService):
        from core.enhanced_state import IntentType
        super().__init__(
            name="report",
            intents=[IntentType.GENERATE_REPORT],
            db_service=db_service,
            llm_service=llm_service
        )
    
    def run(self, state: EnhancedState) -> AgentResponse:
        """处理报告生成请求"""
        try:
            # 获取用户的最后一条消息
            last_user_msg = state.messages[-1] if state.messages else {}
            user_content = last_user_msg.get('content', '')
            
            # 解析报告日期
            report_date = self._parse_report_date(user_content)
            
            # 获取数据
            data = self._get_report_data(report_date)
            
            # 生成报告
            report = self._generate_report(data, report_date)
            
            state.add_message("assistant", report)
            return self._create_success_response(report)
            
        except Exception as e:
            error_msg = f"处理报告请求时发生错误：{str(e)}"
            return self._create_error_response(error_msg)
    
    def _parse_report_date(self, content: str) -> str:
        """解析报告日期"""
        today = datetime.date.today()
        
        if "昨天" in content:
            return (today - datetime.timedelta(days=1)).isoformat()
        elif "前天" in content:
            return (today - datetime.timedelta(days=2)).isoformat()
        elif re.search(r'(\d+)月(\d+)[号日]', content):
            match = re.search(r'(\d+)月(\d+)[号日]', content)
            month, day = int(match.group(1)), int(match.group(2))
            return f"2025-{month:02d}-{day:02d}"
        else:
            return today.isoformat()
    
    def _get_report_data(self, report_date: str) -> dict:
        """获取报告数据"""
        try:
            # 这里应该从数据库获取数据
            # meals = await self.db_service.get_meals_by_date(report_date)
            # exercises = await self.db_service.get_exercises_by_date(report_date)
            
            return {
                'date': report_date,
                'meals': [],  # 暂时返回空列表
                'exercises': [],  # 暂时返回空列表
            }
        except Exception as e:
            print(f"获取报告数据时发生错误：{e}")
            return {'date': report_date, 'meals': [], 'exercises': []}
    
    def _generate_report(self, data: dict, report_date: str) -> str:
        """生成健康报告"""
        try:
            # 格式化日期
            date_obj = datetime.datetime.strptime(report_date, '%Y-%m-%d').date()
            today = datetime.date.today()
            
            if date_obj == today:
                date_str = "今天"
            elif date_obj == today - datetime.timedelta(days=1):
                date_str = "昨天"
            elif date_obj == today - datetime.timedelta(days=2):
                date_str = "前天"
            else:
                date_str = f"{date_obj.month}月{date_obj.day}日"
            
            meals = data.get('meals', [])
            exercises = data.get('exercises', [])
            
            if not meals and not exercises:
                return f"📊 {date_str}没有任何饮食或运动记录。建议您开始记录日常的饮食和运动情况，以便更好地管理健康。"
            
            # 使用 LLM 生成报告
            prompt = f"""请基于以下数据生成健康报告：
            
日期：{report_date}
饮食记录：{data['meals']}
运动记录：{data['exercises']}

请生成一份详细的健康分析报告。"""
            
            response = self.llm_service.generate_response(prompt, "")
            
            return response
            
        except Exception as e:
            return f"生成报告时发生错误：{str(e)}"
    
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