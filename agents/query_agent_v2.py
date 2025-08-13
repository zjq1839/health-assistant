import json
import re
import datetime
from core.agent_protocol import BaseAgent
from core.agent_protocol import AgentResponse  # ensure annotation resolves at import time
from core.enhanced_state import EnhancedState
from core.agent_protocol import LLMService
from core.agent_protocol import DatabaseService
from utils.common_parsers import parse_date_from_text, parse_time_range

class QueryAgentV2(BaseAgent):
    """查询智能体 V2 - 处理数据查询请求"""
    
    def __init__(self, db_service: DatabaseService, llm_service: LLMService):
        from core.enhanced_state import IntentType
        super().__init__(
            name="query",
            intents=[IntentType.QUERY],
            db_service=db_service,
            llm_service=llm_service
        )
    
    def run(self, state: EnhancedState) -> AgentResponse:
        """处理查询请求"""
        try:
            # 获取对话历史和用户消息
            messages = state.get('messages', [])
            last_user_msg = messages[-1] if messages else {}
            user_content = last_user_msg.get('content', '')
            
            # 解析查询参数
            query_params = self._parse_query_params(user_content, state)
            
            # 执行查询
            result = self._execute_query(query_params)
           
            # 格式化响应
            response = self._format_response(result, query_params)
            
            return self._create_success_response(response)
            
        except Exception as e:
            error_msg = f"处理查询请求时发生错误：{str(e)}"
            return self._create_error_response(error_msg)
    
    def _parse_query_params(self, content: str, state: EnhancedState) -> dict:
        """解析查询参数"""
        today = datetime.date.today()
        
        # 优先解析相对或绝对日期
        parsed = parse_date_from_text(content, base_date=today)
        if parsed is None:
            # 尝试解析范围，如本周、本月等
            start_date, end_date = parse_time_range(content, base_date=today)
            if start_date and end_date:
                # 范围查询：此版本先简单取开始日
                query_date = start_date.isoformat()
            else:
                query_date = today.isoformat()
        else:
            query_date = parsed.isoformat()
        
        # 查询类型解析
        if any(kw in content for kw in ['吃', '喝', '饮食', '早餐', '午餐', '晚餐']):
            query_type = 'dietary'
        elif any(kw in content for kw in ['运动', '跑步', '锻炼']):
            query_type = 'exercise'
        else:
            # 基于上下文推断
            messages = state.get('messages', [])
            context = ' '.join([msg.get('content', '') for msg in messages[-3:]])
            if any(kw in context for kw in ['饮食记录', '吃了什么', '早餐', '午餐', '晚餐']):
                query_type = 'dietary'
            elif any(kw in context for kw in ['运动', '跑步', '锻炼']):
                query_type = 'exercise'
            else:
                query_type = 'both'
        
        return {
            'date': query_date,
            'query_type': query_type,
            'original_content': content
        }
    
    def _execute_query(self, params: dict) -> dict:
        """执行数据库查询"""
        query_date = params['date']
        query_type = params['query_type']
        
        result = {
            'date': query_date,
            'dietary_records': [],
            'exercise_records': []
        }
        
        try:
            if query_type in ['dietary', 'both']:
                # 使用已实现的数据库查询方法
                result['dietary_records'] = self.db_service.query_meals(query_date, limit=50)
            
            if query_type in ['exercise', 'both']:
                # 使用已实现的数据库查询方法
                result['exercise_records'] = self.db_service.query_exercises(query_date, limit=50)
                
        except Exception as e:
            print(f"查询数据库时发生错误：{e}")
        
        return result
    
    def _format_response(self, result: dict, params: dict) -> str:
        """格式化查询响应"""
        query_date = result['date']
        dietary_records = result['dietary_records']
        exercise_records = result['exercise_records']
        
        # 格式化日期显示
        date_obj = datetime.datetime.strptime(query_date, '%Y-%m-%d').date()
        today = datetime.date.today()
        
        if date_obj == today:
            date_str = "今天"
        elif date_obj == today - datetime.timedelta(days=1):
            date_str = "昨天"
        elif date_obj == today - datetime.timedelta(days=2):
            date_str = "前天"
        else:
            date_str = f"{date_obj.month}月{date_obj.day}日"
        
        response_parts = [f"📊 {date_str}的记录："]
        
        # 饮食记录
        if params['query_type'] in ['dietary', 'both']:
            if dietary_records:
                response_parts.append("\n🍽️ 饮食记录：")
                for record in dietary_records:
                    response_parts.append(f"  • {record}")
            else:
                response_parts.append("\n🍽️ 饮食记录：暂无记录")
        
        # 运动记录
        if params['query_type'] in ['exercise', 'both']:
            if exercise_records:
                response_parts.append("\n🏃 运动记录：")
                for record in exercise_records:
                    response_parts.append(f"  • {record}")
            else:
                response_parts.append("\n🏃 运动记录：暂无记录")
        
        return ''.join(response_parts)