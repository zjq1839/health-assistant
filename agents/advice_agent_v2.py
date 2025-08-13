"""健康建议代理 V2，负责根据用户输入生成可行的饮食或运动建议。"""
from __future__ import annotations

import re
from typing import Dict, Any

from core.agent_protocol import BaseAgent, AgentResponse
from core.enhanced_state import EnhancedState, IntentType
from core.agent_protocol import LLMService

class AdviceAgentV2(BaseAgent):
    """健康建议智能体 V2 - 提供个性化健康建议"""
    
    def __init__(self, llm_service: LLMService):
        super().__init__(
            name="advice",
            intents=[IntentType.ADVICE],
            llm_service=llm_service
        )
    
    def run(self, state: EnhancedState) -> AgentResponse:
        """处理健康建议请求"""
        try:
            # 获取用户的最后一条消息
            messages = state.get('messages', [])
            last_user_msg = messages[-1] if messages else {}
            user_content = last_user_msg.get('content', '')
            
            # 获取上下文信息
            context_entities = self._extract_context_entities(state)
            
            # 构建提示词
            prompt = self._build_prompt(user_content, context_entities)
            
            # 生成建议
            advice = self._generate_advice(prompt)
            
            return self._create_success_response(advice)
            
        except Exception as e:
            error_msg = f"生成健康建议时发生错误：{str(e)}"
            return self._create_error_response(error_msg)
    
    def _extract_context_entities(self, state: EnhancedState) -> Dict[str, Any]:
        """从对话状态中提取上下文实体"""
        context_entities = {}
        
        # 从最近的几条消息中提取关键信息
        messages = state.get('messages', [])
        recent_messages = messages[-5:] if len(messages) >= 5 else messages
        
        for msg in recent_messages:
            content = msg.get('content', '')
            
            # 提取饮食相关信息
            if any(keyword in content for keyword in ['吃', '喝', '饮食', '早餐', '午餐', '晚餐']):
                context_entities['recent_diet'] = '用户最近有饮食相关讨论'
            
            # 提取运动相关信息
            if any(keyword in content for keyword in ['运动', '跑步', '锻炼', '健身']):
                context_entities['recent_exercise'] = '用户最近有运动相关讨论'
            
            # 提取健康目标
            if any(keyword in content for keyword in ['减肥', '增重', '健康', '目标']):
                context_entities['health_goal'] = '用户有健康目标'
        
        return context_entities
    
    def _build_prompt(self, user_input: str, context_entities: Dict[str, Any]) -> str:
        """根据用户输入和上下文实体构造提示词"""
        entity_section = ""
        if context_entities:
            entity_pairs = ", ".join(f"{k}: {v}" for k, v in context_entities.items() if v)
            entity_section = f"\n\n已知相关信息：{entity_pairs}"
        
        return (
            "你是一名专业且富有同理心的健康顾问。请基于以下用户需求，结合上下文信息，提供3条具体、可执行、积极鼓励的健康建议，涵盖饮食或运动层面。"
            f"{entity_section}\n\n用户需求：{user_input}\n\n请使用简体中文回答，建议要具体可行。"
        )
    
    def _generate_advice(self, prompt: str) -> str:
        """生成健康建议"""
        try:
            response = self.llm_service.generate_response(prompt, "")
            
            return response
            
        except Exception as e:
            return f"生成建议时发生错误：{str(e)}"