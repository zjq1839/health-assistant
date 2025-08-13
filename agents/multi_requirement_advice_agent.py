"""多需求RAG增强建议代理

集成多需求解析功能的高级健康建议智能体
支持处理包含多个要求的复杂查询，为每个需求提供专门的RAG检索和生成
"""

from typing import Dict, Any, List, Optional
from core.agent_protocol import BaseAgent, AgentResponse
from core.enhanced_state import EnhancedState, IntentType
from core.agent_protocol import LLMService, DatabaseService
from core.rag_service import RAGService, create_rag_service
from core.multi_requirement_parser import MultiRequirementParser, MultiRequirementParseResult, ParsedRequirement
from core.multi_requirement_rag_service import MultiRequirementRAGService, create_multi_requirement_rag_service
from utils.logger import logger


class MultiRequirementAdviceAgent(BaseAgent):
    """多需求RAG增强的健康建议智能体
    
    特性：
    1. 自动识别查询中的多个需求
    2. 使用轻量级模型进行需求解析
    3. 为每个需求单独进行RAG检索
    4. 综合生成最终回答
    """
    
    def __init__(self, 
                 llm_service: LLMService,
                 db_service: DatabaseService = None,
                 rag_service: RAGService = None,
                 multi_rag_service: MultiRequirementRAGService = None):
        super().__init__(
            name="multi_requirement_advice",
            intents=[IntentType.ADVICE, IntentType.QUERY, IntentType.GENERATE_REPORT],
            llm_service=llm_service,
            db_service=db_service
        )
        
        # 初始化服务
        self.rag_service = rag_service or create_rag_service()
        self.multi_rag_service = multi_rag_service
        self.requirement_parser = MultiRequirementParser(llm_service)
        
        # 复杂度阈值
        self.complexity_threshold = {
            'use_multi_requirement': 2,  # 超过2个需求时使用多需求处理
            'use_lite_model': 'medium',  # 中等复杂度以上使用轻量级模型
        }
        
        # 用户画像缓存
        self._user_profiles = {}
    
    async def run_async(self, state: EnhancedState) -> AgentResponse:
        """异步处理健康建议请求"""
        try:
            # 获取用户输入
            messages = state.get('messages', [])
            last_user_msg = messages[-1] if messages else {}
            user_input = last_user_msg.get('content', '')
            
            # 构建用户画像
            user_profile = self._build_user_profile(state)
            
            print(f"\n🎯 启动多需求RAG增强建议生成")
            print(f"🔄 用户输入: {user_input}")
            print(f"👤 用户画像: {user_profile}")
            
            # Step 1: 解析需求
            parse_result = self.requirement_parser.parse(user_input)
            
            print(f"\n📋 需求解析结果:")
            print(f"  - 复杂度: {parse_result.complexity}")
            print(f"  - 解析方法: {parse_result.parsing_method}")
            print(f"  - 识别需求数: {len(parse_result.requirements)}")
            print(f"  - 总置信度: {parse_result.total_confidence:.2f}")
            
            # Step 2: 根据复杂度选择处理策略
            if len(parse_result.requirements) >= self.complexity_threshold['use_multi_requirement']:
                # 使用多需求RAG服务
                return await self._handle_multi_requirement_query(
                    user_input, user_profile, parse_result, state
                )
            else:
                # 使用标准RAG处理
                return await self._handle_single_requirement_query(
                    user_input, user_profile, parse_result, state
                )
                
        except Exception as e:
            error_msg = f"生成多需求RAG增强建议时发生错误：{str(e)}"
            logger.error(error_msg, exc_info=True)
            return self._create_error_response(error_msg)
    
    def run(self, state: EnhancedState) -> AgentResponse:
        """同步处理方法（兼容基类）"""
        import asyncio
        return asyncio.run(self.run_async(state))
    
    async def _handle_multi_requirement_query(self, 
                                           user_input: str, 
                                           user_profile: Dict[str, Any],
                                           parse_result: MultiRequirementParseResult,
                                           state: EnhancedState) -> AgentResponse:
        """处理多需求查询"""
        print(f"\n🔄 使用多需求RAG服务处理...")
        
        try:
            # 初始化多需求RAG服务（如果需要）
            if self.multi_rag_service is None:
                self.multi_rag_service = await create_multi_requirement_rag_service()
            
            # 处理多需求查询
            multi_result = await self.multi_rag_service.process_multi_requirement_query(
                query=user_input,
                user_profile=user_profile,
                conversation_history=self._extract_conversation_history(state.get('messages', []))
            )
            
            print(f"\n✅ 多需求处理完成:")
            print(f"  - 成功处理: {multi_result.successful_requirements}/{len(multi_result.requirement_results)}")
            print(f"  - 处理时间: {multi_result.total_processing_time:.2f}s")
            
            # 添加个性化补充
            personalized_tips = self._generate_personalized_tips(user_profile, parse_result.requirements)
            
            # 格式化最终回答
            final_response = self._format_multi_requirement_response(
                multi_result, personalized_tips, parse_result
            )
            
            return self._create_success_response(final_response)
            
        except Exception as e:
            logger.error(f"多需求处理失败: {e}", exc_info=True)
            # 降级到单需求处理
            return await self._handle_single_requirement_query(
                user_input, user_profile, parse_result, state
            )
    
    async def _handle_single_requirement_query(self, 
                                            user_input: str,
                                            user_profile: Dict[str, Any],
                                            parse_result: MultiRequirementParseResult,
                                            state: EnhancedState) -> AgentResponse:
        """处理单需求查询（降级处理）"""
        print(f"\n🔄 使用标准RAG服务处理...")
        
        try:
            # 获取对话历史
            conversation_history = self._extract_conversation_history(state.get('messages', []))
            
            # 确定主要需求类型
            main_requirement = parse_result.requirements[0] if parse_result.requirements else None
            advice_type = self._map_requirement_to_advice_type(main_requirement) if main_requirement else 'general'
            
            # 使用标准RAG检索和生成
            rag_context = self.rag_service.retrieve_context(
                query=user_input,
                user_profile=user_profile,
                conversation_history=conversation_history,
                domain_context=advice_type,
                k=5
            )
            
            print(f"\n🚀 生成RAG增强回答...")
            enhanced_response = self.rag_service.generate_with_context(rag_context)
            
            # 添加个性化补充建议
            personalized_tips = self._generate_personalized_tips(user_profile, parse_result.requirements)
            
            # 整合最终回答
            final_response = self._format_single_requirement_response(
                enhanced_response, personalized_tips, rag_context, parse_result
            )
            
            return self._create_success_response(final_response)
            
        except Exception as e:
            logger.error(f"标准RAG处理失败: {e}", exc_info=True)
            return self._create_error_response(f"处理请求时发生错误：{str(e)}")
    
    def _build_user_profile(self, state: EnhancedState) -> Dict[str, Any]:
        """构建用户画像（继承自原RAGEnhancedAdviceAgent）"""
        user_profile = {}
        
        # 从历史对话中提取用户信息
        messages = state.get('messages', [])
        recent_messages = messages[-10:] if len(messages) >= 10 else messages
        
        for msg in recent_messages:
            content = msg.get('content', '').lower()
            
            # 年龄推断
            if '岁' in content:
                import re
                age_match = re.search(r'(\d+)岁', content)
                if age_match:
                    user_profile['age'] = int(age_match.group(1))
            
            # 性别推断
            if any(keyword in content for keyword in ['我是女生', '女性', '女士']):
                user_profile['gender'] = '女'
            elif any(keyword in content for keyword in ['我是男生', '男性', '先生']):
                user_profile['gender'] = '男'
            
            # 健康目标推断
            if any(keyword in content for keyword in ['减肥', '瘦身', '减重']):
                user_profile['health_goal'] = '减肥'
            elif any(keyword in content for keyword in ['增重', '增肌', '长肌肉']):
                user_profile['health_goal'] = '增重增肌'
            elif any(keyword in content for keyword in ['保持健康', '维持体重']):
                user_profile['health_goal'] = '维持健康'
            
            # 运动偏好
            if any(keyword in content for keyword in ['跑步', '慢跑']):
                user_profile['exercise_preference'] = '有氧运动'
            elif any(keyword in content for keyword in ['力量训练', '举重', '健身房']):
                user_profile['exercise_preference'] = '力量训练'
            elif any(keyword in content for keyword in ['瑜伽', '普拉提']):
                user_profile['exercise_preference'] = '柔性运动'
        
        # 如果有数据库服务，分析用户历史记录
        if self.db_service:
            try:
                user_profile.update(self._analyze_user_history(state))
            except Exception as e:
                logger.warning(f"分析用户历史失败: {e}")
        
        return user_profile
    
    def _analyze_user_history(self, state: EnhancedState) -> Dict[str, Any]:
        """分析用户历史数据"""
        profile_update = {}
        
        try:
            import datetime
            today = datetime.date.today()
            week_ago = today - datetime.timedelta(days=7)
            
            # 查询最近7天的记录
            recent_meals = self.db_service.query_meals(
                date=week_ago.strftime('%Y-%m-%d'),
                limit=50
            )
            
            recent_exercises = self.db_service.query_exercises(
                date=week_ago.strftime('%Y-%m-%d'),
                limit=50
            )
            
            # 分析饮食习惯
            if recent_meals:
                total_calories = sum(meal.get('calories', 0) for meal in recent_meals)
                avg_daily_calories = total_calories / 7 if total_calories > 0 else 0
                profile_update['avg_daily_calories'] = avg_daily_calories
                
                # 分析饮食偏好
                meal_descriptions = [meal.get('description', '') for meal in recent_meals]
                if any('素食' in desc for desc in meal_descriptions):
                    profile_update['diet_preference'] = '素食'
                elif any('肉' in desc for desc in meal_descriptions):
                    profile_update['diet_preference'] = '荤食'
            
            # 分析运动习惯
            if recent_exercises:
                exercise_frequency = len(recent_exercises) / 7
                profile_update['exercise_frequency'] = exercise_frequency
                
                exercise_types = [ex.get('description', '') for ex in recent_exercises]
                if any('跑步' in desc for desc in exercise_types):
                    profile_update['primary_exercise'] = '跑步'
                elif any('力量' in desc for desc in exercise_types):
                    profile_update['primary_exercise'] = '力量训练'
        
        except Exception as e:
            logger.warning(f"分析用户历史失败: {e}")
        
        return profile_update
    
    def _extract_conversation_history(self, messages: List[Dict]) -> List[Dict[str, str]]:
        """提取对话历史"""
        history = []
        
        for msg in messages[-6:]:  # 最近3轮对话
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if content.strip():
                history.append({
                    'role': role,
                    'content': content[:200]  # 限制长度
                })
        
        return history
    
    def _map_requirement_to_advice_type(self, requirement: ParsedRequirement) -> str:
        """将需求映射到建议类型"""
        if not requirement:
            return 'general'
        
        mapping = {
            'nutrition': ['饮食', '营养', '食物', '餐食'],
            'exercise': ['运动', '锻炼', '健身', '训练'],
            'weight_management': ['减肥', '增重', '体重', '减重'],
            'health': ['健康', '养生', '保健', '预防']
        }
        
        description = requirement.description.lower()
        
        for advice_type, keywords in mapping.items():
            if any(keyword in description for keyword in keywords):
                return advice_type
        
        return 'general'
    
    def _generate_personalized_tips(self, 
                                  user_profile: Dict[str, Any], 
                                  requirements: List[ParsedRequirement]) -> List[str]:
        """生成个性化小贴士"""
        tips = []
        
        # 基于年龄的建议
        age = user_profile.get('age', 0)
        if age > 0:
            if age < 25:
                tips.append("年轻人新陈代谢快，可以适当增加运动强度")
            elif age > 50:
                tips.append("注意关节保护，建议选择低冲击性运动")
        
        # 基于性别的建议
        gender = user_profile.get('gender', '')
        if gender == '女':
            tips.append("女性要特别注意铁和钙的补充")
        elif gender == '男':
            tips.append("男性通常需要更多的蛋白质支持肌肉发育")
        
        # 基于需求类型的建议
        requirement_types = [req.type.value for req in requirements]
        if 'advice' in requirement_types:
            tips.append("建议遵循循序渐进的原则，避免过度急进")
        if 'query' in requirement_types:
            tips.append("定期记录和回顾有助于更好地了解自己的健康状况")
        
        return tips[:3]  # 最多返回3个小贴士
    
    def _format_multi_requirement_response(self,
                                         multi_result,
                                         personalized_tips: List[str],
                                         parse_result: MultiRequirementParseResult) -> str:
        """格式化多需求处理结果"""
        response_parts = []
        
        # 添加解析概述
        response_parts.append(f"📋 **您的查询包含 {len(parse_result.requirements)} 个需求，我为您逐一解答：**\n")
        
        # 添加每个需求的回答
        for i, req_result in enumerate(multi_result.requirement_results, 1):
            requirement = req_result.requirement
            response_parts.append(f"**{i}. {requirement.content}**")
            response_parts.append(f"🔍 类型: {requirement.requirement_type}")
            response_parts.append(f"📚 基于 {len(req_result.retrieved_documents)} 个专业文档")
            response_parts.append(f"💡 {req_result.response}\n")
        
        # 添加综合回答
        if multi_result.final_response:
            response_parts.append("🎯 **综合建议**：")
            response_parts.append(multi_result.final_response)
        
        # 添加个性化小贴士
        if personalized_tips:
            response_parts.append("\n💡 **个性化小贴士**：")
            for i, tip in enumerate(personalized_tips, 1):
                response_parts.append(f"{i}. {tip}")
        
        # 添加处理信息
        response_parts.append(f"\n⚡ *处理耗时: {multi_result.total_processing_time:.2f}秒，解析方法: {parse_result.parsing_method}*")
        
        return "\n".join(response_parts)
    
    def _format_single_requirement_response(self,
                                          enhanced_response: str,
                                          personalized_tips: List[str],
                                          rag_context,
                                          parse_result: MultiRequirementParseResult) -> str:
        """格式化单需求处理结果"""
        response_parts = [enhanced_response]
        
        # 添加个性化小贴士
        if personalized_tips:
            response_parts.append("\n💡 **个性化小贴士**：")
            for i, tip in enumerate(personalized_tips, 1):
                response_parts.append(f"{i}. {tip}")
        
        # 添加知识来源说明
        if rag_context.retrieved_docs:
            response_parts.append(f"\n📚 *以上建议基于 {len(rag_context.retrieved_docs)} 个专业知识来源*")
        
        # 添加解析信息
        if parse_result.parsing_method != 'rule':
            response_parts.append(f"⚡ *解析方法: {parse_result.parsing_method}*")
        
        return "\n".join(response_parts)


# 便捷函数
async def create_multi_requirement_advice_agent(llm_service: LLMService, 
                                              db_service: DatabaseService = None) -> MultiRequirementAdviceAgent:
    """创建多需求RAG增强的建议代理"""
    # 预初始化多需求RAG服务
    multi_rag_service = await create_multi_requirement_rag_service()
    
    return MultiRequirementAdviceAgent(
        llm_service=llm_service,
        db_service=db_service,
        multi_rag_service=multi_rag_service
    )


# 测试代码
if __name__ == "__main__":
    import asyncio
    from core.service_container import get_llm_service, get_database_service
    
    async def test_agent():
        """测试多需求建议代理"""
        print("🚀 测试多需求RAG增强建议代理")
        
        # 获取服务
        llm_service = get_llm_service()
        db_service = get_database_service()
        
        # 创建代理
        agent = await create_multi_requirement_advice_agent(llm_service, db_service)
        
        # 测试案例
        test_cases = [
            {
                "scenario": "简单单需求",
                "query": "我想了解减肥的饮食建议",
                "expected_method": "单需求处理"
            },
            {
                "scenario": "复杂多需求",
                "query": "我有高血压，请推荐降压食物和适合的运动方式，还要告诉我注意事项",
                "expected_method": "多需求处理"
            },
            {
                "scenario": "综合查询",
                "query": "分析我的健康状况，制定减肥计划，推荐运动方案，计算每日热量需求",
                "expected_method": "多需求处理"
            }
        ]
        
        for test_case in test_cases:
            print(f"\n{'='*60}")
            print(f"🧪 测试场景: {test_case['scenario']}")
            print(f"📝 查询: {test_case['query']}")
            print(f"🎯 预期方法: {test_case['expected_method']}")
            print("-" * 60)
            
            # 构建测试状态
            state = EnhancedState()
            state.update('messages', [{
                'role': 'user',
                'content': test_case['query']
            }])
            
            try:
                # 执行测试
                response = await agent.run_async(state)
                
                print(f"✅ 处理成功")
                print(f"📄 回答: {response.content[:200]}...")
                
            except Exception as e:
                print(f"❌ 处理失败: {e}")
        
        print(f"\n✅ 所有测试完成")
    
    # 运行测试
    asyncio.run(test_agent())