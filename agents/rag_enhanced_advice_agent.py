"""RAG增强的建议代理

展示如何将RAG集成到现有的Agent架构中
"""

from typing import Dict, Any, List, Optional
from core.agent_protocol import BaseAgent, AgentResponse
from core.enhanced_state import EnhancedState, IntentType
from core.agent_protocol import LLMService, DatabaseService
from core.rag_service import RAGService, create_rag_service
from utils.logger import logger


class RAGEnhancedAdviceAgent(BaseAgent):
    """RAG增强的健康建议智能体"""
    
    def __init__(self, 
                 llm_service: LLMService,
                 db_service: DatabaseService = None,
                 rag_service: RAGService = None):
        super().__init__(
            name="rag_advice",
            intents=[IntentType.ADVICE],
            llm_service=llm_service,
            db_service=db_service
        )
        
        # 初始化RAG服务
        self.rag_service = rag_service or create_rag_service()
        
        # 用户画像缓存
        self._user_profiles = {}
    
    def run(self, state: EnhancedState) -> AgentResponse:
        """处理健康建议请求 - RAG增强版"""
        try:
            # 获取用户输入
            messages = state.get('messages', [])
            last_user_msg = messages[-1] if messages else {}
            user_input = last_user_msg.get('content', '')
            
            # 构建用户画像
            user_profile = self._build_user_profile(state)
            
            # 获取对话历史
            conversation_history = self._extract_conversation_history(messages)
            
            # 检测建议类型
            advice_type = self._detect_advice_type(user_input)
            
            # 使用RAG检索相关知识并生成建议
            print(f"\n🎯 启动RAG增强建议生成 - 建议类型: {advice_type}")
            print(f"🔄 当前处理的用户输入: {user_input}")
            
            rag_context = self.rag_service.retrieve_context(
                query=user_input,
                user_profile=user_profile,
                conversation_history=conversation_history,
                domain_context=advice_type,
                k=5
            )
            
            # 生成增强回答
            print(f"\n🚀 开始生成RAG增强回答...")
            enhanced_response = self.rag_service.generate_with_context(rag_context)
            
            # 添加个性化补充建议
            personalized_tips = self._generate_personalized_tips(user_profile, advice_type)
            
            # 整合最终回答
            final_response = self._format_final_response(
                enhanced_response, 
                personalized_tips, 
                rag_context
            )
            
            return self._create_success_response(final_response)
            
        except Exception as e:
            error_msg = f"生成RAG增强建议时发生错误：{str(e)}"
            logger.error(error_msg, exc_info=True)
            return self._create_error_response(error_msg)
    
    def _build_user_profile(self, state: EnhancedState) -> Dict[str, Any]:
        """构建用户画像"""
        user_profile = {}
        
        # 从历史对话中提取用户信息
        messages = state.get('messages', [])
        
        # 分析最近的消息来推断用户特征
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
        
        # 如果有数据库服务，可以查询用户的历史记录来丰富画像
        if self.db_service:
            user_profile.update(self._analyze_user_history(state))
        
        return user_profile
    
    def _analyze_user_history(self, state: EnhancedState) -> Dict[str, Any]:
        """分析用户历史数据"""
        profile_update = {}
        
        try:
            # 查询最近7天的饮食记录
            import datetime
            today = datetime.date.today()
            week_ago = today - datetime.timedelta(days=7)
            
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
            logger.warning(f"Failed to analyze user history: {e}")
        
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
    
    def _detect_advice_type(self, user_input: str) -> str:
        """检测建议类型"""
        user_input = user_input.lower()
        
        nutrition_keywords = ['饮食', '营养', '食物', '吃什么', '餐食', '膳食']
        exercise_keywords = ['运动', '锻炼', '健身', '训练', '活动']
        weight_keywords = ['减肥', '瘦身', '减重', '增重', '增肌']
        health_keywords = ['健康', '养生', '保健', '预防']
        
        if any(keyword in user_input for keyword in nutrition_keywords):
            return 'nutrition'
        elif any(keyword in user_input for keyword in exercise_keywords):
            return 'exercise'
        elif any(keyword in user_input for keyword in weight_keywords):
            return 'weight_management'
        elif any(keyword in user_input for keyword in health_keywords):
            return 'health_advice'
        else:
            return 'general'
    
    def _generate_personalized_tips(self, user_profile: Dict[str, Any], advice_type: str) -> List[str]:
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
            if advice_type == 'nutrition':
                tips.append("女性要特别注意铁和钙的补充")
            elif advice_type == 'exercise':
                tips.append("建议加强核心力量训练，有助于改善体态")
        elif gender == '男':
            if advice_type == 'nutrition':
                tips.append("男性通常需要更多的蛋白质支持肌肉发育")
        
        # 基于健康目标的建议
        health_goal = user_profile.get('health_goal', '')
        if health_goal == '减肥':
            tips.append("建议采用循序渐进的方式，每周减重0.5-1公斤最健康")
        elif health_goal == '增重增肌':
            tips.append("增肌期间要保证充足的蛋白质摄入和休息")
        
        # 基于运动偏好的建议
        exercise_pref = user_profile.get('exercise_preference', '')
        if exercise_pref == '有氧运动':
            tips.append("有氧运动要注意心率区间，建议保持在最大心率的60-80%")
        
        return tips[:3]  # 最多返回3个小贴士
    
    def _format_final_response(self, 
                             enhanced_response: str, 
                             personalized_tips: List[str], 
                             rag_context) -> str:
        """格式化最终回答"""
        
        response_parts = [enhanced_response]
        
        # 添加个性化小贴士
        if personalized_tips:
            response_parts.append("\n\n💡 **个性化小贴士**：")
            for i, tip in enumerate(personalized_tips, 1):
                response_parts.append(f"{i}. {tip}")
        
        # 添加知识来源说明
        if rag_context.retrieved_docs:
            response_parts.append(f"\n\n📚 *以上建议基于 {len(rag_context.retrieved_docs)} 个专业知识来源*")
        
        return "\n".join(response_parts)
    
    def add_knowledge_to_rag(self, content: str, metadata: Dict[str, Any]) -> None:
        """向RAG知识库添加内容"""
        try:
            self.rag_service.add_knowledge_document(content, metadata)
            logger.info(f"Added knowledge to RAG: {metadata.get('title', 'Unknown')}")
        except Exception as e:
            logger.error(f"Failed to add knowledge to RAG: {e}")
    
    def load_knowledge_from_directory(self, directory_path: str) -> None:
        """从目录加载知识库"""
        try:
            self.rag_service.load_knowledge_base(directory_path)
            logger.info(f"Loaded knowledge base from: {directory_path}")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")


# 便捷函数
def create_rag_enhanced_advice_agent(llm_service: LLMService, 
                                   db_service: DatabaseService = None) -> RAGEnhancedAdviceAgent:
    """创建RAG增强的建议代理"""
    return RAGEnhancedAdviceAgent(
        llm_service=llm_service,
        db_service=db_service
    )


# 测试代码
if __name__ == "__main__":
    # 这里可以添加测试代码
    pass