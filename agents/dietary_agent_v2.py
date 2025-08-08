"""饮食Agent V2 - 基于新协议的实现

展示如何使用：
1. 统一的Agent协议
2. 依赖注入
3. 结构化营养计算
4. 证据绑定
"""

from typing import Dict, List
import json
from datetime import datetime

from core.agent_protocol import BaseAgent, AgentResponse, AgentResult
from core.enhanced_state import IntentType, EnhancedState
from core.agent_protocol import DatabaseService, LLMService, NutritionService
from core.nutrition_service import StructuredNutritionService, NutritionFact, NutritionEvidence


class DietaryAgentV2(BaseAgent):
    """饮食记录Agent V2
    
    特点：
    1. 实现统一协议
    2. 依赖注入设计
    3. 强制结构化营养计算
    4. 完整的证据链
    """
    
    def __init__(self, 
                 db_service: DatabaseService,
                 llm_service: LLMService,
                 nutrition_service: NutritionService = None):
        super().__init__(
            name="dietary",
            intents=[IntentType.RECORD_MEAL],
            db_service=db_service,
            llm_service=llm_service,
            nutrition_service=nutrition_service
        )
        
        # 如果没有提供营养服务，使用默认的结构化服务
        if self.nutrition_service is None:
            self.nutrition_service = StructuredNutritionService()
    
    def validate_input(self, state: EnhancedState) -> bool:
        """验证输入"""
        if not super().validate_input(state):
            return False
        
        # 检查是否包含饮食相关内容
        user_input = self._extract_user_input(state['messages'][-1])
        
        # 简单的关键词检查
        food_keywords = ['吃', '喝', '餐', '食', '饭', '面', '肉', '菜', '果', '奶', '蛋']
        return any(keyword in user_input for keyword in food_keywords)
    
    def run(self, state: EnhancedState) -> AgentResponse:
        """执行饮食记录逻辑"""
        try:
            # 1. 提取用户输入
            user_input = self._extract_user_input(state['messages'][-1])
            
            # 2. 使用LLM提取餐食信息
            meal_info = self._extract_meal_info_with_llm(user_input)
            
            if not meal_info:
                return self._create_error_response(
                    "抱歉，我无法从您的输入中识别出具体的餐食信息。请提供更详细的描述，比如：'我吃了一碗白米饭和100克鸡胸肉'。",
                    "meal_extraction_failed"
                )
            
            # 3. 使用结构化营养服务计算营养成分
            nutrition, evidences = self.nutrition_service.calculate_meal_nutrition(
                meal_info['description']
            )
            
            # 4. 验证营养数据
            warnings = self.nutrition_service.validate_nutrition_data(nutrition)
            
            # 5. 准备保存的数据
            meal_data = {
                'date': meal_info.get('date', datetime.now().strftime('%Y-%m-%d')),
                'meal_type': meal_info.get('meal_type', ''),
                'description': meal_info['description'],
                'calories': nutrition.calories,
                'nutrients': json.dumps(nutrition.to_dict(), ensure_ascii=False),
                'raw_input': user_input,
                'extraction_method': 'llm_structured',
                'confidence': self._calculate_overall_confidence(evidences)
            }
            
            # 6. 保存到数据库
            save_result = self.db_service.save_meal(meal_data)
            
            if save_result.get('status') != 'success':
                return self._create_error_response(
                    f"保存餐食记录时出错：{save_result.get('message', '未知错误')}",
                    "database_save_failed"
                )
            
            # 7. 创建成功响应
            response = self._create_success_response(
                self._format_meal_response(meal_info, nutrition, warnings),
                {
                    'meal_id': save_result.get('id'),
                    'nutrition': nutrition.to_dict(),
                    'warnings': warnings,
                    'evidence_count': len(evidences)
                }
            )
            
            # 8. 添加证据链
            for evidence in evidences:
                response.add_evidence(
                    evidence.source,
                    f"{evidence.method}: {evidence.raw_data.get('food_name', 'unknown')}",
                    evidence.confidence
                )
            
            # 9. 记录执行日志
            self._log_execution(state, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"DietaryAgent execution failed: {str(e)}", exc_info=True)
            return self._create_error_response(
                "处理餐食记录时发生错误，请稍后重试。",
                "internal_error"
            )
    
    def _extract_meal_info_with_llm(self, user_input: str) -> Dict:
        """使用LLM提取餐食信息"""
        # 定义提取模式
        extraction_schema = {
            "description": "餐食的详细描述，包含食物名称和数量",
            "meal_type": "餐食类型：早餐/午餐/晚餐/加餐",
            "date": "餐食日期，格式YYYY-MM-DD"
        }
        
        try:
            # 使用LLM服务提取信息
            extracted_info = self.llm_service.extract_entities(user_input, extraction_schema)
            
            # 验证提取结果
            if not extracted_info or not extracted_info.get('description'):
                return None
            
            # 标准化餐食类型
            meal_type = extracted_info.get('meal_type', '')
            meal_type = self._normalize_meal_type(meal_type)
            
            # 标准化日期
            date = extracted_info.get('date')
            if not date or date == 'today':
                date = datetime.now().strftime('%Y-%m-%d')
            
            return {
                'description': extracted_info['description'],
                'meal_type': meal_type,
                'date': date
            }
            
        except Exception as e:
            self.logger.warning(f"LLM extraction failed: {str(e)}")
            
            # 降级到简单解析
            return self._simple_meal_extraction(user_input)
    
    def _simple_meal_extraction(self, user_input: str) -> Dict:
        """简单的餐食信息提取（降级方案）"""
        # 检测餐食类型
        meal_type = ""
        if any(word in user_input for word in ['早餐', '早饭', '早上']):
            meal_type = "早餐"
        elif any(word in user_input for word in ['午餐', '午饭', '中午']):
            meal_type = "午餐"
        elif any(word in user_input for word in ['晚餐', '晚饭', '晚上']):
            meal_type = "晚餐"
        else:
            meal_type = "加餐"
        
        return {
            'description': user_input,
            'meal_type': meal_type,
            'date': datetime.now().strftime('%Y-%m-%d')
        }
    
    def _normalize_meal_type(self, meal_type: str) -> str:
        """标准化餐食类型"""
        meal_type = meal_type.lower().strip()
        
        if meal_type in ['早餐', '早饭', 'breakfast']:
            return '早餐'
        elif meal_type in ['午餐', '午饭', '中餐', 'lunch']:
            return '午餐'
        elif meal_type in ['晚餐', '晚饭', '晚上', 'dinner']:
            return '晚餐'
        else:
            return '加餐'
    
    def _calculate_overall_confidence(self, evidences: List[NutritionEvidence]) -> float:
        """计算整体置信度"""
        if not evidences:
            return 0.0
        
        # 加权平均置信度
        total_weight = 0
        weighted_confidence = 0
        
        for evidence in evidences:
            # 根据证据来源设置权重
            weight = 1.0
            if evidence.source == "food_database":
                weight = 1.0
            elif evidence.source == "api":
                weight = 0.8
            elif evidence.source == "estimation":
                weight = 0.5
            
            weighted_confidence += evidence.confidence * weight
            total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _format_meal_response(self, meal_info: Dict, nutrition: NutritionFact, warnings: Dict) -> str:
        """格式化餐食记录响应"""
        response_parts = []
        
        # 基本信息
        response_parts.append(f"✅ 已记录您的{meal_info['meal_type']}：{meal_info['description']}")
        
        # 营养信息
        response_parts.append("\n📊 营养成分分析：")
        response_parts.append(f"• 卡路里：{nutrition.calories:.1f} kcal")
        response_parts.append(f"• 蛋白质：{nutrition.protein:.1f}g")
        response_parts.append(f"• 碳水化合物：{nutrition.carbs:.1f}g")
        response_parts.append(f"• 脂肪：{nutrition.fat:.1f}g")
        
        if nutrition.fiber > 0:
            response_parts.append(f"• 纤维：{nutrition.fiber:.1f}g")
        
        # 警告信息
        if warnings:
            response_parts.append("\n⚠️ 注意事项：")
            for field, warning in warnings.items():
                response_parts.append(f"• {warning}")
        
        # 建议
        response_parts.append("\n💡 小贴士：")
        if nutrition.calories > 800:
            response_parts.append("• 这餐卡路里较高，建议搭配适量运动")
        if nutrition.protein < 10:
            response_parts.append("• 蛋白质含量较低，建议增加蛋白质摄入")
        if nutrition.fiber < 5:
            response_parts.append("• 纤维含量较低，建议多吃蔬菜水果")
        
        return "\n".join(response_parts)
    
    def get_nutrition_suggestions(self, query: str) -> List[Dict]:
        """获取营养建议（可被其他组件调用）"""
        try:
            foods = self.nutrition_service.get_food_suggestions(query)
            
            suggestions = []
            for food in foods:
                suggestions.append({
                    'name': food.name,
                    'category': food.category,
                    'calories_per_100g': food.nutrition_per_100g.calories,
                    'protein_per_100g': food.nutrition_per_100g.protein,
                    'confidence': food.confidence
                })
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Failed to get nutrition suggestions: {str(e)}")
            return []
    
    def analyze_meal_balance(self, nutrition: NutritionFact) -> Dict[str, str]:
        """分析餐食营养平衡"""
        analysis = {}
        
        # 计算营养比例
        total_calories = nutrition.calories
        if total_calories > 0:
            protein_ratio = (nutrition.protein * 4) / total_calories
            carbs_ratio = (nutrition.carbs * 4) / total_calories
            fat_ratio = (nutrition.fat * 9) / total_calories
            
            # 评估营养比例
            if protein_ratio < 0.15:
                analysis['protein'] = "蛋白质比例偏低，建议增加优质蛋白质"
            elif protein_ratio > 0.35:
                analysis['protein'] = "蛋白质比例偏高，注意营养均衡"
            
            if carbs_ratio < 0.45:
                analysis['carbs'] = "碳水化合物比例偏低，可能影响能量供应"
            elif carbs_ratio > 0.65:
                analysis['carbs'] = "碳水化合物比例偏高，建议控制精制糖摄入"
            
            if fat_ratio < 0.20:
                analysis['fat'] = "脂肪比例偏低，适量增加健康脂肪"
            elif fat_ratio > 0.35:
                analysis['fat'] = "脂肪比例偏高，建议减少油脂摄入"
        
        return analysis


# 测试代码
if __name__ == "__main__":
    # 这里可以添加单元测试
    pass