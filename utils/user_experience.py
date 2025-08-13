import re
import datetime
from typing import Dict, List, Optional, Tuple

class InputValidator:
    """输入验证器"""
    
    @staticmethod
    def validate_date(date_str: str) -> Tuple[bool, Optional[str]]:
        """验证日期格式"""
        if not date_str:
            return False, "日期不能为空"
        
        # 支持多种日期格式
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # 2024-01-15
            r'^\d{4}/\d{2}/\d{2}$',  # 2024/01/15
            r'^\d{2}-\d{2}$',       # 01-15 (当年)
            r'^\d{2}/\d{2}$',       # 01/15 (当年)
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, date_str):
                try:
                    if len(date_str) == 5:  # MM-DD 或 MM/DD
                        current_year = datetime.datetime.now().year
                        date_str = f"{current_year}-{date_str.replace('/', '-')}"
                    elif '/' in date_str:
                        date_str = date_str.replace('/', '-')
                    
                    datetime.datetime.strptime(date_str, '%Y-%m-%d')
                    return True, date_str
                except ValueError:
                    continue
        
        return False, "日期格式不正确，请使用 YYYY-MM-DD 或 MM-DD 格式"
    
    @staticmethod
    def validate_meal_type(meal_type: str) -> Tuple[bool, Optional[str]]:
        """验证餐食类型"""
        valid_types = ['早餐', '午餐', '晚餐', '加餐', '夜宵']
        
        if not meal_type:
            return False, "餐食类型不能为空"
        
        # 模糊匹配
        meal_type = meal_type.strip()
        for valid_type in valid_types:
            if valid_type in meal_type or meal_type in valid_type:
                return True, valid_type
        
        return False, f"餐食类型不正确，请使用：{', '.join(valid_types)}"
    
    @staticmethod
    def validate_exercise_duration(duration_str: str) -> Tuple[bool, Optional[int]]:
        """验证运动时长"""
        if not duration_str:
            return False, "运动时长不能为空"
        
        try:
            duration = int(duration_str)
            if duration <= 0:
                return False, "运动时长必须大于0"
            if duration > 1440:  # 24小时
                return False, "运动时长不能超过24小时"
            return True, duration
        except ValueError:
            return False, "运动时长必须是数字"
    
    @staticmethod
    def validate_image_path(image_path: str) -> Tuple[bool, Optional[str]]:
        """验证图片路径"""
        import os
        
        if not image_path:
            return False, "图片路径不能为空"
        
        if not os.path.exists(image_path):
            return False, "图片文件不存在"
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
            return False, f"不支持的图片格式，请使用：{', '.join(valid_extensions)}"
        
        return True, image_path

class MessageFormatter:
    """消息格式化器"""
    
    @staticmethod
    def format_meal_record(meal_data: Dict) -> str:
        """格式化餐食记录"""
        if not meal_data:
            return "没有找到餐食记录"
        
        formatted = f"📅 日期：{meal_data.get('date', 'N/A')}\n"
        formatted += f"🍽️ 餐食类型：{meal_data.get('meal_type', 'N/A')}\n"
        formatted += f"📝 描述：{meal_data.get('description', 'N/A')}\n"
        
        if meal_data.get('calories'):
            formatted += f"🔥 卡路里：{meal_data['calories']} kcal\n"
        
        return formatted
    
    @staticmethod
    def format_exercise_record(exercise_data: Dict) -> str:
        """格式化运动记录"""
        if not exercise_data:
            return "没有找到运动记录"
        
        formatted = f"📅 日期：{exercise_data.get('date', 'N/A')}\n"
        formatted += f"🏃 运动类型：{exercise_data.get('exercise_type', 'N/A')}\n"
        formatted += f"⏱️ 时长：{exercise_data.get('duration', 'N/A')} 分钟\n"
        
        if exercise_data.get('description'):
            formatted += f"📝 描述：{exercise_data['description']}\n"
        
        if exercise_data.get('calories_burned'):
            formatted += f"🔥 消耗卡路里：{exercise_data['calories_burned']} kcal\n"
        
        if exercise_data.get('intensity'):
            intensity_emoji = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
            emoji = intensity_emoji.get(exercise_data['intensity'], '⚪')
            formatted += f"{emoji} 强度：{exercise_data['intensity']}\n"
        
        return formatted
    
    @staticmethod
    def format_query_results(results: List[Dict], query_type: str) -> str:
        """格式化查询结果"""
        if not results:
            type_name = "餐食" if query_type == "dietary" else "运动"
            return f"没有找到相关的{type_name}记录"
        
        formatted = f"找到 {len(results)} 条记录：\n\n"
        
        for i, record in enumerate(results, 1):
            formatted += f"--- 记录 {i} ---\n"
            if query_type == "dietary":
                formatted += MessageFormatter.format_meal_record(record)
            else:
                formatted += MessageFormatter.format_exercise_record(record)
            formatted += "\n"
        
        return formatted
    
    @staticmethod
    def format_error_message(error_type: str, details: str = "") -> str:
        """格式化错误消息"""
        error_messages = {
            "validation_error": "❌ 输入验证失败",
            "database_error": "💾 数据库操作失败",
            "llm_error": "🤖 AI 服务暂时不可用",
            "ocr_error": "📷 图片识别失败",
            "network_error": "🌐 网络连接失败",
            "file_error": "📁 文件操作失败",
            "unknown_error": "❓ 未知错误"
        }
        
        base_message = error_messages.get(error_type, error_messages["unknown_error"])
        
        if details:
            return f"{base_message}\n详细信息：{details}"
        
        return base_message
    
    @staticmethod
    def format_success_message(operation: str, details: str = "") -> str:
        """格式化成功消息"""
        success_messages = {
            "add_meal": "✅ 餐食记录添加成功",
            "add_exercise": "✅ 运动记录添加成功",
            "record_meal": "🍽️ 饮食记录成功",
            "record_exercise": "🏃 运动记录成功",
            "generate_report": "📊 报告生成完成",
            "query": "✅ 数据查询完成",
            "advice": "💡 建议已生成",
        }
        
        base_message = success_messages.get(operation, "✅ 操作完成")
        
        if details:
            return f"{base_message}\n{details}"
        
        return base_message
    
    @staticmethod
    def format_response(response: str) -> str:
        """格式化AI响应"""
        if not response:
            return "抱歉，我没有理解您的请求，请重新描述。"
        
        # 清理响应中的多余空白
        response = response.strip()
        
        # 如果响应太长，添加换行以提高可读性
        if len(response) > 200:
            # 在句号后添加换行
            response = response.replace('。', '。\n')
            # 移除多余的换行
            response = '\n'.join(line.strip() for line in response.split('\n') if line.strip())
        
        return response

class UserGuidance:
    """用户指导"""
    
    @staticmethod
    def get_help_message() -> str:
        """获取帮助信息"""
        return """
🤖 健康助手使用指南

📝 记录餐食：
• "我今天早餐吃了鸡蛋和牛奶"
• "记录午餐：米饭、青菜、鸡肉"
• "添加晚餐记录"

🏃 记录运动：
• "我今天跑步30分钟"
• "记录运动：游泳45分钟"
• "添加运动记录"

📷 图片识别：
• "分析这张运动图片" + 上传图片
• "识别图片中的运动信息"

📊 查询数据：
• "今天吃了什么？"
• "本周的运动记录"
• "查询1月15日的数据"

📈 生成报告：
• "生成健康报告"
• "分析我的运动数据"
• "总结本周的健康状况"

💡 提示：
• 日期格式：2024-01-15 或 01-15
• 支持自然语言输入
• 可以上传图片进行识别
        """
    
    @staticmethod
    def get_examples_by_intent(intent: str) -> List[str]:
        """根据意图获取示例"""
        examples = {
            "record_meal": [
                "我今天早餐吃了鸡蛋和牛奶",
                "记录午餐：米饭、青菜、鸡肉",
                "添加晚餐：面条和蔬菜"
            ],
            "record_exercise": [
                "我今天跑步30分钟",
                "记录运动：游泳45分钟",
                "添加运动：骑车1小时"
            ],
            "query": [
                "今天吃了什么？",
                "本周的运动记录",
                "查询1月15日的数据"
            ],
            "generate_report": [
                "生成健康报告",
                "分析我的运动数据",
                "总结本周的健康状况"
            ],
            "advice": [
                "给我一些健康建议",
                "如何制定运动计划",
                "推荐一些健康食谱"
            ]
        }
        
        return examples.get(intent, ["请参考帮助信息"])
    
    @staticmethod
    def suggest_corrections(user_input: str, error_type: str) -> str:
        """建议修正方法"""
        suggestions = {
            "date_format": "请使用正确的日期格式，如：2024-01-15 或 01-15",
            "meal_type": "请指定餐食类型：早餐、午餐、晚餐、加餐或夜宵",
            "exercise_duration": "请提供有效的运动时长（分钟），如：30、45、60",
            "missing_info": "请提供更详细的信息，如具体的食物或运动类型",
            "image_format": "请上传支持的图片格式：jpg、png、bmp等"
        }
        
        suggestion = suggestions.get(error_type, "请检查输入格式并重试")
        return f"💡 建议：{suggestion}"

# 全局实例
validator = InputValidator()
formatter = MessageFormatter()
guidance = UserGuidance()