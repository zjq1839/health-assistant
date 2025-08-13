"""
通用解析工具模块，提供日期、时长等常用解析功能
消除各 Agent 中的重复正则表达式和解析逻辑
"""
import re
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Tuple


def parse_duration(content: str) -> int:
    """
    从文本中解析运动时长（分钟）
    支持多种格式：
    - "30分钟" -> 30
    - "1小时30分钟" -> 90  
    - "2小时" -> 120
    - "45分" -> 45
    """
    duration = 30  # 默认值
    
    # 匹配小时+分钟格式: "1小时30分钟"
    hour_min_pattern = r'(\d+)小时(\d+)分钟'
    hour_min_match = re.search(hour_min_pattern, content)
    if hour_min_match:
        hours = int(hour_min_match.group(1))
        minutes = int(hour_min_match.group(2))
        duration = hours * 60 + minutes
    else:
        # 匹配纯小时格式: "2小时"
        hour_pattern = r'(\d+)小时'
        hour_match = re.search(hour_pattern, content)
        if hour_match:
            hours = int(hour_match.group(1))
            duration = hours * 60
        else:
            # 匹配分钟格式: "30分钟" 或 "30分"
            min_pattern = r'(\d+)分钟?'
            min_match = re.search(min_pattern, content)
            if min_match:
                duration = int(min_match.group(1))
    
    # 合理性检查：限制在1到1440分钟（24小时）
    if duration < 1:
        duration = 1
    elif duration > 1440:
        duration = 1440
        
    return duration


def parse_exercise_type(content: str) -> str:
    """从文本中解析运动类型"""
    exercise_mapping = {
        '跑步': 'running',
        '跑': 'running',
        '游泳': 'swimming',
        '瑜伽': 'yoga',
        '健身': 'fitness',
        '举重': 'weightlifting',
        '骑行': 'cycling',
        '骑车': 'cycling',
        '散步': 'walking',
        '走路': 'walking'
    }
    
    for keyword, exercise_type in exercise_mapping.items():
        if keyword in content:
            return exercise_type
    
    return 'other'


def parse_date_from_text(content: str, base_date: Optional[date] = None) -> Optional[date]:
    """
    从文本中解析日期
    支持相对日期："今天"、"昨天"、"前天"
    支持绝对日期："2024-01-15"、"1月15日"
    """
    if base_date is None:
        base_date = date.today()
    
    content = content.lower().strip()
    
    # 相对日期
    if '今天' in content or '今日' in content:
        return base_date
    elif '昨天' in content or '昨日' in content:
        return base_date - timedelta(days=1)
    elif '前天' in content:
        return base_date - timedelta(days=2)
    elif '明天' in content:
        return base_date + timedelta(days=1)
    elif '后天' in content:
        return base_date + timedelta(days=2)
    
    # 绝对日期格式：YYYY-MM-DD
    date_pattern = r'(\d{4})-(\d{1,2})-(\d{1,2})'
    match = re.search(date_pattern, content)
    if match:
        try:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            return date(year, month, day)
        except ValueError:
            pass
    
    # 中文日期格式：X月X日
    chinese_date_pattern = r'(\d{1,2})月(\d{1,2})日'
    match = re.search(chinese_date_pattern, content)
    if match:
        try:
            month = int(match.group(1))
            day = int(match.group(2))
            # 假设是当年
            return date(base_date.year, month, day)
        except ValueError:
            pass
    
    return None


def parse_query_type(content: str) -> str:
    """
    从查询文本中解析查询类型
    返回: 'meal', 'exercise', 'all'
    """
    content = content.lower()
    
    if any(keyword in content for keyword in ['饮食', '吃', '餐', '食物', '营养']):
        return 'meal'
    elif any(keyword in content for keyword in ['运动', '锻炼', '健身', '跑步']):
        return 'exercise'
    else:
        return 'all'


def parse_meal_type(content: str) -> str:
    """从文本中解析餐食类型"""
    meal_mapping = {
        '早餐': 'breakfast',
        '早饭': 'breakfast', 
        '早上': 'breakfast',
        '午餐': 'lunch',
        '午饭': 'lunch',
        '中午': 'lunch',
        '晚餐': 'dinner',
        '晚饭': 'dinner',
        '晚上': 'dinner',
        '夜宵': 'supper',
        '宵夜': 'supper'
    }
    
    for keyword, meal_type in meal_mapping.items():
        if keyword in content:
            return meal_type
    
    return 'other'


def extract_food_items(content: str) -> list:
    """从文本中提取食物列表（简单实现）"""
    # 简单的食物关键词提取
    food_keywords = ['米饭', '面条', '面包', '鸡蛋', '牛奶', '苹果', '香蕉', 
                    '鸡肉', '牛肉', '猪肉', '鱼', '蔬菜', '沙拉', '汤']
    
    found_foods = []
    for food in food_keywords:
        if food in content:
            found_foods.append(food)
    
    return found_foods if found_foods else ['未指定食物']


def parse_time_range(content: str, base_date: Optional[date] = None) -> Tuple[Optional[date], Optional[date]]:
    """
    解析时间范围
    支持："本周"、"上周"、"本月"、"上个月"等
    返回 (start_date, end_date) 元组
    """
    if base_date is None:
        base_date = date.today()
    
    content = content.lower().strip()
    
    if '本周' in content:
        # 计算本周的开始和结束
        days_since_monday = base_date.weekday()
        start_date = base_date - timedelta(days=days_since_monday)
        end_date = start_date + timedelta(days=6)
        return start_date, end_date
    
    elif '上周' in content:
        # 计算上周的开始和结束
        days_since_monday = base_date.weekday()
        this_monday = base_date - timedelta(days=days_since_monday)
        start_date = this_monday - timedelta(days=7)
        end_date = start_date + timedelta(days=6)
        return start_date, end_date
    
    elif '本月' in content:
        # 本月1号到月末
        start_date = base_date.replace(day=1)
        if base_date.month == 12:
            next_month = base_date.replace(year=base_date.year + 1, month=1, day=1)
        else:
            next_month = base_date.replace(month=base_date.month + 1, day=1)
        end_date = next_month - timedelta(days=1)
        return start_date, end_date
    
    elif '上个月' in content or '上月' in content:
        # 上个月1号到月末
        if base_date.month == 1:
            last_month = base_date.replace(year=base_date.year - 1, month=12, day=1)
        else:
            last_month = base_date.replace(month=base_date.month - 1, day=1)
        
        start_date = last_month
        # 计算上个月的最后一天
        if last_month.month == 12:
            next_month = last_month.replace(year=last_month.year + 1, month=1, day=1)
        else:
            next_month = last_month.replace(month=last_month.month + 1, day=1)
        end_date = next_month - timedelta(days=1)
        return start_date, end_date
    
    return None, None


def intent_to_agent_mapping(intent):
    """
    统一的意图到 Agent 映射函数
    消除 planner 和 main 中的重复映射逻辑
    """
    from core.enhanced_state import IntentType
    
    mapping = {
        IntentType.RECORD_MEAL: "dietary",
        IntentType.RECORD_EXERCISE: "exercise", 
        IntentType.QUERY: "query",
        IntentType.GENERATE_REPORT: "report",
        IntentType.ADVICE: "advice",
        IntentType.UNKNOWN: "general"
    }
    
    return mapping.get(intent, "general")