"""通用解析器工具

提供各种文本解析功能，包括时间、运动、食物等信息的提取
"""

import re
import datetime
from typing import Optional, Tuple
from core.enhanced_state import IntentType


def intent_to_agent_mapping(intent: IntentType) -> str:
    """意图到 Agent 的映射"""
    mapping = {
        IntentType.RECORD_MEAL: "dietary",
        IntentType.RECORD_EXERCISE: "exercise",
        IntentType.GENERATE_REPORT: "report",
        IntentType.QUERY: "query",
        IntentType.ADVICE: "advice",
    }
    # 对于UNKNOWN意图，返回None而不是兜底到advice
    if intent == IntentType.UNKNOWN:
        return None
    return mapping.get(intent, None)


def parse_duration(text: str) -> int:
    """从文本中解析运动时长（分钟）"""
    # 匹配各种时长表达
    patterns = [
        r'(\d+)\s*分钟',
        r'(\d+)\s*min',
        r'(\d+)\s*minutes?',
        r'(\d+)分',
        r'锻炼了?(\d+)',
        r'跑了?(\d+)',
        r'(\d+)\s*小时',  # 小时转换为分钟
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            duration = int(match.group(1))
            # 如果是小时，转换为分钟
            if '小时' in pattern:
                duration *= 60
            return duration
    
    # 默认返回 30 分钟
    return 30


def parse_exercise_type(text: str) -> str:
    """从文本中解析运动类型"""
    exercise_keywords = {
        '跑步': ['跑步', '跑', 'run', 'running', '慢跑'],
        '游泳': ['游泳', 'swim', 'swimming'],
        '健身': ['健身', '举重', '力量训练', '撸铁', 'gym', 'weight'],
        '瑜伽': ['瑜伽', 'yoga'],
        '篮球': ['篮球', 'basketball'],
        '足球': ['足球', 'football', 'soccer'],
        '骑车': ['骑车', '自行车', 'bike', 'cycling'],
        '走路': ['走路', '散步', 'walk', 'walking'],
        '爬山': ['爬山', '登山', 'hiking'],
        '羽毛球': ['羽毛球', 'badminton'],
        '乒乓球': ['乒乓球', 'ping pong', 'table tennis'],
        '网球': ['网球', 'tennis'],
    }
    
    text_lower = text.lower()
    for exercise_type, keywords in exercise_keywords.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return exercise_type
    
    # 默认返回"运动"
    return "运动"


def parse_date_from_text(text: str, base_date: datetime.date = None) -> Optional[datetime.date]:
    """从文本中解析日期"""
    if base_date is None:
        base_date = datetime.date.today()
    
    text = text.lower()
    
    # 相对日期
    if '今天' in text or '今日' in text:
        return base_date
    elif '昨天' in text or '昨日' in text:
        return base_date - datetime.timedelta(days=1)
    elif '前天' in text:
        return base_date - datetime.timedelta(days=2)
    elif '明天' in text:
        return base_date + datetime.timedelta(days=1)
    elif '后天' in text:
        return base_date + datetime.timedelta(days=2)
    
    # 绝对日期模式
    date_patterns = [
        r'(\d{4})[年\-/](\d{1,2})[月\-/](\d{1,2})[日号]?',
        r'(\d{1,2})[月\-/](\d{1,2})[日号]?',
        r'(\d{1,2})[日号]',
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            groups = match.groups()
            if len(groups) == 3:  # 年月日
                year, month, day = map(int, groups)
                return datetime.date(year, month, day)
            elif len(groups) == 2:  # 月日
                month, day = map(int, groups)
                return datetime.date(base_date.year, month, day)
            elif len(groups) == 1:  # 日
                day = int(groups[0])
                return datetime.date(base_date.year, base_date.month, day)
    
    return None


def parse_time_range(text: str, base_date: datetime.date = None) -> Tuple[Optional[datetime.date], Optional[datetime.date]]:
    """从文本中解析时间范围"""
    if base_date is None:
        base_date = datetime.date.today()
    
    text = text.lower()
    
    # 本周
    if '本周' in text or '这周' in text:
        # 获取本周的开始和结束日期（周一到周日）
        days_since_monday = base_date.weekday()
        start_date = base_date - datetime.timedelta(days=days_since_monday)
        end_date = start_date + datetime.timedelta(days=6)
        return start_date, end_date
    
    # 上周
    elif '上周' in text or '上一周' in text:
        days_since_monday = base_date.weekday()
        this_monday = base_date - datetime.timedelta(days=days_since_monday)
        start_date = this_monday - datetime.timedelta(days=7)
        end_date = start_date + datetime.timedelta(days=6)
        return start_date, end_date
    
    # 本月
    elif '本月' in text or '这个月' in text:
        start_date = base_date.replace(day=1)
        # 获取下个月的第一天，然后减去一天得到本月最后一天
        if base_date.month == 12:
            end_date = datetime.date(base_date.year + 1, 1, 1) - datetime.timedelta(days=1)
        else:
            end_date = datetime.date(base_date.year, base_date.month + 1, 1) - datetime.timedelta(days=1)
        return start_date, end_date
    
    # 上月
    elif '上月' in text or '上个月' in text:
        if base_date.month == 1:
            start_date = datetime.date(base_date.year - 1, 12, 1)
            end_date = datetime.date(base_date.year, 1, 1) - datetime.timedelta(days=1)
        else:
            start_date = datetime.date(base_date.year, base_date.month - 1, 1)
            end_date = base_date.replace(day=1) - datetime.timedelta(days=1)
        return start_date, end_date
    
    # 最近N天
    recent_days_match = re.search(r'最近(\d+)天', text)
    if recent_days_match:
        days = int(recent_days_match.group(1))
        end_date = base_date
        start_date = base_date - datetime.timedelta(days=days - 1)
        return start_date, end_date
    
    return None, None