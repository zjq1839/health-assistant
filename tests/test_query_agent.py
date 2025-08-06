import pytest
import datetime
from unittest.mock import Mock, patch
from agents.query_agent import parse_llm_response, extract_query_params
from core.state import State

class TestQueryAgent:
    """测试查询代理的功能"""
    
    def test_parse_llm_response_valid_json(self):
        """测试解析有效的 JSON 响应"""
        response_text = '{"date": "2024-01-15", "query_type": "dietary"}'
        content = "今天吃了什么"
        
        query_date, query_type = parse_llm_response(response_text, content)
        
        assert query_date == "2024-01-15"
        assert query_type == "dietary"
    
    def test_parse_llm_response_invalid_json(self):
        """测试解析无效的 JSON 响应时的备选方法"""
        response_text = "invalid json"
        content = "今天吃了什么"
        
        query_date, query_type = parse_llm_response(response_text, content)
        
        assert query_date == datetime.date.today().isoformat()
        assert query_type == "dietary"
    
    def test_parse_llm_response_exercise_keywords(self):
        """测试运动关键词识别"""
        response_text = "invalid json"
        content = "今天跑步了吗"
        
        query_date, query_type = parse_llm_response(response_text, content)
        
        assert query_type == "exercise"
    
    def test_parse_llm_response_unknown_type(self):
        """测试未知类型的处理"""
        response_text = "invalid json"
        content = "天气怎么样"
        
        query_date, query_type = parse_llm_response(response_text, content)
        
        assert query_type == "unknown"
    
    @patch('agents.query_agent.llm_lite')
    def test_extract_query_params(self, mock_llm):
        """测试提取查询参数"""
        # 模拟 LLM 响应
        mock_response = Mock()
        mock_response.content = '{"date": "2024-01-15", "query_type": "dietary"}'
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response
        
        state = {
            "messages": [("user", "今天吃了什么")]
        }
        
        result = extract_query_params(state)
        
        assert result["query_date"] == "2024-01-15"
        assert result["query_type"] == "dietary"
    
    def test_dietary_keywords(self):
        """测试饮食关键词识别"""
        test_cases = [
            ("今天吃了什么", "dietary"),
            ("早餐吃了鸡蛋", "dietary"),
            ("午餐喝了汤", "dietary"),
            ("晚餐很丰盛", "dietary"),
        ]
        
        for content, expected_type in test_cases:
            _, query_type = parse_llm_response("invalid json", content)
            assert query_type == expected_type
    
    def test_exercise_keywords(self):
        """测试运动关键词识别"""
        test_cases = [
            ("今天运动了吗", "exercise"),
            ("跑步30分钟", "exercise"),
            ("锻炼身体", "exercise"),
        ]
        
        for content, expected_type in test_cases:
            _, query_type = parse_llm_response("invalid json", content)
            assert query_type == expected_type