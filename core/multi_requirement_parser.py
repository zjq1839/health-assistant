"""多需求解析器

处理包含多个要求的复杂查询，分解为子任务并为每个子任务生成相应的响应
支持使用轻量级模型进行需求分解，主模型进行内容生成
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from core.enhanced_state import IntentType
from utils.logger import logger


class RequirementType(Enum):
    """需求类型"""
    QUERY = "query"  # 查询类
    RECORD = "record"  # 记录类
    ADVICE = "advice"  # 建议类
    ANALYSIS = "analysis"  # 分析类
    GENERATE = "generate"  # 生成类
    CALCULATE = "calculate"  # 计算类
    UNKNOWN = "unknown"


@dataclass
class ParsedRequirement:
    """解析后的单个需求"""
    id: str  # 需求ID
    type: RequirementType  # 需求类型
    description: str  # 需求描述
    intent: IntentType  # 对应的意图
    priority: int  # 优先级 (1-5)
    dependencies: List[str]  # 依赖的其他需求ID
    keywords: List[str]  # 关键词
    entities: Dict[str, Any]  # 提取的实体
    confidence: float  # 解析置信度


@dataclass
class MultiRequirementParseResult:
    """多需求解析结果"""
    original_query: str  # 原始查询
    requirements: List[ParsedRequirement]  # 解析出的需求列表
    execution_order: List[str]  # 执行顺序（需求ID列表）
    complexity: str  # 复杂度等级: simple, medium, complex
    parsing_method: str  # 解析方法: lite_model, full_model, hybrid
    total_confidence: float  # 总体置信度


class MultiRequirementParser:
    """多需求解析器
    
    功能：
    1. 识别查询中的多个需求
    2. 分解为独立的子任务
    3. 确定执行顺序和依赖关系
    4. 为每个需求生成对应的处理策略
    """
    
    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        
        # 需求识别的关键词模式
        self.requirement_patterns = {
            RequirementType.QUERY: [
                r'查询|查看|显示|统计|多少|什么时候|哪些|列出',
                r'.*吃了什么|.*做了什么运动|.*情况如何',
                r'数据|记录|历史|趋势'
            ],
            RequirementType.RECORD: [
                r'记录|添加|输入|保存',
                r'我.*吃了|我.*喝了|我.*做了',
                r'今天.*餐|运动.*分钟'
            ],
            RequirementType.ADVICE: [
                r'建议|推荐|应该|怎么办|如何',
                r'吃什么好|做什么运动|怎么减肥',
                r'方案|计划'
            ],
            RequirementType.ANALYSIS: [
                r'分析|评估|比较|总结',
                r'趋势|变化|效果|原因',
                r'为什么|怎么样|如何看待'
            ],
            RequirementType.GENERATE: [
                r'生成|制作|创建|写',
                r'报告|计划|清单|图表',
                r'帮我.*制定|给我.*方案'
            ],
            RequirementType.CALCULATE: [
                r'计算|算|消耗|燃烧',
                r'卡路里|热量|BMI|体重',
                r'多少.*卡|消耗.*热量'
            ]
        }
        
        # 连接词模式（用于分割多个需求）
        self.connectors = [
            r'，|,|;|；',  # 标点符号
            r'还有|另外|还要|以及|而且|并且',  # 并列连词
            r'然后|接着|之后|最后|其次',  # 顺序连词
            r'同时|一起|一边.*一边',  # 同时性连词
        ]
    
    def parse(self, query: str, context: str = "") -> MultiRequirementParseResult:
        """解析多需求查询"""
        logger.info(f"开始解析多需求查询: {query[:50]}...")
        
        # 1. 预处理和复杂度评估
        complexity = self._assess_complexity(query)
        logger.debug(f"查询复杂度: {complexity}")
        
        # 2. 选择解析方法
        parsing_method = self._choose_parsing_method(complexity, query)
        logger.debug(f"选择解析方法: {parsing_method}")
        
        # 3. 执行解析
        if parsing_method == "lite_model":
            requirements = self._parse_with_lite_model(query, context)
        elif parsing_method == "full_model":
            requirements = self._parse_with_full_model(query, context)
        else:  # hybrid
            requirements = self._parse_with_hybrid_method(query, context)
        
        # 4. 后处理：确定执行顺序和依赖关系
        execution_order = self._determine_execution_order(requirements)
        
        # 5. 计算总体置信度
        total_confidence = self._calculate_total_confidence(requirements)
        
        result = MultiRequirementParseResult(
            original_query=query,
            requirements=requirements,
            execution_order=execution_order,
            complexity=complexity,
            parsing_method=parsing_method,
            total_confidence=total_confidence
        )
        
        logger.info(f"解析完成，识别出 {len(requirements)} 个需求，总置信度: {total_confidence:.2f}")
        return result
    
    def _assess_complexity(self, query: str) -> str:
        """评估查询复杂度"""
        # 计算各种复杂度指标
        word_count = len(query.split())
        connector_count = sum(1 for pattern in self.connectors 
                            if re.search(pattern, query))
        
        # 检查是否包含多种需求类型
        type_matches = 0
        for req_type, patterns in self.requirement_patterns.items():
            if any(re.search(pattern, query, re.IGNORECASE) for pattern in patterns):
                type_matches += 1
        
        # 复杂度判定
        if word_count <= 10 and connector_count == 0 and type_matches <= 1:
            return "simple"
        elif word_count <= 25 and connector_count <= 2 and type_matches <= 2:
            return "medium"
        else:
            return "complex"
    
    def _choose_parsing_method(self, complexity: str, query: str) -> str:
        """选择解析方法"""
        if complexity == "simple":
            return "lite_model"
        elif complexity == "medium":
            return "lite_model"  # 优先使用轻量级模型
        else:
            return "hybrid"  # 复杂查询使用混合方法
    
    def _parse_with_lite_model(self, query: str, context: str) -> List[ParsedRequirement]:
        """使用轻量级模型解析"""
        logger.debug("使用轻量级模型进行需求解析")
        
        try:
            prompt = self._build_lite_parsing_prompt(query, context)
            
            if self.llm_service and hasattr(self.llm_service, 'lite_llm'):
                response = self.llm_service.lite_llm.invoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)
            else:
                # 降级到规则解析
                return self._parse_with_rules(query)
            
            return self._parse_lite_model_response(content, query)
            
        except Exception as e:
            logger.warning(f"轻量级模型解析失败: {e}，降级到规则解析")
            return self._parse_with_rules(query)
    
    def _parse_with_full_model(self, query: str, context: str) -> List[ParsedRequirement]:
        """使用完整模型解析"""
        logger.debug("使用完整模型进行需求解析")
        
        try:
            prompt = self._build_full_parsing_prompt(query, context)
            
            if self.llm_service:
                response = self.llm_service.generate_response(prompt, context)
            else:
                # 降级到规则解析
                return self._parse_with_rules(query)
            
            return self._parse_full_model_response(response, query)
            
        except Exception as e:
            logger.warning(f"完整模型解析失败: {e}，降级到规则解析")
            return self._parse_with_rules(query)
    
    def _parse_with_hybrid_method(self, query: str, context: str) -> List[ParsedRequirement]:
        """使用混合方法解析（先轻量级，后完整模型）"""
        logger.debug("使用混合方法进行需求解析")
        
        # 先用轻量级模型进行初步分解
        lite_requirements = self._parse_with_lite_model(query, context)
        
        # 如果轻量级模型置信度不够，使用完整模型
        avg_confidence = sum(req.confidence for req in lite_requirements) / len(lite_requirements) if lite_requirements else 0
        
        if avg_confidence < 0.7:
            logger.debug("轻量级模型置信度不足，切换到完整模型")
            return self._parse_with_full_model(query, context)
        
        return lite_requirements
    
    def _parse_with_rules(self, query: str) -> List[ParsedRequirement]:
        """使用规则解析（降级方案）"""
        logger.debug("使用规则进行需求解析")
        
        requirements = []
        
        # 按连接词分割查询
        segments = self._split_by_connectors(query)
        
        for i, segment in enumerate(segments):
            req_type = self._classify_requirement_type(segment)
            intent = self._map_type_to_intent(req_type)
            
            requirement = ParsedRequirement(
                id=f"req_{i+1}",
                type=req_type,
                description=segment.strip(),
                intent=intent,
                priority=i+1,
                dependencies=[],
                keywords=self._extract_keywords(segment),
                entities={},
                confidence=0.6  # 规则解析的基础置信度
            )
            
            requirements.append(requirement)
        
        return requirements
    
    def _split_by_connectors(self, query: str) -> List[str]:
        """按连接词分割查询"""
        # 构建完整的分割模式
        full_pattern = '|'.join(self.connectors)
        
        # 分割并清理
        segments = re.split(full_pattern, query)
        return [seg.strip() for seg in segments if seg.strip()]
    
    def _classify_requirement_type(self, text: str) -> RequirementType:
        """分类需求类型"""
        scores = {}
        
        for req_type, patterns in self.requirement_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 1
            
            if score > 0:
                scores[req_type] = score / len(patterns)
        
        if scores:
            return max(scores, key=scores.get)
        
        return RequirementType.UNKNOWN
    
    def _map_type_to_intent(self, req_type: RequirementType) -> IntentType:
        """将需求类型映射到意图"""
        mapping = {
            RequirementType.QUERY: IntentType.QUERY,
            RequirementType.RECORD: IntentType.RECORD_MEAL,  # 默认，可以根据内容细化
            RequirementType.ADVICE: IntentType.ADVICE,
            RequirementType.ANALYSIS: IntentType.GENERATE_REPORT,
            RequirementType.GENERATE: IntentType.GENERATE_REPORT,
            RequirementType.CALCULATE: IntentType.ADVICE,
            RequirementType.UNKNOWN: IntentType.UNKNOWN
        }
        
        return mapping.get(req_type, IntentType.UNKNOWN)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取（基于正则表达式，避免额外依赖）
        import re
        
        # 使用正则表达式分割中文文本
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+\d*', text)
        
        # 过滤停用词和短词
        stopwords = {'我', '的', '是', '在', '了', '和', '与', '或', '但', '然后', '还有', '这', '那', '有', '没有', '可以', '需要', '应该'}
        keywords = [word for word in words 
                   if len(word) > 1 and word not in stopwords]
        
        return keywords[:5]  # 限制关键词数量
    
    def _build_lite_parsing_prompt(self, query: str, context: str) -> str:
        """构建轻量级模型解析提示词"""
        return f"""分析以下查询，注意其中可能包含单个或多个需求，识别其中包含的需求并分解。

查询: {query}
上下文: {context}

需求类型:
- query: 查询数据
- record: 记录信息
- advice: 获取建议
- analysis: 分析数据
- generate: 生成内容
- calculate: 计算数值

请返回JSON格式:
{{
  "requirements": [
    {{
      "id": "req_1",
      "type": "query",
      "description": "具体需求描述",
      "priority": 1,
      "confidence": 0.8
    }}
  ]
}}

只返回JSON，不要解释："""
    
    def _build_full_parsing_prompt(self, query: str, context: str) -> str:
        """构建完整模型解析提示词"""
        return f"""你是一个智能需求分析专家，请分析以下复杂查询，识别其中包含的多个需求并详细分解。

用户查询: {query}
上下文信息: {context}

请按以下步骤分析：

1. **需求识别**: 识别查询中包含的所有独立需求
2. **类型分类**: 为每个需求确定类型（查询、记录、建议、分析、生成、计算等）
3. **依赖关系**: 分析需求之间的依赖关系和执行顺序
4. **实体提取**: 提取每个需求相关的关键实体（时间、食物、运动等）

需求类型定义:
- query: 查询历史数据或信息
- record: 记录新的饮食或运动信息
- advice: 获取健康建议或推荐
- analysis: 分析数据趋势或效果
- generate: 生成报告、计划等内容
- calculate: 计算卡路里、BMI等数值

请返回详细的JSON格式结果，包含每个需求的完整信息。"""
    
    def _parse_lite_model_response(self, response: str, query: str) -> List[ParsedRequirement]:
        """解析轻量级模型响应"""
        try:
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("未找到JSON格式响应")
            
            data = json.loads(json_match.group())
            requirements = []
            
            for i, req_data in enumerate(data.get('requirements', [])):
                req_type = RequirementType(req_data.get('type', 'unknown'))
                intent = self._map_type_to_intent(req_type)
                
                requirement = ParsedRequirement(
                    id=req_data.get('id', f'req_{i+1}'),
                    type=req_type,
                    description=req_data.get('description', ''),
                    intent=intent,
                    priority=req_data.get('priority', i+1),
                    dependencies=req_data.get('dependencies', []),
                    keywords=self._extract_keywords(req_data.get('description', '')),
                    entities=req_data.get('entities', {}),
                    confidence=req_data.get('confidence', 0.7)
                )
                
                requirements.append(requirement)
            
            return requirements
            
        except Exception as e:
            logger.warning(f"解析轻量级模型响应失败: {e}")
            return self._parse_with_rules(query)
    
    def _parse_full_model_response(self, response: str, query: str) -> List[ParsedRequirement]:
        """解析完整模型响应"""
        # 这里可以实现更复杂的响应解析逻辑
        # 暂时使用类似轻量级模型的解析方法
        return self._parse_lite_model_response(response, query)
    
    def _determine_execution_order(self, requirements: List[ParsedRequirement]) -> List[str]:
        """确定执行顺序"""
        # 简单的优先级排序
        sorted_reqs = sorted(requirements, key=lambda x: x.priority)
        return [req.id for req in sorted_reqs]
    
    def _calculate_total_confidence(self, requirements: List[ParsedRequirement]) -> float:
        """计算总体置信度"""
        if not requirements:
            return 0.0
        
        return sum(req.confidence for req in requirements) / len(requirements)


# 便捷函数
def create_multi_requirement_parser() -> MultiRequirementParser:
    """创建多需求解析器实例"""
    try:
        from core.service_container import ConfigurableServiceContainer
        from core.agent_protocol import LLMService
        container = ConfigurableServiceContainer()
        llm_service = container.get(LLMService)
        return MultiRequirementParser(llm_service)
    except Exception as e:
        logger.warning(f"创建解析器时无法获取LLM服务: {e}")
        return MultiRequirementParser(None)


# 使用示例
if __name__ == "__main__":
    # 创建解析器
    parser = create_multi_requirement_parser()
    
    # 测试案例
    test_queries = [
        "查询我昨天的饮食记录，并给我一些今天的健康建议",
        "记录我今天早餐吃了鸡蛋和牛奶，然后分析我这周的营养摄入情况",
        "生成我的健康报告，同时推荐适合我的运动计划",
        "计算我今天消耗的卡路里，查看我的运动历史，还要制定减肥计划"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        print("-" * 50)
        
        result = parser.parse(query)
        
        print(f"复杂度: {result.complexity}")
        print(f"解析方法: {result.parsing_method}")
        print(f"总置信度: {result.total_confidence:.2f}")
        print(f"识别需求数: {len(result.requirements)}")
        
        for req in result.requirements:
            print(f"  {req.id}: {req.type.value} - {req.description[:30]}... (置信度: {req.confidence:.2f})")
        
        print(f"执行顺序: {' -> '.join(result.execution_order)}")