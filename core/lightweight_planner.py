"""轻量级规划器实现

解决问题：Supervisor 每轮都进行大模型决策的性能问题
方案：规则引擎 -> 小模型 -> 大模型的三级决策机制
"""

import re
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import OrderedDict

from .enhanced_state import IntentType, EnhancedState
# from core.intent_recognizer import IntentCache  # removed: module not present
# from agents.config import get_llm  # removed: not used here
from utils.logger import logger
from utils.common_parsers import intent_to_agent_mapping
from config import config


@dataclass
class PlanResult:
    """规划结果"""
    intent: IntentType
    confidence: float
    agent: str
    method: str  # 'rule', 'lite_model', 'llm'
    processing_time: float
    entities: Dict = None
    
    @property
    def needs_llm(self) -> bool:
        """是否需要大模型处理"""
        return self.confidence < 0.6


class IntentCache:
    """简单LRU意图缓存，按 (user_input, context) 作为键缓存 (intent, confidence, entities)。"""
    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self._store: "OrderedDict[Tuple[str, str], Tuple[IntentType, float, Dict]]" = OrderedDict()

    def _make_key(self, user_input: str, context: str) -> Tuple[str, str]:
        ui = (user_input or "").strip()
        ctx = (context or "").strip()
        return (ui, ctx)

    def get(self, user_input: str, context: str):
        key = self._make_key(user_input, context)
        if key in self._store:
            value = self._store.pop(key)
            # 访问后移动到末尾，表示最近使用
            self._store[key] = value
            return value
        return None

    def put(self, user_input: str, intent: IntentType, confidence: float, entities: Dict, context: str):
        key = self._make_key(user_input, context)
        value = (intent, confidence, entities)
        if key in self._store:
            self._store.pop(key)
        elif len(self._store) >= self.max_size:
            # 弹出最久未使用
            self._store.popitem(last=False)
        self._store[key] = value


class RuleBasedClassifier:
    """基于规则的快速意图分类器
    
    优势：
    - 响应时间 < 10ms
    - 零成本
    - 高准确率（针对明确意图）
    """
    
    # 预编译正则表达式，提升性能
    INTENT_PATTERNS = {
        IntentType.RECORD_MEAL: [
            # 修改此规则，避免与"吃了什么"等查询冲突
            re.compile(r'(我|今天|昨天|刚才).*(吃了|喝了)(?!.*(什么|吗|呢))', re.IGNORECASE),
            re.compile(r'(早餐|午餐|晚餐|夜宵)', re.IGNORECASE),
            re.compile(r'记录.*饮食', re.IGNORECASE),
            re.compile(r'(鸡蛋|牛奶|米饭|面条|面包|水果|苹果|香蕉|橙子|土豆|胡萝卜|白菜|肉|鱼|虾)', re.IGNORECASE),
            re.compile(r'吃.*[的了]', re.IGNORECASE),  # 新增：匹配"吃了"、"吃的"等模式
        ],
        IntentType.RECORD_EXERCISE: [
            re.compile(r'(我|今天|昨天|刚才).*(跑步|游泳|健身|锻炼)', re.IGNORECASE),
            re.compile(r'运动.*分钟', re.IGNORECASE),
            re.compile(r'记录.*运动', re.IGNORECASE),
            re.compile(r'(图片|截图|识别).*运动', re.IGNORECASE),
        ],
        IntentType.QUERY: [
            # 增强查询规则，加入"搜索"、"吗"等关键词
            re.compile(r'(查询|查看|显示|搜索).*(记录|数据)', re.IGNORECASE),
            re.compile(r'.*吃了什么', re.IGNORECASE),
            re.compile(r'.*做了什么运动', re.IGNORECASE),
            # 移除过于宽泛的"昨天呢"匹配，改为在_context_enhanced_classification中处理
            re.compile(r'.*吗\??$', re.IGNORECASE), # 只处理"...吗"等问句
        ],
        IntentType.GENERATE_REPORT: [
            re.compile(r'生成.*报告', re.IGNORECASE),
            re.compile(r'(分析|总结).*(饮食|运动|健康)', re.IGNORECASE),
            re.compile(r'健康.*报告', re.IGNORECASE),
        ],
        IntentType.ADVICE: [
            re.compile(r'(推荐|建议|介绍).*(食物|运动|方法)', re.IGNORECASE),
            re.compile(r'怎么.*减肥', re.IGNORECASE),
            re.compile(r'如何.*健身', re.IGNORECASE),
        ]
    }
    
    # 否定词检测
    NEGATION_PATTERN = re.compile(r'(不是|没有|别|不要|不想)', re.IGNORECASE)
    
    def classify(self, text: str, context: str = "") -> PlanResult:
        """基于规则的快速分类"""
        start_time = time.time()
        
        # 检查否定词
        if self.NEGATION_PATTERN.search(text):
            return PlanResult(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                agent="general",
                method="rule",
                processing_time=time.time() - start_time
            )
        
        # 计算各意图的匹配分数
        scores = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0
            matches = 0
            
            for pattern in patterns:
                if pattern.search(text):
                    matches += 1
                    score += 1
            
            if matches > 0:
                # 归一化分数：匹配数量 / 总模式数量
                base_score = score / len(patterns)
                
                # 改进的置信度计算
                # 1. 文本长度权重（短文本匹配可能更准确）
                words = text.split()
                length_weight = min(1.0, 20 / len(words)) if len(words) > 0 else 1.0
                
                # 2. 匹配质量权重（多个模式匹配更可靠）
                quality_weight = min(1.0, matches / 2)  # 2个或以上匹配给满分
                
                # 3. 最终分数
                final_score = base_score * length_weight * quality_weight
                scores[intent] = final_score
        
        # 上下文增强
        if context and not scores:
            scores = self._context_enhanced_classification(text, context)
        
        processing_time = time.time() - start_time
        
        if scores:
            best_intent = max(scores, key=scores.get)
            confidence = scores[best_intent]
            
            # 更保守的置信度增强（避免过度自信）
            enhanced_confidence = min(confidence + 0.3, 0.85)  # 最高0.85，避免过度自信
            
            # 调整阈值，要求更高的基础置信度
            if enhanced_confidence >= 0.6:  # 提高阈值到0.6
                return PlanResult(
                    intent=best_intent,
                    confidence=enhanced_confidence,
                    agent=intent_to_agent_mapping(best_intent),
                    method="rule",
                    processing_time=processing_time
                )
        
        return PlanResult(
            intent=IntentType.UNKNOWN,
            confidence=0.0,
            agent="general",
            method="rule",
            processing_time=processing_time
        )
    
    def _context_enhanced_classification(self, text: str, context: str) -> Dict[IntentType, float]:
        """基于上下文的增强分类"""
        scores = {}
        
        # 处理模糊查询（如"昨天呢"）
        if re.search(r'(昨天|前天|那天)呢?$', text, re.IGNORECASE):
            if '饮食' in context or '吃' in context:
                scores[IntentType.QUERY] = 0.8
            elif '运动' in context or '锻炼' in context:
                scores[IntentType.QUERY] = 0.8
        
        return scores


class LiteModelClassifier:
    """小模型分类器
    
    用于处理规则引擎无法处理的复杂情况
    成本比大模型低 80%，速度快 5 倍
    """
    
    def __init__(self):
        # 使用模拟的轻量级模型
        class MockLiteModel:
            def invoke(self, prompt):
                class MockResponse:
                    def __init__(self):
                        self.content = '{"intent": "unknown", "confidence": 0.5}'
                return MockResponse()
        
        self.lite_model = MockLiteModel()
    
    def classify(self, text: str, context: str = "") -> PlanResult:
        """使用小模型进行分类"""
        start_time = time.time()
        
        prompt = self._build_lite_prompt(text, context)
        
        try:
            response = self.lite_model.invoke(prompt)
            result = self._parse_lite_response(response.content)
            
            processing_time = time.time() - start_time
            
            return PlanResult(
                intent=result['intent'],
                confidence=result['confidence'],
                agent=intent_to_agent_mapping(result['intent']),
                method="lite_model",
                processing_time=processing_time,
                entities=result.get('entities', {})
            )
            
        except Exception as e:
            logger.warning(f"Lite model classification failed: {e}")
            return PlanResult(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                agent="general",
                method="lite_model",
                processing_time=time.time() - start_time
            )
    
    def _build_lite_prompt(self, text: str, context: str) -> str:
        """构建小模型提示词（简化版）"""
        return f"""分析用户意图，返回JSON格式。

用户输入: {text}
上下文: {context}

意图类型:
- record_meal: 记录饮食
- record_exercise: 记录运动  
- query: 查询记录
- generate_report: 生成报告
- advice: 寻求建议
- unknown: 未知

返回格式: {{"intent": "意图", "confidence": 0.8}}
只返回JSON，不要解释："""
    
    def _parse_lite_response(self, response: str) -> Dict:
        """解析小模型响应"""
        try:
            import json
            # 清理响应
            cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            data = json.loads(cleaned)
            
            # 验证和转换
            intent_str = data.get('intent', 'unknown')
            intent = IntentType(intent_str) if intent_str in [e.value for e in IntentType] else IntentType.UNKNOWN
            
            return {
                'intent': intent,
                'confidence': float(data.get('confidence', 0.0)),
                'entities': data.get('entities', {})
            }
        except Exception:
            return {
                'intent': IntentType.UNKNOWN,
                'confidence': 0.0,
                'entities': {}
            }


class LightweightPlanner:
    """轻量级规划器
    
    三级决策机制：
    1. 规则引擎（快速路径）- 90% 的明确意图
    2. 小模型（中等路径）- 复杂但常见的情况
    3. 大模型（慢路径）- 兜底处理
    """
    
    def __init__(self, cache_size: int = 200, rule_classifier=None, lite_classifier=None, llm_service=None):
        self.rule_classifier = rule_classifier or RuleBasedClassifier()
        self.lite_classifier = lite_classifier or LiteModelClassifier()
        self.cache = IntentCache(cache_size)
        
        # 使用模拟的大模型服务
        if llm_service:
            self.llm = llm_service
        else:
            class MockLLMService:
                def invoke(self, prompt):
                    class MockResponse:
                        def __init__(self):
                            self.content = '{"intent": "unknown", "confidence": 0.5}'
                    return MockResponse()
            self.llm = MockLLMService()
        
        # 性能统计
        self.stats = {
            'rule_hits': 0,
            'lite_hits': 0,
            'llm_hits': 0,
            'cache_hits': 0
        }
    
    def plan(self, user_input: str, context: str = "", state: Optional[EnhancedState] = None) -> PlanResult:
        """智能规划，选择最优决策路径；当配置启用 llm-only 时，直接用LLM结合state识别意图"""
        logger.debug(
            f"Starting planning for user_input: '{user_input[:30]}...' with context: '{context[:50]}...'"
        )
        start_time = time.time()

        # 0. LLM-only 模式：绕过规则与小模型
        if getattr(config.intent, 'llm_only_mode', False):
            return self._llm_intent_with_state(user_input, context, state, start_time)

        # 1. 缓存查询（最快路径）
        cached_result = self.cache.get(user_input, context)
        if cached_result:
            self.stats['cache_hits'] += 1
            intent, confidence, entities = cached_result
            logger.info(
                f"Cache hit for '{user_input[:30]}...'. Intent: {intent.value}, Confidence: {confidence:.2f}"
            )
            return PlanResult(
                intent=intent,
                confidence=confidence,
                agent=intent_to_agent_mapping(intent),
                method="cache",
                processing_time=time.time() - start_time,
                entities=entities
            )
        
        # 2. 规则引擎（快速路径）
        rule_result = self.rule_classifier.classify(user_input, context)
        logger.debug(
            f"Rule engine result for '{user_input[:30]}...': Intent: {rule_result.intent.value}, Confidence: {rule_result.confidence:.2f}"
        )
        # 使用可配置阈值
        if rule_result.confidence >= getattr(config.intent, 'rule_confidence_threshold', 0.6):  # 高置信度
            self.stats['rule_hits'] += 1
            self._cache_result(user_input, context, rule_result)
            logger.info(f"Rule engine hit: {rule_result.intent.value} (confidence: {rule_result.confidence:.2f})")
            return rule_result
        
        # 3. 小模型（中等路径）
        lite_result = self.lite_classifier.classify(user_input, context)
        logger.debug(
            f"Lite model result for '{user_input[:30]}...': Intent: {lite_result.intent.value}, Confidence: {lite_result.confidence:.2f}"
        )
        # 使用可配置阈值
        if lite_result.confidence >= getattr(config.intent, 'lite_confidence_threshold', 0.65):  # 中等置信度
            self.stats['lite_hits'] += 1
            self._cache_result(user_input, context, lite_result)
            logger.info(f"Lite model hit: {lite_result.intent.value} (confidence: {lite_result.confidence:.2f})")
            return lite_result
        
        # 4. 大模型兜底（慢路径）
        llm_result = self._fallback_to_llm(user_input, context)
        self.stats['llm_hits'] += 1
        self._cache_result(user_input, context, llm_result)
        logger.info(f"LLM fallback: {llm_result.intent.value} (confidence: {llm_result.confidence:.2f})")
        return llm_result

    def _llm_intent_with_state(self, user_input: str, context: str, state: Optional[EnhancedState], start_time: float) -> PlanResult:
        """直接通过大模型进行意图识别，并可利用 EnhancedState 提供的上下文与当前意图。"""
        # 构造更丰富的上下文
        dialog_summary = ""
        recent_intents = []
        entities = {}
        turn_id = None
        has_meaningful_context = False
        ds = None
        
        if state and getattr(config.intent, 'state_enhanced_llm', True):
            try:
                ds = state.get('dialog_state')
                turn_id = state.get('turn_id')
                if ds:
                    # 收集最近意图与实体
                    try:
                        recent_intents = [i.value for i in ds.get_recent_intents() if i and i != IntentType.UNKNOWN]
                    except Exception:
                        recent_intents = []
                    try:
                        entities = ds.get_context_entities() or {}
                    except Exception:
                        entities = {}
                    try:
                        # 如果 EnhancedState 提供 summary，优先使用
                        if hasattr(ds, 'context_summary') and ds.context_summary:
                            dialog_summary = ds.context_summary
                    except Exception:
                        pass
                    
                    # 判断是否有有意义的上下文
                    has_meaningful_context = (
                        bool(recent_intents) or 
                        bool(entities) or 
                        bool(dialog_summary) or
                        ('饮食' in context or '吃' in context or '运动' in context or '锻炼' in context)
                    )
            except Exception:
                pass

        # 特殊处理：如果是"昨天呢"类似的查询，但没有有意义的上下文，直接返回unknown
        if re.search(r'(昨天|前天|那天)呢?$', user_input, re.IGNORECASE):
            if not has_meaningful_context:
                return PlanResult(
                    intent=IntentType.UNKNOWN,
                    confidence=0.95,
                    agent="general",
                    method="llm_only",
                    processing_time=time.time() - start_time
                )

        # 生成带有历史意图标注的对话上下文块
        annotated_history_block = ""
        # 优先从 DialogState.turn_history 生成用户消息 + 意图标注
        try:
            if ds and hasattr(ds, 'get_recent_turns'):
                recent_turns = ds.get_recent_turns(5)
                if recent_turns:
                    lines = []
                    for t in recent_turns:
                        intent_val = t.intent.value if getattr(t, 'intent', None) else 'unknown'
                        try:
                            conf_val = f"{t.confidence:.2f}"
                        except Exception:
                            conf_val = ""
                        conf_part = f", 置信度: {conf_val}" if conf_val else ""
                        # 在用户消息后面追加识别出的意图类型
                        lines.append(f"用户: {t.user_input} （识别意图: {intent_val}{conf_part}）")
                    if lines:
                        annotated_history_block = "对话上下文（含历史意图）:\n" + "\n".join(lines)
        except Exception:
            pass

        # 拼装提示词
        context_block = f"上下文: {context}" if context else ""
        intents_block = f"最近意图: {', '.join(recent_intents)}" if recent_intents else ""
        entities_block = f"上下文实体: {entities}" if entities else ""
        # 将带意图标注的历史对话一起加入提示词
        extra_block = "\n".join([b for b in [intents_block, annotated_history_block] if b])

        prompt = f"""你是意图识别专家。请基于用户输入以及给定的state上下文，输出JSON。

{extra_block}
用户输入: {user_input}

意图类型:
- record_meal: 记录饮食信息
- record_exercise: 记录运动信息
- query: 查询历史记录、包括查询饮食记录和运动记录
- generate_report: 生成健康报告
- advice: 寻求健康建议
- unknown: 无法确定意图

注意：如果用户进行模糊查询，但没有明确的上下文（如最近没有饮食或运动相关对话），应该识别为unknown。

返回格式:
{{
  "intent": "意图类型",
  "confidence": 0.9,
  "entities": {{"key": "value"}},
  "reasoning": "简要说明"
}}

只返回JSON，不要其他内容："""
        try:
            print(prompt)
            response = self.llm.invoke(prompt)
            result = self._parse_llm_response(response.content)
            # 命中计数
            self.stats['llm_hits'] += 1
            plan = PlanResult(
                intent=result['intent'],
                confidence=result['confidence'],
                agent=intent_to_agent_mapping(result['intent']),
                method="llm_only",
                processing_time=time.time() - start_time,
                entities=result.get('entities', {})
            )
            # 根据置信度决定是否写入缓存
            self._cache_result(user_input, context, plan)
            return plan
        except Exception as e:
            logger.error(f"LLM intent with state failed: {e}")
            return PlanResult(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                agent="general",
                method="llm_only",
                processing_time=time.time() - start_time
            )
    
    def _fallback_to_llm(self, user_input: str, context: str) -> PlanResult:
        """大模型兜底处理"""
        start_time = time.time()
        
        prompt = f"""你是意图识别专家。分析用户输入，返回JSON格式结果。

上下文: {context}
用户输入: {user_input}

意图类型:
- record_meal: 记录饮食信息
- record_exercise: 记录运动信息
- query: 查询历史记录
- generate_report: 生成健康报告
- advice: 寻求健康建议
- unknown: 无法确定意图

返回格式:
{{
  "intent": "意图类型",
  "confidence": 0.95,
  "entities": {{"key": "value"}},
  "reasoning": "判断理由"
}}

只返回JSON，不要其他内容："""
        
        try:
            response = self.llm.invoke(prompt)
            result = self._parse_llm_response(response.content)
            
            return PlanResult(
                intent=result['intent'],
                confidence=result['confidence'],
                agent=intent_to_agent_mapping(result['intent']),
                method="llm",
                processing_time=time.time() - start_time,
                entities=result.get('entities', {})
            )
            
        except Exception as e:
            logger.error(f"LLM fallback failed: {e}")
            return PlanResult(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                agent="general",
                method="llm",
                processing_time=time.time() - start_time
            )
    
    def _parse_llm_response(self, response: str) -> Dict:
        """解析大模型响应"""
        try:
            import json
            cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            data = json.loads(cleaned)
            
            intent_str = data.get('intent', 'unknown')
            intent = IntentType(intent_str) if intent_str in [e.value for e in IntentType] else IntentType.UNKNOWN
            
            return {
                'intent': intent,
                'confidence': float(data.get('confidence', 0.0)),
                'entities': data.get('entities', {})
            }
        except Exception:
            return {
                'intent': IntentType.UNKNOWN,
                'confidence': 0.0,
                'entities': {}
            }
    
    def _cache_result(self, user_input: str, context: str, result: PlanResult):
        """缓存结果"""
        if result.confidence >= 0.7:  # 只缓存高置信度结果
            self.cache.put(
                user_input=user_input,
                intent=result.intent,
                confidence=result.confidence,
                entities=result.entities or {},
                context=context
            )
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        total = sum(self.stats.values())
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'total_requests': total,
            'rule_percentage': self.stats['rule_hits'] / total * 100,
            'lite_percentage': self.stats['lite_hits'] / total * 100,
            'llm_percentage': self.stats['llm_hits'] / total * 100,
            'cache_percentage': self.stats['cache_hits'] / total * 100
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'rule_hits': 0,
            'lite_hits': 0,
            'llm_hits': 0,
            'cache_hits': 0
        }


# 使用示例
if __name__ == "__main__":
    planner = LightweightPlanner()
    
    # 测试用例
    test_cases = [
        "我今天早餐吃了鸡蛋和牛奶",  # 应该命中规则引擎
        "昨天呢？",  # 需要上下文，可能命中小模型
        "帮我分析一下最近的健康状况",  # 复杂查询，可能需要大模型
    ]
    
    for case in test_cases:
        result = planner.plan(case)
        print(f"输入: {case}")
        print(f"结果: {result.intent.value} (置信度: {result.confidence:.2f}, 方法: {result.method})")
        print(f"处理时间: {result.processing_time*1000:.1f}ms")
        print("---")
    
    # 性能统计
    print("性能统计:")
    stats = planner.get_performance_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")