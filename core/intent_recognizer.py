import re
import hashlib
import json
import threading
from typing import Dict, Tuple, Optional, Any, List
from datetime import datetime, timedelta, timezone

from langchain_ollama import ChatOllama

from utils.performance import LRUCache
from .enhanced_state import IntentType, EnhancedState, DialogTurn
from utils.logger import logger
from config import IntentConfig

# --- 文本规范化与正则预编译 ---

def normalize_text(text: str) -> str:
    """统一处理文本，包括转半角、去多余空格等。"""
    # Mapping from full-width to half-width characters
    full_width = "！＂＃＄％＆＇（）＊＋，－．／０１２３４５６７８９：；＜＝＞？＠ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ［＼］＾＿｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝～"
    half_width = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"

    # Create the translation table
    translation_table = str.maketrans(full_width, half_width)

    # Translate the text
    text = text.translate(translation_table)

    # Replace full-width space with a standard space
    text = text.replace('\u3000', ' ')

    # Normalize other whitespace and punctuation
    text = re.sub(r'[\s,.;!?]+', ' ', text).strip()
    return text


PRECOMPILED_PATTERNS = {
    IntentType.RECORD_MEAL: [
        re.compile(p) for p in [
            r'(吃|早餐|午餐|晚餐|饮食|食物|菜|饭)',
            r'(今天|昨天|前天).*(吃|饮食)',
            r'(记录|添加).*(饮食|食物)',
            r'(减脂餐|想吃|健康餐|营养餐)',
            r'(减脂|减肥|健康).*(想|要).*(吃|饮食|食物)'
        ]
    ],
    IntentType.RECORD_EXERCISE: [
        re.compile(p) for p in [
            r'(运动|锻炼|健身|跑步|游泳|瑜伽)',
            r'(图片|截图|OCR|识别)',
            r'(今天|昨天|前天).*(运动|锻炼)',
            r'(记录|添加).*(运动|锻炼)'
        ]
    ],
    IntentType.GENERATE_REPORT: [
        re.compile(p) for p in [
            r'(生成|查看|显示).*(报告|分析|总结)',
            r'(今天|昨天|本周|本月).*(报告|分析)',
            r'(健康|饮食|运动).*(报告|分析)'
        ]
    ],
    IntentType.QUERY: [
        re.compile(p) for p in [
            r'(查询|查(询)?(记录|数据)|查看).*(吃了|饮食|运动|锻炼|记录|数据)',
            r'(哪天|什么时候).*(饮食|运动|锻炼)',
            r'(昨天|前天|那天)呢?$',
            r'(我的|查看我的).*(记录|数据)'
        ]
    ],
    IntentType.ADVICE: [
        re.compile(p) for p in [
            r'(分享|推荐|给我|介绍).*(菜谱|食谱|方法|建议)',
            r'(怎么|如何).*(减脂|减肥|健身|锻炼)',
            r'(什么|哪些).*(食物|运动|方法).*(好|有效|推荐)',
            r'(请|能否|可以).*(推荐|建议|分享)',
            r'(减脂|减肥|健康).*(菜谱|食谱|方法|计划)'
        ]
    ],
    'negation': re.compile(r'(不是|没有|别|不要)'),
    'conflict': re.compile(r'(但是|不过|其实|实际上)'),
    'context_query': re.compile(r'(昨天|前天|那天)呢?$')
}

# --- 实体抽取相关 ---

TIME_ENTITY_PATTERNS = {
    'today': re.compile(r'今天'),
    'yesterday': re.compile(r'昨天'),
    'day_before_yesterday': re.compile(r'前天'),
    'tomorrow': re.compile(r'明天'),
}

MEAL_TYPE_PATTERNS = {
    'breakfast': re.compile(r'早餐|早上'),
    'lunch': re.compile(r'午餐|中午'),
    'dinner': re.compile(r'晚餐|晚上'),
    'supper': re.compile(r'夜宵'),
}

EXERCISE_TYPE_PATTERNS = {
    'running': re.compile(r'跑步'),
    'swimming': re.compile(r'游泳'),
    'yoga': re.compile(r'瑜伽'),
    'fitness': re.compile(r'健身'),
    'cycling': re.compile(r'骑行'),
}

UNIT_PATTERNS = {
    'duration_minutes': re.compile(r'(\d+)\s*(分钟|min)'),
    'distance_km': re.compile(r'(\d+(\.\d+)?)\s*(公里|km)'),
    'calories_kcal': re.compile(r'(\d+)\s*(大卡|千卡|kcal)'),
    'quantity': re.compile(r'(\d+)\s*(份|个|碗|盘)'),
}


class IntentCache:
    """线程安全的意图识别缓存，支持TTL。"""

    def __init__(self, max_size: int = 200, ttl_seconds: int = 3600 * 24):
        self.cache = LRUCache(max_size=max_size)
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()

    def _generate_key(self, user_input: str, context: str = "") -> str:
        """生成缓存键。"""
        content = f"{user_input}|{context}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, user_input: str, context: str = "") -> Optional[Tuple[IntentType, float, Dict[str, Any]]]:
        """获取缓存，并检查是否过期。"""
        key = self._generate_key(user_input, context)
        with self._lock:
            cached_data = self.cache.get(key)

        if cached_data:
            intent, confidence, entities, expires_at = cached_data
            if datetime.now(timezone.utc) < expires_at:
                logger.debug(f"Intent cache hit for key: {key[:8]}...")
                return intent, confidence, entities
            else:
                logger.debug(f"Intent cache expired for key: {key[:8]}...")
                self.delete(user_input, context) # 惰性删除

        return None

    def put(self, user_input: str, intent: IntentType, confidence: float,
            entities: Dict[str, Any], context: str = ""):
        """缓存意图识别结果。"""
        key = self._generate_key(user_input, context)
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds)
        cached_data = (intent, confidence, entities, expires_at)
        with self._lock:
            self.cache.put(key, cached_data)
        logger.debug(f"Intent cached for key: {key[:8]}... Expires at {expires_at}")

    def delete(self, user_input: str, context: str = ""):
        """显式删除缓存条目。"""
        key = self._generate_key(user_input, context)
        with self._lock:
            self.cache.delete(key)
            logger.debug(f"Intent cache deleted for key: {key[:8]}...")

    def clear(self):
        """清空缓存。"""
        with self._lock:
            self.cache.clear()
            logger.info("Intent cache cleared.")


class EnhancedIntentRecognizer:
    """增强的意图识别器（重构版）"""

    def __init__(self, llm: ChatOllama, cache_size: int = 200, config: Optional[IntentConfig] = None):
        self.llm = llm
        self.intent_cache = IntentCache(max_size=cache_size, ttl_seconds=config.cache_ttl_seconds if config else 3600 * 24)
        self.config: IntentConfig = config or IntentConfig()
        self.keyword_patterns = PRECOMPILED_PATTERNS

    def recognize_intent(self, state: EnhancedState, current_tz: Optional[timezone] = None) -> Tuple[IntentType, float, Dict[str, Any]]:
        """识别用户意图（重构版）。"""
        user_input_raw = get_user_input(state)
        if not user_input_raw:
            return IntentType.UNKNOWN, 0.0, {}

        user_input = normalize_text(user_input_raw)
        context = self._build_context(state)

        # 尝试从缓存获取
        cached_result = self.intent_cache.get(user_input, context)
        if cached_result:
            return cached_result

        # 关键词与实体识别
        keyword_result = self._keyword_based_recognition(user_input, state, current_tz)

        # 若关键词置信度足够高，则直接返回
        if keyword_result[1] >= self.config.keyword_direct_return_threshold:
            self.intent_cache.put(user_input, *keyword_result, context)
            return keyword_result

        # LLM增强识别
        llm_result = self._llm_based_recognition(user_input, context)

        # 融合策略
        final_result = self._combine_results(keyword_result, llm_result, user_input)

        # 缓存最终结果
        self.intent_cache.put(user_input, *final_result, context)
        return final_result

    def _build_context(self, state: EnhancedState) -> str:
        """构建用于LLM和缓存的上下文。"""
        summary = get_context_summary(state)
        history = get_dialog_history(state, 5)
        
        context_parts = []
        if summary:
            context_parts.append(f"摘要: {summary}")

        for turn in history:
            role = "用户" if turn.role == "user" else "助手"
            context_parts.append(f"{role}: {turn.content}")
        
        return "\n".join(context_parts)

    def _keyword_based_recognition(self, user_input: str, state: EnhancedState, current_tz: Optional[timezone]) -> Tuple[IntentType, float, Dict[str, Any]]:
        """基于关键词的意图识别（重构版）。"""
        best_intent = IntentType.UNKNOWN
        best_confidence = 0.0

        # 否定词与冲突信号检测
        negation_detected = bool(self.keyword_patterns['negation'].search(user_input))
        conflict_detected = bool(self.keyword_patterns['conflict'].search(user_input))

        for intent_type, patterns in self.keyword_patterns.items():
            if not isinstance(patterns, list):
                continue

            confidence = sum(0.35 for p in patterns if p.search(user_input))
            if confidence == 0:
                continue

            # 上下文加权（时间衰减）
            confidence += self._contextual_weighting(intent_type, state)

            if confidence > best_confidence:
                best_confidence = confidence
                best_intent = intent_type

        # 特殊处理 "昨天呢" 等情况
        if self.keyword_patterns['context_query'].search(user_input):
            context_intent = self._infer_from_context(state, window=5)
            if context_intent != IntentType.UNKNOWN:
                best_intent, best_confidence = context_intent, 0.9

        # 提取实体
        entities = self._extract_entities(user_input, best_intent, current_tz)

        # 惩罚项
        if negation_detected:
            best_confidence *= 0.5
        if conflict_detected:
            best_confidence *= 0.8

        return best_intent, min(best_confidence, 1.0), entities

    def _contextual_weighting(self, intent_type: IntentType, state: EnhancedState) -> float:
        """上下文加权，带有时间衰减。"""
        history = get_dialog_history(state, 5)
        weight = 0.0
        # 权重从最近的0.2, 0.15, 0.1, 0.05, 0.025递减
        decay_factors = [0.2, 0.15, 0.1, 0.05, 0.025]
        
        for i, turn in enumerate(reversed(history)):
            if turn.intent == intent_type:
                weight += decay_factors[i]
        return weight

    def _infer_from_context(self, state: EnhancedState, window: int = 5) -> IntentType:
        """从上下文中推断意图，用于处理'昨天呢'等。"""
        history = get_dialog_history(state, window)
        for turn in reversed(history):
            if turn.intent not in [IntentType.UNKNOWN, IntentType.QUERY]:
                return turn.intent
        return IntentType.UNKNOWN

    def _extract_entities(self, user_input: str, intent: IntentType, current_tz: Optional[timezone]) -> Dict[str, Any]:
        """提取实体（重构版），包括时间和单位。"""
        entities = {}
        now = datetime.now(current_tz or timezone.utc)

        # 提取时间实体并解析为绝对日期
        for key, pattern in TIME_ENTITY_PATTERNS.items():
            if pattern.search(user_input):
                if key == 'today':
                    entities['time'] = now.isoformat()
                elif key == 'yesterday':
                    entities['time'] = (now - timedelta(days=1)).isoformat()
                elif key == 'day_before_yesterday':
                    entities['time'] = (now - timedelta(days=2)).isoformat()
                elif key == 'tomorrow':
                    entities['time'] = (now + timedelta(days=1)).isoformat()
                break
        
        # 提取其他实体
        if intent == IntentType.RECORD_MEAL:
            for key, pattern in MEAL_TYPE_PATTERNS.items():
                if pattern.search(user_input):
                    entities['meal_type'] = key
                    break
        elif intent == IntentType.RECORD_EXERCISE:
            for key, pattern in EXERCISE_TYPE_PATTERNS.items():
                if pattern.search(user_input):
                    entities['exercise_type'] = key
                    break
        
        # 提取带单位的数值
        for key, pattern in UNIT_PATTERNS.items():
            match = pattern.search(user_input)
            if match:
                entities[key] = float(match.group(1))

        return entities

    def _llm_based_recognition(self, user_input: str, context: str) -> Tuple[IntentType, float, Dict[str, Any]]:
        """基于LLM的意图识别（重构版），增强鲁棒性。"""
        prompt = self._build_llm_prompt(user_input, context)
        try:
            response = self.llm.with_config({
                "temperature": self.config.llm_temperature,
                "stop": ["}", "\n\n"], # 提前停止
            }).invoke(prompt)
            
            content = response.content.strip()
            if len(content) > self.config.llm_max_response_length:
                content = content[:self.config.llm_max_response_length]

            result = self._parse_llm_json(content)
            if not result:
                return IntentType.UNKNOWN, 0.0, {}

            intent = IntentType.from_string(result.get('intent', 'unknown'))
            confidence = float(result.get('confidence', 0.0))
            entities = result.get('entities', {})
            
            logger.debug(f"LLM raw response: {content}")
            logger.info(f"LLM parsed intent: {intent.value}, confidence: {confidence}")
            return intent, confidence, entities

        except Exception as e:
            logger.error(f"LLM intent recognition failed: {e}")
            return IntentType.UNKNOWN, 0.0, {}

    def _build_llm_prompt(self, user_input: str, context: str) -> str:
        return f"""你是一个意图识别专家。基于用户输入和对话上下文，识别用户意图。

对话上下文:
{context}

当前用户输入: {user_input}

意图类型: {', '.join([i.value for i in IntentType])}

请严格按照以下JSON格式返回，不要添加任何额外说明:
{{
    "intent": "意图类型",
    "confidence": 0.0-1.0的置信度,
    "entities": {{"实体名": "实体值"}},
    "reasoning": "简要说明判断理由"
}}"""

    def _parse_llm_json(self, content: str) -> Optional[Dict[str, Any]]:
        """三段式解析LLM返回的JSON，增强鲁棒性。

        通过三种方式尝试解析JSON:
        1. 从代码块中提取;
        2. 直接解析单行JSON;
        3. 通过括号平衡提取。
        """
        # 1. 尝试直接解析代码块中的JSON
        match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 2. 尝试解析单行JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 3. 尝试平衡括号抽取
        try:
            # 查找第一个左花括号作为JSON开始的位置
            start = content.find('{')
            if start != -1:
                # 寻找与之匹配的 '}'
                brace_level = 1
                for i in range(start + 1, len(content)):
                    if content[i] == '{':
                        brace_level += 1
                    elif content[i] == '}':
                        brace_level -= 1
                    if brace_level == 0:
                        return json.loads(content[start:i+1])
        except (json.JSONDecodeError, IndexError):
            pass
        
        logger.warning(f"Failed to parse JSON from LLM response: {content[:200]}...")
        return None

    def _combine_results(
        self, 
        keyword_result: Tuple[IntentType, float, Dict[str, Any]], 
        llm_result: Tuple[IntentType, float, Dict[str, Any]],
        user_input: str
    ) -> Tuple[IntentType, float, Dict[str, Any]]:
        """融合关键词和LLM结果（重构版），带冲突惩罚。"""
        kw_intent, kw_conf, kw_entities = keyword_result
        llm_intent, llm_conf, llm_entities = llm_result

        final_intent, final_conf, final_entities = kw_intent, kw_conf, kw_entities
        combined_entities = {**kw_entities, **llm_entities}

        # 一致性检查
        if kw_intent == llm_intent and kw_intent != IntentType.UNKNOWN:
            final_intent = kw_intent
            final_conf = kw_conf * self.config.keyword_weight + llm_conf * self.config.llm_weight
            final_entities = combined_entities
            logger.info(f"Intents consistent: {kw_intent.value}. Combined confidence.")
        # 冲突处理
        elif llm_intent != IntentType.UNKNOWN:
            margin = llm_conf - kw_conf
            # 如果LLM置信度显著高于关键词，且超过margin阈值，采纳LLM
            if margin > self.config.llm_override_margin:
                final_intent, final_conf, final_entities = llm_intent, llm_conf, combined_entities
                logger.info(f"LLM overrides keyword. Margin: {margin:.2f}")
            else:
                # 否则，选择分数更高者，并对冲突施加惩罚
                kw_score = kw_conf * self.config.keyword_weight
                llm_score = llm_conf * self.config.llm_weight
                
                if llm_score > kw_score:
                    final_intent, final_conf = llm_intent, llm_score
                else:
                    final_intent, final_conf = kw_intent, kw_score
                
                final_conf *= (1 - self.config.conflict_penalty) # 施加惩罚
                final_entities = combined_entities
                logger.info(f"Conflict detected. Winner: {final_intent.value}, penalized confidence: {final_conf:.2f}")
        
        # 记录可观测性数据
        observability_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'user_input': user_input,
            'keyword_intent': kw_intent.value,
            'keyword_confidence': kw_conf,
            'llm_intent': llm_intent.value,
            'llm_confidence': llm_conf,
            'final_intent': final_intent.value,
            'final_confidence': final_conf,
            'fusion_strategy': 'consistent' if kw_intent == llm_intent else ('llm_override' if 'margin' in locals() and margin > self.config.llm_override_margin else 'penalized_winner')
        }
        logger.info(f"Intent recognition details: {json.dumps(observability_data, ensure_ascii=False)}")

        return final_intent, min(final_conf, 1.0), final_entities



def get_user_input(state: EnhancedState) -> str:
    """安全地从state中提取最新用户输入。"""
    if not state.get("messages"):
        return ""
    last_message = state["messages"][-1]
    # 兼容多种消息格式
    if isinstance(last_message, tuple):
        return last_message[1]
    return getattr(last_message, 'content', '')

def get_context_summary(state: EnhancedState) -> str:
    """安全地从state中提取上下文摘要。"""
    return state.get("context_summary", "")

def get_dialog_history(state: EnhancedState, num_turns: int = 5) -> List[DialogTurn]:
    """安全地从state中提取最近的对话历史。"""
    dialog_state = state.get("dialog_state")
    if not dialog_state or not hasattr(dialog_state, 'get_recent_turns'):
        return []
    return dialog_state.get_recent_turns(num_turns)


# 全局意图识别器实例
_intent_recognizer_instance = None
_instance_lock = threading.Lock()

def get_intent_recognizer(llm: ChatOllama, config: Optional[IntentConfig] = None) -> 'EnhancedIntentRecognizer':
    """获取意图识别器单例。"""
    global _intent_recognizer_instance
    if _intent_recognizer_instance is None:
        with _instance_lock:
            if _intent_recognizer_instance is None:
                _intent_recognizer_instance = EnhancedIntentRecognizer(llm, config=config)
    return _intent_recognizer_instance