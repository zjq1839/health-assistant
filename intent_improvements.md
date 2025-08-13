# 意图识别系统改进方案

## 1. 规则引擎置信度优化

### 问题
当前规则引擎存在置信度人为提升的问题：
```python
enhanced_confidence = min(confidence + 0.6, 1.0)  # 过于激进
```

### 解决方案
```python
# 改进的置信度计算
def calculate_rule_confidence(self, matches: int, total_patterns: int, text: str) -> float:
    base_confidence = matches / total_patterns
    
    # 文本长度权重（短文本匹配可能更准确）
    length_weight = min(1.0, 20 / len(text.split())) if len(text.split()) > 0 else 1.0
    
    # 否定词惩罚
    negation_penalty = 0.3 if self.NEGATION_PATTERN.search(text) else 0.0
    
    # 最终置信度：基础 * 长度权重 - 否定惩罚
    final_confidence = base_confidence * length_weight - negation_penalty
    
    return max(0.0, min(0.9, final_confidence))  # 限制在0-0.9范围
```

## 2. 增强配置参数

### 添加缺失的IntentConfig参数
```python
class IntentConfig(BaseModel):
    # ... 现有参数 ...
    
    # 新增参数
    llm_override_margin: float = Field(default=0.2, alias='LLM_OVERRIDE_MARGIN')
    conflict_penalty: float = Field(default=0.15, alias='CONFLICT_PENALTY')
    rule_confidence_cap: float = Field(default=0.9, alias='RULE_CONFIDENCE_CAP')
    lite_confidence_threshold: float = Field(default=0.65, alias='LITE_CONFIDENCE_THRESHOLD')
    rule_confidence_threshold: float = Field(default=0.6, alias='RULE_CONFIDENCE_THRESHOLD')
```

## 3. 改进正则表达式规则

### 问题
当前规则可能存在冲突，如"吃了什么"被误分类为RECORD_MEAL

### 解决方案
```python
INTENT_PATTERNS = {
    IntentType.RECORD_MEAL: [
        # 更精确的记录模式，排除疑问句
        re.compile(r'(我|今天|昨天|刚才).*(吃了|喝了)(?!.*(什么|吗|呢|\?))', re.IGNORECASE),
        re.compile(r'(早餐|午餐|晚餐|夜宵).*[：:]', re.IGNORECASE),
        re.compile(r'记录.*饮食|饮食.*记录', re.IGNORECASE),
    ],
    IntentType.QUERY: [
        # 明确的查询模式
        re.compile(r'.*(什么|吗|呢|\?)$', re.IGNORECASE),
        re.compile(r'(查询|查看|显示|搜索).*(记录|数据)', re.IGNORECASE),
        re.compile(r'(昨天|前天|那天)呢\??$', re.IGNORECASE),
    ]
}
```

## 4. 实施A/B测试框架

### 测试不同配置的效果
```python
class IntentTestFramework:
    def __init__(self):
        self.test_cases = [
            ("我今天吃了苹果", IntentType.RECORD_MEAL),
            ("我今天吃了什么？", IntentType.QUERY),
            ("帮我分析健康状况", IntentType.GENERATE_REPORT),
            # ... 更多测试用例
        ]
    
    def evaluate_configuration(self, config: IntentConfig) -> Dict[str, float]:
        correct = 0
        total = len(self.test_cases)
        
        planner = LightweightPlanner(config)
        
        for text, expected_intent in self.test_cases:
            result = planner.plan(text)
            if result.intent == expected_intent:
                correct += 1
        
        accuracy = correct / total
        return {
            'accuracy': accuracy,
            'precision': self._calculate_precision(),
            'recall': self._calculate_recall(),
            'f1_score': self._calculate_f1()
        }
```

## 5. 引入机器学习辅助

### 基于历史数据的模式学习
```python
class MLEnhancedClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.classifier = LogisticRegression()
        self.is_trained = False
    
    def train_from_logs(self, log_data: List[Tuple[str, IntentType]]):
        """从历史日志训练模型"""
        texts, labels = zip(*log_data)
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_trained = True
    
    def predict_with_confidence(self, text: str) -> Tuple[IntentType, float]:
        if not self.is_trained:
            return IntentType.UNKNOWN, 0.0
        
        X = self.vectorizer.transform([text])
        probabilities = self.classifier.predict_proba(X)[0]
        
        best_idx = probabilities.argmax()
        confidence = probabilities[best_idx]
        intent = self.classifier.classes_[best_idx]
        
        return intent, confidence
```

## 6. 监控和告警机制

### 实时性能监控
```python
class IntentMonitor:
    def __init__(self):
        self.metrics = {
            'accuracy_trend': [],
            'confidence_distribution': [],
            'processing_time': [],
            'fallback_rate': 0.0
        }
    
    def log_prediction(self, text: str, predicted: IntentType, 
                      confidence: float, actual: IntentType = None):
        """记录预测结果用于监控"""
        # 记录指标
        # 检测异常模式
        # 触发告警
        pass
    
    def get_health_score(self) -> float:
        """计算系统健康分数"""
        # 基于准确率、置信度分布、处理时间等计算
        pass
```

## 7. 优化建议优先级

### 高优先级（立即实施）
1. 修复规则引擎置信度计算逻辑
2. 添加缺失的配置参数
3. 改进正则表达式规则

### 中优先级（1-2周内）
1. 增加边界测试用例
2. 实施A/B测试框架
3. 添加性能监控

### 低优先级（长期优化）
1. 引入机器学习辅助
2. 构建完整的评估体系
3. 实施自适应调优

## 8. 预期改进效果

- **准确率提升**：从当前85%提升至92%以上
- **响应时间**：保持在当前水平（<50ms）
- **鲁棒性**：减少边界情况误判50%
- **可维护性**：通过配置化提升系统可调节性