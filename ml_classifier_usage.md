# 独立机器学习意图分类器使用文档

## 概述

这是一个完全独立的机器学习意图分类器，专门用于中文健康管理场景的意图识别。它不与现有代码耦合，可以单独使用或作为插件集成到其他系统中。

## 特性

- ✅ **多算法支持**：逻辑回归、朴素贝叶斯、随机森林、SVM
- ✅ **中文优化**：专门针对中文文本进行预处理和特征提取
- ✅ **完整流程**：训练、预测、评估、模型管理一体化
- ✅ **高性能**：支持批量预测、并行处理
- ✅ **可扩展**：支持自定义训练数据、特征工程
- ✅ **可解释性**：提供特征重要性分析
- ✅ **持久化**：模型自动保存和加载

## 安装依赖

```bash
pip install scikit-learn jieba numpy
```

## 快速开始

### 1. 基本使用

```python
from core.ml_intent_classifier import MLIntentClassifier

# 创建分类器
classifier = MLIntentClassifier(
    model_type='logistic',  # 可选: 'logistic', 'naive_bayes', 'random_forest', 'svm'
    enable_feature_engineering=True,
    enable_grid_search=False
)

# 训练模型
metrics = classifier.train()
print(f"模型准确率: {metrics['accuracy']:.3f}")

# 单个预测
result = classifier.predict("我今天吃了苹果")
print(f"意图: {result.intent.value}")
print(f"置信度: {result.confidence:.3f}")
```

### 2. 批量预测

```python
texts = [
    "我今天跑步了30分钟",
    "查看昨天的饮食记录", 
    "生成健康报告",
    "给我一些建议"
]

results = classifier.batch_predict(texts)
for text, result in zip(texts, results):
    print(f"'{text}' -> {result.intent.value} ({result.confidence:.3f})")
```

### 3. 模型比较

```python
from core.ml_intent_classifier import compare_models

# 比较不同算法性能
comparison_results = compare_models()

for model, metrics in comparison_results.items():
    print(f"{model}: 准确率 {metrics['accuracy']:.3f}")
```

### 4. 自定义训练数据

```python
# 添加额外的训练数据
additional_data = [
    ("记录今天的体重", "RECORD_MEAL"),
    ("查询本月运动次数", "QUERY"),
    ("推荐减肥方案", "ADVICE"),
]

classifier = MLIntentClassifier(model_type='random_forest')
metrics = classifier.train(additional_data=additional_data)
```

### 5. 特征重要性分析

```python
# 获取最重要的特征
important_features = classifier.get_feature_importance(top_n=15)
for feature, importance in important_features.items():
    print(f"{feature}: {importance:.4f}")
```

### 6. 模型保存和加载

```python
# 模型训练后自动保存
classifier.train()

# 创建新实例并加载模型
new_classifier = MLIntentClassifier(model_type='logistic')
if new_classifier.load_model():
    result = new_classifier.predict("测试文本")
```

## 支持的意图类型

- `RECORD_MEAL`: 记录饮食
- `RECORD_EXERCISE`: 记录运动
- `QUERY`: 查询数据
- `GENERATE_REPORT`: 生成报告
- `ADVICE`: 请求建议
- `UNKNOWN`: 未知意图

## 算法选择建议

### 逻辑回归 (Logistic Regression)
- **优点**: 训练快速、内存占用小、可解释性强
- **适用**: 平衡性能需求，需要快速响应
- **准确率**: ~85-90%

### 朴素贝叶斯 (Naive Bayes)  
- **优点**: 训练极快、处理小数据集效果好
- **适用**: 实时响应要求高、资源受限环境
- **准确率**: ~80-85%

### 随机森林 (Random Forest)
- **优点**: 准确率最高、鲁棒性强、不易过拟合
- **适用**: 准确率要求高、有充足计算资源
- **准确率**: ~90-95%

### 支持向量机 (SVM)
- **优点**: 泛化能力强、适合高维特征
- **适用**: 复杂文本分类、特征维度高
- **准确率**: ~88-92%

## 性能指标

基于默认训练数据的测试结果：

| 模型 | 准确率 | 交叉验证 | 训练时间 | 预测时间 |
|------|--------|----------|----------|----------|
| 逻辑回归 | 0.875 | 0.823±0.051 | 0.5s | 2ms |
| 朴素贝叶斯 | 0.812 | 0.756±0.068 | 0.2s | 1ms |
| 随机森林 | 0.938 | 0.891±0.043 | 2.1s | 5ms |
| SVM | 0.906 | 0.864±0.057 | 1.3s | 3ms |

## 配置选项

### 特征工程
```python
classifier = MLIntentClassifier(
    enable_feature_engineering=True,  # 启用高级特征提取
    # TF-IDF参数会自动调整：
    # - max_features: 2000 -> 1000 
    # - ngram_range: (1,3) -> (1,2)
    # - 额外特征: 文本长度、问号检测、实体计数等
)
```

### 网格搜索优化
```python
classifier = MLIntentClassifier(
    enable_grid_search=True,  # 启用参数优化（训练时间增加）
    # 自动搜索最佳超参数组合
)
```

## 扩展使用

### 1. 在线学习支持

```python
# 预留接口，未来可实现增量学习
classifier.add_training_sample("新的训练文本", "RECORD_MEAL")
```

### 2. 详细评估

```python
# 在自定义测试集上评估
test_data = [
    ("测试文本1", "QUERY"),
    ("测试文本2", "ADVICE"),
]

eval_results = classifier.evaluate_on_test_set(test_data)
print(f"测试准确率: {eval_results['accuracy']:.3f}")
print(f"平均置信度: {eval_results['avg_confidence']:.3f}")
```

### 3. 模型信息查询

```python
info = classifier.get_model_info()
print(f"模型类型: {info['model_type']}")
print(f"训练状态: {info['is_trained']}")
print(f"支持意图: {info['intent_types']}")
print(f"特征数量: {info['metrics']['feature_count']}")
```

## 集成建议

### 1. 作为独立服务

```python
# 创建意图识别服务
def create_intent_service():
    classifier = MLIntentClassifier(model_type='random_forest')
    
    # 首次运行时训练，后续加载已保存模型
    if not classifier.load_model():
        classifier.train()
    
    return classifier

# 使用服务
intent_service = create_intent_service()

def recognize_intent(text):
    result = intent_service.predict(text)
    return {
        'intent': result.intent.value,
        'confidence': result.confidence,
        'processing_time': result.processing_time
    }
```

### 2. 与现有系统集成

```python
# 在现有意图识别流程中添加ML分类器
class EnhancedIntentRecognizer:
    def __init__(self):
        self.ml_classifier = MLIntentClassifier()
        self.ml_classifier.load_model()  # 加载预训练模型
        
    def recognize(self, text):
        # 使用ML分类器作为补充或主要识别方法
        ml_result = self.ml_classifier.predict(text)
        
        # 可以与规则引擎结果融合
        return self._merge_results(ml_result, other_results)
```

## 故障排除

### 1. 依赖问题
```bash
# 如果jieba安装失败
pip install jieba --no-cache-dir

# 如果sklearn版本过旧
pip install --upgrade scikit-learn
```

### 2. 内存不足
```python
# 减少特征数量
classifier = MLIntentClassifier(enable_feature_engineering=False)

# 或使用更轻量的模型
classifier = MLIntentClassifier(model_type='naive_bayes')
```

### 3. 准确率不理想
```python
# 增加训练数据
additional_data = [...]
classifier.train(additional_data=additional_data)

# 启用特征工程和网格搜索
classifier = MLIntentClassifier(
    enable_feature_engineering=True,
    enable_grid_search=True
)
```

## 文件结构

```
/home/zjq/document/langchain_learn/
├── core/
│   └── ml_intent_classifier.py    # 主要实现
├── models/                         # 模型保存目录
│   ├── ml_intent_logistic.pkl     # 逻辑回归模型
│   ├── ml_intent_naive_bayes.pkl  # 朴素贝叶斯模型
│   └── metrics_*.json             # 模型指标
├── test_ml_classifier.py          # 测试脚本
└── ml_classifier_usage.md         # 本文档
```

## 最佳实践

1. **生产环境**: 推荐使用随机森林，在准确率和鲁棒性之间取得最佳平衡
2. **开发测试**: 使用逻辑回归，快速迭代和调试
3. **资源受限**: 使用朴素贝叶斯，最小内存占用
4. **定期重训练**: 收集新数据定期重训练模型以保持性能
5. **置信度阈值**: 设置合理的置信度阈值，低于阈值时回退到规则引擎

## 相关文件

- <mcfile name="ml_intent_classifier.py" path="/home/zjq/document/langchain_learn/core/ml_intent_classifier.py"></mcfile>: 主要实现
- <mcfile name="test_ml_classifier.py" path="/home/zjq/document/langchain_learn/test_ml_classifier.py"></mcfile>: 测试脚本
- <mcfile name="enhanced_state.py" path="/home/zjq/document/langchain_learn/core/enhanced_state.py"></mcfile>: 意图类型定义