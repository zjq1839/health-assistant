#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮对话感知的机器学习意图分类器
将ML分类器与现有的多轮对话系统集成
"""

import numpy as np
# 移除未使用的 pandas 导入
# import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import joblib
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score

try:
    import jieba
    jieba.setLogLevel(20)  # 减少日志输出
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ jieba未安装，请运行: pip install jieba")

from .enhanced_state import IntentType, EnhancedState, DialogState, DialogTurn
from utils.logger import logger

@dataclass
class MultiTurnPredictionResult:
    """多轮预测结果"""
    intent: IntentType
    confidence: float
    processing_time: float
    context_influence: float  # 上下文对结果的影响程度
    base_ml_confidence: float  # 纯ML模型的置信度
    context_boost: float  # 上下文提升的置信度
    evidence: Dict[str, Any]  # 决策证据

class ContextAwareTextPreprocessor:
    """上下文感知的文本预处理器"""
    
    def __init__(self):
        self.stop_words = set([
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', 
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', 
            '你', '会', '着', '没有', '看', '好', '自己', '这'
        ])
    
    def preprocess_with_context(self, text: str, context_info: Dict[str, Any] = None) -> str:
        """预处理文本，考虑上下文信息"""
        # 基础预处理
        text = text.strip().lower()
        
        # 分词
        words = list(jieba.cut(text))
        
        # 去停用词
        words = [w for w in words if w not in self.stop_words and len(w) > 1]
        
        # 如果有上下文信息，添加上下文特征
        if context_info:
            # 添加最近意图作为特征
            recent_intents = context_info.get('recent_intents', [])
            if recent_intents:
                words.extend([f"prev_intent_{intent.value}" for intent in recent_intents[-2:]])
            
            # 添加时间特征
            time_features = context_info.get('time_features', [])
            words.extend(time_features)
            
            # 添加实体特征
            entities = context_info.get('entities', {})
            for entity_type, entity_value in entities.items():
                words.append(f"entity_{entity_type}")
        
        return ' '.join(words)

class MultiTurnMLIntentClassifier:
    """多轮对话感知的ML意图分类器"""
    
    def __init__(self, 
                 model_type: str = 'random_forest',
                 enable_context_features: bool = True,
                 context_weight: float = 0.3,
                 time_decay_factor: float = 0.8):
        """
        初始化分类器
        
        Args:
            model_type: 模型类型 ('logistic', 'random_forest', 'naive_bayes', 'svm')
            enable_context_features: 是否启用上下文特征
            context_weight: 上下文权重(0-1)
            time_decay_factor: 时间衰减因子
        """
        self.model_type = model_type
        self.enable_context_features = enable_context_features
        self.context_weight = context_weight
        self.time_decay_factor = time_decay_factor
        
        self.preprocessor = ContextAwareTextPreprocessor()
        self.vectorizer = None
        self.model = None
        self.label_encoder = {}
        self.is_trained = False
        
        # 模型映射
        self.model_classes = {
            'logistic': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'naive_bayes': MultinomialNB,
            'svm': SVC
        }
        
        # 训练数据（增强版，包含上下文场景）
        self.base_training_data = [
            # 直接意图表达
            ("我今天吃了苹果", "record_meal"),
            ("记录早餐：牛奶面包", "record_meal"),
            ("今天跑步30分钟", "record_exercise"), 
            ("健身房锻炼1小时", "record_exercise"),
            ("生成健康报告", "generate_report"),
            ("查看昨天的饮食记录", "query"),
            ("给我一些减肥建议", "advice"),
            
            # 上下文相关表达
            ("昨天呢", "query"),  # 需要上下文
            ("那天的情况", "query"),
            ("也是这样", "record_meal"),  # 模糊表达
            ("差不多", "record_meal"),
            ("还有吗", "query"),
            ("继续", "record_meal"),
            ("再来一份", "record_meal"),
            
            # 时间相关
            ("前天也跑了", "record_exercise"),
            ("本周运动情况", "query"),
            ("这个月的报告", "generate_report"),
            
            # 复杂上下文
            ("这次换个口味", "record_meal"),
            ("和上次一样", "record_meal"),
            ("比昨天多一点", "record_exercise"),
            ("效果怎么样", "query"),
            ("有什么好方法", "advice"),
            
            # 否定和修正
            ("不是这个意思", "unknown"),
            ("刚才说错了", "unknown"),
            ("重新记录", "record_meal"),
            
            # 更多样本
            ("午餐吃了什么", "query"),
            ("今天运动了吗", "query"),
            ("推荐健康食谱", "advice"),
            ("本月体重变化", "generate_report"),
        ]
        
        logger.info(f"初始化多轮ML意图分类器: {model_type}")
    
    def _extract_context_features(self, state: EnhancedState) -> Dict[str, Any]:
        """从对话状态中提取上下文特征"""
        if not self.enable_context_features:
            return {}
        
        dialog_state = state.get('dialog_state')
        if not dialog_state:
            return {}
        
        context_features = {}
        
        # 最近意图
        recent_intents = dialog_state.get_recent_intents(3)
        context_features['recent_intents'] = recent_intents
        
        # 时间特征
        time_features = []
        current_time = datetime.now()
        
        # 根据当前时间添加时间特征
        hour = current_time.hour
        if 6 <= hour < 10:
            time_features.append("time_breakfast")
        elif 11 <= hour < 14:
            time_features.append("time_lunch") 
        elif 17 <= hour < 20:
            time_features.append("time_dinner")
        elif 20 <= hour <= 23:
            time_features.append("time_evening")
        
        context_features['time_features'] = time_features
        
        # 实体特征
        context_features['entities'] = dialog_state.get_context_entities()
        
        return context_features
    
    def _calculate_context_boost(self, 
                                base_prediction: np.ndarray, 
                                context_features: Dict[str, Any]) -> float:
        """计算上下文对预测的提升"""
        if not self.enable_context_features or not context_features:
            return 0.0
        
        boost = 0.0
        recent_intents = context_features.get('recent_intents', [])
        
        if recent_intents:
            # 如果最近的意图与预测一致，增加置信度
            predicted_intent_idx = np.argmax(base_prediction)
            predicted_intent = self._idx_to_intent(predicted_intent_idx)
            
            # 时间衰减权重
            for i, recent_intent in enumerate(reversed(recent_intents)):
                if recent_intent == predicted_intent:
                    weight = (self.time_decay_factor ** i) * 0.1
                    boost += weight
        
        return min(boost, 0.2)  # 最大提升20%
    
    def _build_features(self, text: str, context_features: Dict[str, Any] = None) -> str:
        """构建包含上下文的特征"""
        return self.preprocessor.preprocess_with_context(text, context_features)
    
    def train(self, additional_data: List[Tuple[str, str]] = None) -> Dict[str, float]:
        """训练模型"""
        logger.info("开始训练多轮ML意图分类器...")
        
        # 准备训练数据
        training_data = self.base_training_data.copy()
        if additional_data:
            training_data.extend(additional_data)
        
        # 分离文本和标签
        texts = [item[0] for item in training_data]
        labels = [item[1] for item in training_data]
        
        # 构建标签编码
        unique_labels = list(set(labels))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_encoder.items()}
        
        # 预处理文本（暂时不加上下文，训练时使用基础特征）
        processed_texts = [self.preprocessor.preprocess_with_context(text) for text in texts]
        
        # 特征提取
        if self.enable_context_features:
            max_features = 1500
            ngram_range = (1, 3)
        else:
            max_features = 1000 
            ngram_range = (1, 2)
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=1,
            max_df=0.95
        )
        
        X = self.vectorizer.fit_transform(processed_texts)
        y = [self.label_encoder[label] for label in labels]
        
        # 初始化模型
        if self.model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif self.model_type == 'svm':
            self.model = SVC(probability=True, random_state=42)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 训练模型
        self.model.fit(X, y)
        self.is_trained = True
        
        # 评估
        predictions = self.model.predict(X)
        accuracy = accuracy_score(y, predictions)
        
        # 交叉验证
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        
        metrics = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_count': X.shape[1],
            'training_samples': len(training_data)
        }
        
        logger.info(f"多轮ML分类器训练完成: 准确率={accuracy:.3f}, 交叉验证={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        return metrics
    
    def predict(self, text: str, state: EnhancedState = None) -> MultiTurnPredictionResult:
        """预测意图（多轮版本）"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        start_time = datetime.now()
        
        # 提取上下文特征
        context_features = self._extract_context_features(state) if state else {}
        
        # 构建特征
        processed_text = self._build_features(text, context_features)
        
        # 向量化
        X = self.vectorizer.transform([processed_text])
        
        # 基础ML预测
        base_probabilities = self.model.predict_proba(X)[0]
        base_confidence = float(np.max(base_probabilities))
        predicted_idx = np.argmax(base_probabilities)
        
        # 计算上下文提升
        context_boost = self._calculate_context_boost(base_probabilities, context_features)
        
        # 最终置信度
        final_confidence = min(base_confidence + context_boost, 1.0)
        
        # 转换为意图类型
        predicted_label = self.idx_to_label[predicted_idx]
        predicted_intent = self._label_to_intent(predicted_label)
        
        # 计算处理时间
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # 构建证据
        evidence = {
            'base_prediction': predicted_label,
            'all_probabilities': {self.idx_to_label[i]: float(prob) 
                                for i, prob in enumerate(base_probabilities)},
            'context_features_used': list(context_features.keys()) if context_features else [],
            'context_boost_applied': context_boost > 0
        }
        
        return MultiTurnPredictionResult(
            intent=predicted_intent,
            confidence=final_confidence,
            processing_time=processing_time,
            context_influence=context_boost / final_confidence if final_confidence > 0 else 0,
            base_ml_confidence=base_confidence,
            context_boost=context_boost,
            evidence=evidence
        )
    
    def _label_to_intent(self, label: str) -> IntentType:
        """标签转换为意图类型"""
        label_mapping = {
            'record_meal': IntentType.RECORD_MEAL,
            'record_exercise': IntentType.RECORD_EXERCISE,
            'query': IntentType.QUERY,
            'generate_report': IntentType.GENERATE_REPORT,
            'advice': IntentType.ADVICE,
            'unknown': IntentType.UNKNOWN
        }
        return label_mapping.get(label, IntentType.UNKNOWN)
    
    def _intent_to_label(self, intent: IntentType) -> str:
        """意图类型转换为标签"""
        return intent.value
    
    def _idx_to_intent(self, idx: int) -> IntentType:
        """索引转换为意图类型"""
        label = self.idx_to_label[idx]
        return self._label_to_intent(label)
    
    def batch_predict(self, 
                     texts: List[str], 
                     states: List[EnhancedState] = None) -> List[MultiTurnPredictionResult]:
        """批量预测（多轮版本）"""
        if not texts:
            return []
        
        if states is None:
            states = [None] * len(texts)
        
        results = []
        for text, state in zip(texts, states):
            result = self.predict(text, state)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'enable_context_features': self.enable_context_features,
            'context_weight': self.context_weight,
            'time_decay_factor': self.time_decay_factor,
            'intent_types': [intent.value for intent in IntentType],
            'label_mapping': self.label_encoder if self.is_trained else {},
            'feature_count': self.vectorizer.get_feature_names_out().shape[0] if self.is_trained and self.vectorizer else 0
        }
    
    def save_model(self, filepath: str = None) -> str:
        """保存模型"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        
        if filepath is None:
            filepath = f"models/multi_turn_ml_intent_{self.model_type}.pkl"
        
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'idx_to_label': self.idx_to_label,
            'model_type': self.model_type,
            'enable_context_features': self.enable_context_features,
            'context_weight': self.context_weight,
            'time_decay_factor': self.time_decay_factor
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"多轮ML模型已保存到: {filepath}")
        return filepath
    
    def load_model(self, filepath: str = None) -> bool:
        """加载模型"""
        if filepath is None:
            filepath = f"models/multi_turn_ml_intent_{self.model_type}.pkl"
        
        try:
            import os
            if not os.path.exists(filepath):
                logger.warning(f"模型文件不存在: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.label_encoder = model_data['label_encoder']
            self.idx_to_label = model_data['idx_to_label']
            self.enable_context_features = model_data.get('enable_context_features', True)
            self.context_weight = model_data.get('context_weight', 0.3)
            self.time_decay_factor = model_data.get('time_decay_factor', 0.8)
            self.is_trained = True
            
            logger.info(f"多轮ML模型已从 {filepath} 加载")
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False

def create_multi_turn_classifier(model_type: str = 'random_forest') -> MultiTurnMLIntentClassifier:
    """创建多轮ML分类器的便捷函数"""
    classifier = MultiTurnMLIntentClassifier(model_type=model_type)
    
    # 尝试加载已有模型，如果失败则训练新模型
    if not classifier.load_model():
        logger.info("未找到预训练模型，开始训练新模型...")
        classifier.train()
        classifier.save_model()
    
    return classifier

# 使用示例
if __name__ == "__main__":
    # 创建分类器
    classifier = MultiTurnMLIntentClassifier(model_type='random_forest')
    
    # 训练
    metrics = classifier.train()
    print(f"训练完成: {metrics}")
    
    # 模拟对话状态
    from .enhanced_state import create_enhanced_state
    
    state = create_enhanced_state()
    state['dialog_state'].turn_history = [
        # 模拟之前的对话
    ]
    
    # 测试预测
    test_cases = [
        ("我今天吃了苹果", None),  # 无上下文
        ("昨天呢", state),  # 需要上下文
        ("也是这样", state),  # 模糊表达
    ]
    
    for text, test_state in test_cases:
        result = classifier.predict(text, test_state)
        print(f"'{text}' -> {result.intent.value} "
              f"(置信度: {result.confidence:.3f}, "
              f"上下文影响: {result.context_influence:.3f})")