"""独立的机器学习意图分类器

专门用于意图识别的机器学习解决方案
支持多种算法：逻辑回归、朴素贝叶斯、随机森林
提供完整的训练、预测、评估和模型管理功能
"""

import os
import json
import pickle
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
from datetime import datetime

# 机器学习库
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.pipeline import Pipeline
    import jieba
    ML_AVAILABLE = True
except ImportError as e:
    print(f"机器学习库未安装: {e}")
    ML_AVAILABLE = False

from .enhanced_state import IntentType
from utils.logger import logger


@dataclass
class MLPredictionResult:
    """ML模型预测结果"""
    intent: IntentType
    confidence: float
    probabilities: Dict[str, float]
    features_used: int
    processing_time: float
    model_used: str


class ChineseTextPreprocessor:
    """中文文本预处理器"""
    
    def __init__(self):
        # 停用词列表
        self.stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', 
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '用',
            '还', '又', '能', '这样', '那样', '什么', '怎么', '为什么'
        }
        
        # 意图相关的重要词汇（不会被过滤）
        self.intent_keywords = {
            '吃', '喝', '食物', '饮食', '餐', '早餐', '午餐', '晚餐', '夜宵',
            '运动', '锻炼', '跑步', '游泳', '健身', '瑜伽', '散步', '骑车',
            '查询', '查看', '显示', '搜索', '记录', '统计', '数据',
            '报告', '分析', '统计', '建议', '推荐', '如何', '怎样',
            '卡路里', '体重', '健康', '营养', '蛋白质', '维生素'
        }
        
        # 情感词汇
        self.emotion_words = {
            '好', '坏', '棒', '差', '满意', '不错', '很好', '不好'
        }
    
    def preprocess(self, text: str) -> List[str]:
        """预处理文本，返回分词结果"""
        if not text:
            return []
        
        # 分词
        words = list(jieba.cut(text.strip().lower()))
        
        # 过滤停用词，但保留意图关键词和情感词
        filtered_words = []
        for word in words:
            if len(word) > 1 or word in self.intent_keywords or word in self.emotion_words:
                if word not in self.stop_words or word in self.intent_keywords or word in self.emotion_words:
                    filtered_words.append(word)
        
        return filtered_words
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """提取额外特征"""
        features = {}
        
        # 文本长度特征
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        
        # 问号特征（查询意图的强信号）
        features['has_question'] = '？' in text or '吗' in text or '呢' in text or '如何' in text or '怎么' in text
        
        # 时间词特征
        time_words = ['今天', '昨天', '明天', '刚才', '现在', '早上', '中午', '晚上', '上周', '本周', '最近']
        features['has_time'] = any(word in text for word in time_words)
        
        # 数字特征（可能表示记录意图）
        import re
        features['has_number'] = bool(re.search(r'\d+', text))
        
        # 食物词汇特征
        food_words = ['吃', '喝', '餐', '食物', '米饭', '面条', '水果', '蔬菜', '肉', '鱼', '蛋', '奶']
        features['food_count'] = sum(1 for word in food_words if word in text)
        
        # 运动词汇特征
        exercise_words = ['跑', '游泳', '健身', '锻炼', '运动', '瑜伽', '散步', '骑车', '球']
        features['exercise_count'] = sum(1 for word in exercise_words if word in text)
        
        # 查询词汇特征
        query_words = ['查', '看', '显示', '搜索', '统计', '多少', '什么']
        features['query_count'] = sum(1 for word in query_words if word in text)
        
        return features


class MLIntentClassifier:
    """独立的机器学习意图分类器"""
    
    def __init__(self, 
                 model_type: str = 'logistic',
                 enable_feature_engineering: bool = True,
                 enable_grid_search: bool = False):
        """
        初始化分类器
        
        Args:
            model_type: 模型类型 ('logistic', 'naive_bayes', 'random_forest', 'svm')
            enable_feature_engineering: 是否启用特征工程
            enable_grid_search: 是否启用网格搜索优化
        """
        if not ML_AVAILABLE:
            raise ImportError("机器学习库未安装，请安装 scikit-learn 和 jieba")
        
        self.model_type = model_type
        self.enable_feature_engineering = enable_feature_engineering
        self.enable_grid_search = enable_grid_search
        
        # 文本预处理
        self.preprocessor = ChineseTextPreprocessor()
        
        # 特征提取器
        self.vectorizer = None
        
        # 模型
        self.model = None
        self.pipeline = None
        self.is_trained = False
        
        # 模型性能指标
        self.metrics = {
            'accuracy': 0.0,
            'training_samples': 0,
            'last_update': None,
            'cross_validation_score': 0.0,
            'feature_count': 0
        }
        
        # 标签映射
        self.label_to_intent = {}
        self.intent_to_label = {}
        
        # 模型文件路径
        self.model_dir = '/home/zjq/document/langchain_learn/models'
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger.info(f"ML意图分类器初始化完成 - 模型类型: {model_type}")
    
    def _create_vectorizer(self):
        """创建文本向量化器"""
        if self.enable_feature_engineering:
            return TfidfVectorizer(
                max_features=2000,
                ngram_range=(1, 3),  # 1-3gram特征
                min_df=2,
                max_df=0.8,
                sublinear_tf=True,
                use_idf=True
            )
        else:
            return TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=1
            )
    
    def _create_model(self):
        """创建机器学习模型"""
        models = {
            'logistic': LogisticRegression(
                random_state=42,
                max_iter=2000,
                class_weight='balanced',
                solver='liblinear'
            ),
            'naive_bayes': MultinomialNB(alpha=0.1),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight='balanced',
                max_depth=10,
                min_samples_split=5
            ),
            'svm': SVC(
                kernel='linear',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        }
        return models.get(self.model_type, models['logistic'])
    
    def get_training_data(self) -> List[Tuple[str, str]]:
        """获取训练数据集"""
        training_data = [
            # 记录餐食 - 更多样本
            ("我今天早餐吃了鸡蛋和牛奶", "RECORD_MEAL"),
            ("早上喝了一杯豆浆", "RECORD_MEAL"),
            ("中午吃的米饭和青菜", "RECORD_MEAL"),
            ("晚餐有鱼和汤", "RECORD_MEAL"),
            ("我吃了一个苹果", "RECORD_MEAL"),
            ("喝了200ml水", "RECORD_MEAL"),
            ("午餐是麻辣烫", "RECORD_MEAL"),
            ("刚吃完晚饭", "RECORD_MEAL"),
            ("今天吃了很多水果", "RECORD_MEAL"),
            ("早餐吃面包和果酱", "RECORD_MEAL"),
            ("中午点了外卖", "RECORD_MEAL"),
            ("晚上煮了面条", "RECORD_MEAL"),
            ("吃了一份沙拉", "RECORD_MEAL"),
            ("喝了咖啡", "RECORD_MEAL"),
            ("吃了巧克力", "RECORD_MEAL"),
            ("喝了蜂蜜水", "RECORD_MEAL"),
            ("吃了烧烤", "RECORD_MEAL"),
            ("喝了果汁", "RECORD_MEAL"),
            ("吃了火锅", "RECORD_MEAL"),
            ("早餐喝了粥", "RECORD_MEAL"),
            
            # 记录运动 - 更多样本  
            ("我今天跑步了30分钟", "RECORD_EXERCISE"),
            ("刚才游泳1小时", "RECORD_EXERCISE"),
            ("今天健身房锻炼", "RECORD_EXERCISE"),
            ("做了瑜伽", "RECORD_EXERCISE"),
            ("骑车上班", "RECORD_EXERCISE"),
            ("散步半小时", "RECORD_EXERCISE"),
            ("举重训练", "RECORD_EXERCISE"),
            ("晨跑结束", "RECORD_EXERCISE"),
            ("篮球运动1小时", "RECORD_EXERCISE"),
            ("爬山2小时", "RECORD_EXERCISE"),
            ("今天踢足球", "RECORD_EXERCISE"),
            ("做了俯卧撑", "RECORD_EXERCISE"),
            ("跳绳10分钟", "RECORD_EXERCISE"),
            ("练习太极", "RECORD_EXERCISE"),
            ("跳了广场舞", "RECORD_EXERCISE"),
            ("做了拉伸运动", "RECORD_EXERCISE"),
            ("爬楼梯锻炼", "RECORD_EXERCISE"),
            ("走了1万步", "RECORD_EXERCISE"),
            ("做了平板支撑", "RECORD_EXERCISE"),
            ("练习瑜伽45分钟", "RECORD_EXERCISE"),
            
            # 查询数据 - 更多样本
            ("我今天吃了什么？", "QUERY"),
            ("查看昨天的饮食记录", "QUERY"),
            ("显示本周运动情况", "QUERY"),
            ("我的体重变化如何？", "QUERY"),
            ("最近吃了多少卡路里？", "QUERY"),
            ("运动时长统计", "QUERY"),
            ("昨天的数据", "QUERY"),
            ("搜索健康记录", "QUERY"),
            ("查询营养摄入", "QUERY"),
            ("看看我的进步", "QUERY"),
            ("本月运动了多少次？", "QUERY"),
            ("查看卡路里消耗", "QUERY"),
            ("显示体重趋势", "QUERY"),
            ("今天走了多少步？", "QUERY"),
            ("查询睡眠时间", "QUERY"),
            ("看看营养搭配", "QUERY"),
            ("检查运动目标", "QUERY"),
            ("昨天吃了多少蛋白质？", "QUERY"),
            ("查看健康评分", "QUERY"),
            ("搜索运动记录", "QUERY"),
            
            # 生成报告 - 更多样本
            ("生成健康报告", "GENERATE_REPORT"),
            ("分析我的饮食习惯", "GENERATE_REPORT"),
            ("制作运动总结", "GENERATE_REPORT"),
            ("帮我分析体重趋势", "GENERATE_REPORT"),
            ("营养分析报告", "GENERATE_REPORT"),
            ("健康状况评估", "GENERATE_REPORT"),
            ("月度总结", "GENERATE_REPORT"),
            ("制作图表", "GENERATE_REPORT"),
            ("数据可视化", "GENERATE_REPORT"),
            ("健康评分", "GENERATE_REPORT"),
            ("周报告", "GENERATE_REPORT"),
            ("运动效果分析", "GENERATE_REPORT"),
            ("饮食均衡评估", "GENERATE_REPORT"),
            ("卡路里摄入分析", "GENERATE_REPORT"),
            ("健康趋势图", "GENERATE_REPORT"),
            ("身体指标报告", "GENERATE_REPORT"),
            ("营养摄入统计", "GENERATE_REPORT"),
            ("运动强度分析", "GENERATE_REPORT"),
            ("健康对比报告", "GENERATE_REPORT"),
            ("综合健康评估", "GENERATE_REPORT"),
            
            # 建议咨询 - 更多样本
            ("给我一些健康建议", "ADVICE"),
            ("如何减肥？", "ADVICE"),
            ("推荐运动方案", "ADVICE"),
            ("营养搭配建议", "ADVICE"),
            ("怎么改善睡眠？", "ADVICE"),
            ("健身计划推荐", "ADVICE"),
            ("饮食建议", "ADVICE"),
            ("如何增肌？", "ADVICE"),
            ("健康小贴士", "ADVICE"),
            ("运动指导", "ADVICE"),
            ("怎么控制体重？", "ADVICE"),
            ("如何提高免疫力？", "ADVICE"),
            ("推荐健康食谱", "ADVICE"),
            ("怎么缓解疲劳？", "ADVICE"),
            ("如何改善体质？", "ADVICE"),
            ("给我运动建议", "ADVICE"),
            ("怎么保持健康？", "ADVICE"),
            ("营养补充建议", "ADVICE"),
            ("如何预防疾病？", "ADVICE"),
            ("健康生活方式建议", "ADVICE"),
            
            # 未知/其他
            ("你好", "UNKNOWN"),
            ("今天天气不错", "UNKNOWN"),
            ("帮我设置提醒", "UNKNOWN"),
            ("系统设置", "UNKNOWN"),
            ("退出程序", "UNKNOWN"),
            ("谢谢", "UNKNOWN"),
            ("再见", "UNKNOWN"),
            ("什么时间了？", "UNKNOWN"),
            ("今天星期几？", "UNKNOWN"),
            ("打开音乐", "UNKNOWN"),
        ]
        
        return training_data
    
    def preprocess_text(self, text: str) -> str:
        """预处理单个文本"""
        words = self.preprocessor.preprocess(text)
        return ' '.join(words)
    
    def train(self, additional_data: List[Tuple[str, str]] = None, test_size: float = 0.2) -> Dict[str, float]:
        """训练模型"""
        logger.info("开始训练ML意图分类器...")
        
        # 准备训练数据
        training_data = self.get_training_data()
        
        # 添加额外数据
        if additional_data:
            training_data.extend(additional_data)
        
        # 分离文本和标签
        texts, labels = zip(*training_data)
        
        # 文本预处理
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # 创建标签映射
        unique_labels = list(set(labels))
        self.label_to_intent = {}
        self.intent_to_label = {}
         
         # 映射标签到IntentType枚举
        for label in unique_labels:
            try:
                 # 直接映射标签名到枚举值
                intent_type = getattr(IntentType, label)
                self.label_to_intent[label] = intent_type
                self.intent_to_label[intent_type] = label
            except AttributeError:
                 # 如果找不到对应的枚举值，映射到UNKNOWN
                self.label_to_intent[label] = IntentType.UNKNOWN
                self.intent_to_label[IntentType.UNKNOWN] = label
        
        # 分割训练/测试集
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # 创建向量化器和模型
        self.vectorizer = self._create_vectorizer()
        self.model = self._create_model()
        
        # 创建管道
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.model)
        ])
        
        # 网格搜索优化（可选）
        if self.enable_grid_search:
            self._perform_grid_search(X_train, y_train)
        else:
            # 直接训练
            self.pipeline.fit(X_train, y_train)
        
        self.is_trained = True
        
        # 评估模型
        train_score = self.pipeline.score(X_train, y_train)
        test_score = self.pipeline.score(X_test, y_test)
        
        # 交叉验证
        cv_scores = cross_val_score(self.pipeline, processed_texts, labels, cv=5)
        
        # 详细评估
        y_pred = self.pipeline.predict(X_test)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        
        # 更新指标
        self.metrics.update({
            'accuracy': test_score,
            'training_samples': len(training_data),
            'last_update': datetime.now().isoformat(),
            'train_accuracy': train_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_rep,
            'feature_count': len(self.vectorizer.get_feature_names_out()) if hasattr(self.vectorizer, 'get_feature_names_out') else 0
        })
        
        # 保存模型
        self._save_model()
        
        logger.info(f"ML模型训练完成 - 测试准确率: {test_score:.3f}, CV均值: {cv_scores.mean():.3f}")
        
        return self.metrics
    
    def _perform_grid_search(self, X_train: List[str], y_train: List[str]):
        """执行网格搜索优化"""
        logger.info("执行网格搜索优化...")
        
        # 定义参数网格
        if self.model_type == 'logistic':
            param_grid = {
                'vectorizer__max_features': [1000, 2000],
                'vectorizer__ngram_range': [(1, 2), (1, 3)],
                'classifier__C': [0.1, 1, 10]
            }
        elif self.model_type == 'random_forest':
            param_grid = {
                'vectorizer__max_features': [1000, 2000],
                'vectorizer__ngram_range': [(1, 2), (1, 3)],
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20]
            }
        else:
            param_grid = {
                'vectorizer__max_features': [1000, 2000],
                'vectorizer__ngram_range': [(1, 2), (1, 3)]
            }
        
        # 执行网格搜索
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # 使用最佳参数
        self.pipeline = grid_search.best_estimator_
        self.vectorizer = self.pipeline.named_steps['vectorizer']
        self.model = self.pipeline.named_steps['classifier']
        
        logger.info(f"网格搜索完成 - 最佳参数: {grid_search.best_params_}")
    
    def predict(self, text: str) -> MLPredictionResult:
        """预测意图"""
        start_time = time.time()
        
        if not self.is_trained:
            logger.warning("ML模型未训练，返回默认结果")
            return MLPredictionResult(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                probabilities={},
                features_used=0,
                processing_time=time.time() - start_time,
                model_used=self.model_type
            )
        
        try:
            # 预处理文本
            processed_text = self.preprocess_text(text)
            
            # 预测概率
            probabilities = self.pipeline.predict_proba([processed_text])[0]
            classes = self.pipeline.classes_
            
            # 构建概率字典
            prob_dict = {}
            for i, class_name in enumerate(classes):
                if class_name in self.label_to_intent:
                    intent = self.label_to_intent[class_name]
                    prob_dict[intent.value] = float(probabilities[i])
            
            # 获取最高概率的意图
            max_prob_idx = np.argmax(probabilities)
            predicted_label = classes[max_prob_idx]
            confidence = float(probabilities[max_prob_idx])
            
            # 转换为IntentType
            if predicted_label in self.label_to_intent:
                predicted_intent = self.label_to_intent[predicted_label]
            else:
                predicted_intent = IntentType.UNKNOWN
                confidence = 0.0
            
            processing_time = time.time() - start_time
            
            return MLPredictionResult(
                intent=predicted_intent,
                confidence=confidence,
                probabilities=prob_dict,
                features_used=self.metrics.get('feature_count', 0),
                processing_time=processing_time,
                model_used=self.model_type
            )
            
        except Exception as e:
            logger.error(f"ML预测失败: {e}")
            return MLPredictionResult(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                probabilities={},
                features_used=0,
                processing_time=time.time() - start_time,
                model_used=self.model_type
            )
    
    def batch_predict(self, texts: List[str]) -> List[MLPredictionResult]:
        """批量预测"""
        if not self.is_trained:
            return [self.predict(text) for text in texts]
        
        start_time = time.time()
        
        try:
            # 预处理文本
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # 批量预测
            probabilities_batch = self.pipeline.predict_proba(processed_texts)
            classes = self.pipeline.classes_
            
            results = []
            for i, text in enumerate(texts):
                probabilities = probabilities_batch[i]
                
                # 构建概率字典
                prob_dict = {}
                for j, class_name in enumerate(classes):
                    if class_name in self.label_to_intent:
                        intent = self.label_to_intent[class_name]
                        prob_dict[intent.value] = float(probabilities[j])
                
                # 获取最高概率的意图
                max_prob_idx = np.argmax(probabilities)
                predicted_label = classes[max_prob_idx]
                confidence = float(probabilities[max_prob_idx])
                
                # 转换为IntentType
                if predicted_label in self.label_to_intent:
                    predicted_intent = self.label_to_intent[predicted_label]
                else:
                    predicted_intent = IntentType.UNKNOWN
                    confidence = 0.0
                
                results.append(MLPredictionResult(
                    intent=predicted_intent,
                    confidence=confidence,
                    probabilities=prob_dict,
                    features_used=self.metrics.get('feature_count', 0),
                    processing_time=(time.time() - start_time) / len(texts),  # 平均时间
                    model_used=self.model_type
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"批量预测失败: {e}")
            return [self.predict(text) for text in texts]
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """获取特征重要性"""
        if not self.is_trained:
            return {}
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                # Random Forest等
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                # Logistic Regression等
                importances = np.abs(self.model.coef_[0])
            else:
                return {}
            
            feature_names = self.vectorizer.get_feature_names_out()
            
            # 获取top特征
            top_indices = np.argsort(importances)[-top_n:][::-1]
            top_features = {}
            
            for idx in top_indices:
                if idx < len(feature_names):
                    top_features[feature_names[idx]] = float(importances[idx])
            
            return top_features
            
        except Exception as e:
            logger.error(f"获取特征重要性失败: {e}")
            return {}
    
    def evaluate_on_test_set(self, test_data: List[Tuple[str, str]]) -> Dict[str, Any]:
        """在测试集上评估模型"""
        if not self.is_trained:
            return {"error": "模型未训练"}
        
        texts, true_labels = zip(*test_data)
        
        # 预测
        predictions = []
        confidences = []
        
        for text in texts:
            result = self.predict(text)
            # 将IntentType转换为标签字符串
            predicted_label = self.intent_to_label.get(result.intent, "UNKNOWN")
            predictions.append(predicted_label)
            confidences.append(result.confidence)
        
        # 计算指标
        accuracy = accuracy_score(true_labels, predictions)
        classification_rep = classification_report(true_labels, predictions, output_dict=True)
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist(),
            'avg_confidence': np.mean(confidences),
            'predictions': list(zip(texts, true_labels, predictions, confidences))
        }
    
    def add_training_sample(self, text: str, intent_label: str):
        """添加训练样本（用于在线学习）"""
        # 这里可以实现增量学习
        # 暂时存储新样本，达到一定数量后重新训练
        pass
    
    def _save_model(self):
        """保存模型到磁盘"""
        try:
            model_file = os.path.join(self.model_dir, f'ml_intent_{self.model_type}.pkl')
            metrics_file = os.path.join(self.model_dir, f'metrics_{self.model_type}.json')
            
            # 保存模型
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'pipeline': self.pipeline,
                    'vectorizer': self.vectorizer,
                    'model': self.model,
                    'label_to_intent': self.label_to_intent,
                    'intent_to_label': self.intent_to_label,
                    'model_type': self.model_type
                }, f)
            
            # 保存指标（需要处理不可序列化的对象）
            serializable_metrics = {}
            for key, value in self.metrics.items():
                if isinstance(value, (str, int, float, bool, list, dict)):
                    serializable_metrics[key] = value
                else:
                    serializable_metrics[key] = str(value)
            
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_metrics, f, ensure_ascii=False, indent=2)
            
            logger.info(f"ML模型已保存: {model_file}")
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
    
    def load_model(self, model_type: str = None) -> bool:
        """从磁盘加载模型"""
        try:
            if model_type:
                self.model_type = model_type
                
            model_file = os.path.join(self.model_dir, f'ml_intent_{self.model_type}.pkl')
            metrics_file = os.path.join(self.model_dir, f'metrics_{self.model_type}.json')
            
            if os.path.exists(model_file):
                # 加载模型
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.pipeline = model_data['pipeline']
                self.vectorizer = model_data['vectorizer']
                self.model = model_data['model']
                self.label_to_intent = model_data['label_to_intent']
                self.intent_to_label = model_data['intent_to_label']
                self.model_type = model_data['model_type']
                
                # 加载指标
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        self.metrics = json.load(f)
                
                self.is_trained = True
                logger.info(f"ML模型已加载: {model_file}")
                return True
            else:
                logger.warning(f"模型文件不存在: {model_file}")
                return False
        
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'metrics': self.metrics,
            'feature_engineering_enabled': self.enable_feature_engineering,
            'grid_search_enabled': self.enable_grid_search
        }
        
        if self.is_trained:
            info['classes'] = list(self.label_to_intent.keys())
            info['intent_types'] = [intent.value for intent in self.label_to_intent.values()]
        
        return info


def compare_models(test_data: List[Tuple[str, str]] = None) -> Dict[str, Dict]:
    """比较不同模型的性能"""
    if not ML_AVAILABLE:
        print("机器学习库未安装")
        return {}
    
    models = ['logistic', 'naive_bayes', 'random_forest']
    results = {}
    
    for model_type in models:
        print(f"\n训练 {model_type} 模型...")
        
        classifier = MLIntentClassifier(model_type=model_type)
        
        # 训练模型
        metrics = classifier.train()
        
        # 在测试集上评估
        if test_data:
            eval_results = classifier.evaluate_on_test_set(test_data)
            metrics.update(eval_results)
        
        results[model_type] = metrics
        
        print(f"{model_type} 完成 - 准确率: {metrics['accuracy']:.3f}")
    
    return results


# 使用示例
if __name__ == "__main__":
    if not ML_AVAILABLE:
        print("请先安装必要的机器学习库：")
        print("pip install scikit-learn jieba numpy")
        exit(1)
    
    # 创建分类器
    classifier = MLIntentClassifier(
        model_type='logistic',
        enable_feature_engineering=True,
        enable_grid_search=False
    )
    
    # 训练模型
    print("开始训练模型...")
    metrics = classifier.train()
    print("训练完成！")
    print(f"模型准确率: {metrics['accuracy']:.3f}")
    print(f"交叉验证分数: {metrics['cv_mean']:.3f} (±{metrics['cv_std']:.3f})")
    
    # 测试预测
    test_cases = [
        "我今天吃了苹果",
        "我今天吃了什么？",
        "跑步30分钟",
        "生成健康报告",
        "给我一些建议",
        "昨天的运动记录",
        "分析我的饮食习惯",
        "如何减肥？"
    ]
    
    print("\n测试预测：")
    for text in test_cases:
        result = classifier.predict(text)
        print(f"输入: {text}")
        print(f"预测: {result.intent.value} (置信度: {result.confidence:.3f})")
        print(f"处理时间: {result.processing_time*1000:.1f}ms")
        print("---")
    
    # 显示特征重要性
    print("\n重要特征:")
    important_features = classifier.get_feature_importance(top_n=10)
    for feature, importance in important_features.items():
        print(f"{feature}: {importance:.4f}")
    
    # 显示模型信息
    print("\n模型信息:")
    model_info = classifier.get_model_info()
    for key, value in model_info.items():
        if key != 'metrics':
            print(f"{key}: {value}")
    
    # 模型比较（可选）
    print("\n是否要比较不同模型? (y/n): ", end="")
    if input().lower() == 'y':
        print("\n开始模型比较...")
        comparison_results = compare_models()
        
        print("\n模型比较结果:")
        for model, result in comparison_results.items():
            print(f"{model}: 准确率 {result['accuracy']:.3f}, CV {result['cv_mean']:.3f}")