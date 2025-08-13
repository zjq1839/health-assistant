#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习意图分类器演示脚本
展示核心功能的简化使用示例
"""

from core.ml_intent_classifier import MLIntentClassifier, compare_models

def main():
    print("=" * 60)
    print("🤖 独立机器学习意图分类器演示")
    print("=" * 60)
    
    # 1. 创建并训练分类器
    print("\n📚 步骤1: 创建并训练分类器")
    print("-" * 40)
    
    classifier = MLIntentClassifier(
        model_type='random_forest',  # 使用随机森林（最高准确率）
        enable_feature_engineering=True
    )
    
    print("正在训练模型...")
    metrics = classifier.train()
    
    print(f"✅ 训练完成！")
    print(f"   - 准确率: {metrics['accuracy']:.1%}")
    print(f"   - 交叉验证: {metrics['cv_mean']:.1%} ± {metrics['cv_std']:.1%}")
    print(f"   - 特征数量: {metrics['feature_count']}")
    
    # 2. 单个文本预测演示
    print("\n🎯 步骤2: 单个文本预测演示")
    print("-" * 40)
    
    test_texts = [
        "我今天吃了一个苹果和一碗米饭",
        "跑步了半小时，感觉很累",
        "查看一下我本周的运动记录",
        "帮我生成健康分析报告",
        "给我一些减肥的建议",
        "今天天气真不错"  # 未知意图
    ]
    
    for text in test_texts:
        result = classifier.predict(text)
        
        # 根据置信度显示不同颜色的状态
        if result.confidence >= 0.8:
            status = "🟢 高置信度"
        elif result.confidence >= 0.6:
            status = "🟡 中等置信度"
        else:
            status = "🔴 低置信度"
            
        print(f"文本: '{text}'")
        print(f"意图: {result.intent.value} | {status} ({result.confidence:.1%})")
        print()
    
    # 3. 批量预测演示
    print("⚡ 步骤3: 批量预测演示")
    print("-" * 40)
    
    batch_texts = [
        "记录今天的早餐",
        "查询昨天消耗的卡路里",
        "生成本月健康报告"
    ]
    
    print(f"批量处理 {len(batch_texts)} 条文本...")
    batch_results = classifier.batch_predict(batch_texts)
    
    for text, result in zip(batch_texts, batch_results):
        print(f"• {text} -> {result.intent.value}")
    
    # 4. 特征重要性分析
    print("\n🔍 步骤4: 特征重要性分析")
    print("-" * 40)
    
    important_features = classifier.get_feature_importance(top_n=10)
    
    print("最重要的10个特征:")
    for i, (feature, importance) in enumerate(important_features.items(), 1):
        print(f"{i:2d}. {feature:<15} ({importance:.4f})")
    
    # 5. 模型信息展示
    print("\n📊 步骤5: 模型信息")
    print("-" * 40)
    
    info = classifier.get_model_info()
    print(f"模型类型: {info['model_type']}")
    print(f"训练状态: {'✅ 已训练' if info['is_trained'] else '❌ 未训练'}")
    print(f"支持意图类型: {len(info['intent_types'])} 种")
    for intent in info['intent_types']:
        print(f"  • {intent}")
    
    # 6. 模型比较（可选）
    print("\n🏆 步骤6: 不同算法性能比较（可选）")
    print("-" * 40)
    
    user_input = input("是否运行模型比较？这会花费额外时间 (y/n): ").lower().strip()
    
    if user_input in ['y', 'yes', '是', '1']:
        print("正在比较不同算法性能...")
        try:
            comparison_results = compare_models()
            
            print("\n算法性能排行榜:")
            # 按准确率排序
            sorted_results = sorted(
                comparison_results.items(), 
                key=lambda x: x[1]['accuracy'], 
                reverse=True
            )
            
            for rank, (model, metrics) in enumerate(sorted_results, 1):
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
                print(f"{medal} {model:<20} 准确率: {metrics['accuracy']:.1%}")
                
        except Exception as e:
            print(f"❌ 模型比较失败: {e}")
    else:
        print("跳过模型比较")
    
    # 7. 交互式测试
    print("\n🎮 步骤7: 交互式测试")
    print("-" * 40)
    print("输入文本测试分类器，输入 'quit' 退出")
    
    while True:
        try:
            user_text = input("\n请输入测试文本: ").strip()
            
            if not user_text or user_text.lower() in ['quit', 'exit', '退出', 'q']:
                break
                
            result = classifier.predict(user_text)
            
            print(f"预测结果:")
            print(f"  意图: {result.intent.value}")
            print(f"  置信度: {result.confidence:.1%}")
            print(f"  处理时间: {result.processing_time:.1f}ms")
            
            # 给出建议
            if result.confidence < 0.5:
                print("  💡 提示: 置信度较低，建议优化文本表达或增加训练数据")
            elif result.intent.value == 'unknown':
                print("  💡 提示: 未识别意图，可能需要扩展训练数据")
                
        except KeyboardInterrupt:
            print("\n\n用户中断，退出测试")
            break
        except Exception as e:
            print(f"❌ 预测错误: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 演示完成！")
    print("\n📝 总结:")
    print("✅ 独立的机器学习分类器已成功创建")
    print("✅ 支持多种算法和特征工程")
    print("✅ 具备完整的训练、预测、评估能力")
    print("✅ 可以独立使用，不依赖现有代码")
    print("\n📖 更多详细信息请查看: ml_classifier_usage.md")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except ImportError as e:
        print(f"❌ 依赖库缺失: {e}")
        print("💡 请先安装依赖: pip install scikit-learn jieba numpy")
    except Exception as e:
        print(f"❌ 程序运行错误: {e}")
        import traceback
        traceback.print_exc()