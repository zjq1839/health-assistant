#!/usr/bin/env python3
"""调试报告生成意图识别失败的原因"""

import re
from core.enhanced_state import IntentType
from core.lightweight_planner import RuleBasedClassifier

def debug_generate_report_patterns():
    """调试报告生成规则匹配情况"""
    
    # 当前规则引擎的模式
    current_patterns = [
        re.compile(r'生成.*报告', re.IGNORECASE),
        re.compile(r'(分析|总结).*(饮食|运动|健康)', re.IGNORECASE),
        re.compile(r'健康.*报告', re.IGNORECASE),
    ]
    
    # 测试用例（来自 test_comprehensive_intent_recognition.py）
    test_cases = [
        "生成健康报告",        # 应该匹配 ✓
        "制作我的分析报告",    # 不匹配 - 没有"生成"
        "创建健康总结",        # 不匹配 - 没有"生成"，没有"报告"
        "导出数据报告",        # 不匹配 - 没有"生成"
        "生成本周报告",        # 应该匹配 ✓
        "制作月度总结",        # 不匹配 - 没有"生成"，没有"报告"
        "创建年度报告",        # 不匹配 - 没有"生成"
        "导出季度数据",        # 不匹配 - 没有"生成"，没有"报告"
        "生成饮食分析报告",    # 应该匹配 ✓
        "制作运动效果报告",    # 不匹配 - 没有"生成"
        "创建体重变化报告",    # 不匹配 - 没有"生成"
        "导出营养摄入报告",    # 不匹配 - 没有"生成"
        "制作图表分析",        # 不匹配 - 没有"生成"，没有"报告"
        "生成趋势图",          # 不匹配 - 没有"报告"
        "创建可视化报告",      # 不匹配 - 没有"生成"
        "导出统计图表",        # 不匹配 - 没有"生成"，没有"报告"
        "帮我做个总结",        # 可能匹配模式2
        "整理下数据",          # 不匹配
        "分析我的情况",        # 不匹配 - 没有"饮食|运动|健康"
        "汇总健康数据",        # 不匹配 - 没有"生成"，没有"报告"
    ]
    
    print("🔍 调试报告生成规则匹配情况：")
    print("=" * 60)
    
    total_cases = len(test_cases)
    matched_cases = 0
    
    for i, text in enumerate(test_cases, 1):
        matches = []
        for j, pattern in enumerate(current_patterns, 1):
            if pattern.search(text):
                matches.append(f"模式{j}")
        
        if matches:
            matched_cases += 1
            print(f"✅ {i:2d}. '{text}' -> 匹配: {', '.join(matches)}")
        else:
            print(f"❌ {i:2d}. '{text}' -> 无匹配")
    
    print("=" * 60)
    print(f"📊 匹配统计：{matched_cases}/{total_cases} ({matched_cases/total_cases:.1%})")
    
    print("\n🔧 改进建议的规则模式：")
    improved_patterns = [
        r'(生成|制作|创建|导出).*(报告|分析|总结)',
        r'(分析|总结).*(饮食|运动|健康|情况|数据)',
        r'(健康|饮食|运动).*(报告|分析|总结)',
        r'(图表|趋势|可视化).*(分析|报告)',
        r'(汇总|整理).*(数据|健康)',
    ]
    
    print("改进后的模式：")
    for i, pattern in enumerate(improved_patterns, 1):
        print(f"  {i}. {pattern}")
    
    # 测试改进后的模式
    print("\n🧪 测试改进后的匹配效果：")
    print("=" * 60)
    
    improved_compiled = [re.compile(p, re.IGNORECASE) for p in improved_patterns]
    improved_matched = 0
    
    for i, text in enumerate(test_cases, 1):
        matches = []
        for j, pattern in enumerate(improved_compiled, 1):
            if pattern.search(text):
                matches.append(f"模式{j}")
        
        if matches:
            improved_matched += 1
            print(f"✅ {i:2d}. '{text}' -> 匹配: {', '.join(matches)}")
        else:
            print(f"❌ {i:2d}. '{text}' -> 无匹配")
    
    print("=" * 60)
    print(f"📊 改进后匹配统计：{improved_matched}/{total_cases} ({improved_matched/total_cases:.1%})")
    print(f"📈 提升：+{improved_matched - matched_cases} 个匹配 (+{(improved_matched - matched_cases)/total_cases:.1%})")

def test_rule_classifier_scoring():
    """测试规则引擎的评分机制"""
    print("\n🎯 测试规则引擎评分机制：")
    print("=" * 60)
    
    classifier = RuleBasedClassifier()
    
    test_cases = [
        "生成健康报告",
        "制作我的分析报告", 
        "分析我的健康情况",
        "帮我做个总结",
    ]
    
    for text in test_cases:
        result = classifier.classify(text)
        print(f"输入：'{text}'")
        print(f"  意图：{result.intent.value}")
        print(f"  置信度：{result.confidence:.3f}")
        print(f"  方法：{result.method}")
        print()

if __name__ == "__main__":
    debug_generate_report_patterns()
    test_rule_classifier_scoring()