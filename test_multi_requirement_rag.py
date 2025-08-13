#!/usr/bin/env python3
"""
多需求RAG服务测试脚本

测试复杂查询的多需求解析和处理功能，演示如何使用：
1. MultiRequirementParser - 解析包含多个要求的查询
2. MultiRequirementRAGService - 为每个需求检索上下文并生成回答

使用方法:
python test_multi_requirement_rag.py
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.multi_requirement_parser import create_multi_requirement_parser
from core.multi_requirement_rag_service import create_multi_requirement_rag_service
from utils.logger import logger

def test_requirement_parser():
    """测试需求解析器"""
    print("\n" + "="*60)
    print("测试需求解析器 (MultiRequirementParser)")
    print("="*60)
    
    try:
        # 创建需求解析器
        parser = create_multi_requirement_parser()
        
        # 测试查询列表
        test_queries = [
            "查询我昨天的饮食记录，并给我一些今天的健康建议",
            "记录我今天早餐吃了鸡蛋和牛奶，然后分析我这周的营养摄入情况",
            "生成我的健康报告，同时推荐适合我的运动计划"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n【测试查询 {i}】")
            print(f"原始查询: {query}")
            print("-" * 50)
            
            try:
                # 解析需求
                result = parser.parse(query)
                
                print(f"解析出 {len(result.requirements)} 个需求:")
                print(f"复杂度: {result.complexity}")
                print(f"解析方法: {result.parsing_method}")
                print(f"总置信度: {result.total_confidence:.2f}")
                
                for j, req in enumerate(result.requirements, 1):
                    print(f"  需求 {j}: [{req.type.value}] {req.description}")
                    print(f"    意图: {req.intent.value}")
                    print(f"    优先级: {req.priority}")
                    print(f"    置信度: {req.confidence:.2f}")
                    print(f"    关键词: {req.keywords}")
                    if req.dependencies:
                        print(f"    依赖: {req.dependencies}")
                    print()
                
                print(f"执行顺序: {' -> '.join(result.execution_order)}")
                    
            except Exception as e:
                print(f"解析失败: {e}")
                logger.error(f"解析失败: {e}", exc_info=True)
    
    except Exception as e:
        print(f"需求解析器初始化失败: {e}")
        logger.error(f"需求解析器初始化失败: {e}", exc_info=True)

def test_multi_requirement_rag():
    """测试多需求RAG服务"""
    print("\n" + "="*60)
    print("测试多需求RAG服务 (MultiRequirementRAGService)")
    print("="*60)
    
    try:
        # 创建多需求RAG服务
        multi_rag_service = create_multi_requirement_rag_service()
        
        # 可选的知识库加载（如果目录存在）
        kb_path = "/home/zjq/document/langchain_learn/rag_knowledge_base"
        if os.path.exists(kb_path):
            print(f"📚 加载知识库: {kb_path}")
            multi_rag_service.rag_service.load_knowledge_base(kb_path)
        else:
            print(f"⚠️ 知识库目录不存在: {kb_path}，将使用默认响应")
        
        # 测试复杂查询
        complex_queries = [
            {
                "query": "我有高血压，请推荐降压食物和运动方式，并说明注意事项",
                "user_profile": {
                    "age": 45,
                    "gender": "男",
                    "health_conditions": ["高血压"],
                    "preferences": ["中式饮食"]
                }
            },
            {
                "query": "我想减肥，需要低卡路里食谱，适合的运动计划，以及如何计算每日消耗",
                "user_profile": {
                    "age": 30,
                    "gender": "女",
                    "weight": 65,
                    "height": 160,
                    "goal": "减肥"
                }
            }
        ]
        
        for i, test_case in enumerate(complex_queries, 1):
            print(f"\n【多需求RAG测试 {i}】")
            print(f"查询: {test_case['query']}")
            print(f"用户画像: {test_case['user_profile']}")
            print("-" * 50)
            
            try:
                # 处理多需求查询
                result = multi_rag_service.process_multi_requirement_query(
                    query=test_case['query'],
                    user_profile=test_case['user_profile']
                )
                
                print(f"解析出 {len(result.requirement_contexts)} 个需求:")
                
                # 显示每个需求的处理结果
                for j, req_context in enumerate(result.requirement_contexts, 1):
                    print(f"\n  需求 {j}: {req_context.requirement.description}")
                    print(f"    类型: {req_context.requirement.type.value}")
                    print(f"    检索到 {len(req_context.rag_context.retrieved_docs)} 个相关文档")
                    print(f"    回答: {req_context.response[:100]}...")
                    print(f"    处理时间: {req_context.processing_time:.2f}s")
                
                print(f"\n📋 综合回答:")
                print(f"{result.final_response}")
                
                print(f"\n📊 处理统计:")
                print(f"  - 总处理时间: {result.total_processing_time:.2f}s")
                print(f"  - 成功率: {result.success_rate:.2%}")
                
            except Exception as e:
                print(f"处理失败: {e}")
                logger.error(f"多需求RAG处理失败: {e}", exc_info=True)
            
    except Exception as e:
        print(f"多需求RAG服务初始化失败: {e}")
        logger.error(f"多需求RAG服务失败: {e}", exc_info=True)

def test_integration():
    """集成测试：展示完整的多需求处理流程"""
    print("\n" + "="*60)
    print("集成测试：完整多需求处理流程")
    print("="*60)
    
    # 模拟真实的复杂健康咨询场景
    integration_test_cases = [
        {
            "scenario": "糖尿病患者综合咨询",
            "query": "我是糖尿病患者，需要血糖控制饮食建议，适合的运动类型，以及药物管理指导",
            "user_context": {
                "age": 55,
                "condition": "2型糖尿病",
                "current_medication": "二甲双胍",
                "blood_sugar": "空腹8.5mmol/L"
            }
        },
        {
            "scenario": "孕期营养咨询",
            "query": "我怀孕6个月，请推荐孕期营养食谱，适合的运动，以及需要补充的维生素",
            "user_context": {
                "age": 28,
                "pregnancy_stage": "孕中期",
                "weight_gain": "5kg",
                "previous_issues": "无"
            }
        }
    ]
    
    try:
        multi_rag_service = create_multi_requirement_rag_service()
        
        for test_case in integration_test_cases:
            print(f"\n🏥 场景: {test_case['scenario']}")
            print(f"咨询: {test_case['query']}")
            print(f"用户背景: {test_case['user_context']}")
            print("-" * 50)
            
            result = multi_rag_service.process_multi_requirement_query(
                query=test_case['query'],
                user_profile=test_case['user_context']
            )
            
            print(f"✅ 处理完成，共解析 {len(result.requirement_contexts)} 个专业需求")
            print(f"🎯 综合专业建议:\n{result.final_response}")
            print(f"⏱️  处理耗时: {result.total_processing_time:.2f}秒")
            
    except Exception as e:
        print(f"集成测试失败: {e}")
        logger.error(f"集成测试失败: {e}", exc_info=True)

def main():
    """主测试函数"""
    print("🚀 开始多需求RAG系统测试")
    print("此测试将演示如何处理包含多个要求的复杂查询")
    
    try:
        # 1. 测试需求解析器
        test_requirement_parser()
        
        # 2. 测试多需求RAG服务
        test_multi_requirement_rag()
        
        # 3. 集成测试
        test_integration()
        
        print("\n✅ 所有测试完成！")
        print("\n💡 使用说明:")
        print("1. MultiRequirementParser 可以解析复杂查询中的多个需求")
        print("2. MultiRequirementRAGService 为每个需求提供专门的RAG检索和生成")
        print("3. 支持需求优先级排序和依赖关系处理")
        print("4. 可以集成到现有的健康咨询agents中使用")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        logger.error(f"主测试失败: {e}", exc_info=True)

if __name__ == "__main__":
    main()