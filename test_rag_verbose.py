#!/usr/bin/env python3
"""
RAG 详细过程演示脚本

展示RAG检索和生成的完整过程，包括：
1. 查询增强
2. 文档检索
3. 模型思考过程
4. 最终回答生成
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.enhanced_rag_service import create_enhanced_rag_service
from core.rag_service import create_rag_service
from utils.logger import logger


def test_verbose_rag_process():
    """测试详细的RAG过程"""
    
    print("🔥 RAG 详细过程演示")
    print("="*100)
    
    # 创建RAG服务
    print("🔧 初始化RAG服务...")
    try:
        rag_service = create_rag_service(use_hybrid=True)
        print("✅ RAG服务初始化成功（混合检索模式）")
    except Exception as e:
        print(f"❌ RAG服务初始化失败: {e}")
        return False
    
    # 测试查询列表
    test_queries = [
        {
            "query": "我想减肥，应该怎么安排饮食？",
            "user_profile": {"age": 25, "gender": "女", "health_goal": "减肥"},
            "domain": "nutrition"
        },
        {
            "query": "高血压患者的运动建议",
            "user_profile": {"age": 50, "gender": "男", "health_condition": "高血压"},
            "domain": "exercise"
        },
        {
            "query": "儿童营养需要注意什么？",
            "user_profile": {"age": 8, "guardian": "家长咨询"},
            "domain": "nutrition"
        }
    ]
    
    # 逐个测试查询
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n🎯 测试案例 {i}/3")
        print("="*100)
        
        try:
            # 步骤1: 检索上下文
            print(f"📋 测试查询: {test_case['query']}")
            
            context = rag_service.retrieve_context(
                query=test_case["query"],
                user_profile=test_case["user_profile"],
                domain_context=test_case["domain"],
                k=3  # 限制为3个文档以便观察
            )
            
            # 步骤2: 生成回答
            response = rag_service.generate_with_context(context)
            
            # 显示最终结果
            print(f"\n📝 最终生成的回答:")
            print("="*80)
            print(response)
            print("="*80)
            
            print(f"✅ 测试案例 {i} 完成")
            
        except Exception as e:
            print(f"❌ 测试案例 {i} 失败: {e}")
            logger.error(f"Test case {i} failed", exc_info=True)
        
        # 分隔符
        if i < len(test_queries):
            print("\n" + "🔄"*50 + " 下一个测试案例 " + "🔄"*50)
    
    print(f"\n🎉 RAG 详细过程演示完成！")
    return True


def test_rag_agent_integration():
    """测试RAG代理集成"""
    
    print("\n" + "="*100)
    print("🤖 测试RAG增强建议代理")
    print("="*100)
    
    try:
        from agents.rag_enhanced_advice_agent import RAGEnhancedAdviceAgent
        from core.service_container import get_container, LLMService, DatabaseService
        
        # 获取服务
        container = get_container()
        llm_service = container.get(LLMService)
        db_service = container.get(DatabaseService)
        
        # 创建RAG增强代理
        agent = RAGEnhancedAdviceAgent(
            llm_service=llm_service,
            db_service=db_service
        )
        
        # 模拟状态
        from core.enhanced_state import EnhancedState
        state = EnhancedState()
        state['messages'] = [
            {"role": "user", "content": "我是糖尿病患者，应该如何控制饮食？"}
        ]
        
        print("🎯 模拟用户咨询: 我是糖尿病患者，应该如何控制饮食？")
        
        # 运行代理
        response = agent.run(state)
        
        print(f"\n📝 代理回答:")
        print("="*80)
        print(response.message)
        print("="*80)
        
        print("✅ RAG代理测试完成")
        return True
        
    except Exception as e:
        print(f"❌ RAG代理测试失败: {e}")
        logger.error("RAG agent test failed", exc_info=True)
        return False


def main():
    """主函数"""
    print("🚀 开始RAG详细过程测试")
    
    results = []
    
    # 测试基础RAG过程
    results.append(("RAG基础过程", test_verbose_rag_process()))
    
    # 测试RAG代理集成
    results.append(("RAG代理集成", test_rag_agent_integration()))
    
    # 汇总结果
    print("\n" + "="*100)
    print("📊 测试结果汇总:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有测试通过！RAG详细过程展示成功")
    else:
        print("⚠️  部分测试失败，请检查相关组件")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)