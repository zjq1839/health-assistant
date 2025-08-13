#!/usr/bin/env python3
"""
RAG 知识库测试脚本

测试从 "知识库" 目录导入的文档是否可以正确查询
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.enhanced_rag_service import create_enhanced_rag_service, LLAMA_INDEX_AVAILABLE
from core.rag_service import create_rag_service
from utils.logger import logger

def test_langchain_rag():
    """测试 LangChain RAG"""
    print("🔍 测试 LangChain RAG 系统...")
    
    try:
        # 使用增强的 RAG 服务
        rag_service = create_enhanced_rag_service(use_llama_index=False)
        
        # 测试查询
        queries = [
            "什么是平衡膳食？",
            "如何控制体重？",
            "地中海饮食有什么特点？",
            "运动前后应该如何补充营养？",
            "老年人的营养需求有什么特点？"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n📝 查询 {i}: {query}")
            
            # 检索相关上下文
            context = rag_service.retrieve_context(
                query,
                user_profile={"age": 30, "gender": "女"},
                domain_context="nutrition"
            )
            
            print(f"   📄 检索到 {len(context.retrieved_docs)} 个相关文档片段")
            
            # 显示最相关的文档片段
            if context.retrieved_docs:
                top_doc = context.retrieved_docs[0]
                preview = top_doc.page_content[:150].replace('\n', ' ')
                print(f"   💡 最相关内容预览: {preview}...")
                print(f"   📂 来源: {top_doc.metadata.get('source', '未知')}")
            
            # 生成回答（使用 RAG）
            try:
                response = rag_service.generate_with_context(context)
                print(f"   🤖 AI 回答预览: {response[:200]}...")
            except Exception as e:
                print(f"   ⚠️  生成回答失败: {e}")
        
        print("\n✅ LangChain RAG 测试完成")
        return True
        
    except Exception as e:
        print(f"❌ LangChain RAG 测试失败: {e}")
        return False


def test_llamaindex_rag():
    """测试 LlamaIndex RAG"""
    if not LLAMA_INDEX_AVAILABLE:
        print("⚠️  LlamaIndex 未安装，跳过测试")
        return True
    
    print("\n🔍 测试 LlamaIndex RAG 系统...")
    
    try:
        # 使用 LlamaIndex RAG 服务
        rag_service = create_enhanced_rag_service(use_llama_index=True)
        
        # 测试查询
        queries = [
            "膳食纤维有什么作用？",
            "如何预防糖尿病？",
            "孕妇需要注意哪些营养？"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n📝 查询 {i}: {query}")
            
            try:
                # LlamaIndex 查询
                result = rag_service.query(query, top_k=3)
                print(f"   🤖 查询结果: {str(result)[:300]}...")
            except Exception as e:
                print(f"   ⚠️  查询失败: {e}")
        
        print("\n✅ LlamaIndex RAG 测试完成")
        return True
        
    except Exception as e:
        print(f"❌ LlamaIndex RAG 测试失败: {e}")
        return False


def test_knowledge_coverage():
    """测试知识覆盖范围"""
    print("\n🔍 测试知识库覆盖范围...")
    
    try:
        rag_service = create_enhanced_rag_service(use_llama_index=False)
        
        # 测试不同主题的覆盖
        topics = {
            "中国膳食指南": "中国居民膳食指南的核心原则",
            "美国膳食指南": "美国膳食指南2015-2020的建议",
            "膳食宝塔": "中国居民平衡膳食宝塔的结构",
            "慢性病预防": "如何通过饮食预防慢性病",
            "营养素需求": "人体对各种营养素的需求"
        }
        
        coverage_results = {}
        
        for topic, query in topics.items():
            context = rag_service.retrieve_context(query)
            num_docs = len(context.retrieved_docs)
            coverage_results[topic] = num_docs
            
            if num_docs > 0:
                print(f"   ✅ {topic}: 找到 {num_docs} 个相关文档")
            else:
                print(f"   ❌ {topic}: 未找到相关文档")
        
        total_coverage = sum(1 for count in coverage_results.values() if count > 0)
        print(f"\n📊 知识覆盖率: {total_coverage}/{len(topics)} ({total_coverage/len(topics)*100:.1f}%)")
        
        return total_coverage == len(topics)
        
    except Exception as e:
        print(f"❌ 知识覆盖测试失败: {e}")
        return False


def test_document_quality():
    """测试文档质量"""
    print("\n🔍 测试文档质量...")
    
    try:
        rag_service = create_enhanced_rag_service(use_llama_index=False)
        
        # 测试文档内容质量
        test_query = "什么是健康饮食"
        context = rag_service.retrieve_context(test_query)
        
        if not context.retrieved_docs:
            print("❌ 未检索到任何文档")
            return False
        
        # 分析文档质量
        doc_stats = {
            "total_docs": len(context.retrieved_docs),
            "avg_length": sum(len(doc.page_content) for doc in context.retrieved_docs) / len(context.retrieved_docs),
            "sources": set(doc.metadata.get('source', '未知') for doc in context.retrieved_docs)
        }
        
        print(f"   📄 检索文档数: {doc_stats['total_docs']}")
        print(f"   📏 平均长度: {doc_stats['avg_length']:.0f} 字符")
        print(f"   📂 文档来源: {len(doc_stats['sources'])} 个不同来源")
        
        # 显示来源
        for source in sorted(doc_stats['sources']):
            print(f"      - {source}")
        
        # 质量判断
        quality_good = (
            doc_stats['total_docs'] >= 3 and 
            doc_stats['avg_length'] >= 100 and 
            len(doc_stats['sources']) >= 2
        )
        
        if quality_good:
            print("   ✅ 文档质量良好")
        else:
            print("   ⚠️  文档质量可能需要改进")
        
        return quality_good
        
    except Exception as e:
        print(f"❌ 文档质量测试失败: {e}")
        return False


def main():
    """主测试流程"""
    print("🚀 开始 RAG 知识库测试\n")
    print("=" * 60)
    
    test_results = []
    
    # 测试 LangChain RAG
    test_results.append(("LangChain RAG", test_langchain_rag()))
    
    # 测试 LlamaIndex RAG
    test_results.append(("LlamaIndex RAG", test_llamaindex_rag()))
    
    # 测试知识覆盖
    test_results.append(("知识覆盖", test_knowledge_coverage()))
    
    # 测试文档质量
    test_results.append(("文档质量", test_document_quality()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总:")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有测试通过！RAG 知识库系统运行正常")
    else:
        print("⚠️  部分测试失败，请检查相关组件")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)