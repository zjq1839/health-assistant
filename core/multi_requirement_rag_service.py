"""多需求RAG服务

扩展标准RAG服务，支持处理包含多个需求的复杂查询
为每个需求单独检索上下文并生成响应，最后综合成完整回答
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from core.rag_service import RAGService, RAGContext
from core.multi_requirement_parser import (
    MultiRequirementParser, MultiRequirementParseResult, 
    ParsedRequirement, RequirementType
)
from utils.logger import logger


@dataclass
class RequirementRAGContext:
    """单个需求的RAG上下文"""
    requirement: ParsedRequirement
    rag_context: RAGContext
    response: str = ""
    processing_time: float = 0.0


@dataclass
class MultiRequirementRAGResult:
    """多需求RAG处理结果"""
    original_query: str
    parse_result: MultiRequirementParseResult
    requirement_contexts: List[RequirementRAGContext]
    final_response: str
    total_processing_time: float
    success_rate: float  # 成功处理的需求比例


class MultiRequirementRAGService:
    """多需求RAG服务
    
    功能：
    1. 解析包含多个需求的复杂查询
    2. 为每个需求单独进行RAG检索和生成
    3. 综合所有结果生成完整响应
    4. 支持需求间的依赖关系处理
    """
    
    def __init__(self, rag_service: RAGService, requirement_parser: MultiRequirementParser):
        self.rag_service = rag_service
        self.parser = requirement_parser
        
        # 需求类型到RAG策略的映射
        self.requirement_strategies = {
            RequirementType.QUERY: self._handle_query_requirement,
            RequirementType.RECORD: self._handle_record_requirement,
            RequirementType.ADVICE: self._handle_advice_requirement,
            RequirementType.ANALYSIS: self._handle_analysis_requirement,
            RequirementType.GENERATE: self._handle_generate_requirement,
            RequirementType.CALCULATE: self._handle_calculate_requirement,
        }
    
    def process_multi_requirement_query(self, 
                                      query: str, 
                                      user_profile: Dict[str, Any] = None,
                                      context: str = "") -> MultiRequirementRAGResult:
        """处理多需求查询"""
        import time
        start_time = time.time()
        
        print(f"\n🔍 [多需求RAG] 开始处理复杂查询: {query}")
        print(f"📊 [多需求RAG] 用户画像: {user_profile}")
        
        # 1. 解析多需求
        print(f"\n📝 [多需求RAG] 步骤1: 需求解析")
        parse_result = self.parser.parse(query, context)
        
        print(f"✅ [多需求RAG] 解析完成:")
        print(f"   - 识别需求数: {len(parse_result.requirements)}")
        print(f"   - 复杂度: {parse_result.complexity}")
        print(f"   - 解析方法: {parse_result.parsing_method}")
        print(f"   - 总置信度: {parse_result.total_confidence:.2f}")
        
        for i, req in enumerate(parse_result.requirements, 1):
            print(f"   需求{i}: [{req.type.value}] {req.description} (置信度: {req.confidence:.2f})")
        
        # 2. 按执行顺序处理每个需求
        print(f"\n🔄 [多需求RAG] 步骤2: 按序处理需求")
        print(f"📋 [多需求RAG] 执行顺序: {' -> '.join(parse_result.execution_order)}")
        
        requirement_contexts = []
        processed_results = {}  # 存储已处理需求的结果，供后续需求参考
        
        for req_id in parse_result.execution_order:
            requirement = next((req for req in parse_result.requirements if req.id == req_id), None)
            if not requirement:
                continue
            
            print(f"\n🎯 [多需求RAG] 处理需求 {req_id}: [{requirement.type.value}] {requirement.description}")
            
            try:
                req_context = self._process_single_requirement(
                    requirement, user_profile, processed_results
                )
                requirement_contexts.append(req_context)
                processed_results[req_id] = req_context
                
                print(f"✅ [多需求RAG] 需求 {req_id} 处理完成")
                print(f"   - 检索文档数: {len(req_context.rag_context.retrieved_docs)}")
                print(f"   - 响应长度: {len(req_context.response)} 字符")
                print(f"   - 处理时间: {req_context.processing_time:.2f}秒")
                
            except Exception as e:
                logger.error(f"处理需求 {req_id} 失败: {e}")
                print(f"❌ [多需求RAG] 需求 {req_id} 处理失败: {e}")
                
                # 创建错误的上下文
                error_context = RequirementRAGContext(
                    requirement=requirement,
                    rag_context=RAGContext(
                        query=requirement.description,
                        enhanced_query=requirement.description,
                        retrieved_docs=[],
                        domain_context="",
                        user_profile=user_profile or {},
                        retrieval_method="error"
                    ),
                    response=f"抱歉，处理这个需求时遇到了问题：{str(e)}",
                    processing_time=0.0
                )
                requirement_contexts.append(error_context)
                processed_results[req_id] = error_context
        
        # 3. 综合生成最终响应
        print(f"\n🔄 [多需求RAG] 步骤3: 综合生成最终响应")
        final_response = self._synthesize_final_response(
            query, parse_result, requirement_contexts, user_profile
        )
        
        # 4. 计算统计信息
        total_time = time.time() - start_time
        success_count = sum(1 for ctx in requirement_contexts if "抱歉" not in ctx.response)
        success_rate = success_count / len(requirement_contexts) if requirement_contexts else 0
        
        print(f"\n🎉 [多需求RAG] 处理完成!")
        print(f"📊 [多需求RAG] 统计信息:")
        print(f"   - 总处理时间: {total_time:.2f}秒")
        print(f"   - 成功率: {success_rate:.2%} ({success_count}/{len(requirement_contexts)})")
        print(f"   - 最终响应长度: {len(final_response)} 字符")
        
        return MultiRequirementRAGResult(
            original_query=query,
            parse_result=parse_result,
            requirement_contexts=requirement_contexts,
            final_response=final_response,
            total_processing_time=total_time,
            success_rate=success_rate
        )
    
    def _process_single_requirement(self, 
                                   requirement: ParsedRequirement,
                                   user_profile: Dict[str, Any],
                                   processed_results: Dict[str, RequirementRAGContext]) -> RequirementRAGContext:
        """处理单个需求"""
        import time
        start_time = time.time()
        
        print(f"   🔍 [需求处理] 开始处理: {requirement.description}")
        
        # 选择处理策略
        strategy = self.requirement_strategies.get(
            requirement.type, 
            self._handle_default_requirement
        )
        
        # 执行策略
        rag_context, response = strategy(requirement, user_profile, processed_results)
        
        processing_time = time.time() - start_time
        
        print(f"   ⏱️ [需求处理] 处理耗时: {processing_time:.2f}秒")
        
        return RequirementRAGContext(
            requirement=requirement,
            rag_context=rag_context,
            response=response,
            processing_time=processing_time
        )
    
    def _handle_query_requirement(self, 
                                requirement: ParsedRequirement,
                                user_profile: Dict[str, Any],
                                processed_results: Dict) -> tuple[RAGContext, str]:
        """处理查询类需求"""
        print(f"     📊 [查询需求] 检索相关数据")
        
        # 构建查询增强信息
        domain_context = "data_query"
        
        # 检索上下文
        rag_context = self.rag_service.retrieve_context(
            requirement.description,
            user_profile=user_profile,
            domain_context=domain_context
        )
        
        # 生成响应
        response = self.rag_service.generate_with_context(rag_context)
        
        return rag_context, response
    
    def _handle_record_requirement(self, 
                                 requirement: ParsedRequirement,
                                 user_profile: Dict[str, Any],
                                 processed_results: Dict) -> tuple[RAGContext, str]:
        """处理记录类需求"""
        print(f"     📝 [记录需求] 提供记录指导")
        
        # 记录类需求主要提供指导和确认信息
        domain_context = "record_guidance"
        
        # 检索相关的记录指导信息
        rag_context = self.rag_service.retrieve_context(
            f"如何记录 {requirement.description}",
            user_profile=user_profile,
            domain_context=domain_context
        )
        
        # 生成记录指导响应
        enhanced_prompt = f"""基于用户需求：{requirement.description}
        
请提供：
1. 记录确认（如果用户已经描述了具体内容）
2. 记录建议（如果需要更多信息）
3. 相关的健康提示

用户画像：{user_profile}
"""
        
        # 使用增强的提示生成响应
        rag_context.enhanced_query = enhanced_prompt
        response = self.rag_service.generate_with_context(rag_context)
        
        return rag_context, response
    
    def _handle_advice_requirement(self, 
                                 requirement: ParsedRequirement,
                                 user_profile: Dict[str, Any],
                                 processed_results: Dict) -> tuple[RAGContext, str]:
        """处理建议类需求"""
        print(f"     💡 [建议需求] 生成个性化建议")
        
        domain_context = "health_advice"
        
        # 检索建议相关的知识
        rag_context = self.rag_service.retrieve_context(
            requirement.description,
            user_profile=user_profile,
            domain_context=domain_context
        )
        
        # 生成个性化建议
        response = self.rag_service.generate_with_context(rag_context)
        
        return rag_context, response
    
    def _handle_analysis_requirement(self, 
                                   requirement: ParsedRequirement,
                                   user_profile: Dict[str, Any],
                                   processed_results: Dict) -> tuple[RAGContext, str]:
        """处理分析类需求"""
        print(f"     📈 [分析需求] 执行数据分析")
        
        domain_context = "data_analysis"
        
        # 检索分析方法和模板
        rag_context = self.rag_service.retrieve_context(
            f"分析方法 {requirement.description}",
            user_profile=user_profile,
            domain_context=domain_context
        )
        
        # 考虑依赖的查询结果
        dependency_data = self._collect_dependency_data(requirement, processed_results)
        
        if dependency_data:
            enhanced_context = f"""
分析需求：{requirement.description}

相关数据：
{dependency_data}

请基于以上数据进行分析。
"""
            rag_context.enhanced_query = enhanced_context
        
        response = self.rag_service.generate_with_context(rag_context)
        
        return rag_context, response
    
    def _handle_generate_requirement(self, 
                                   requirement: ParsedRequirement,
                                   user_profile: Dict[str, Any],
                                   processed_results: Dict) -> tuple[RAGContext, str]:
        """处理生成类需求"""
        print(f"     📄 [生成需求] 创建内容")
        
        domain_context = "content_generation"
        
        # 检索生成模板和指导
        rag_context = self.rag_service.retrieve_context(
            requirement.description,
            user_profile=user_profile,
            domain_context=domain_context
        )
        
        # 收集依赖数据用于生成
        dependency_data = self._collect_dependency_data(requirement, processed_results)
        
        if dependency_data:
            enhanced_context = f"""
生成需求：{requirement.description}

基础数据：
{dependency_data}

用户画像：{user_profile}

请基于以上信息生成所需内容。
"""
            rag_context.enhanced_query = enhanced_context
        
        response = self.rag_service.generate_with_context(rag_context)
        
        return rag_context, response
    
    def _handle_calculate_requirement(self, 
                                    requirement: ParsedRequirement,
                                    user_profile: Dict[str, Any],
                                    processed_results: Dict) -> tuple[RAGContext, str]:
        """处理计算类需求"""
        print(f"     🧮 [计算需求] 执行数值计算")
        
        domain_context = "calculation"
        
        # 检索计算公式和方法
        rag_context = self.rag_service.retrieve_context(
            f"计算公式 {requirement.description}",
            user_profile=user_profile,
            domain_context=domain_context
        )
        
        # 生成计算响应
        response = self.rag_service.generate_with_context(rag_context)
        
        return rag_context, response
    
    def _handle_default_requirement(self, 
                                  requirement: ParsedRequirement,
                                  user_profile: Dict[str, Any],
                                  processed_results: Dict) -> tuple[RAGContext, str]:
        """处理默认需求"""
        print(f"     ❓ [默认需求] 通用处理")
        
        # 使用通用RAG处理
        rag_context = self.rag_service.retrieve_context(
            requirement.description,
            user_profile=user_profile
        )
        
        response = self.rag_service.generate_with_context(rag_context)
        
        return rag_context, response
    
    def _collect_dependency_data(self, 
                               requirement: ParsedRequirement,
                               processed_results: Dict[str, RequirementRAGContext]) -> str:
        """收集依赖需求的数据"""
        dependency_data = []
        
        for dep_id in requirement.dependencies:
            if dep_id in processed_results:
                dep_context = processed_results[dep_id]
                dependency_data.append(f"{dep_context.requirement.description}: {dep_context.response}")
        
        return "\n".join(dependency_data)
    
    def _synthesize_final_response(self, 
                                 original_query: str,
                                 parse_result: MultiRequirementParseResult,
                                 requirement_contexts: List[RequirementRAGContext],
                                 user_profile: Dict[str, Any]) -> str:
        """综合生成最终响应"""
        print(f"   🔄 [响应综合] 整合所有需求的回答")
        
        if not requirement_contexts:
            return "抱歉，无法处理您的查询。"
        
        # 构建综合提示
        synthesis_prompt = f"""用户原始查询：{original_query}

您已经分别处理了以下需求：

"""
        
        for i, ctx in enumerate(requirement_contexts, 1):
            synthesis_prompt += f"""
需求{i}：{ctx.requirement.description}
回答：{ctx.response}

"""
        
        synthesis_prompt += f"""
请将以上所有回答整合成一个连贯、完整的响应，确保：
1. 回答了用户的所有问题
2. 信息组织清晰，逻辑连贯
3. 语言自然流畅
4. 考虑用户画像：{user_profile}

请直接给出最终的综合回答："""
        
        print(f"   🤖 [响应综合] 调用LLM进行最终整合")
        
        try:
            # 使用RAG服务的LLM进行综合
            final_response = self.rag_service.llm_service.generate_response(synthesis_prompt)
            
            print(f"   ✅ [响应综合] 综合完成，最终响应长度: {len(final_response)} 字符")
            
            return final_response
            
        except Exception as e:
            logger.error(f"综合响应失败: {e}")
            print(f"   ❌ [响应综合] 综合失败，使用简单拼接: {e}")
            
            # 降级：简单拼接各部分响应
            combined_response = f"关于您的查询「{original_query}」，我为您提供以下回答：\n\n"
            
            for i, ctx in enumerate(requirement_contexts, 1):
                combined_response += f"{i}. {ctx.requirement.description}\n"
                combined_response += f"{ctx.response}\n\n"
            
            return combined_response


# 便捷函数
def create_multi_requirement_rag_service() -> MultiRequirementRAGService:
    """创建多需求RAG服务实例"""
    from core.rag_service import create_rag_service
    from core.multi_requirement_parser import create_multi_requirement_parser
    
    rag_service = create_rag_service()
    parser = create_multi_requirement_parser()
    
    return MultiRequirementRAGService(rag_service, parser)


# 使用示例
if __name__ == "__main__":
    # 创建服务
    multi_rag_service = create_multi_requirement_rag_service()
    
    # 加载知识库
    multi_rag_service.rag_service.load_knowledge_base("/home/zjq/document/langchain_learn/rag_knowledge_base")
    
    # 测试复杂查询
    test_queries = [
        "查询我昨天的饮食记录，然后给我一些今天的健康建议",
        "记录我今天早餐吃了鸡蛋和牛奶，并分析我这周的营养摄入情况",
        "生成我的健康报告，同时推荐适合我的运动计划",
    ]
    
    user_profile = {
        "age": 25,
        "gender": "女",
        "health_goal": "减肥",
        "activity_level": "中等"
    }
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"测试查询: {query}")
        print(f"{'='*60}")
        
        result = multi_rag_service.process_multi_requirement_query(
            query, user_profile
        )
        
        print(f"\n📋 最终响应:")
        print(f"{result.final_response}")
        
        print(f"\n📊 处理统计:")
        print(f"成功率: {result.success_rate:.2%}")
        print(f"总耗时: {result.total_processing_time:.2f}秒")