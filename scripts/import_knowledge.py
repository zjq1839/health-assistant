#!/usr/bin/env python3
"""
知识库导入脚本

用途：
1. 将指定目录的文档导入到 RAG 知识库
2. 支持 LangChain 和 LlamaIndex 两种方式
3. 支持多种文档格式：PDF、TXT、MD、JSON

使用方法：
python scripts/import_knowledge.py --source /path/to/documents --method langchain
python scripts/import_knowledge.py --source /path/to/documents --method llamaindex
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

from core.enhanced_rag_service import (
    import_knowledge_from_directory, 
    create_enhanced_rag_service,
    EnhancedDocumentLoader,
    LLAMA_INDEX_AVAILABLE
)
from utils.logger import logger
from config import config as cfg


def check_dependencies():
    """检查依赖项"""
    issues = []
    
    # 检查基本依赖
    try:
        import langchain
        import faiss
    except ImportError as e:
        issues.append(f"Missing LangChain dependencies: {e}")
    
    # 检查 PDF 支持
    try:
        import pypdf
    except ImportError:
        issues.append("Missing PyPDF2 for PDF support. Install: pip install pypdf")
    
    # 检查 LlamaIndex（可选）
    if not LLAMA_INDEX_AVAILABLE:
        issues.append("LlamaIndex not available. Install: pip install llama-index")
    
    return issues


def preview_documents(source_dir: str):
    """预览源目录中的文档"""
    loader = EnhancedDocumentLoader()
    source_path = Path(source_dir)
    
    if not source_path.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return
    
    print(f"\n📁 扫描目录: {source_dir}")
    print("=" * 50)
    
    file_types = {}
    total_files = 0
    
    for file_path in source_path.rglob("*"):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext in loader.supported_extensions:
                file_types[ext] = file_types.get(ext, 0) + 1
                total_files += 1
                print(f"✓ {file_path.name}")
            else:
                print(f"⚠ {file_path.name} (不支持的格式)")
    
    print("\n📊 文件统计:")
    for ext, count in file_types.items():
        print(f"  {ext}: {count} 个文件")
    
    print(f"\n总计: {total_files} 个支持的文件")
    return total_files


def import_with_langchain(source_dir: str):
    """使用 LangChain 方式导入"""
    logger.info("使用 LangChain + FAISS 方式导入知识库...")
    
    try:
        rag_service = create_enhanced_rag_service(use_llama_index=False)
        success = rag_service.import_from_directory(source_dir)
        
        if success:
            print("✅ LangChain 导入成功！")
            print(f"📚 知识库位置: {cfg.knowledge_base.path}")
            print(f"🔍 向量存储: {cfg.knowledge_base.path}/vector_store")
        else:
            print("❌ LangChain 导入失败")
            
        return success
        
    except Exception as e:
        logger.error(f"LangChain 导入失败: {e}")
        print(f"❌ 错误: {e}")
        return False


def import_with_llamaindex(source_dir: str):
    """使用 LlamaIndex 方式导入"""
    if not LLAMA_INDEX_AVAILABLE:
        print("❌ LlamaIndex 未安装，请运行: pip install llama-index")
        return False
    
    logger.info("使用 LlamaIndex + FAISS 方式导入知识库...")
    
    try:
        rag_service = create_enhanced_rag_service(use_llama_index=True)
        rag_service.load_documents_from_directory(source_dir)
        
        print("✅ LlamaIndex 导入成功！")
        print(f"📚 知识库位置: {cfg.knowledge_base.path}")
        print(f"🔍 索引存储: {cfg.knowledge_base.path}/llama_index_storage")
        return True
        
    except Exception as e:
        logger.error(f"LlamaIndex 导入失败: {e}")
        print(f"❌ 错误: {e}")
        return False


def test_query(method: str):
    """测试查询功能"""
    print(f"\n🧪 测试 {method} 查询功能...")
    
    try:
        use_llama = (method == "llamaindex")
        rag_service = create_enhanced_rag_service(use_llama_index=use_llama)
        
        test_query = "什么是健康的饮食习惯？"
        
        if use_llama:
            result = rag_service.query(test_query)
            print(f"✅ 查询测试成功")
            print(f"🔍 查询: {test_query}")
            print(f"📝 结果: {str(result)[:200]}...")
        else:
            context = rag_service.retrieve_context(test_query)
            num_chunks = len(context.retrieved_docs)
            print(f"✅ 查询测试成功")
            print(f"🔍 查询: {test_query}")
            print(f"📄 检索到 {num_chunks} 个相关片段")
        
    except Exception as e:
        print(f"❌ 查询测试失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="RAG 知识库导入工具")
    parser.add_argument(
        "--source", 
        required=True, 
        help="源文档目录路径"
    )
    parser.add_argument(
        "--method", 
        choices=["langchain", "llamaindex"], 
        default="langchain",
        help="导入方法: langchain 或 llamaindex (默认: langchain)"
    )
    parser.add_argument(
        "--preview", 
        action="store_true",
        help="仅预览文档，不执行导入"
    )
    parser.add_argument(
        "--test", 
        action="store_true",
        help="导入后测试查询功能"
    )
    parser.add_argument(
        "--check-deps", 
        action="store_true",
        help="检查依赖项"
    )
    
    args = parser.parse_args()
    
    # 检查依赖项
    if args.check_deps:
        print("🔍 检查依赖项...")
        issues = check_dependencies()
        if issues:
            print("⚠️  发现问题:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ 所有依赖项正常")
        return
    
    # 预览文档
    total_files = preview_documents(args.source)
    if total_files == 0:
        print("❌ 没有找到支持的文档文件")
        return
    
    if args.preview:
        return
    
    # 确认导入
    print(f"\n📥 准备使用 {args.method} 方式导入 {total_files} 个文件")
    confirm = input("是否继续？(y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("操作已取消")
        return
    
    # 执行导入
    if args.method == "langchain":
        success = import_with_langchain(args.source)
    else:
        success = import_with_llamaindex(args.source)
    
    # 测试查询
    if success and args.test:
        test_query(args.method)
    
    if success:
        print(f"\n🎉 知识库导入完成！")
        print(f"💡 现在可以使用 RAG 功能查询这些知识了")


if __name__ == "__main__":
    main()