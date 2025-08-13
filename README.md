# 健康助手 V2

一个基于“轻量级规划 + 统一 Agent 协议 + 依赖注入”的命令行健康助手，支持记录餐食、记录运动、查询数据、生成报告与获取建议。项目默认使用 Ollama 本地大模型（ChatOllama），数据持久化采用 SQLite。

## 功能特性
- 🎯 统一意图与 Agent 映射：record_meal / record_exercise / query / generate_report / advice
- 🧭 三级决策：规则 → 小模型 → 大模型 的轻量级规划器（LightweightPlanner）
- 🤝 统一协议：所有 Agent 实现 AgentProtocol，复用 BaseAgent 的通用能力
- 🧩 依赖注入：ServiceContainer 统一管理 Database/LLM/Nutrition 等服务
- 🗃️ 数据库：SQLite 自动建表，记录饮食与运动明细
- 🔤 Token 统计：自动收集并在 CLI 中展示本轮调用的 Token 用量

## 项目结构
```
langchain_learn/
├── main_v2.py                  # 程序入口（CLI）
├── agents/                     # 具体智能体（V2）
│   ├── dietary_agent_v2.py     # 记录餐食
│   ├── exercise_agent_v2.py    # 记录运动
│   ├── query_agent_v2.py       # 查询数据
│   ├── report_agent_v2.py      # 生成报告
│   ├── advice_agent_v2.py      # 建议与问答
│   ├── general_agent_v2.py     # 兜底通用
│   └── config.py               # LLM 工厂（ChatOllama）
├── core/
│   ├── agent_protocol.py       # 协议、BaseAgent、AgentFactory、SQLite 实现
│   ├── enhanced_state.py       # 会话状态（EnhancedState）
│   ├── lightweight_planner.py  # 轻量级规划器（含规则/小模型）
│   └── service_container.py    # 依赖注入容器（setup_container / get_container）
├── utils/
│   ├── user_experience.py      # 示例与引导
│   └── logger.py               # 结构化日志
├── config.py                   # Pydantic 配置（支持 .env / config.yml）
├── config.yml                  # 可选 YAML 配置
├── requirements.txt
└── tests/                      # 单测（数据库、Agent 等）
```

## 安装与准备
1) 安装依赖
```bash
pip install -r requirements.txt
```
2) 安装并启动 Ollama（本地大模型）
- 安装见 Ollama 官方文档
- 预拉取默认模型（以 qwen3:32b 为例）：
```bash
ollama pull qwen3:32b
```
3) 配置环境变量（可选，亦可改 config.yml）
- 复制并编辑 .env（如不存在可新建）
```bash
cp .env.example .env
```
- 常用变量（与 config.py 对应）：
```bash
# LLM
LLM_MODEL=qwen3:32b
LLM_LITE_MODEL=qwen3:32b
LLM_TEMPERATURE=0.0

# 数据库
DB_PATH=./diet.db

# 对话/缓存
MAX_MESSAGES=20
ENABLE_CACHE=true
CACHE_SIZE=100
```

## 运行
```bash
python main_v2.py
```
启动后可使用命令：
- help：查看使用指南与示例
- stats：查看统计信息（对话轮次、意图分布、Agent 使用、Token 用量、数据条目）
- health：执行健康检查（DB/LLM/Planner/Agent 工厂）
- reset：重置会话
- quit：退出

示例交互：
- “我早餐吃了鸡蛋和牛奶”（record_meal）
- “我跑步30分钟”（record_exercise）
- “查询我昨天的饮食记录”（query）
- “生成本周健康报告”（generate_report）
- “推荐一些健康食谱”（advice）

## 使用详解

### 命令行使用示例

#### 启动与健康检查
```bash
$ python main_v2.py
🏥 健康助手 V2 启动成功！
💡 输入 'help' 查看使用指南，输入 'quit' 退出
📊 输入 'stats' 查看系统统计信息
--------------------------------------------------

🔍 系统健康检查中...
  ✅ 数据库连接正常
  ✅ LLM服务正常
  ✅ 意图规划器正常
  ✅ Agent工厂正常
✅ 健康检查完成

👤 您：health
🔍 系统健康检查中...
  ✅ 数据库连接正常
  ✅ LLM服务正常
  ✅ 意图规划器正常
  ✅ Agent工厂正常
✅ 健康检查完成
```

#### 统计信息展示
```bash
👤 您：stats
📊 系统统计信息：
  💬 对话轮次：5
  🎯 意图分布：
    - record_meal: 2
    - record_exercise: 1
    - query: 1
    - generate_report: 1
  🤖 Agent使用情况：
    - dietary: 2
    - exercise: 1
    - query: 1
    - report: 1
  🍽️ 饮食记录数：8
  🏃 运动记录数：3
  🔤 当前会话Token使用：
    - 输入Token: 1245
    - 输出Token: 892
    - 总Token: 2137
```

#### 实际对话示例
```bash
👤 您：我早餐吃了鸡蛋和牛奶
🤖 助手：✅ 餐食记录添加成功
📅 日期：2024-01-15
🍽️ 餐食类型：早餐
📝 描述：鸡蛋和牛奶

👤 您：我跑步30分钟
🤖 助手：✅ 运动记录添加成功
📅 日期：2024-01-15
🏃 运动类型：跑步
⏱️ 时长：30 分钟
🔥 消耗热量：约 240 卡路里

👤 您：查询我今天的饮食记录
🤖 助手：📊 找到 1 条记录：
--- 记录 1 ---
📅 日期：2024-01-15
🍽️ 餐食类型：早餐
📝 描述：鸡蛋和牛奶
```

### 配置项说明

#### 环境变量 & config.py 对应表

| 环境变量 | config.py 字段 | 默认值 | 说明 |
|---------|---------------|-------|------|
| **LLM 配置** |
| `LLM_MODEL` | `llm.model` | `qwen3:32b` | 主要 LLM 模型名称 |
| `LLM_LITE_MODEL` | `llm.lite_model` | `qwen3:32b` | 轻量级模型（意图分类用） |
| `LLM_TEMPERATURE` | `llm.temperature` | `0.0` | 生成温度（0.0-1.0） |
| **数据库配置** |
| `DB_PATH` | `database.path` | `/home/zjq/document/langchain_learn/diet.db` | SQLite 数据库文件路径 |
| **知识库配置** |
| `KNOWLEDGE_BASE_PATH` | `knowledge_base.path` | `/home/zjq/document/langchain_learn/rag_knowledge_base` | RAG 知识库路径 |
| `EMBEDDING_MODEL` | `knowledge_base.embedding_model` | `nn200433/text2vec-bge-large-chinese:latest` | 向量化模型 |
| `VECTOR_SEARCH_K` | `vector_search.k` | `2` | 向量检索返回结果数 |
| **对话配置** |
| `MAX_MESSAGES` | `dialogue.max_messages` | `20` | 最大历史消息数 |
| `IMMEDIATE_CONTEXT_SIZE` | `dialogue.immediate_context_size` | `3` | 即时上下文大小 |
| `ENABLE_INTELLIGENT_TRUNCATION` | `dialogue.enable_intelligent_truncation` | `true` | 智能截断开关 |
| `ENABLE_CONTEXT_COMPRESSION` | `dialogue.enable_context_compression` | `false` | 上下文压缩开关 |
| **缓存配置** |
| `ENABLE_CACHE` | `cache.enable` | `true` | 启用缓存 |
| `CACHE_SIZE` | `cache.size` | `100` | 缓存大小 |
| `CACHE_TYPE` | `cache.type` | `sqlite` | 缓存类型 |
| `INTENT_CACHE_SIZE` | `cache.intent_cache_size` | `200` | 意图缓存大小 |
| `INTENT_CACHE_TTL_HOURS` | `cache.intent_cache_ttl_hours` | `24` | 意图缓存过期时间（小时） |
| **意图识别配置** |
| `ENABLE_ENHANCED_RECOGNITION` | `intent.enable_enhanced_recognition` | `true` | 增强意图识别 |
| `INTENT_CONFIDENCE_THRESHOLD` | `intent.confidence_threshold` | `0.7` | 意图识别置信度阈值 |
| `ENABLE_CONTEXT_ENHANCEMENT` | `intent.enable_context_enhancement` | `true` | 上下文增强 |
| `KEYWORD_WEIGHT` | `intent.keyword_weight` | `0.6` | 关键词权重 |
| `LLM_WEIGHT` | `intent.llm_weight` | `0.4` | LLM 权重 |
| `KEYWORD_DIRECT_RETURN_THRESHOLD` | `intent.keyword_direct_return_threshold` | `0.95` | 关键词直接返回阈值 |
| `INTENT_LLM_TEMPERATURE` | `intent.llm_temperature` | `0.1` | 意图分类 LLM 温度 |
| `INTENT_LLM_TIMEOUT_SECONDS` | `intent.llm_timeout_seconds` | `5` | 意图分类超时（秒） |
| `LLM_MAX_RESPONSE_LENGTH` | `intent.llm_max_response_length` | `1024` | LLM 最大响应长度 |
| **日志配置** |
| `LOG_LEVEL` | `logging.level` | `INFO` | 日志级别 |
| `LOG_FILE` | `logging.file` | `app.log` | 日志文件名 |

## 测试
```bash
pytest -q
```

## 开发者指南

### 如何新增一个 Agent

#### 步骤 1：定义新意图类型
在 <mcfile name="enhanced_state.py" path="core/enhanced_state.py"></mcfile> 中新增意图枚举：
```python
class IntentType(Enum):
    # 现有意图...
    NEW_FEATURE = "new_feature"  # 新增意图
```

#### 步骤 2：创建 Agent 实现
在 `agents/` 目录下创建新文件（如 `new_feature_agent_v2.py`）：
```python
from core.agent_protocol import BaseAgent, AgentResponse, AgentResult
from core.enhanced_state import IntentType

class NewFeatureAgentV2(BaseAgent):
    def __init__(self, llm_service):
        super().__init__(name="new_feature")
        self.llm_service = llm_service
    
    def can_handle(self, intent: IntentType) -> bool:
        return intent == IntentType.NEW_FEATURE
    
    def run(self, state: dict) -> AgentResponse:
        # 实现具体逻辑
        user_input = state['messages'][-1]['content']
        
        # 处理逻辑...
        
        return self._create_success_response(
            message="新功能处理完成",
            evidence={"processed": user_input}
        )
```

#### 步骤 3：注册到 AgentFactory
在 <mcfile name="agent_protocol.py" path="core/agent_protocol.py"></mcfile> 的 <mcsymbol name="create_agent" filename="agent_protocol.py" path="core/agent_protocol.py" startline="239" type="function"></mcsymbol> 方法中添加：
```python
elif agent_name == "new_feature":
    from agents.new_feature_agent_v2 import NewFeatureAgentV2
    agent = NewFeatureAgentV2(
        llm_service=self.container.get(LLMService)
    )
```

#### 步骤 4：更新意图映射
在 <mcfile name="common_parsers.py" path="utils/common_parsers.py"></mcfile> 的 <mcsymbol name="intent_to_agent_mapping" filename="common_parsers.py" path="utils/common_parsers.py" startline="230" type="function"></mcsymbol> 中添加：
```python
mapping = {
    # 现有映射...
    IntentType.NEW_FEATURE: "new_feature",
}
```

#### 步骤 5：更新用户示例
在 <mcfile name="user_experience.py" path="utils/user_experience.py"></mcfile> 中添加示例：
```python
def get_examples_by_intent(self, intent: str) -> List[str]:
    examples = {
        # 现有示例...
        "new_feature": [
            "触发新功能的示例输入",
            "另一个示例"
        ]
    }
```

### 如何接入新的 LLM 服务

#### 步骤 1：实现 LLMService 接口
创建新的 LLM 服务实现（如在 <mcfile name="service_container.py" path="core/service_container.py"></mcfile> 中）：
```python
class CustomLLMService(LLMService):
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.token_usage_callback = TokenUsageCallback()
        # 初始化您的 LLM 客户端
    
    def extract_entities(self, text: str, schema: Dict) -> Dict:
        # 实现实体提取逻辑
        pass
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        # 实现响应生成逻辑
        pass
    
    def classify_intent(self, text: str, context: str = "") -> Dict:
        # 实现意图分类逻辑
        pass
    
    def invoke(self, prompt: str) -> Any:
        # 实现直接调用逻辑
        pass
```

#### 步骤 2：注册到服务容器
在 <mcsymbol name="ConfigurableServiceContainer" filename="service_container.py" path="core/service_container.py" startline="174" type="class"></mcsymbol> 的 `_setup_services` 方法中添加：
```python
def _setup_services(self):
    # 现有设置...
    
    # LLM服务
    llm_config = self.config.get('llm', {})
    if llm_config.get('type') == 'custom':
        self.register_factory(
            LLMService,
            lambda: CustomLLMService(
                api_key=llm_config.get('api_key'),
                model_name=llm_config.get('model_name')
            )
        )
    # elif 其他类型...
```

#### 步骤 3：更新配置
在 <mcfile name="config.py" path="config.py"></mcfile> 或 `config.yml` 中添加相关配置字段：
```python
class LLMConfig(BaseModel):
    model: str = Field(default='qwen3:32b', alias='LLM_MODEL')
    # 新增字段
    api_key: str = Field(default='', alias='LLM_API_KEY')
    provider: str = Field(default='ollama', alias='LLM_PROVIDER')
```

#### 步骤 4：测试新服务
```bash
# 设置环境变量
export LLM_PROVIDER=custom
export LLM_API_KEY=your_api_key

# 运行健康检查
python main_v2.py
# 在 CLI 中输入：health
```

### 架构扩展建议

1. **新增数据源**：实现 `DatabaseService` 接口支持其他数据库
2. **多模态输入**：扩展 `BaseAgent` 支持图片、语音输入
3. **插件系统**：实现动态加载 Agent 的插件机制
4. **监控仪表板**：基于统计数据创建 Web 界面
5. **分布式部署**：将服务容器改造为微服务架构

## 关键设计
- LightweightPlanner：规则优先，结合小模型辅助判别，降低延迟与成本
- AgentProtocol + BaseAgent：统一 run/校验/日志/响应构造，便于测试与扩展
- ServiceContainer：集中式依赖注入，提供 setup_container/get_container 便捷方法
- SQLiteDatabaseService：首次运行自动建表，开箱即用
- ChatOllama：通过 agents/config.get_llm 按任务类型动态配置温度与模型，并记录 Token 用量

## 常见问题
- 首次运行较慢：如本地模型未就绪，请先通过 ollama pull 拉取模型
- 无法连接 LLM：确认 Ollama 已启动，或修改 .env 的 LLM_MODEL/LLM_LITE_MODEL
- DB 路径权限：修改 DB_PATH 到可写目录

## 许可证
MIT