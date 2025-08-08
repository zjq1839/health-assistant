# 数据库初始化完成指南

## 🎉 初始化状态

✅ **数据库初始化已完成！**

### 📊 当前数据库状态

#### 主数据库 (`/home/zjq/document/langchain_learn/diet.db`)
- ✅ `meals` 表 - 餐食记录 (8条记录)
- ✅ `exercises` 表 - 运动记录 (1条记录)
- ✅ 索引已创建

#### 营养数据库 (`/home/zjq/document/langchain_learn/data/nutrition.db`)
- ✅ `foods` 表 - 食物营养数据 (5条默认记录)
- ✅ `exercises` 表 - 运动MET值数据 (8条默认记录)
- ✅ 索引已创建

#### 缓存数据库 (`/home/zjq/document/langchain_learn/.cache/`)
- ✅ 缓存目录已创建
- ✅ LLM缓存配置完成

## 🛠️ 可用工具

### 1. 数据库初始化脚本
```bash
python init_database.py
```
- 创建所有必要的数据库表
- 初始化默认数据
- 检查数据库状态

### 2. 数据库检查脚本
```bash
# 查看统计信息
python check_database.py stats

# 查看餐食记录
python check_database.py meals

# 查看运动记录
python check_database.py exercises

# 查看今日记录
python check_database.py today

# 查看所有信息
python check_database.py all
```

### 3. 主程序
```bash
# 启动健康助手
python main_v2.py

# 测试餐食记录
echo '我今天早餐吃了鸡蛋和牛奶' | python main_v2.py

# 测试运动记录
echo '我今天跑步30分钟' | python main_v2.py
```

## 📋 数据库表结构

### meals 表 (餐食记录)
```sql
CREATE TABLE meals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    meal_type TEXT,
    description TEXT,
    calories REAL,
    nutrients TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### exercises 表 (运动记录)
```sql
CREATE TABLE exercises (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    exercise_type TEXT,
    duration INTEGER,
    description TEXT,
    calories_burned REAL,
    intensity TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### foods 表 (食物营养数据)
```sql
CREATE TABLE foods (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT,
    calories REAL,
    protein REAL,
    carbs REAL,
    fat REAL,
    fiber REAL,
    sugar REAL,
    sodium REAL,
    source TEXT,
    confidence REAL,
    aliases TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🔧 配置文件

### config.py
- 数据库路径配置
- LLM模型配置
- 缓存配置

### config.yml
- 应用程序默认配置
- 可通过环境变量覆盖

## ✅ 验证步骤

1. **检查数据库文件是否存在**
   ```bash
   ls -la /home/zjq/document/langchain_learn/diet.db
   ls -la /home/zjq/document/langchain_learn/data/nutrition.db
   ```

2. **检查表结构**
   ```bash
   sqlite3 /home/zjq/document/langchain_learn/diet.db ".schema"
   ```

3. **检查数据**
   ```bash
   python check_database.py stats
   ```

4. **测试程序功能**
   ```bash
   echo 'help' | python main_v2.py
   ```

## 🚀 下一步

数据库初始化已完成，现在可以：

1. **正常使用健康助手**
   - 记录餐食：`python main_v2.py`
   - 记录运动
   - 查询历史记录

2. **添加更多食物数据**
   - 可以通过程序界面添加
   - 或直接操作数据库

3. **配置优化**
   - 调整LLM模型参数
   - 优化缓存设置
   - 自定义数据库路径

## 📞 故障排除

如果遇到问题：

1. **重新初始化数据库**
   ```bash
   python init_database.py
   ```

2. **检查权限**
   ```bash
   chmod +x init_database.py check_database.py
   ```

3. **查看日志**
   - 检查 `app.log` 文件
   - 运行程序时查看控制台输出

---

**🎉 恭喜！数据库初始化完成，健康助手已准备就绪！**