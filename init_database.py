#!/usr/bin/env python3
"""数据库初始化脚本

创建所有必要的数据库表和索引
"""

import sqlite3
import os
from pathlib import Path
from config import config

def init_main_database():
    """初始化主数据库（饮食和运动记录）"""
    db_path = config.database.path
    
    # 确保目录存在
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"正在初始化主数据库: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 创建 meals 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                meal_type TEXT,
                description TEXT,
                calories REAL,
                nutrients TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("✓ meals 表创建成功")
        
        # 创建 exercises 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exercises (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                exercise_type TEXT,
                duration INTEGER,
                description TEXT,
                calories_burned REAL,
                intensity TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("✓ exercises 表创建成功")
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_meals_date ON meals(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exercises_date ON exercises(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_meals_created_at ON meals(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exercises_created_at ON exercises(created_at)")
        print("✓ 索引创建成功")
        
        conn.commit()
        print("✓ 主数据库初始化完成")
        
    except Exception as e:
        print(f"✗ 主数据库初始化失败: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def init_nutrition_database():
    """初始化营养数据库（食物和运动数据）"""
    # 使用默认路径
    nutrition_db_path = Path(__file__).parent / "data" / "nutrition.db"
    
    # 确保目录存在
    nutrition_db_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"正在初始化营养数据库: {nutrition_db_path}")
    
    conn = sqlite3.connect(str(nutrition_db_path))
    cursor = conn.cursor()
    
    try:
        # 创建 foods 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS foods (
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
            )
        """)
        print("✓ foods 表创建成功")
        
        # 创建 exercises 表（营养数据库中的运动数据）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exercises (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT,
                met_value REAL,
                source TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("✓ exercises 表创建成功")
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_food_name ON foods(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_food_category ON foods(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exercise_name ON exercises(name)")
        print("✓ 索引创建成功")
        
        conn.commit()
        print("✓ 营养数据库初始化完成")
        
    except Exception as e:
        print(f"✗ 营养数据库初始化失败: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def init_cache_database():
    """初始化缓存数据库"""
    cache_dir = Path(__file__).parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_db_path = cache_dir / "llm_cache.db"
    print(f"正在初始化缓存数据库: {cache_db_path}")
    
    # LangChain 会自动创建缓存表，这里只需要确保目录存在
    print("✓ 缓存数据库目录创建成功")

def check_database_status():
    """检查数据库状态"""
    print("\n=== 数据库状态检查 ===")
    
    # 检查主数据库
    main_db_path = config.database.path
    if os.path.exists(main_db_path):
        conn = sqlite3.connect(main_db_path)
        cursor = conn.cursor()
        
        # 检查表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"主数据库 ({main_db_path}):")
        print(f"  - 表: {', '.join(tables)}")
        
        # 检查记录数量
        if 'meals' in tables:
            cursor.execute("SELECT COUNT(*) FROM meals")
            meal_count = cursor.fetchone()[0]
            print(f"  - meals 记录数: {meal_count}")
            
        if 'exercises' in tables:
            cursor.execute("SELECT COUNT(*) FROM exercises")
            exercise_count = cursor.fetchone()[0]
            print(f"  - exercises 记录数: {exercise_count}")
            
        conn.close()
    else:
        print(f"主数据库不存在: {main_db_path}")
    
    # 检查营养数据库
    nutrition_db_path = Path(__file__).parent / "data" / "nutrition.db"
    if nutrition_db_path.exists():
        conn = sqlite3.connect(str(nutrition_db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"\n营养数据库 ({nutrition_db_path}):")
        print(f"  - 表: {', '.join(tables)}")
        
        # 检查记录数量
        if 'foods' in tables:
            cursor.execute("SELECT COUNT(*) FROM foods")
            food_count = cursor.fetchone()[0]
            print(f"  - foods 记录数: {food_count}")
            
        if 'exercises' in tables:
            cursor.execute("SELECT COUNT(*) FROM exercises")
            exercise_count = cursor.fetchone()[0]
            print(f"  - exercises 记录数: {exercise_count}")
            
        conn.close()
    else:
        print(f"营养数据库不存在: {nutrition_db_path}")

def main():
    """主函数"""
    print("=== 健康助手数据库初始化 ===")
    
    try:
        # 初始化各个数据库
        init_main_database()
        init_nutrition_database()
        init_cache_database()
        
        print("\n✓ 所有数据库初始化完成")
        
        # 检查状态
        check_database_status()
        
        print("\n=== 初始化完成 ===")
        print("现在可以运行 python main_v2.py 启动健康助手")
        
    except Exception as e:
        print(f"\n✗ 初始化失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())