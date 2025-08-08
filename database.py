import sqlite3
from typing import List, Tuple
from config import config

DB_PATH = config.database.path

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS meals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            meal_type TEXT NOT NULL, -- '早餐', '午餐', '晚餐'
            description TEXT NOT NULL,
            calories REAL DEFAULT 0, -- 卡路里
            nutrients TEXT, -- JSON格式存储营养成分
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exercises (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            exercise_type TEXT NOT NULL,
            duration INTEGER NOT NULL, -- in minutes
            description TEXT,
            calories_burned REAL DEFAULT 0, -- 消耗的卡路里
            intensity TEXT DEFAULT 'medium', -- 强度: low, medium, high
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 创建索引提升查询性能
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_meals_date ON meals(date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_exercises_date ON exercises(date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_meals_type ON meals(meal_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_exercises_type ON exercises(exercise_type)')
    
    conn.commit()
    conn.close()

def add_meal(date: str, meal_type: str, description: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO meals (date, meal_type, description) VALUES (?, ?, ?)", (date, meal_type, description))
    conn.commit()
    conn.close()

def get_meals_by_date(date: str) -> List[Tuple]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM meals WHERE date = ?", (date,))
    meals = cursor.fetchall()
    conn.close()
    return meals

def add_exercise(date: str, exercise_type: str, duration: int, description: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO exercises (date, exercise_type, duration, description) VALUES (?, ?, ?, ?)", (date, exercise_type, duration, description))
    conn.commit()
    conn.close()

def get_exercises_by_date(date: str) -> List[Tuple]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM exercises WHERE date = ?", (date,))
    exercises = cursor.fetchall()
    conn.close()
    return exercises

# 为了兼容测试，添加函数别名
def query_meals(date: str = None) -> List[Tuple]:
    """查询餐食记录"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if date:
        cursor.execute("SELECT * FROM meals WHERE date = ?", (date,))
    else:
        cursor.execute("SELECT * FROM meals ORDER BY date DESC LIMIT 10")
    meals = cursor.fetchall()
    conn.close()
    return meals

def query_exercises(date: str = None) -> List[Tuple]:
    """查询运动记录"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if date:
        cursor.execute("SELECT * FROM exercises WHERE date = ?", (date,))
    else:
        cursor.execute("SELECT * FROM exercises ORDER BY date DESC LIMIT 10")
    exercises = cursor.fetchall()
    conn.close()
    return exercises

if __name__ == '__main__':
    init_db()
    print("数据库和表已成功创建。")