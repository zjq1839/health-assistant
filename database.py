import sqlite3
from typing import List, Tuple

DB_PATH = '/home/zjq/document/langchain_learn/diet.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS meals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            meal_type TEXT NOT NULL, -- '早餐', '午餐', '晚餐'
            description TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exercises (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            exercise_type TEXT NOT NULL,
            duration INTEGER NOT NULL, -- in minutes
            description TEXT
        )
    ''')
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

if __name__ == '__main__':
    init_db()
    print("数据库和表已成功创建。")