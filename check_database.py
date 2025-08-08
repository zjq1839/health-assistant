#!/usr/bin/env python3
"""数据库状态检查脚本

查看数据库中的记录和统计信息
"""

import sqlite3
import sys
from datetime import datetime, date
from config import config

def show_meals(limit=10, date_filter=None):
    """显示餐食记录"""
    conn = sqlite3.connect(config.database.path)
    cursor = conn.cursor()
    
    try:
        if date_filter:
            cursor.execute("""
                SELECT id, date, meal_type, description, calories, created_at 
                FROM meals 
                WHERE date = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (date_filter, limit))
        else:
            cursor.execute("""
                SELECT id, date, meal_type, description, calories, created_at 
                FROM meals 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        
        if not rows:
            print("📭 没有找到餐食记录")
            return
        
        print(f"🍽️ 餐食记录 (最近 {len(rows)} 条):")
        print("-" * 80)
        
        for row in rows:
            id_, date_, meal_type, description, calories, created_at = row
            print(f"ID: {id_}")
            print(f"📅 日期: {date_}")
            print(f"🍽️ 类型: {meal_type or '未分类'}")
            print(f"📝 描述: {description}")
            print(f"🔥 卡路里: {calories or 0.0} kcal")
            print(f"⏰ 创建时间: {created_at}")
            print("-" * 40)
            
    finally:
        conn.close()

def show_exercises(limit=10, date_filter=None):
    """显示运动记录"""
    conn = sqlite3.connect(config.database.path)
    cursor = conn.cursor()
    
    try:
        if date_filter:
            cursor.execute("""
                SELECT id, date, exercise_type, duration, description, calories_burned, intensity, created_at 
                FROM exercises 
                WHERE date = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (date_filter, limit))
        else:
            cursor.execute("""
                SELECT id, date, exercise_type, duration, description, calories_burned, intensity, created_at 
                FROM exercises 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        
        if not rows:
            print("📭 没有找到运动记录")
            return
        
        print(f"🏃 运动记录 (最近 {len(rows)} 条):")
        print("-" * 80)
        
        for row in rows:
            id_, date_, exercise_type, duration, description, calories_burned, intensity, created_at = row
            print(f"ID: {id_}")
            print(f"📅 日期: {date_}")
            print(f"🏃 类型: {exercise_type or '未分类'}")
            print(f"⏱️ 时长: {duration or 0} 分钟")
            print(f"📝 描述: {description}")
            print(f"🔥 消耗卡路里: {calories_burned or 0.0} kcal")
            print(f"💪 强度: {intensity or '中等'}")
            print(f"⏰ 创建时间: {created_at}")
            print("-" * 40)
            
    finally:
        conn.close()

def show_statistics():
    """显示统计信息"""
    conn = sqlite3.connect(config.database.path)
    cursor = conn.cursor()
    
    try:
        # 餐食统计
        cursor.execute("SELECT COUNT(*) FROM meals")
        total_meals = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT date) FROM meals")
        meal_days = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(calories) FROM meals WHERE calories IS NOT NULL")
        total_calories = cursor.fetchone()[0] or 0
        
        # 运动统计
        cursor.execute("SELECT COUNT(*) FROM exercises")
        total_exercises = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT date) FROM exercises")
        exercise_days = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(calories_burned) FROM exercises WHERE calories_burned IS NOT NULL")
        total_burned = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT SUM(duration) FROM exercises WHERE duration IS NOT NULL")
        total_duration = cursor.fetchone()[0] or 0
        
        # 今日统计
        today = date.today().isoformat()
        cursor.execute("SELECT COUNT(*) FROM meals WHERE date = ?", (today,))
        today_meals = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM exercises WHERE date = ?", (today,))
        today_exercises = cursor.fetchone()[0]
        
        print("📊 数据库统计信息")
        print("=" * 50)
        print(f"🍽️ 餐食记录:")
        print(f"  - 总记录数: {total_meals}")
        print(f"  - 记录天数: {meal_days}")
        print(f"  - 总卡路里: {total_calories:.1f} kcal")
        print(f"  - 今日记录: {today_meals}")
        
        print(f"\n🏃 运动记录:")
        print(f"  - 总记录数: {total_exercises}")
        print(f"  - 运动天数: {exercise_days}")
        print(f"  - 总消耗: {total_burned:.1f} kcal")
        print(f"  - 总时长: {total_duration} 分钟")
        print(f"  - 今日记录: {today_exercises}")
        
        print(f"\n📅 今日日期: {today}")
        
    finally:
        conn.close()

def show_today():
    """显示今日记录"""
    today = date.today().isoformat()
    print(f"📅 今日记录 ({today})")
    print("=" * 50)
    
    show_meals(limit=20, date_filter=today)
    print()
    show_exercises(limit=20, date_filter=today)

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法:")
        print("  python check_database.py stats    - 显示统计信息")
        print("  python check_database.py meals    - 显示最近的餐食记录")
        print("  python check_database.py exercises - 显示最近的运动记录")
        print("  python check_database.py today    - 显示今日记录")
        print("  python check_database.py all      - 显示所有信息")
        return 1
    
    command = sys.argv[1].lower()
    
    try:
        if command == "stats":
            show_statistics()
        elif command == "meals":
            show_meals()
        elif command == "exercises":
            show_exercises()
        elif command == "today":
            show_today()
        elif command == "all":
            show_statistics()
            print("\n")
            show_today()
            print("\n")
            show_meals()
            print("\n")
            show_exercises()
        else:
            print(f"未知命令: {command}")
            return 1
            
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())