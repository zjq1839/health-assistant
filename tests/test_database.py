import pytest
import sqlite3
import tempfile
import os
from unittest.mock import patch
from database import init_db, add_meal, add_exercise, query_meals, query_exercises

class TestDatabase:
    """测试数据库操作功能"""
    
    @pytest.fixture
    def temp_db(self):
        """创建临时数据库用于测试"""
        # 创建临时文件
        fd, temp_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        # 使用临时数据库路径
        with patch('database.DB_PATH', temp_path):
            init_db()
            yield temp_path
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_init_db_creates_tables(self, temp_db):
        """测试数据库初始化是否创建了正确的表"""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        # 检查 meals 表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='meals'")
        assert cursor.fetchone() is not None
        
        # 检查 exercises 表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='exercises'")
        assert cursor.fetchone() is not None
        
        # 检查索引
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_meals_date'")
        assert cursor.fetchone() is not None
        
        conn.close()
    
    def test_add_meal(self, temp_db):
        """测试添加餐食记录"""
        with patch('database.DB_PATH', temp_db):
            add_meal("2024-01-15", "早餐", "鸡蛋和牛奶")
            
            # 验证数据是否正确插入
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM meals WHERE date='2024-01-15'")
            result = cursor.fetchone()
            
            assert result is not None
            assert result[1] == "2024-01-15"  # date
            assert result[2] == "早餐"  # meal_type
            assert result[3] == "鸡蛋和牛奶"  # description
            
            conn.close()
    
    def test_add_exercise(self, temp_db):
        """测试添加运动记录"""
        with patch('database.DB_PATH', temp_db):
            add_exercise("2024-01-15", "跑步", 30, "晨跑")
            
            # 验证数据是否正确插入
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM exercises WHERE date='2024-01-15'")
            result = cursor.fetchone()
            
            assert result is not None
            assert result[1] == "2024-01-15"  # date
            assert result[2] == "跑步"  # exercise_type
            assert result[3] == 30  # duration
            assert result[4] == "晨跑"  # description
            
            conn.close()
    
    def test_query_meals(self, temp_db):
        """测试查询餐食记录"""
        with patch('database.DB_PATH', temp_db):
            # 添加测试数据
            add_meal("2024-01-15", "早餐", "鸡蛋")
            add_meal("2024-01-15", "午餐", "米饭")
            add_meal("2024-01-16", "早餐", "面包")
            
            # 查询特定日期的记录
            results = query_meals("2024-01-15")
            assert len(results) == 2
            
            # 查询所有记录
            all_results = query_meals()
            assert len(all_results) == 3
    
    def test_query_exercises(self, temp_db):
        """测试查询运动记录"""
        with patch('database.DB_PATH', temp_db):
            # 添加测试数据
            add_exercise("2024-01-15", "跑步", 30, "晨跑")
            add_exercise("2024-01-15", "游泳", 45, "下午游泳")
            add_exercise("2024-01-16", "骑车", 60, "骑车上班")
            
            # 查询特定日期的记录
            results = query_exercises("2024-01-15")
            assert len(results) == 2
            
            # 查询所有记录
            all_results = query_exercises()
            assert len(all_results) == 3
    
    def test_database_schema(self, temp_db):
        """测试数据库表结构"""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        # 检查 meals 表结构
        cursor.execute("PRAGMA table_info(meals)")
        meals_columns = [row[1] for row in cursor.fetchall()]
        expected_meals_columns = ['id', 'date', 'meal_type', 'description', 'calories', 'nutrients', 'created_at', 'updated_at']
        
        for col in expected_meals_columns:
            assert col in meals_columns
        
        # 检查 exercises 表结构
        cursor.execute("PRAGMA table_info(exercises)")
        exercises_columns = [row[1] for row in cursor.fetchall()]
        expected_exercises_columns = ['id', 'date', 'exercise_type', 'duration', 'description', 'calories_burned', 'intensity', 'created_at', 'updated_at']
        
        for col in expected_exercises_columns:
            assert col in exercises_columns
        
        conn.close()