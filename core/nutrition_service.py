"""营养计算和食物数据库服务

解决问题：输出非结构化，工具调用不"硬约束"
方案：强制使用结构化数据，绑定证据来源
"""

import json
import sqlite3
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import requests
from pathlib import Path


@dataclass
class NutritionFact:
    """营养成分数据结构"""
    calories: float = 0.0
    protein: float = 0.0  # 蛋白质 (g)
    carbs: float = 0.0    # 碳水化合物 (g)
    fat: float = 0.0      # 脂肪 (g)
    fiber: float = 0.0    # 纤维 (g)
    sugar: float = 0.0    # 糖 (g)
    sodium: float = 0.0   # 钠 (mg)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def __add__(self, other: 'NutritionFact') -> 'NutritionFact':
        """支持营养成分相加"""
        return NutritionFact(
            calories=self.calories + other.calories,
            protein=self.protein + other.protein,
            carbs=self.carbs + other.carbs,
            fat=self.fat + other.fat,
            fiber=self.fiber + other.fiber,
            sugar=self.sugar + other.sugar,
            sodium=self.sodium + other.sodium
        )
    
    def multiply(self, factor: float) -> 'NutritionFact':
        """按比例缩放营养成分"""
        return NutritionFact(
            calories=self.calories * factor,
            protein=self.protein * factor,
            carbs=self.carbs * factor,
            fat=self.fat * factor,
            fiber=self.fiber * factor,
            sugar=self.sugar * factor,
            sodium=self.sodium * factor
        )


@dataclass
class FoodItem:
    """食物条目"""
    name: str
    nutrition_per_100g: NutritionFact
    category: str = "unknown"
    source: str = "database"
    confidence: float = 1.0
    aliases: List[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
    
    def calculate_nutrition(self, amount_g: float) -> NutritionFact:
        """计算指定重量的营养成分"""
        factor = amount_g / 100.0
        return self.nutrition_per_100g.multiply(factor)


@dataclass
class ExerciseActivity:
    """运动活动"""
    name: str
    met_value: float  # 代谢当量
    category: str = "unknown"
    source: str = "database"
    confidence: float = 1.0
    
    def calculate_calories_burned(self, weight_kg: float, duration_minutes: float) -> float:
        """计算消耗的卡路里
        
        公式：卡路里 = MET × 体重(kg) × 时间(小时)
        """
        hours = duration_minutes / 60.0
        return self.met_value * weight_kg * hours


@dataclass
class NutritionEvidence:
    """营养计算证据"""
    source: str
    method: str
    confidence: float
    raw_data: Dict
    timestamp: str
    
    @classmethod
    def create(cls, source: str, method: str, confidence: float, raw_data: Dict) -> 'NutritionEvidence':
        return cls(
            source=source,
            method=method,
            confidence=confidence,
            raw_data=raw_data,
            timestamp=datetime.now().isoformat()
        )


class FoodDatabase(ABC):
    """食物数据库抽象接口"""
    
    @abstractmethod
    def search_food(self, query: str) -> List[FoodItem]:
        """搜索食物"""
        pass
    
    @abstractmethod
    def get_food_by_id(self, food_id: str) -> Optional[FoodItem]:
        """根据ID获取食物"""
        pass
    
    @abstractmethod
    def add_food(self, food: FoodItem) -> str:
        """添加食物到数据库"""
        pass


class ExerciseDatabase(ABC):
    """运动数据库抽象接口"""
    
    @abstractmethod
    def search_exercise(self, query: str) -> List[ExerciseActivity]:
        """搜索运动"""
        pass
    
    @abstractmethod
    def get_exercise_by_id(self, exercise_id: str) -> Optional[ExerciseActivity]:
        """根据ID获取运动"""
        pass


class LocalFoodDatabase(FoodDatabase):
    """本地食物数据库实现"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # 使用默认路径
            db_path = Path(__file__).parent.parent / "data" / "nutrition.db"
        
        self.db_path = str(db_path)
        self._init_database()
        # 确保数据库表创建成功后再加载默认数据
        self._ensure_default_foods()
    
    def _init_database(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_food_name ON foods(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_food_category ON foods(category)")
        
        conn.commit()
        conn.close()
    
    def _ensure_default_foods(self):
        """加载默认食物数据"""
        default_foods = [
            FoodItem(
                name="白米饭",
                nutrition_per_100g=NutritionFact(
                    calories=130, protein=2.7, carbs=28, fat=0.3, fiber=0.4
                ),
                category="主食",
                aliases=["米饭", "大米", "白饭"]
            ),
            FoodItem(
                name="鸡胸肉",
                nutrition_per_100g=NutritionFact(
                    calories=165, protein=31, carbs=0, fat=3.6, fiber=0
                ),
                category="肉类",
                aliases=["鸡肉", "鸡胸"]
            ),
            FoodItem(
                name="苹果",
                nutrition_per_100g=NutritionFact(
                    calories=52, protein=0.3, carbs=14, fat=0.2, fiber=2.4, sugar=10
                ),
                category="水果",
                aliases=["红苹果", "青苹果"]
            ),
            FoodItem(
                name="西兰花",
                nutrition_per_100g=NutritionFact(
                    calories=34, protein=2.8, carbs=7, fat=0.4, fiber=2.6
                ),
                category="蔬菜",
                aliases=["花椰菜", "绿花菜"]
            ),
            FoodItem(
                name="鸡蛋",
                nutrition_per_100g=NutritionFact(
                    calories=155, protein=13, carbs=1.1, fat=11, fiber=0
                ),
                category="蛋类",
                aliases=["鸡蛋", "蛋"]
            ),
            FoodItem(
                name="牛奶",
                nutrition_per_100g=NutritionFact(
                    calories=65, protein=3.5, carbs=5, fat=3.5, fiber=0
                ),
                category="饮品",
                aliases=["纯牛奶", "鲜牛奶"]
            )
        ]
        
        # 检查是否已有数据
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM foods")
            count = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            # 表不存在，说明初始化失败，直接返回
            conn.close()
            return
            
        if count == 0:
            # 插入默认数据
            for food in default_foods:
                self.add_food(food)
        
        conn.close()
    
    def search_food(self, query: str) -> List[FoodItem]:
        """搜索食物"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 模糊搜索
        cursor.execute("""
            SELECT * FROM foods 
            WHERE name LIKE ? OR aliases LIKE ?
            ORDER BY 
                CASE 
                    WHEN name = ? THEN 1
                    WHEN name LIKE ? THEN 2
                    ELSE 3
                END,
                confidence DESC
            LIMIT 10
        """, (f"%{query}%", f"%{query}%", query, f"{query}%"))
        
        rows = cursor.fetchall()
        conn.close()
        
        foods = []
        for row in rows:
            nutrition = NutritionFact(
                calories=row[3] or 0,
                protein=row[4] or 0,
                carbs=row[5] or 0,
                fat=row[6] or 0,
                fiber=row[7] or 0,
                sugar=row[8] or 0,
                sodium=row[9] or 0
            )
            
            aliases = json.loads(row[12]) if row[12] else []
            
            food = FoodItem(
                name=row[1],
                nutrition_per_100g=nutrition,
                category=row[2] or "unknown",
                source=row[10] or "database",
                confidence=row[11] or 1.0,
                aliases=aliases
            )
            foods.append(food)
        
        return foods
    
    def get_food_by_id(self, food_id: str) -> Optional[FoodItem]:
        """根据ID获取食物"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM foods WHERE id = ?", (food_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        nutrition = NutritionFact(
            calories=row[3] or 0,
            protein=row[4] or 0,
            carbs=row[5] or 0,
            fat=row[6] or 0,
            fiber=row[7] or 0,
            sugar=row[8] or 0,
            sodium=row[9] or 0
        )
        
        aliases = json.loads(row[12]) if row[12] else []
        
        return FoodItem(
            name=row[1],
            nutrition_per_100g=nutrition,
            category=row[2] or "unknown",
            source=row[10] or "database",
            confidence=row[11] or 1.0,
            aliases=aliases
        )
    
    def add_food(self, food: FoodItem) -> str:
        """添加食物到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO foods (
                name, category, calories, protein, carbs, fat, fiber, sugar, sodium,
                source, confidence, aliases
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            food.name,
            food.category,
            food.nutrition_per_100g.calories,
            food.nutrition_per_100g.protein,
            food.nutrition_per_100g.carbs,
            food.nutrition_per_100g.fat,
            food.nutrition_per_100g.fiber,
            food.nutrition_per_100g.sugar,
            food.nutrition_per_100g.sodium,
            food.source,
            food.confidence,
            json.dumps(food.aliases, ensure_ascii=False)
        ))
        
        food_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return str(food_id)


class LocalExerciseDatabase(ExerciseDatabase):
    """本地运动数据库实现"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "nutrition.db"
        
        self.db_path = str(db_path)
        self._init_database()
        # 确保数据库表创建成功后再加载默认数据
        self._ensure_default_exercises()
    
    def _init_database(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exercise_name ON exercises(name)")
        
        conn.commit()
        conn.close()
    
    def _ensure_default_exercises(self):
        """加载默认运动数据"""
        default_exercises = [
            ExerciseActivity(name="跑步", met_value=8.0, category="有氧运动"),
            ExerciseActivity(name="快走", met_value=4.0, category="有氧运动"),
            ExerciseActivity(name="游泳", met_value=6.0, category="有氧运动"),
            ExerciseActivity(name="骑自行车", met_value=5.5, category="有氧运动"),
            ExerciseActivity(name="俯卧撑", met_value=3.8, category="力量训练"),
            ExerciseActivity(name="仰卧起坐", met_value=3.8, category="力量训练"),
            ExerciseActivity(name="瑜伽", met_value=2.5, category="柔韧性训练"),
            ExerciseActivity(name="举重", met_value=6.0, category="力量训练")
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM exercises")
            count = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            # 表不存在，说明初始化失败，直接返回
            conn.close()
            return
            
        if count == 0:
            for exercise in default_exercises:
                cursor.execute("""
                    INSERT INTO exercises (name, category, met_value, source, confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    exercise.name,
                    exercise.category,
                    exercise.met_value,
                    exercise.source,
                    exercise.confidence
                ))
            conn.commit()
        
        conn.close()
    
    def search_exercise(self, query: str) -> List[ExerciseActivity]:
        """搜索运动"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM exercises 
            WHERE name LIKE ?
            ORDER BY 
                CASE 
                    WHEN name = ? THEN 1
                    WHEN name LIKE ? THEN 2
                    ELSE 3
                END,
                confidence DESC
            LIMIT 10
        """, (f"%{query}%", query, f"{query}%"))
        
        rows = cursor.fetchall()
        conn.close()
        
        exercises = []
        for row in rows:
            exercise = ExerciseActivity(
                name=row[1],
                category=row[2] or "unknown",
                met_value=row[3] or 3.0,
                source=row[4] or "database",
                confidence=row[5] or 1.0
            )
            exercises.append(exercise)
        
        return exercises
    
    def get_exercise_by_id(self, exercise_id: str) -> Optional[ExerciseActivity]:
        """根据ID获取运动"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM exercises WHERE id = ?", (exercise_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return ExerciseActivity(
            name=row[1],
            category=row[2] or "unknown",
            met_value=row[3] or 3.0,
            source=row[4] or "database",
            confidence=row[5] or 1.0
        )


class StructuredNutritionService:
    """结构化营养计算服务
    
    强制使用结构化数据，绑定证据来源
    """
    
    def __init__(self, 
                 food_db: FoodDatabase = None,
                 exercise_db: ExerciseDatabase = None):
        self.food_db = food_db or LocalFoodDatabase()
        self.exercise_db = exercise_db or LocalExerciseDatabase()
        
        # 导入日志
        from utils.logger import logger
        self.logger = logger
    
    def calculate_meal_nutrition(self, meal_description: str) -> Tuple[NutritionFact, List[NutritionEvidence]]:
        """计算餐食营养成分
        
        返回：(营养成分, 证据列表)
        """
        total_nutrition = NutritionFact()
        evidences = []
        
        # 解析餐食描述
        food_items = self._parse_meal_description(meal_description)
        
        for food_name, amount in food_items:
            # 搜索食物
            foods = self.food_db.search_food(food_name)
            
            if not foods:
                # 记录未找到的食物
                evidence = NutritionEvidence.create(
                    source="food_database",
                    method="search_failed",
                    confidence=0.0,
                    raw_data={
                        "source": "food_database",
                        "method": "search_failed",
                        "query": food_name,
                        "amount": amount,
                        "error": "Food not found in database"
                    }
                )
                evidences.append(evidence)
                continue
            
            # 使用最匹配的食物
            best_food = foods[0]
            food_nutrition = best_food.calculate_nutrition(amount)
            total_nutrition = total_nutrition + food_nutrition
            
            # 记录证据
            evidence = NutritionEvidence.create(
                source="food_database",
                method="database_lookup",
                confidence=best_food.confidence,
                raw_data={
                    "source": "food_database",
                    "method": "database_lookup",
                    "food_name": best_food.name,
                    "amount_g": amount,
                    "nutrition_per_100g": best_food.nutrition_per_100g.to_dict(),
                    "calculated_nutrition": food_nutrition.to_dict(),
                    "category": best_food.category
                }
            )
            evidences.append(evidence)
            
            self.logger.info(
                f"Calculated nutrition for {food_name}",
                extra={
                    "food_name": best_food.name,
                    "amount": amount,
                    "calories": food_nutrition.calories,
                    "confidence": best_food.confidence
                }
            )
        
        return total_nutrition, evidences
    
    def calculate_exercise_calories(self, exercise_description: str, 
                                  weight_kg: float = 70.0) -> Tuple[float, List[NutritionEvidence]]:
        """计算运动消耗的卡路里
        
        返回：(消耗卡路里, 证据列表)
        """
        total_calories = 0.0
        evidences = []
        
        # 解析运动描述
        exercise_items = self._parse_exercise_description(exercise_description)
        
        for exercise_name, duration in exercise_items:
            # 搜索运动
            exercises = self.exercise_db.search_exercise(exercise_name)
            
            if not exercises:
                evidence = NutritionEvidence.create(
                    source="exercise_database",
                    method="search_failed",
                    confidence=0.0,
                    raw_data={
                        "query": exercise_name,
                        "duration": duration,
                        "error": "Exercise not found in database"
                    }
                )
                evidences.append(evidence)
                continue
            
            # 使用最匹配的运动
            best_exercise = exercises[0]
            calories_burned = best_exercise.calculate_calories_burned(weight_kg, duration)
            total_calories += calories_burned
            
            # 记录证据
            evidence = NutritionEvidence.create(
                source="exercise_database",
                method="met_calculation",
                confidence=best_exercise.confidence,
                raw_data={
                    "exercise_name": best_exercise.name,
                    "duration_minutes": duration,
                    "weight_kg": weight_kg,
                    "met_value": best_exercise.met_value,
                    "calories_burned": calories_burned,
                    "category": best_exercise.category
                }
            )
            evidences.append(evidence)
            
            self.logger.info(
                f"Calculated calories for {exercise_name}",
                extra={
                    "exercise_name": best_exercise.name,
                    "duration": duration,
                    "calories_burned": calories_burned,
                    "met_value": best_exercise.met_value
                }
            )
        
        return total_calories, evidences
    
    def _parse_meal_description(self, description: str) -> List[Tuple[str, float]]:
        """解析餐食描述，提取食物和重量
        
        返回：[(食物名称, 重量(克))]
        """
        import re
        
        # 简单的解析逻辑，实际应该更复杂
        items = []
        
        # 匹配模式："食物名称 数量单位"
        patterns = [
            r'(\S+)\s*(\d+(?:\.\d+)?)\s*克',
            r'(\S+)\s*(\d+(?:\.\d+)?)\s*g',
            r'(\S+)\s*(\d+(?:\.\d+)?)\s*份',
            r'(\S+)\s*(\d+(?:\.\d+)?)\s*个',
            r'(\S+)\s*(\d+(?:\.\d+)?)\s*碗'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, description)
            for food_name, amount_str in matches:
                amount = float(amount_str)
                
                # 根据单位转换重量
                if '份' in pattern:
                    amount *= 150  # 假设一份150克
                elif '个' in pattern:
                    amount *= 100  # 假设一个100克
                elif '碗' in pattern:
                    amount *= 200  # 假设一碗200克
                
                items.append((food_name, amount))
        
        # 如果没有匹配到，尝试简单分词
        if not items:
            words = description.split()
            for word in words:
                if len(word) > 1:  # 过滤单字
                    items.append((word, 100))  # 默认100克
        
        return items
    
    def _parse_exercise_description(self, description: str) -> List[Tuple[str, float]]:
        """解析运动描述，提取运动和时长
        
        返回：[(运动名称, 时长(分钟))]
        """
        import re
        
        items = []
        
        # 匹配模式："运动名称 时长单位"
        patterns = [
            r'(\S+)\s*(\d+(?:\.\d+)?)\s*分钟',
            r'(\S+)\s*(\d+(?:\.\d+)?)\s*分',
            r'(\S+)\s*(\d+(?:\.\d+)?)\s*小时',
            r'(\S+)\s*(\d+(?:\.\d+)?)\s*次'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, description)
            for exercise_name, duration_str in matches:
                duration = float(duration_str)
                
                # 根据单位转换时长
                if '小时' in pattern:
                    duration *= 60  # 转换为分钟
                elif '次' in pattern:
                    duration *= 5   # 假设每次5分钟
                
                items.append((exercise_name, duration))
        
        # 如果没有匹配到，尝试简单分词
        if not items:
            words = description.split()
            for word in words:
                if len(word) > 1:
                    items.append((word, 30))  # 默认30分钟
        
        return items
    
    def get_food_suggestions(self, query: str, limit: int = 5) -> List[FoodItem]:
        """获取食物建议"""
        return self.food_db.search_food(query)[:limit]
    
    def get_exercise_suggestions(self, query: str, limit: int = 5) -> List[ExerciseActivity]:
        """获取运动建议"""
        return self.exercise_db.search_exercise(query)[:limit]
    
    def validate_nutrition_data(self, nutrition: NutritionFact) -> Dict[str, str]:
        """验证营养数据的合理性"""
        warnings = {}
        
        if nutrition.calories < 0:
            warnings['calories'] = "卡路里不能为负数"
        elif nutrition.calories > 5000:
            warnings['calories'] = "单餐卡路里过高，请检查数据"
        
        if nutrition.protein < 0:
            warnings['protein'] = "蛋白质不能为负数"
        elif nutrition.protein > 200:
            warnings['protein'] = "蛋白质含量过高，请检查数据"
        
        if nutrition.carbs < 0:
            warnings['carbs'] = "碳水化合物不能为负数"
        
        if nutrition.fat < 0:
            warnings['fat'] = "脂肪不能为负数"
        
        # 检查营养比例
        total_macro = nutrition.protein * 4 + nutrition.carbs * 4 + nutrition.fat * 9
        if total_macro > 0 and abs(total_macro - nutrition.calories) > nutrition.calories * 0.2:
            warnings['consistency'] = "营养成分与总卡路里不一致"
        
        return warnings


# 使用示例
if __name__ == "__main__":
    # 创建服务
    nutrition_service = StructuredNutritionService()
    
    # 测试餐食营养计算
    meal_desc = "白米饭150克 鸡胸肉100克 西兰花80克"
    nutrition, evidences = nutrition_service.calculate_meal_nutrition(meal_desc)
    
    print(f"总营养成分：{nutrition.to_dict()}")
    print(f"证据数量：{len(evidences)}")
    
    for evidence in evidences:
        print(f"证据：{evidence.source} - {evidence.method} (置信度: {evidence.confidence})")
    
    # 测试运动卡路里计算
    exercise_desc = "跑步30分钟 俯卧撑20次"
    calories, ex_evidences = nutrition_service.calculate_exercise_calories(exercise_desc, 70)
    
    print(f"\n消耗卡路里：{calories}")
    print(f"证据数量：{len(ex_evidences)}")
    
    for evidence in ex_evidences:
        print(f"证据：{evidence.source} - {evidence.method} (置信度: {evidence.confidence})")