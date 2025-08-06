from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    docs: list
    intent: Literal["record_meal", "record_exercise", "generate_report", "unknown"]
    meal_type: str
    meal_description: str
    meal_date: str
    exercise_type: str
    exercise_duration: int
    exercise_description: str
    report_date: str
    full_meal_description: str
    full_exercise_description: str
    next_agent: str
    query_date: str
    query_type: str