from langgraph.graph import StateGraph, START, END
from core.state import State
from agents.dietary_agent import extract_meal_info, record_meal
from agents.exercise_agent import extract_exercise_info, record_exercise
from agents.report_agent import extract_report_date, prepare_report_data, prompt_divide, retriever, generator
from agents.general_agent import unknown_intent
from agents.supervisor_agent import supervisor
from agents.query_agent import extract_query_params, query_database
from database import init_db
from agents.advice_agent import provide_advice

# Initialize the database
init_db()

# Create a new state graph
graph_builder = StateGraph(State)

# Add nodes for each agent and function
graph_builder.add_node("extract_meal_info", extract_meal_info)
graph_builder.add_node("record_meal", record_meal)
graph_builder.add_node("extract_exercise_info", extract_exercise_info)
graph_builder.add_node("record_exercise", record_exercise)
graph_builder.add_node("extract_report_date", extract_report_date)
graph_builder.add_node("prepare_report_data", prepare_report_data)
graph_builder.add_node("prompt_divide_node", prompt_divide)
graph_builder.add_node("retriever_node", retriever)
graph_builder.add_node("generator_node", generator)
graph_builder.add_node("extract_query_params", extract_query_params)
graph_builder.add_node("query_database", query_database)
graph_builder.add_node("unknown_intent", unknown_intent)
graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("provide_advice", provide_advice)

# Define the entry point of the graph
graph_builder.add_edge(START, "supervisor")

# Define routing logic based on the supervisor's decision
def route_agent(state: State):
    next_agent = state.get("next_agent", "general")
    if next_agent == "dietary":
        return "extract_meal_info"
    elif next_agent == "exercise":
        return "extract_exercise_info"
    elif next_agent == "report":
        return "extract_report_date"
    elif next_agent == "query":
        return "extract_query_params"
    elif next_agent == "advice":
        return "provide_advice"
    else:
        return "unknown_intent"

graph_builder.add_conditional_edges("supervisor", route_agent, {
    "extract_meal_info": "extract_meal_info",
    "extract_exercise_info": "extract_exercise_info",
    "extract_report_date": "extract_report_date",
    "extract_query_params": "extract_query_params",
    "provide_advice": "provide_advice",
    "unknown_intent": "unknown_intent"
})

# Define routing after meal information is extracted
def route_after_meal_extraction(state: State):
    if state.get("next_agent") == "query":
        return "extract_query_params"
    return "record_meal"

graph_builder.add_conditional_edges(
    "extract_meal_info",
    route_after_meal_extraction,
    {
        "record_meal": "record_meal",
        "extract_query_params": "extract_query_params",
    }
)
graph_builder.add_edge("record_meal", END)

# Define routing after exercise information is extracted
def route_after_exercise_extraction(state: State):
    if state.get("next_agent") == "query":
        return "extract_query_params"
    return "record_exercise"

graph_builder.add_conditional_edges(
    "extract_exercise_info",
    route_after_exercise_extraction,
    {
        "record_exercise": "record_exercise",
        "extract_query_params": "extract_query_params",
    }
)
graph_builder.add_edge("record_exercise", END)

# Define the flow for generating reports
graph_builder.add_edge("extract_report_date", "prepare_report_data")

def should_generate_report(state: State):
    if not state.get('full_meal_description'):
        return END
    return "prompt_divide_node"

graph_builder.add_conditional_edges("prepare_report_data", should_generate_report, {
    "prompt_divide_node": "prompt_divide_node",
    END: END
})

graph_builder.add_edge("prompt_divide_node", "retriever_node")
graph_builder.add_edge("retriever_node", "generator_node")
graph_builder.add_edge("generator_node", END)

# Define the flow for querying the database
graph_builder.add_edge("extract_query_params", "query_database")
graph_builder.add_edge("query_database", END)

# Define the end for unknown intents
graph_builder.add_edge("unknown_intent", END)

# Compile the graph
graph = graph_builder.compile()
