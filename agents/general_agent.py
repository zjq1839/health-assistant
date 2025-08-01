from core.state import State

def unknown_intent(state: State):
    return {"messages": [("ai", "抱歉，我不太明白你的意思。你可以说‘我早餐吃了...’来记录饮食，或者说‘生成今天的报告’来获取分析。")]}