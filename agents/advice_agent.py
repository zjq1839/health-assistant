"""健康建议代理，负责根据用户输入与上下文实体生成可行的饮食或运动建议。"""
from __future__ import annotations

import re
from typing import Dict, Any

from agents.config import llm
from core.state import State
from core.enhanced_state import DialogState
from utils.logger import logger


def _build_prompt(user_input: str, context_entities: Dict[str, Any]) -> str:
    """根据用户输入和上下文实体构造提示词。"""
    entity_section = ""
    if context_entities:
        entity_pairs = ", ".join(f"{k}: {v}" for k, v in context_entities.items() if v)
        entity_section = f"\n\n已知相关信息：{entity_pairs}"

    return (
        "你是一名专业且富有同理心的健康顾问。请基于以下用户需求，结合上下文信息，提供3条具体、可执行、积极鼓励的健康建议，涵盖饮食或运动层面。"
        f"{entity_section}\n\n用户需求：{user_input}\n\n请使用简体中文回答。"
    )


def provide_advice(state: State):  # noqa: D401
    """生成健康建议并追加到响应。"""
    # 获取用户最近输入
    last_message = state["messages"][-1]
    if isinstance(last_message, tuple):
        user_input = last_message[1]
    else:
        user_input = getattr(last_message, "content", "")

    # 复用对话状态中已解析的实体
    dialog_state: DialogState | None = state.get("dialog_state")  # type: ignore[arg-type]
    context_entities: Dict[str, Any] = dialog_state.get_context_entities() if dialog_state else {}

    prompt = _build_prompt(user_input, context_entities)
    logger.debug("AdviceAgent Prompt: %s", prompt)

    response = llm.invoke(prompt)
    cleaned_response = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()

    # 返回格式与其他代理保持一致
    return {
        "messages": [("ai", cleaned_response)],
        "next_agent": "advice",
    }