from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen3:4b", temperature=0)
llm_lite = ChatOllama(model="qwen3:1.7b", temperature=0,reasoning=True)