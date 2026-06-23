"""Step2 Agent V1 — 使用 LangGraph create_react_agent 预构建版。

学习目标：看懂 LangGraph 帮你隐藏了什么（对照 V2 手写版）。
"""
from config import DEEPSEEK_MODEL,DEEPSEEK_API_KEY
from tools import TOOLS

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# --- 1. 创建模型 ---
# ChatDeepSeek 底层用 OpenAI SDK，需要处理代理
model=ChatDeepSeek(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    # 代理配置：会传给底层的 httpx 客户端
    http_async_client=None, # 先不加代理，跑不通再加
    http_client=None
)

# --- 2. 创建 ReAct graph ---
# 这一行 = agent node + tools node + conditional edge + state 定义
memory=MemorySaver()
graph=create_react_agent(model,TOOLS,checkpointer=memory)

# --- 3. 入口 ---
def main():
    # 使用同一个 thread_id，MemorySaver 自动保持消息连续
    #thread_id 是 session 标识。同一个 thread_id → MemorySaver 自动叠加消息
    config={"configurable":{"thread_id":"user-session-1"}}

    print("=== 对话 1 ===")
    task1 = "我叫张三，我是Python后端工程师，最近在学习AI Agent开发。"
    
    result=graph.invoke({"messages":[HumanMessage(content=task1)]},config=config)
    #config 传给每次 invoke()，LangGraph 用它找到之前的消息
    print(f"Agent: {result['messages'][-1].content}\n")

    print("=== 对话 2（检查记忆）===")
    task2="我叫什么名字？我的职业是什么？我在学什么？"
    print(f"用户: {task2}\n")

    result=graph.invoke({"messages":[HumanMessage(content=task2)]},config=config)
    #config 传给每次 invoke()，LangGraph 用它找到之前的消息
    print(f"Agent: {result['messages'][-1].content}")

if __name__=="__main__":
    main()