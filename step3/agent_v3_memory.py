"""Step3 Agent V3 — ReAct + 双层持久记忆（recall + save_memory 节点）。

改造：在 create_react_agent 前后加入记忆检索和存储节点。
"""
import uuid
from typing import Annotated, TypedDict

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from config import DEEPSEEK_MODEL, DEEPSEEK_API_KEY
from memory import MemoryManager
from tools import TOOLS

# --- 1. 状态 ---
class State(TypedDict):
    messages:Annotated[list,add_messages]
    conv_id:int # 对话 ID，用于关联记忆

# --- 2. 记忆管理器 ---
memory_manager=MemoryManager()

# --- 3. recall 节点 ---
def recall_node(state:State):
    """检索记忆，作为 SystemMessage 注入消息开头。"""
    # 用最新用户消息作为检索 query
    user_query=""
    for m in reversed(state["messages"]):
        if getattr(m,"type",None)=="human":
            user_query=m.content
            break

    memory_text=memory_manager.recall(user_query)
    if memory_text:
        system_msg=SystemMessage(
            content=f"以下是关于这位用户的已知信息，请在回答时参考：\n\n{memory_text}"
        )
        # 插入到消息列表最前面
        return {"messages":[system_msg]}
    return {}

# --- 4. save_memory 节点 ---
def save_memory_node(state:State):
    """对话结束后保存记忆。"""
    conv_id=state.get("conv_id",str(uuid.uuid4())[:8]) #生成对话id
    memory_manager.save_conversation(state["messages"],conv_id) #保存对话内容
    print(f"[memory] 对话 {conv_id} 记忆已保存")
    return {}


# --- 5. 构建图 ---
model=ChatDeepSeek(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
)
react_graph=create_react_agent(model,TOOLS)

builder=StateGraph(State)
builder.add_node("recall",recall_node)
builder.add_node("react",react_graph)
builder.add_node("save_memory",save_memory_node)

builder.add_edge(START,"recall")
builder.add_edge("recall","react")
builder.add_edge("react","save_memory")
builder.add_edge("save_memory",END)

memory=MemorySaver()
graph=builder.compile(checkpointer=memory)


# --- 6. 入口 ---
def main():
    session_id = "user-session-persist"
    config = {"configurable": {"thread_id": session_id}}

    # 第一轮对话：自我介绍
    conv_id_1 = "conv-" + str(uuid.uuid4())[:8]
    print(f"=== 对话 1 ({conv_id_1}) ===\n")
    task1 = "我叫张三，我是Python后端工程师，最近在学习AI Agent开发。"
    print(f"用户: {task1}")
    result = graph.invoke(
        {"messages": [HumanMessage(content=task1)], "conv_id": conv_id_1},
        config=config,
    )
    print(f"Agent: {result['messages'][-1].content[:200]}...\n")

    # 第二轮对话：询问之前的信息
    conv_id_2 = "conv-" + str(uuid.uuid4())[:8]
    print(f"=== 对话 2 ({conv_id_2}) ===\n")
    task2 = "我叫什么名字？我的职业是什么？我在学什么？"
    print(f"用户: {task2}")
    result = graph.invoke(
        {"messages": [HumanMessage(content=task2)], "conv_id": conv_id_2},
        config=config,
    )
    print(f"Agent: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
