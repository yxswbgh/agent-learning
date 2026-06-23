from typing import Annotated, TypedDict

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from config import DEEPSEEK_MODEL, DEEPSEEK_API_KEY
from tools import TOOLS


'''
1. State      — 图的共享数据（消息列表）
2. agent node — 调 LLM，返回 AIMessage
3. tools node — 执行 tool_calls，返回 ToolMessage 列表
4. 路由      — 检查最后一条消息有没有 tool_calls，决定继续还是结束
'''

# --- 1. 定义 State ---图的共享数据，消息列表
class State(TypedDict):
    messages:Annotated[list,add_messages]
 # add_messages 自动合并消息列表

# --- 2. 创建模型 ---
model=ChatDeepSeek(model=DEEPSEEK_MODEL,api_key=DEEPSEEK_API_KEY)

# --- 3. 节点函数 ---
def call_model(state:State):
    """Agent 节点：调 LLM 思考下一步做什么。"""
    messages=state["messages"]
    response=model.invoke(messages)
    return {"messages":[response]}

def call_tools(state:State):
    """Tools 节点：执行 LLM 请求的工具调用。"""
    messages=state["messages"]
    last_message=messages[-1]

    # 按 name 建立工具索引
    tool_map={t.name:t for t in TOOLS}

    tool_messages=[] #保存结果
    for tc in last_message.tool_calls:
        tool=tool_map.get(tc["name"]) #通过name返回工具
        if tool is None:
            result = f"错误：未知工具 {tc['name']}"
        else:
            try:
                result=tool.invoke(tc["args"])
            except Exception as e:
                result = f"工具执行失败: {type(e).__name__} — {e}"
        
        tool_messages.append(ToolMessage(content=result,tool_call_id=tc["id"]))

        return {"messages":tool_messages}
    
# --- 4. 路由函数 ---
def route_after_agent(state:State):
    """判断 Agent 输出后该去哪：继续调 tools，还是结束。"""
    last_message=state["messages"][-1]
    if hasattr(last_message,"tool_calls") and last_message.tool_calls:
        return "tools"

    return END

# --- 5. 构建 Graph ---
builder=StateGraph(State)
builder.add_node("agent",call_model)
builder.add_node("tools",call_tools)
builder.add_edge(START,"agent")
builder.add_conditional_edges("agent",route_after_agent)
builder.add_edge("tools","agent") # tools 执行完回到 agent 继续思考

memory=MemorySaver()

graph=builder.compile(checkpointer=memory)

'''
增加checkpointer=memory，保存同一个session下的对话
start ---->agent(call_model)<-----tools(call_tools)
                |                   | 
                |有tool_call---------
                |
                |没有tool_call---->END
'''

# --- 6. 入口 ---
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