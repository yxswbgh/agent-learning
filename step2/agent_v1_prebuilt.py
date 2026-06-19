"""Step2 Agent V1 — 使用 LangGraph create_react_agent 预构建版。

学习目标：看懂 LangGraph 帮你隐藏了什么（对照 V2 手写版）。
"""
from config import DEEPSEEK_MODEL,DEEPSEEK_API_KEY
from tools import TOOLS

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

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
graph=create_react_agent(model,TOOLS)

# --- 3. 入口 ---
def main():
    task=(
        "请帮我做一个简单的调研：搜索 2026 年 AI 领域最重要的技术趋势，"
        "整理成一份简短的报告，保存到 ./output/ai-trends-2026.md。"
        "要求：先搜索获取信息，再整理写入文件。"
        "注意：禁止删除任何文件！！！"
    )
    print(f"任务: {task}\n")
    print("Agent 运行中...\n")

    result=graph.invoke({"messages":[HumanMessage(content=task)]})

    # 打印最后一条消息（最终回答）
    final=result["messages"][-1]
    print(f"\n{'='*50}")
    print(f"最终输出:\n{final.content}")
    print(f"{'='*50}")


if __name__=="__main__":
    main()