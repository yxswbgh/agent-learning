"""Step1 Agent 入口 — 组装模型、工具、Agent，运行并输出结果。"""
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from config import DEEPSEEK_API_KEY,DEEPSEEK_BASE_URL,DEEPSEEK_MODEL
from tools import search_tavily

# --- 1. 创建模型 ---
# provider='deepseek' 使用 Pydantic AI 内建的 DeepSeekProvider
# 它会自动从环境变量读 DEEPSEEK_API_KEY，base_url 指向 https://api.deepseek.com
model=OpenAIChatModel(
    model_name=DEEPSEEK_MODEL,
    provider="deepseek"
)

# --- 2. 创建 Agent ---
agent=Agent(
    model=model,
    system_prompt=(
        "你是一个智能助手，可以搜索网络获取最新信息。"
        "当你不确定答案、需要查证事实、或问题涉及实时信息时，"
        "使用 search_tavily 工具搜索。"
        "基于搜索结果给出准确、简洁的回答。"
    ),
    tools=[search_tavily]
)

# --- 3. 入口 ---
def main():
    question="2026年世界杯在哪里举办？"
    print(f"问题: {question}\n")
    print("Agent 运行中...\n")

    result=agent.run_sync(question)
    print(result.output)

if __name__=="__main__":
    main()