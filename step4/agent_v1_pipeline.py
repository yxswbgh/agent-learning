"""Step4 系统A：3-Agent 流水线 — Researcher → Writer → Editor

用 LangGraph StateGraph 串行连接三个 create_react_agent 子图。
每个 Agent 有专用 system prompt 和工具集。
"""

import uuid
from typing import Annotated, TypedDict

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL
from memory import MemoryManager
from tools import TOOLS, search_tool, read_file_tool, write_file_tool



# --- 1. 流水线 State ---
class PipelineState(TypedDict):
    messages: Annotated[list, add_messages]  # 当前阶段的消息
    task: str                                 # 原始任务
    research_notes: str                       # Researcher → Writer
    draft: str                                # Writer → Editor
    final_report: str                         # Editor 产出


# --- 2. 创建三个专用模型 ---
model = ChatDeepSeek(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY)

# 研究员：可以搜索和读文件
researcher_agent = create_react_agent(
    ChatDeepSeek(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY),
    [search_tool, read_file_tool],
)


# 写手：可以读文件和写文件
writer_agent = create_react_agent(
    ChatDeepSeek(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY),
    [read_file_tool, write_file_tool],
)

# 编辑：可以读文件和写文件
editor_agent = create_react_agent(
    ChatDeepSeek(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY),
    [read_file_tool, write_file_tool],
)


# --- 3. 流水线节点 ---

def researcher_node(state: PipelineState) -> dict:
    """研究员节点：调研主题 → 产出结构化笔记。"""
    task = state["task"]
    print(f"\n{'='*50}")
    print(f"[流水线] Phase 1: 研究员开始工作")
    print(f"[流水线] 任务: {task}")

    prompt = f"""你是一名资深研究员。请研究以下主题并产出结构化的调研笔记。

调研主题：{task}

要求：
1. 使用搜索工具获取最新信息（至少搜索 2 次）
2. 整理关键发现、数据、观点和趋势
3. 输出结构化的调研笔记（Markdown 格式，包含标题、分类、要点）
4. 笔记要足够详细，让写手能基于此撰写完整报告
5. 用 read_file_tool 读取已有文件作为参考（如需要）"""

    result = researcher_agent.invoke(
        {"messages": [HumanMessage(content=prompt)]}
    )
    notes = result["messages"][-1].content
    print(f"[流水线] 研究员产出: {len(notes)} 字符")
    return {"research_notes": notes}


def writer_node(state: PipelineState) -> dict:
    """写手节点：基于调研笔记 → 撰写完整报告。"""
    notes = state.get("research_notes", "")
    print(f"\n{'='*50}")
    print(f"[流水线] Phase 2: 写手开始工作")

    prompt = f"""你是一名专业写手。请根据以下调研笔记撰写一份完整的报告。

调研笔记：
{notes[:8000]}  # 截断防止上下文溢出

要求：
1. 报告结构清晰：标题 → 摘要 → 正文（分章节）→ 结论
2. 语言流畅、逻辑严谨、面向非技术读者
3. 使用 write_file_tool 将初稿保存到 output/draft_report.md
4. 保留调研笔记中的关键数据和引用来源"""

    result = writer_agent.invoke(
        {"messages": [HumanMessage(content=prompt)]}
    )
    draft = result["messages"][-1].content
    print(f"[流水线] 写手产出: {len(draft)} 字符")
    return {"draft": draft}


def editor_node(state: PipelineState) -> dict:
    """编辑节点：审校初稿 → 产出终稿。"""
    draft = state.get("draft", "")
    print(f"\n{'='*50}")
    print(f"[流水线] Phase 3: 编辑开始工作")

    prompt = f"""你是一名资深编辑。请审校以下报告并产出终稿。

初稿：
{draft[:8000]}

要求：
1. 检查事实准确性和逻辑一致性
2. 改善表达流畅度和结构
3. 修正语法错误和格式问题
4. 使用 write_file_tool 将终稿保存到 output/final_report.md
5. 确认初稿中所有需要保留的关键信息都已包含"""

    result = editor_agent.invoke(
        {"messages": [HumanMessage(content=prompt)]}
    )
    final = result["messages"][-1].content
    print(f"[流水线] 编辑产出: {len(final)} 字符")
    return {"final_report": final}



# --- 3.5 记忆节点 ---
memory_manager = MemoryManager()


def recall_node(state: PipelineState) -> dict:
    """流水线开始前检索相关记忆，注入为 system message。"""
    task = state.get("task", "")
    memory_text = memory_manager.search(query=task, top_k=3)

    if memory_text:
        print(f"[记忆] 检索到相关记忆，注入上下文")
        return {
            "messages": [
                SystemMessage(
                    content=f"以下是关于这位用户的已知信息，请在执行任务时参考：\n\n{memory_text}"
                )
            ]
        }
    return {}


def save_memory_node(state: PipelineState) -> dict:
    """流水线结束后保存对话记忆。"""
    # 收集所有阶段的输出作为记忆来源
    messages_to_save = []
    task = state.get("task", "")
    if task:
        messages_to_save.append(
            HumanMessage(content=f"任务: {task}")
        )

    final = state.get("final_report", "")
    if final:
        messages_to_save.append(
            HumanMessage(content=f"产出报告摘要: {final[:500]}")
        )

    if messages_to_save:
        memory_manager.add(messages_to_save, infer=True)

    return {}


# --- 4. 构建流水线图 ---

builder = StateGraph(PipelineState)
builder.add_node("recall", recall_node)
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)
builder.add_node("editor", editor_node)
builder.add_node("save_memory", save_memory_node)

builder.add_edge(START, "recall")
builder.add_edge("recall", "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", "editor")
builder.add_edge("editor", "save_memory")
builder.add_edge("save_memory", END)

pipeline_graph = builder.compile()



# --- 5. 入口 ---

def main():
    """运行 3-Agent 流水线。"""
    task = "调研2026年AI Agent开发框架的最新趋势，重点关注LangGraph、CrewAI和AutoGen的对比"

    print("=" * 60)
    print("  3-Agent 流水线: Researcher → Writer → Editor")
    print("=" * 60)
    print(f"\n任务: {task}\n")

    config = {"configurable": {"thread_id": str(uuid.uuid4())[:8]}}

    result = pipeline_graph.invoke(
        {"task": task},
        config=config,
    )

    print(f"\n{'='*60}")
    print("  流水线执行完成")
    print(f"{'='*60}")
    print(f"调研笔记: {len(result.get('research_notes', ''))} 字符")
    print(f"初稿: {len(result.get('draft', ''))} 字符")
    print(f"终稿: {len(result.get('final_report', ''))} 字符")

    # 输出终稿预览
    final = result.get("final_report", "")
    if final:
        print(f"\n--- 终稿预览 (前 500 字符) ---")
        print(final[:500])


if __name__ == "__main__":
    main()