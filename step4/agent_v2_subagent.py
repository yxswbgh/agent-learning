"""Step4 系统B：Sub-Agent 模式 — Supervisor 动态委派

主 Agent（Supervisor）不直接执行任务，而是通过委派工具
动态调用三个子 Agent：Researcher、Writer、Editor。
"""

import uuid
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL
from memory import MemoryManager
from tools import search_tool, read_file_tool, write_file_tool


# --- 1. 创建三个子 Agent（复用流水线的定义）---
def _make_researcher():
    return create_react_agent(
        ChatDeepSeek(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY),
        [search_tool, read_file_tool],
    )


def _make_writer():
    return create_react_agent(
        ChatDeepSeek(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY),
        [read_file_tool, write_file_tool],
    )


def _make_editor():
    return create_react_agent(
        ChatDeepSeek(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY),
        [read_file_tool, write_file_tool],
    )


# 模块级实例化（每个 Sub-Agent 调用时复用）
researcher_graph = _make_researcher()
writer_graph = _make_writer()
editor_graph = _make_editor()



# --- 2. 委派工具 ---

@tool
def delegate_research(topic: str) -> str:
    """委派研究任务给专业研究员。

    研究员可以搜索网络和读取文件，会产出结构化的调研笔记。
    当你需要收集信息、查证事实、了解某个主题时使用此工具。

    Args:
        topic: 研究主题，越具体越好
    """
    prompt = f"""你是一名资深研究员。请研究以下主题并产出结构化的调研笔记。

研究主题：{topic}

要求：
1. 使用搜索工具获取最新信息（至少搜索 2 次）
2. 整理关键发现、数据、观点和趋势
3. 输出结构化的调研笔记（Markdown 格式）
4. 笔记要足够详细，让写手能基于此撰写完整报告"""
    
    result = researcher_graph.invoke(
        {"messages": [HumanMessage(content=prompt)]}
    )
    return result["messages"][-1].content


@tool
def delegate_writing(research_notes: str) -> str:
    """委派写作任务给专业写手。

    写手可以读取文件和写入文件，会基于调研笔记撰写完整报告。
    当你有了足够的调研资料后使用此工具。

    Args:
        research_notes: 调研笔记内容，作为写作素材
    """
    prompt = f"""你是一名专业写手。请根据以下调研笔记撰写一份完整的报告。

调研笔记：
{research_notes[:8000]}

要求：
1. 报告结构清晰：标题 → 摘要 → 正文 → 结论
2. 语言流畅、逻辑严谨
3. 使用 write_file_tool 将报告保存到 output/subagent_report.md"""
    
    result = writer_graph.invoke(
        {"messages": [HumanMessage(content=prompt)]}
    )
    return result["messages"][-1].content


@tool
def delegate_editing(draft: str) -> str:
    """委派编辑任务给资深编辑。

    编辑可以读取文件和写入文件，会审校报告并产出终稿。
    当你有了初稿后，使用此工具进行最终审校。

    Args:
        draft: 报告初稿内容
    """
    prompt = f"""你是一名资深编辑。请审校以下报告并产出终稿。

初稿：
{draft[:8000]}

要求：
1. 检查事实准确性和逻辑一致性
2. 改善表达和结构
3. 修正语法和格式问题
4. 使用 write_file_tool 将终稿保存到 output/subagent_final.md"""
    
    result = editor_graph.invoke(
        {"messages": [HumanMessage(content=prompt)]}
    )
    return result["messages"][-1].content


SUPERVISOR_TOOLS = [delegate_research, delegate_writing, delegate_editing]


# --- 3. 创建 Supervisor Agent ---

SUPERVISOR_SYSTEM_PROMPT = """你是一个多 Agent 系统的调度者（Supervisor）。你有三个专家可以调用：

1. **delegate_research** — 研究员，负责搜索网络、收集信息
2. **delegate_writing** — 写手，基于调研撰写报告
3. **delegate_editing** — 编辑，审校和润色报告

工作流程：
1. 收到用户任务 → 先调用 researcher 收集信息
2. 拿到调研笔记 → 调用 writer 撰写报告
3. 拿到初稿 → 调用 editor 审校产出终稿
4. 如果某个阶段的结果不理想，可以重新委派或要求修改

注意：
- 每一步的结果要传递给下一步
- 如果子 Agent 返回的信息不完整，可以要求补充
- 最终要向用户汇报完成的报告"""

supervisor_model = ChatDeepSeek(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY)
memory = MemorySaver()

supervisor_graph = create_react_agent(
    supervisor_model,
    SUPERVISOR_TOOLS,
    checkpointer=memory,
)


# --- 4. 记忆集成 ---
memory_manager = MemoryManager()

# --- 5. 入口 ---

def main():
    """运行 Sub-Agent 模式。"""
    task = "调研2026年AI Agent开发框架的最新趋势，重点关注LangGraph、CrewAI和AutoGen的对比，最后写一份报告"

    print("=" * 60)
    print("  Sub-Agent 模式: Supervisor → 动态委派")
    print("=" * 60)
    print(f"\n任务: {task}\n")

    config = {"configurable": {"thread_id": str(uuid.uuid4())[:8]}}

    # 检索相关记忆
    memory_context = memory_manager.search(query=task, top_k=3)
    messages = []
    if memory_context:
        messages.append(SystemMessage(content=memory_context))
    messages.append(HumanMessage(content=task))

    result = supervisor_graph.invoke(
        {"messages": messages},
        config=config,
    )

    final_response = result["messages"][-1].content
    print(f"\n{'='*60}")
    print("  Supervisor 最终回复")
    print(f"{'='*60}")
    print(final_response[:1000])

    # 保存记忆
    memory_manager.add(
        [
            HumanMessage(content=f"任务: {task}"),
            HumanMessage(content=f"产出: {final_response[:500]}"),
        ],
        infer=True,
    )


if __name__ == "__main__":
    main()