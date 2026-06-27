# Step 2: 完整 ReAct 循环

> 用 LangGraph 手写一个完整的 Reasoning + Acting 循环，让 Agent 能执行多步任务。

## 目标

从 Step 1 的单轮 tool calling，升级到**真正的 ReAct 循环**：Agent 可以在一个任务中多次思考、多次调用工具、最终产出结果。典型场景：搜索资料 → 整理 → 写报告。

## 技术栈

| 组件 | 选型 | 原因 |
|------|------|------|
| LLM | DeepSeek (OpenAI 兼容) | 同 Step 1 |
| Agent 框架 | LangGraph | 显式构建 StateGraph，比 Pydantic AI 更底层、更可控 |
| 工具定义 | LangChain `@tool` | 自动生成 tool schema，与 LangGraph 无缝集成 |
| 追踪 | LangSmith | 可视化 Agent 的每一步决策 |

## 架构

### ReAct 循环

```
         ┌──────────────────────────────────────┐
         │                                      │
         ▼                                      │
    ┌─────────┐    tool_calls     ┌──────────┐  │
    │call_model├─────────────────►│call_tools │  │
    │  (LLM)  │                  │  (执行)   │  │
    └────┬─────┘                  └────┬─────┘  │
         │                             │        │
         │ 无 tool_calls               │        │
         │ (最终文本回复)               └────────┘
         ▼                               (回到 call_model)
       END
```

每个循环中：
1. `call_model` — LLM 接收 messages，返回文本回复或 tool_call 请求
2. `route_after_agent` — 条件判断：有 tool_calls → 去 `call_tools`；无 → 结束
3. `call_tools` — 执行工具，结果追加到 messages，回到 `call_model`

### 两种实现对比

| | V1: Prebuilt | V2: Handcrafted |
|---|---|---|
| 文件 | `agent_v1_prebuilt.py` | `agent_v2_handcrafted.py` |
| 代码行数 | ~20 行 | ~80 行 |
| 核心调用 | `create_react_agent(model, tools)` | 手动定义 StateGraph + nodes + edges |
| 透明度 | 黑盒 | 完全透明 |
| 适用场景 | 快速原型 | 生产定制 |

学习策略：先用 V1 理解"LangGraph 做了什么"，再用 V2 亲手实现每一个节点和边。

## 三个工具与"铁律"

```python
TOOLS = [search_tool, read_file_tool, write_file_tool]
```

每个工具的设计遵循三条"铁律"：

| 铁律 | 含义 | 示例 |
|------|------|------|
| **粒度** | 每个工具只做一件事，人类能理解的最小语义单位 | `search` 只搜不写，`write_file` 只写不搜 |
| **可逆性** | 读操作不产生副作用，写操作尽可能校验 | `write_file` 先打印确认 dir 存在，失败返回原因 |
| **错误返回** | 出错了返回"这是什么错误 + 建议"，而不是抛异常 | 返回 `f"搜索失败: {e}。建议检查网络连接"` |

## 文件说明

```
step2/
├── config.py                 # 加载配置 + LangSmith 自动追踪
├── tools.py                  # 三个 LangChain @tool + TOOLS 列表
├── agent_v1_prebuilt.py      # V1: 一行 create_react_agent()
├── agent_v2_handcrafted.py   # V2: 手写 StateGraph (call_model + call_tools + router)
├── output/
│   └── ai-trends-2026.md    # Agent 生成的示例报告（155 行 Markdown）
└── README.md                 # 本文件
```

## 运行

```bash
# 1. 安装依赖
pip install langgraph langchain langchain-openai python-dotenv tavily-python

# 2. 配置（可选：LangSmith 追踪）
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY, TAVILY_API_KEY
# （可选）LANGSMITH_API_KEY — 用于可视化 Agent 执行过程

# 3. 运行
cd step2
python agent_v1_prebuilt.py     # 快速体验
python agent_v2_handcrafted.py   # 理解内部机制
```

## 关键知识点

### 1. 为什么需要 StateGraph

ReAct 循环是一个**有状态的循环图**：
- `State` 是 `{"messages": [...]}` — 贯穿所有节点的共享状态
- `add_node` 添加处理节点，`add_edge` / `add_conditional_edges` 定义流程
- 条件路由的核心代码只有 3 行：

```python
def route_after_agent(state: State):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "call_tools"
    return END
```

### 2. Prebuilt vs Handcrafted 的取舍

- **Prebuilt** 适合原型验证。但它把 tool 执行逻辑、消息拼接、循环控制全藏起来了
- **Handcrafted** 让你拥有完全控制权：可以在任意节点前后插入逻辑（这就是 Step 3 要做的 — 在 LLM 调用前插入记忆召回）

### 3. LangSmith 的价值

在 `.env` 中填入 `LANGSMITH_API_KEY`，每次 Agent 运行都会生成完整的执行追踪：
- 每个 LLM 调用的 prompt/completion
- 每个 tool_call 的参数和返回值
- 整个 graph 的节点流转路径

这对调试 Agent 的"为什么调了这个工具"、"为什么没有调那个工具"问题至关重要。

## 验收标准

运行 V2，Agent 能成功执行多步任务（如 "搜索 2026 年 AI 趋势，整理成 Markdown 报告并写入文件"），输出文件内容正确完整。

---

**上一步**：[Step 1 — 最小 Agent](../step1/) | **下一步**：[Step 3 — Agent 记忆系统](../step3/)
