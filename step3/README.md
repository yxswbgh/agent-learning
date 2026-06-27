# Step 3: Agent 记忆系统

> 给 Agent 加上两层记忆：会话记忆（同一线程）和持久记忆（跨进程存活）。

## 目标

让 Agent 拥有"记性"：
- **短期**：同一会话内，Agent 记得你刚说过什么（ChatGPT 式的上下文）
- **长期**：重启进程后，Agent 仍然记得你的名字、偏好、之前聊过的话题

## 技术栈

| 组件 | 选型 | 原因 |
|------|------|------|
| LLM | DeepSeek (OpenAI 兼容) | 同前两步 |
| Agent 框架 | LangGraph | StateGraph + checkpointer + 子图组合 |
| 会话记忆 | LangGraph `MemorySaver` | 内置的 messages 持久化，零代码集成 |
| 结构化事实 | SQLite (`UserFactsStore`) | 精确匹配用户属性（姓名、职位、偏好） |
| 语义记忆 | ChromaDB (`ChromaMemoryStore`) | 向量检索对话摘要，支持模糊语义搜索 |
| 嵌入模型 | DeepSeek Embedding (`text-embedding-3-large` via proxy) | 与 LLM 同厂商，成本最低 |
| 记忆提取 | LLM 本身 | 用 LLM 从对话中抽取可存储的事实和摘要 |

## 架构

### 两层记忆系统

```
┌─────────────────────────────────────────────────────┐
│                   Layer 1: 会话记忆                    │
│  MemorySaver (in-process state)                      │
│  生命周期：同一 thread_id + 同一进程                    │
│  存储内容：完整 messages 列表                          │
│  API: config={"configurable":{"thread_id":"..."}}     │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                  Layer 2: 持久记忆                     │
│  跨进程存活，存于磁盘                                  │
│  ┌──────────────────┐  ┌───────────────────────────┐ │
│  │  UserFactsStore   │  │   ChromaMemoryStore       │ │
│  │  (SQLite)         │  │   (ChromaDB + Embedding)  │ │
│  │  结构化 key-value  │  │   语义向量搜索              │ │
│  │  "name=张三"       │  │   "上次聊过 AI Agent..."    │ │
│  │  模糊精确匹配      │  │   相关性排序                │ │
│  └──────────────────┘  └───────────────────────────┘ │
│                    MemoryManager                      │
│           (LLM 提取事实 + 生成摘要)                     │
└─────────────────────────────────────────────────────┘
```

### Agent 执行流程

```
START
  │
  ▼
┌──────────┐    1. 从 UserFactsStore + Chroma 检索相关记忆
│  recall  │       拼接成 "previous memories: ..." 前缀
└────┬─────┘
     │
     ▼
┌──────────┐    2. 原有的 ReAct 循环（Step 2 的完整 StateGraph 作为子图）
│  react   │       LLM 在思考时已经能看到前面检索到的记忆
└────┬─────┘
     │
     ▼
┌──────────────┐  3. LLM 从最终对话中提取新事实 + 生成摘要
│ save_memory  │     分别写入 SQLite 和 ChromaDB
└──────┬───────┘
       │
       ▼
      END
```

**关键设计**：记忆的召回和存储不是 LLM 的 tool，而是 Graph 的节点。这保证了：
- LLM 在思考时**必然**能看到记忆（不需要它自己"记得"去查）
- 记忆存储**必然**在每轮对话结束后执行（不依赖 LLM 的判断）

## 四个 Agent 变体

| 变体 | 文件 | 做了什么 |
|------|------|----------|
| V1 | `agent_v1_session.py` | Step2 V1 + `MemorySaver`，单会话内记忆 |
| V2 | `agent_v2_session.py` | Step2 V2 + `MemorySaver`，手写图 + 会话记忆 |
| V3 | `agent_v3_memory.py` | V1 + recall/react/save_memory 三层管道，完整持久记忆 |
| V4 | `agent_v4_memory_handcrafted.py` | V2 + 三层管道，最完整的手动实现 |

**学习路径**：V1 → V2（理解 MemorySaver）→ V3（理解记忆管道架构）→ V4（理解如何在手写图上扩展）

## 文件说明

```
step3/
├── config.py                      # 配置 + 记忆相关常量（CHROMA_DIR, EMBEDDING_MODEL...）
├── tools.py                       # 与 Step2 相同的三个工具
├── memory.py                      # 记忆引擎核心（290 行）
│   ├── UserFactsStore             #   SQLite CRUD + 模糊搜索
│   ├── ChromaMemoryStore          #   ChromaDB 向量存储 + 语义搜索
│   └── MemoryManager              #   协调器：LLM 提取事实 + 写摘要
├── agent_v1_session.py            # Phase A: prebuilt + MemorySaver
├── agent_v2_session.py            # Phase A: handcrafted + MemorySaver
├── agent_v3_memory.py             # Phase B: prebuilt + 三层管道
├── agent_v4_memory_handcrafted.py # Phase B: handcrafted + 三层管道（最完整）
├── chroma_data/                   # ChromaDB 持久化目录（.gitignore 忽略）
├── MEMORY_NOTES.md                # MemorySaver vs 持久记忆的对比笔记
└── README.md                      # 本文件
```

## 运行

```bash
# 1. 安装依赖
pip install langgraph langchain langchain-openai chromadb python-dotenv

# 2. 配置
cp .env.example .env
# 编辑 .env：
#   DEEPSEEK_API_KEY=
#   TAVILY_API_KEY=
#   EMBEDDING_API_KEY=   # 嵌入模型的 API Key（本项目中通过代理使用）
#   EMBEDDING_BASE_URL=  # 嵌入模型端点

# 3. 运行（按学习顺序）
cd step3

# Phase A — 体验会话记忆
python agent_v1_session.py    # 预置的两次对话，第二次 Agent 能引用第一次的内容

# Phase B — 体验持久记忆
python agent_v3_memory.py     # 第一次对话告诉 Agent 你的信息
                               # 第二次对话（新 thread_id）Agent 能回忆起来
```

## 关键知识点

### 1. MemorySaver 的代价

`MemorySaver` 把所有 messages 原样存在内存里。对话长 → 状态大 → 每次 LLM 调用都要带全量历史 → Token 消耗线性增长。这是为什么需要 Layer 2 的**主动召回**：只把相关的记忆注入 context，而不是全量携带。

### 2. 为什么记忆不是 Tool

一种常见设计是把"记东西"做成 `save_memory` tool，让 LLM 自己决定何时调用。问题：
- LLM 可能"忘了"去记
- LLM 可能记错（提取了无用信息）
- 不可靠

本项目的选择是：**记忆是 Graph 的节点，必定执行**。LLM 只是用来格式化要存的内容，但"存不存"这个决策不在 LLM 手里。

### 3. SQLite + ChromaDB 双存储的取舍

| | SQLite (UserFactsStore) | ChromaDB |
|---|---|---|
| 查什么 | 精确事实：姓名、职位、偏好 | 模糊语义：话题、观点、总结 |
| 怎么查 | SQL LIKE / = | 向量相似度 |
| 为什么不用一个 | SQL 做语义搜索是灾难 | Chroma 做精确匹配是浪费 |

### 4. 嵌入模型的代理

本项目通过 HTTP 代理将 `text-embedding-3-large` 格式的请求转发给 DeepSeek Embedding。这是踩过的坑 — 在 `MEMORY_NOTES.md` 中有记录。

## 验收标准

**最终验证**：运行 V3 或 V4，执行两次对话：
1. 第一次："我叫张三，我是 Python 后端工程师，我在学习 AI Agent"
2. 第二次（新 thread_id，模拟新会话）："我叫什么名字？我是做什么的？"

Agent 应该能回答出 "张三" 和 "Python 后端工程师"。

---

**上一步**：[Step 2 — 完整 ReAct 循环](../step2/)
