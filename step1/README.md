# Step 1: Minimal Agent

> 用最少的代码跑通一个能搜索网页的 AI Agent。

## 目标

验证"LLM + Tool Calling"的最小闭环：给模型一个搜索工具，让它能回答需要实时信息的问题。

## 技术栈

| 组件 | 选型 | 原因 |
|------|------|------|
| LLM | DeepSeek (OpenAI 兼容) | 便宜、中文友好、Function Calling 稳定 |
| Agent 框架 | Pydantic AI | 高层抽象，自动生成 tool schema，管理对话上下文 |
| 搜索工具 | Tavily Search API | 专为 AI Agent 设计的搜索 API |

## 架构

```
用户问题 → Agent(model + tools) → LLM 决定是否调 tool
                                    ├── 需要搜索 → 调用 search_tavily() → 拿到结果 → LLM 生成答案
                                    └── 不需要搜索 → 直接生成答案
```

这是一个**单轮 ReAct**：LLM 最多调一次工具就返回最终答案。完整的循环式 ReAct 见 Step 2。

## 文件说明

```
step1/
├── config.py         # 加载 .env，验证 API Key
├── tools.py          # search_tavily() — 唯一的工具函数
├── agent.py          # 组装 Agent 并运行示例问题
├── .env.example      # API Key 模板
├── ERRORS.md         # 踩坑记录（SSL 代理、Pydantic AI API 变更）
└── README.md         # 本文件
```

## 运行

```bash
# 1. 安装依赖（在项目根目录）
pip install pydantic-ai python-dotenv tavily-python openai

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY 和 TAVILY_API_KEY

# 3. 运行
cd step1
python agent.py
```

## 关键知识点

### 1. Pydantic AI 隐藏了什么

- **Tool Schema 生成**：框架自动从函数签名 + docstring 生成 OpenAI function calling 格式
- **对话管理**：`Agent.run()` 内部处理了 messages 拼接、tool_call 解析、tool_result 回传
- **重试**：默认自动重试一次失败的 LLM 调用

### 2. 为什么"10 行代码"就够了

因为 Pydantic AI 承担了所有胶水代码。没有它，你需要手写：
- OpenAI API 调用 + tool_call 解析
- 工具执行调度
- 多轮对话的 messages 管理

这正是 Step 2 要亲手做的事情。

### 3. 踩坑记录

详见 [ERRORS.md](./ERRORS.md)，核心三个坑：
- WSL2 下 HTTPS 代理导致 SSL 错误 → 需要 `verify=False` 或正确配代理
- Pydantic AI 从 `data` 属性变更为 `output` → API 不稳定是正常现象
- `AgentRunResult.data` 不存在 → 看文档比看博客靠谱

## 验收标准

运行 `agent.py`，能正确回答 "2026 年世界杯在哪里举办？"，答案中包含通过搜索获取的事实信息。

---

**下一步**：[Step 2 — 完整 ReAct 循环](../step2/)
