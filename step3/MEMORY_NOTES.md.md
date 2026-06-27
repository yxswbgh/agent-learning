# Step3 记忆系统学习笔记

## 两层记忆的区别

| | Session Memory | Persistent Memory |
|---|---|---|
| 实现 | MemorySaver | UserFactsStore + Chroma |
| 存储 | 内存 | SQLite + Chroma |
| 生命周期 | 进程内 | 持久化 |
| 检索方式 | 自动追加 messages | recall 节点主动检索 |
| 适合 | 连贯对话 | 跨天/跨设备 |

## 关键 API

- `MemorySaver()` — 零成本 session 记忆
- `config = {"configurable": {"thread_id": "xxx"}}` — 同一个 thread_id 关联同一 session
- `add_messages` reducer — 新消息追加，不替换
- Chroma `PersistentClient` — 本地持久化
