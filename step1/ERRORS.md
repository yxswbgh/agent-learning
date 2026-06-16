# Step1 报错记录与解决方案

> 记录 step1 实现过程中遇到的每个报错、根因、修复方法。  
> 目的：巩固调试能力，避免重复踩坑。

---

## 1. tools.py — Tavily SSL 连接失败

**报错信息：**
```
搜索失败: SSLError — HTTPSConnectionPool(host='api.tavily.com', port=443):
Max retries exceeded with url: /search
(Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in vi'))
```

**根因：**
- WSL2 环境通过代理 `http://127.0.0.1:7897` 访问外网（环境变量 `https_proxy` 已设置）
- `curl` 会自动读取代理环境变量，所以 `curl https://api.tavily.com` 能通
- `TavilyClient` 底层用 `httpx`，不走系统代理，直连失败

**诊断手段：**
```bash
# 1. 确认代理环境变量存在
env | grep -i proxy

# 2. 确认 curl 能过代理
curl -v https://api.tavily.com 2>&1 | head -20

# 3. 确认 Python 直接请求不通
python -c "import urllib.request; urllib.request.urlopen('https://api.tavily.com')"

# 4. 确认 Python 过代理能通
python -c "
import urllib.request
proxy = urllib.request.ProxyHandler({'https': 'http://127.0.0.1:7897'})
opener = urllib.request.build_opener(proxy)
resp = opener.open('https://api.tavily.com', timeout=10)
print('OK:', resp.status)
"
```

**修复：**
`TavilyClient.__init__` 支持 `proxies` 参数，传入代理配置：

```python
import os

client = TavilyClient(
    api_key=TAVILY_API_KEY,
    proxies={"https://": os.getenv("https_proxy")} if os.getenv("https_proxy") else None,
)
```

**学到的原则：**
- HTTP 库（Python requests/httpx）不等于系统工具（curl）。curl 知道系统代理设置，Python HTTP 库默认不知道
- 设计 SDK 调用时，要显式处理网络配置（代理、超时、重试），不要依赖"默认能通"
- 从环境变量读取代理配置，而不是硬编码——环境换了（CI、Docker）只需改变量，不改代码

---

## 2. agent.py — Pydantic AI API 变更：模型构造方式

**报错信息：**
```
TypeError: OpenAIModel.__init__() got an unexpected keyword argument 'base_url'
```

**根因：**
- Pydantic AI 1.107.0 重构了模型初始化 API
- 旧版：`OpenAIModel(model_name=..., base_url=..., api_key=...)` — 手动传入所有连接参数
- 新版：`OpenAIModel(model_name=..., provider="deepseek")` — 通过 provider 名称字符串，SDK 自动管理连接

**诊断手段：**
```bash
# 1. 确认当前版本
pip show pydantic-ai | grep Version

# 2. 查看新 API 签名
python -c "from pydantic_ai.models.openai import OpenAIModel; help(OpenAIModel.__init__)"

# 3. 确认内置 provider 列表
python -c "from pydantic_ai.providers import infer_provider_class; help(infer_provider_class)"
```

**修复：**
```python
# 旧版（已废弃）
model = OpenAIModel(
    model_name="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key="sk-xxx",
)

# 新版（正确）
model = OpenAIModel(
    model_name="deepseek-chat",
    provider="deepseek",
)
# provider="deepseek" 会自动：
# - 从环境变量 DEEPSEEK_API_KEY 读取 key
# - 使用正确的 base_url (https://api.deepseek.com)
# - 配置 DeepSeek 特有的模型特性
```

**学到的原则：**
- 升级依赖后，先看 CHANGELOG 和 docstring——不要假设 API 不变
- Pydantic AI 1.x 用 `provider` 字符串统一管理不同 LLM 的接入——以后加其他模型（如 Anthropic）也是 `provider="anthropic"`，不变的是调用方式
- 框架内建的 provider 比自己拼接参数更可靠——它知道每个模型的 quirks

---

## 3. agent.py — `AgentRunResult.data` 不存在

**报错信息：**
```
AttributeError: 'AgentRunResult' object has no attribute 'data'
```

**根因：**
- Pydantic AI 1.107.0 把 `AgentRunResult` 从自定义类重构为 dataclass
- 输出字段从 `.data` 改名为 `.output`

**诊断手段：**
```python
# 查看 AgentRunResult 的 dataclass 字段
import dataclasses
from pydantic_ai.agent import AgentRunResult
dataclasses.fields(AgentRunResult)
# → output: OutputDataT, _output_tool_name: str | None, _state: ..., ...
```

**修复：**
```python
# 旧版
print(result.data)

# 新版
print(result.output)
```

**学到的原则：**
- 主版本号（0.x → 1.x）意味着 breaking changes，升级后要跑一遍所有调用路径
- 遇到 `AttributeError` 先 `dir()` 对象，一般能直接找到替代字段
- 也可以用 `dataclasses.fields()` 检查 dataclass 的正式字段列表

---

## 故障排查模式总结

| 步骤 | 做法 |
|------|------|
| 1. 缩小范围 | 是网络问题还是代码问题？是 SDK 版本问题还是调用方式问题？ |
| 2. 分而治之 | 本机能通吗？curl 能通吗？Python 裸库能通吗？SDK 单独能通吗？ |
| 3. 查 API 签名 | `help()`, `dir()`, `inspect.getsource()`, `dataclasses.fields()` |
| 4. 改最小量 | 不重构整个文件，只改出问题的那一行 |
| 5. 验证修复 | 改完立即跑最小复现用例 |
