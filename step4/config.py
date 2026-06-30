"""Step4 配置模块 — 加载 .env，配置 LangSmith Trace，记忆系统常量。"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --- 1. 加载 .env ---
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# --- 2. 读取 API key ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")

# --- 3. 常量 ---
DEEPSEEK_MODEL = "deepseek-chat"
OUTPUT_DIR = Path(__file__).parent / "output"

# --- 代理环境变量大小写同步 ---
# httpx/openai SDK 只认大写的 HTTPS_PROXY / HTTP_PROXY
_proxy_url = os.environ.get("https_proxy") or os.environ.get("http_proxy")
if _proxy_url:
    os.environ.setdefault("HTTPS_PROXY", _proxy_url)
    os.environ.setdefault("HTTP_PROXY", _proxy_url)


# --- 4. 启动检查 ---
if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY 未设置。请检查 step4/.env 文件。")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY 未设置。请检查 step4/.env 文件。")

# --- 5. LangSmith 配置 ---
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    print("[config] LangSmith tracing 已开启")
else:
    print("[config] LANGSMITH_API_KEY 未设置，跳过 LangSmith tracing")

# --- 6. 记忆系统常量 ---
EMBEDDING_MODEL = "Embedding-V1"
EMBEDDING_BASE_URL = "https://nangeai.top/v1"
CHROMA_DIR = str(Path(__file__).parent / "chroma_data")
MEMORY_COLLECTION = "memories"  # step4 新 collection，与 step3 的 conversation_summaries 区分