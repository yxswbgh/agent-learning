"""Step2 配置模块 — 加载 .env，配置 LangSmith Trace，导出常量和 API key。"""

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
ZHIPU_API_KEY=os.getenv("ZHIPU_API_KEY")

# --- 3. 常量 ---
DEEPSEEK_MODEL = "deepseek-chat"
OUTPUT_DIR = Path(__file__).parent / "output"

# --- 4. 启动检查 ---
if not DEEPSEEK_API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY 未设置。请检查 step2/.env 文件。"
    )
if not TAVILY_API_KEY:
    raise RuntimeError(
        "TAVILY_API_KEY 未设置。请检查 step2/.env 文件。"
    )

# --- 5. LangSmith 配置 ---
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    print("[config] LangSmith tracing 已开启")
else:
    print("[config] LANGSMITH_API_KEY 未设置，跳过 LangSmith tracing")


# --- 6. 记忆系统常量 ---
EMBEDDING_MODEL = "embedding-3"

CHROMA_DIR = str(Path(__file__).parent / "chroma_data")