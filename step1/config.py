"""Step1 配置模块 — 加载 .env，导出常量和 API key。

职责：
1. 调用 load_dotenv() 加载 step1/.env
2. 从环境变量读取 API key
3. 定义 DeepSeek 连接常量
4. 启动时检查 key 是否存在
"""

import os
from  pathlib import Path

from dotenv import load_dotenv

# --- 1. 加载 .env ---
# __file__ 是当前文件的路径:
# .parent 就是 step1/ 目录
env_path=Path(__file__).parent/".env"
load_dotenv(dotenv_path=env_path)

# --- 2. 读取 API key ---
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")

# --- 3. DeepSeek 连接常量 ---
DEEPSEEK_BASE_URL="https://api.deepseek.com"
DEEPSEEK_MODEL="deepseek-chat"


# --- 4. 启动检查 ---
if not DEEPSEEK_API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY 未设置。请检查 step1/.env 文件，"
        "确保 DEEPSEEK_API_KEY=你的key 已填写。"
    )

if not TAVILY_API_KEY:
    raise RuntimeError(
        "TAVILY_API_KEY 未设置。请检查 step1/.env 文件，"
        "确保 TAVILY_API_KEY=你的key 已填写。"
    )
