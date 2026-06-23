import os
from langchain_core.tools import tool
from tavily import TavilyClient

from config import TAVILY_API_KEY

from pathlib import Path

import subprocess
from config import OUTPUT_DIR
MAX_READ_LINES = 200  # 大文件截断，防止上下文溢出


# TavilyClient 提到模块级别，每个 tool 调用共享同一个实例
_tavily = TavilyClient(
    api_key=TAVILY_API_KEY,
    proxies={"https://": os.getenv("https_proxy")} if os.getenv("https_proxy") else None,
)


@tool
def search_tool(query: str) -> str:
    """搜索网络获取最新信息。当需要查证事实、获取实时数据时使用。

    铁律1 — 粒度：只执行一次搜索，不负责整理/归纳/写报告。
    铁律2 — 副作用：只读操作，天然安全。
    铁律3 — 错误返回：搜索失败时返回失败原因和改进建议。

    Args:
        query: 搜索关键词，中英文均可。越具体越好。
    """
    try:
        response = _tavily.search(query)
    except Exception as e:
        return (
            f"搜索失败: {type(e).__name__}。\n"
            f"建议：1) 检查网络连接 2) 尝试缩短关键词 3) 换用英文关键词重试\n"
            f"原始错误: {e}"
        )

    results = response.get("results", [])
    if not results:
        return (
            f'搜索 "{query}" 没有返回结果。\n'
            f"建议：1) 尝试更通用的关键词 2) 去掉特殊符号 3) 改用英文搜索"
        )

    lines = []
    for i, item in enumerate(results, start=1):
        title = item.get("title", "无标题")
        url = item.get("url", "")
        content = item.get("content", "")
        lines.append(f"{i}. {title}\n   URL: {url}\n   {content}\n")

    return "\n".join(lines)


@tool
def read_file_tool(path: str) -> str:
    """读取本地文件内容。当你需要查看已有文件、或其他工具的输出结果时使用。

    铁律1 — 粒度：只读一个文件，不读整个目录。
    铁律2 — 副作用：只读操作，天然安全。
    铁律3 — 错误返回：文件不存在时列出同目录相似文件，给出"did you mean?"建议。

    Args:
        path: 文件路径，可以是相对路径或绝对路径。
    """
    file_path = Path(path).expanduser().resolve()

    if not file_path.exists():
        # 铁律3：列出同目录文件，给出 did you mean 建议
        parent = file_path.parent
        if parent.exists():
            siblings = [f.name for f in parent.iterdir() if f.is_file()]
            if siblings:
                hint = f"目录 {parent} 中有以下文件：{', '.join(siblings)}"
            else:
                hint = f"目录 {parent} 是空的"
        else:
            hint = f"目录 {parent} 也不存在"

        return (
            f"错误：文件不存在 — {file_path}\n"
            f"{hint}\n"
            f"建议：检查路径拼写是否正确，或先用 search_tool 确认文件位置。"
        )

    if not file_path.is_file():
        return (
            f"错误：路径存在但不是文件 — {file_path}\n"
            f"建议：这是一个目录，请指定具体文件名。"
        )

    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        total = len(lines)

        if total > MAX_READ_LINES:
            truncated = "\n".join(lines[:MAX_READ_LINES])
            return (
                f"{truncated}\n\n"
                f"--- 文件共 {total} 行，已截断显示前 {MAX_READ_LINES} 行 ---\n"
                f"建议：如需查看后续内容，请指定更小的范围或多次读取。"
            )

        return content

    except UnicodeDecodeError:
        return (
            f"错误：无法以 UTF-8 编码读取 — {file_path}\n"
            f"建议：此文件可能是二进制文件，请使用其他工具处理。"
        )
    except PermissionError:
        return (
            f"错误：权限不足，无法读取 — {file_path}\n"
            f"建议：检查文件权限（ls -la）或使用有权限的用户运行。"
        )
    

@tool
def write_file_tool(path: str, content: str, dry_run: bool = False) -> str:
    """将内容写入文件。当你需要保存搜索结果、生成报告、记录信息时使用。

    铁律1 — 粒度：只写一个文件。
    铁律2 — 副作用可逆：dry_run=True 时只预览不写入；写前自动 git stash 备份。
    铁律3 — 错误返回：目录不存在时提示创建；权限不足时提示路径。

    Args:
        path: 文件路径。如果只写文件名，默认保存到 ./output 目录。
        content: 要写入的内容（Markdown 格式最佳）。
        dry_run: 设为 True 只预览不写入，用于检查效果。默认 False。
    """
    file_path = Path(path)

    # 相对路径默认放到 output/ 下
    if not file_path.is_absolute():
        #file_path = OUTPUT_DIR / file_path
        file_path=file_path #trace时发现目录重复一次

    # --- Dry-run 模式 ---
    if dry_run:
        preview = content[:500] + ("..." if len(content) > 500 else "")
        return (
            f"[DRY-RUN] 将写入 {file_path}，内容预览：\n"
            f"---\n{preview}\n---\n"
            f"共 {len(content)} 字符，{len(content.splitlines())} 行。\n"
            f"确认无误后请设置 dry_run=False 执行实际写入。"
        )

    # --- 目录检查 ---
    parent = file_path.parent
    if not parent.exists():
        return (
            f"错误：目录不存在 — {parent}\n"
            f"建议：请先执行 mkdir -p {parent} 创建目录，或换一个有权限的路径。"
        )

    # --- Git 备份 ---
    '''
    stash_message = f"step2-auto-stash-before-write-{file_path.name}"
    try:
        subprocess.run(
            ["git", "-C", str(OUTPUT_DIR.parent), "stash", "push", "--include-untracked",
             "-m", stash_message],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        pass  # git 不可用时跳过备份，不阻塞写入
    '''

    # --- 写入 ---
    try:
        file_path.write_text(content, encoding="utf-8")
    except PermissionError:
        return (
            f"错误：权限不足，无法写入 — {file_path}\n"
            f"建议：检查目标目录的写权限，或换一个路径。"
        )
    except Exception as e:
        return (
            f"错误：写入失败 — {type(e).__name__}: {e}\n"
            f"文件：{file_path}\n"
            f"建议：检查磁盘空间、路径合法性。"
        )

    return (
        f"写入成功: {file_path}\n"
        f"大小: {len(content)} 字符, {len(content.splitlines())} 行\n"
        f"回滚方法: git stash pop (恢复写入前的状态)"
    )

# 工具列表 — agent 文件直接 import 这个
TOOLS = [search_tool, read_file_tool, write_file_tool]