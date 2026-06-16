"""Step1 工具模块 — Tavily 网络搜索。

Pydantic AI 会自动根据函数签名和 docstring 生成 tool schema，
所以函数名、参数类型、docstring 就是你的工具定义，不需要额外配置。
"""

from config import TAVILY_API_KEY
from tavily import TavilyClient
import os

async def search_tavily(query:str)->str:
    """搜索网络获取最新信息。当你需要查证事实或获取实时数据时使用此工具。

    Args:
        query: 搜索关键词，用中文或英文均可。
    """

    client=TavilyClient(api_key=TAVILY_API_KEY,
                        proxies={"https://":os.getenv("https_proxy")} if os.getenv("https_proxy") else None
                        ) 

    try:
        response=client.search(query)

    except Exception as e:
        # 不要把原始异常堆栈返回给 Agent——Agent 看不懂。
        # 返回一个清晰的失败描述，让 Agent 决定下一步（如实告诉用户、换关键词重试）
        return f"搜索失败: {type(e).__name__} — {e}"
    
    # response 是一个 dict，核心字段是 'results' 列表
    results=response.get("results",[])

    if not results:
        return f"搜索 {query} 没有返回结果。尝试换更具体或更通用的关键词。"

    
    # 格式化为可读文本
    lines=[]
    for i,item in enumerate(results,start=1):
        title=item.get("title","无标题")
        url=item.get("url","")
        content=item.get("content","")
        lines.append(f"{i}.{title}\n URL:{url}\n  {content}\n")

    return "\n".join(lines)