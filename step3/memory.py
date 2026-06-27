"""Step3 记忆模块 — 双层记忆：UserFactsStore (SQLite) + ChromaMemoryStore (向量)"""

"""
Phase A 的局限：关掉 Python 进程后 MemorySaver 数据消失。Phase B 实现持久化——下次启动还能记得。
"""
import json
import os
import sqlite3
import uuid
from datetime import datetime,timezone
from pathlib import Path

from config import (
    CHROMA_DIR,
    DEEPSEEK_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL
)


import chromadb
from chromadb.utils import embedding_functions

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage,SystemMessage

# ═══════════════════════════════════════════
# 第一层：UserFactsStore（结构化事实，SQLite）
# ═══════════════════════════════════════════
class UserFactsStore:
    """SQLite 持久化的用户事实表。

    每个事实是一个 key-value 对，来源可追溯。
    """

    def __init__(self,db_path:str="user_facts.db"):
        self.db_path=db_path
        self._init_table()
    
    def _init_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_facts (
                    id TEXT PRIMARY KEY,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )#建表
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_fact_key
                ON user_facts (key)
                """
            )#建索引

    def upsert(self,key:str,value:str,source:str="")->str:
        """插入或更新事实。key 冲突时更新 value。返回事实 ID。"""
        now=datetime.now(timezone.utc).isoformat()
        fact_id=str(uuid.uuid4())[:8]
        with sqlite3.connect(self.db_path) as conn:
            existing=conn.execute(
                "SELECT id FROM user_facts WHERE key = ?",(key,)
            ).fetchone()

            if existing:#更新
                conn.execute(
                    "UPDATE user_facts SET value = ?, updated_at = ?, source = ? WHERE key = ?",
                    (value,now,source,key)
                )
                return existing[0]
            
            else:#插入新纪录
                conn.execute(
                    "INSERT INTO user_facts (id, key, value, source, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)"
                    (fact_id,key,value,source,now,now)
                )
                return fact_id
            
    def get(self,key:str)->str|None:
        """按 key 精确查询。"""
        with sqlite3.connect(self.db_path) as conn:
            row=conn.execute("SELECT value FROM user_facts WHERE key = ?", (key,)).fetchone()
            return row[0] if row else None
        
    
    def search(self,query:str)->list[dict]:
        """模糊搜索 value 中包含关键词的事实。"""
        with sqlite3.connect(self.db_path) as conn:
            rows=conn.execute(
                "SELECT key, value, source FROM user_facts WHERE value LIKE ? OR key LIKE ?",
                (f"%{query}%", f"%{query}%"),
            ).fetchall()

            return [{"key":r[0],"value":r[1],"source":r[2]} for r in rows]
        
    
    def get_all(self)->list[dict]:
        """获取所有事实。"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT key, value, source, updated_at FROM user_facts ORDER BY updated_at DESC"
            ).fetchall()
            return [{"key": r[0], "value": r[1], "source": r[2], "updated_at": r[3]} for r in rows]
        


# ═══════════════════════════════════════════
# 第二层：ChromaMemoryStore（对话摘要向量，Chroma）
# ═══════════════════════════════════════════
class ChromaMemoryStore:
    """Chroma 向量存储——对话摘要的语义检索。

    每条记录 = 对话摘要 + embedding 向量 + metadata。
    """
    def __init__(self,collection_name:str="conversation_summaries"):
        os.makedirs(CHROMA_DIR,exist_ok=True)
        self.client=chromadb.PersistentClient(path=CHROMA_DIR)

        self.embed_fn=embedding_functions.OpenAIEmbeddingFunction(
            api_key=EMBEDDING_API_KEY,
            api_base=EMBEDDING_BASE_URL,
            model_name=EMBEDDING_MODEL
        )

        try:
            self.collection=self.client.get_collection(
                name=collection_name,
                embedding_function=self.embed_fn
            )
        except Exception:
            self.collection=self.client.create_collection(
                name=collection_name,
                embedding_function=self.embed_fn,
                metadata={"hnsw:space":"cosine"}
            )

    def add_summary(self,conv_id:str,summary:str,metadata:dict|None=None)->None:
        """存入对话摘要向量。"""
        meta=metadata or {}
        meta["stored_at"]=datetime.now(timezone.utc).isoformat()

        self.collection.add(
            ids=[conv_id],
            documents=[summary],
            metadatas=[meta]
        )

    
    def search(self,query:str,top_k:int=3)->list[dict]:
        """语义检索相关历史摘要。"""
        try:
            results=self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
        
        except Exception:
            return [] #embedding失败时降级返回空
    
        if not results["ids"] or not results["ids"][0]:
            return []
        
        out=[]
        for i,doc_id in enumerate(results["ids"][0]):
            out.append(
                {
                    "id":doc_id,
                    "summary":results["documents"][0][i],
                    "metadata":results["metadatas"][0][i] if results["metadatas"] else {},
                }
            )
        return out
    

# ═══════════════════════════════════════════
# 第三层：MemoryManager（协调 + LLM 提取/摘要）
# ═══════════════════════════════════════════
class MemoryManager:
    """协调双层记忆：recall 检索 + save 提取存储。

    recall() — 从 user_facts + Chroma 检索相关记忆，返回格式化文本
    save_conversation() — 从对话中提取事实 + 生成摘要，存入两层
    """
    def __init__(self):
        self.facts=UserFactsStore()
        self.chroma=ChromaMemoryStore()
        self.llm=ChatDeepSeek(
            model='deepseek-chat',
            api_key=DEEPSEEK_API_KEY
        )

    #检索
    def recall(self,query:str)->str:
        """检索所有相关记忆，返回可注入 system prompt 的文本。"""

        parts=[]

        # 1. 精确匹配 user_facts
        all_facts=self.facts.get_all()
        if all_facts:
            lines=[f"- {f['key']:{f['value']}}" for f in all_facts]
            parts.append("已知用户信息：\n" + "\n".join(lines))


        # 2. 语义检索 Chroma
        chroma_results=self.chroma.search(query,top_k=2)
        if chroma_results:
            lines=[f"- {r['summary']}" for r in chroma_results]
            parts.append("历史相关对话：\n" + "\n".join(lines))

            if not parts:
                return ""
            
            return "\n\n".join(parts)
        
    #存储
    def save_conversation(self,messages:list,conv_id:int)->None:
        """对话结束后：提取事实 + 生成摘要 → 持久化。"""
        # 把 messages 转成可读文本
        


    def _messages_to_text(self,messages:list)->str:
        """消息列表 → 可读文本。"""
        lines=[]
        for m in messages:
            role=getattr(m,"type", getattr(m,"role","unknown"))
            content = getattr(m, "content", str(m))
            if content:
                lines.append(f"[{role}]: {content[:500]}")
        return "\n".join(lines)
    


    
    def _extract_facts(self, dialog_text: str) -> list[dict]:
        """用 LLM 从对话中提取用户事实。"""
        prompt = f"""从以下对话中提取关于用户的关键事实。
                    每个事实是一个 key-value 对（key 用英文标签，value 保持原文）。
                    只提取用户明确说过的信息，不要推理。
                    返回 JSON 数组格式。如果没有新事实，返回空数组 []。

                    对话：
                    {dialog_text}

                    输出（只输出 JSON，不要其他文字）：
                """
        try:
            response=self.llm.invoke(
                [SystemMessage(content="你是一个信息提取助手。只输出JSON。"),
                 HumanMessage(content=prompt)]
            )
            text=response.content.strip()
            # 去掉可能的 markdown 代码块标记
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]
            facts=json.loads(text)
            return facts if isinstance(facts,list) else []
        except Exception:
            return []
        

    def _summarize(self, dialog_text: str) -> str:
        """用 LLM 生成对话摘要（100 字以内）。"""
        prompt = f"""用一句话（不超过100字）总结以下对话的核心内容和关键信息。
                    重点记录：用户身份、偏好、讨论的话题、做出的决策。

                    对话：
                    {dialog_text}

                    摘要："""
        
        try:
            response=self.llm.invoke(
                [HumanMessage(content=prompt)]
            )
            return response.content.strip()
        except Exception:
            return ""

    