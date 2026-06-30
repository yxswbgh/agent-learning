"""Step4 记忆模块 — Mem0 风格：add() + search() 双层记忆"""

import json
import math
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    CHROMA_DIR,
    DEEPSEEK_API_KEY,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    MEMORY_COLLECTION,
)





# ═══════════════════════════════════════════
# ChromaMemoryStore — 向量记忆存储
# ═══════════════════════════════════════════

class ChromaMemoryStore:
    """Chroma 向量存储 — 存储单条记忆，支持 CRUD + 语义检索。

    每条记忆 = id + 文本内容 + embedding + metadata。
    metadata 包含: key, type, importance, source, created_at, updated_at
    """
    def __init__(self, collection_name: str = MEMORY_COLLECTION):
        os.makedirs(CHROMA_DIR, exist_ok=True)
        self.client=chromadb.PersistentClient(path=CHROMA_DIR)

        self.embed_fn=embedding_functions.OpenAIEmbeddingFunction(
            api_key=EMBEDDING_API_KEY,
            api_base=EMBEDDING_BASE_URL,
            model_name=EMBEDDING_MODEL
        )

        try:
            self.collection=self.client.get_collection(
                name=collection_name,
                embedding_function=self.embed_fn,
            )
        except Exception:
            self.collection=self.client.create_collection(
                name=collection_name,
                embedding_function=self.embed_fn,
                metadata={"hnsw:space":"cosine"}
            )


    # --- CRUD ---

    def add_memory(
        self, mem_id: str, content: str, metadata: dict | None = None
    ) -> None:
        """插入一条新记忆。"""
        meta = metadata or {}
        meta.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        meta.setdefault("updated_at", meta["created_at"])
        meta.setdefault("importance", 0.5)
        meta.setdefault("type", "fact")

        self.collection.add(
            ids=[mem_id],
            documents=[content],
            metadatas=[meta],
        )

    def update_memory(
        self, mem_id: str, content: str, metadata: dict | None = None
    ) -> None:
        """更新一条已有记忆。"""
        meta = metadata or {}
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()

        self.collection.update(
            ids=[mem_id],
            documents=[content],
            metadatas=[meta],
        )

    def delete_memory(self, mem_id: str) -> None:
        """删除一条记忆。"""
        self.collection.delete(ids=[mem_id])

    def get_by_id(self, mem_id: str) -> dict | None:
        """按 ID 获取记忆。"""
        result = self.collection.get(ids=[mem_id])
        if result and result["ids"]:
            return {
                "id": result["ids"][0],
                "content": result["documents"][0] if result["documents"] else "",
                "metadata": result["metadatas"][0] if result["metadatas"] else {},
            }
        return None
    

    # --- 检索 ---
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """语义检索相关记忆，返回 distance 分数用于后续排序。"""
        try:
            results=self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents","metadatas","distances"]
            )

        except Exception:
            return []
        
        if not results["ids"] or not results["ids"][0]:
            return []
        
        out=[]
        for i,doc_id in enumerate(results["ids"][0]):
            distance=results["distances"][0][i] if results["distances"] else 0.0
            out.append(
                {
                    "id":doc_id,
                    "content":results["documents"][0][i] if results["documents"] else "",
                    "metadata":results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance":distance # cosine distance，越小越相关
                }
            )
        return out
    
    def count(self) -> int:
        """返回记忆总数。"""
        return self.collection.count()







# ═══════════════════════════════════════════
# MemoryManager — Mem0 风格记忆协调器
# ═══════════════════════════════════════════
class MemoryManager:
    """Mem0 风格记忆管理器。

    对外暴露两个核心函数：
    - add(messages, infer=True, prompt=None) → int  # 返回存储的记忆数
    - search(query, top_k=5, ...) → list[dict]      # 返回排序后的记忆列表
    """
    def __init__(self):
        self.chroma = ChromaMemoryStore()
        self.llm = ChatDeepSeek(
            model="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
        )
        self.DEFAULT_EXTRACT_PROMPT = """你是一个信息提取助手。从以下对话中识别并抽取有价值的 Facts。

            Facts 包括但不限于：
            - 用户偏好（喜欢/不喜欢什么）
            - 关键信息（姓名、职业、技能、项目）
            - 待办事项（用户提到要做的事）
            - 决策（用户做出的选择和决定）
            - 个人资料（任何关于用户的描述）

            要求：
            1. 只提取用户明确说过的信息，不要推理或编造
            2. 每个 Fact 是一个独立的、可被检索的知识片段
            3. 为每个 Fact 评估重要性（0.0~1.0）

            返回 JSON 数组格式。如果没有值得记录的事实，返回空数组 []。

            输出格式：
            [{"key": "英文标签", "value": "事实内容（保持原文）", "type": "profile|preference|todo|decision|fact", "importance": 0.0-1.0}]"""
        
        self.DEFAULT_DECISION_PROMPT = """你是一个记忆管理助手。对于每一对新旧记忆对，判断应该执行什么操作。

操作定义：
- ADD: 新记忆是全新的信息，旧记忆不存在或不相关
- UPDATE: 新记忆是旧记忆的演进、修正或补充（新旧描述同一件事但内容不同）
- DELETE: 新记忆表明旧记忆已过时/无效（例如用户改了名字、换了工作）
- NONE: 新旧记忆重复或无重要变化，不需要任何操作

输入是 JSON 数组，每个元素包含 "new" (候选新记忆) 和 "old" (旧记忆，可能为 null)。

返回 JSON 数组：
[{"event": "ADD|UPDATE|DELETE|NONE", "new_key": "新记忆的key", "old_id": "旧记忆ID或null", "reason": "简短理由"}]"""

    # ═══════════════════════════════════════
    # add() — 记忆存储入口
    # ═══════════════════════════════════════
    def add(self, messages, infer: bool = True, prompt: str | None = None) -> int:
        """存储新记忆。

        Args:
            messages: LangChain 消息列表 或 dict 列表 (含 role/content)
            infer: True=LLM 智能提取+去重，False=直接 embedding
            prompt: infer=True 时传给 LLM 的自定义 system prompt

        Returns:
            成功存储的记忆数量
        """
        if not infer:
            return self._add_direct(messages)
        return self._add_with_infer(messages, prompt)
    

    def _add_direct(self, messages) -> int:
        """将每条消息内容直接 embedding 存入 Chroma，不做 LLM 处理。"""
        count=0
        for msg in messages:
            content=self._extract_content(msg)
            if not content or not content.strip():
                continue

            mem_id=str(uuid.uuid4())[:8]
            self.chroma.add_memory(
                mem_id=mem_id,
                content=content.strip(),
                metadata={
                    "type": "raw_message",
                    "importance": 0.3,
                    "source": "direct_add",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            count+=1
        print(f"[memory] 直接存入 {count} 条记忆")
        return count


    # --- 辅助: 从各种消息格式提取文本 ---
    def _extract_content(self, msg) -> str:
        """从 LangChain message / dict / 字符串提取文本内容。"""
        if isinstance(msg, str):
            return msg
        if hasattr(msg, "content"):
            return msg.content
        if isinstance(msg, dict):
            return msg.get("content", "") or str(msg)
        return str(msg)
    
    def _messages_to_text(self, messages: list) -> str:
        """消息列表 → 可读文本（用于 LLM 提示）。"""
        lines = []
        for m in messages:
            role = getattr(m, "type", getattr(m, "role", "unknown"))
            content = self._extract_content(m)
            if content:
                lines.append(f"[{role}]: {content[:500]}")
        return "\n".join(lines)
    

    # ═══════════════════════════════════════
    # _add_with_infer — 智能记忆提取
    # ═══════════════════════════════════════
    def _add_with_infer(self,messages,custom_prompt:str|None)->int:
        """智能存储：LLM 提取事实 → 检索旧记忆 → 决策 → 执行。"""
        dialog_text=self._messages_to_text(messages)

        # Step 1: 事实提取 → 候选记忆列表
        
        candidates = self._extract_facts(dialog_text, custom_prompt)
        if not candidates:
            print("[memory] 未提取到新事实，跳过存储")
            return 0

        print(f"[memory] Step 1: 提取到 {len(candidates)} 条候选记忆")

        # Step 2: 旧记忆检索 → 新旧记忆对
        pairs = self._build_memory_pairs(candidates)
        print(f"[memory] Step 2: 构建了 {len(pairs)} 对新旧记忆对")

        # Step 3-4 在 Task B4 实现
        decisions = self._decide_updates(pairs)
        count = self._execute_decisions(decisions)
        return count



    def _extract_facts(self,dialog_text: str, custom_prompt: str | None = None)-> list[dict]:
        """Step 1: LLM 从对话中提取候选事实。"""
        system_prompt = custom_prompt or self.DEFAULT_EXTRACT_PROMPT

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"对话内容：\n{dialog_text}"),
                ]
            )
            text = response.content.strip()
            # 去掉可能的 markdown 代码块标记
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:])
                if text.endswith("```"):
                    text = text[:-3]
            facts = json.loads(text)
            if isinstance(facts, list):
                return facts
            return []
        except Exception as e:
            print(f"[memory] 事实提取失败: {e}")
            return []
        

    def _build_memory_pairs(self, candidates: list[dict]) -> list[dict]:
        """Step 2: 每个候选记忆去 Chroma 检索最相似的旧记忆，构建新旧对。"""
        pairs = []
        for candidate in candidates:
            query = candidate.get("value", "")
            if not query:
                continue

            old_results = self.chroma.search(query, top_k=3)
            if old_results:
                for old in old_results:
                    pairs.append({"new": candidate, "old": old})
            else:
                pairs.append({"new": candidate, "old": None})

        return pairs
    
    def _decide_updates(self, pairs: list[dict]) -> list[dict]:
        """Step 3: LLM 对每对新旧记忆对判断 ADD/UPDATE/DELETE/NONE。"""
        if not pairs:
            return []

        # 构造给 LLM 的简化版 pairs（避免 content 过长）
        simplified = []
        for p in pairs:
            item = {
                "new": {
                    "key": p["new"].get("key", ""),
                    "value": p["new"].get("value", ""),
                    "type": p["new"].get("type", "fact"),
                    "importance": p["new"].get("importance", 0.5),
                },
                "old": None,
            }
            if p["old"]:
                item["old"] = {
                    "id": p["old"]["id"],
                    "content": p["old"]["content"],
                }
            simplified.append(item)

        pairs_json = json.dumps(simplified, ensure_ascii=False, indent=2)

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=self.DEFAULT_DECISION_PROMPT),
                    HumanMessage(content=pairs_json),
                ]
            )
            text = response.content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:])
                if text.endswith("```"):
                    text = text[:-3]
            decisions = json.loads(text)
            if isinstance(decisions, list):
                print(f"[memory] Step 3: LLM 决策 — {json.dumps(decisions, ensure_ascii=False)}")
                return decisions
            return []
        except Exception as e:
            print(f"[memory] 决策失败: {e}")
            # 降级：全部 ADD
            return [
                {
                    "event": "ADD",
                    "new_key": p["new"].get("key", ""),
                    "old_id": p["old"]["id"] if p["old"] else None,
                    "reason": "降级策略：决策失败，默认 ADD",
                }
                for p in pairs
            ]

    def _execute_decisions(self, decisions: list[dict]) -> int:
        """Step 4: 执行决策 — ADD/UPDATE/DELETE/NONE 对应的 Chroma 操作。"""
        if not decisions:
            return 0

        count = 0
        for i, decision in enumerate(decisions):
            event = decision.get("event", "NONE")
            old_id = decision.get("old_id")
            new_key = decision.get("new_key", "")

            if event == "NONE":
                continue

            elif event == "DELETE":
                if old_id:
                    self.chroma.delete_memory(old_id)
                    print(f"[memory] DELETE: {old_id}")

            elif event == "UPDATE":
                if old_id:
                    # 更新旧记忆的内容
                    self.chroma.update_memory(
                        mem_id=old_id,
                        content=new_key,
                        metadata={
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    count += 1
                    print(f"[memory] UPDATE: {old_id}")

            elif event == "ADD":
                mem_id = str(uuid.uuid4())[:8]
                self.chroma.add_memory(
                    mem_id=mem_id,
                    content=new_key,
                    metadata={
                        "key": new_key,
                        "type": "fact",
                        "importance": 0.5,
                        "source": "infer_add",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                count += 1
                print(f"[memory] ADD: {mem_id} — {new_key[:50]}")

        print(f"[memory] Step 4: 执行完成，共 {count} 条变更")
        return count


    # ═══════════════════════════════════════
    # search() — 记忆检索入口
    # ═══════════════════════════════════════
    def search(
        self,
        query: str,
        top_k: int = 5,
        w_relevance: float = 0.5,
        w_recency: float = 0.3,
        w_importance: float = 0.2,
        decay_days: float = 30.0,
    ) -> list[dict]:
        """检索相关记忆并统一排序。

        排序公式:
          final_score = w_relevance × 相关性 + w_recency × 时近性 + w_importance × 重要性

        Args:
            query: 检索查询
            top_k: 返回数量
            w_relevance: 相关性权重
            w_recency: 时近性权重
            w_importance: 重要性权重
            decay_days: 时近性衰减天数（越久远的记忆权重越低）

        Returns:
            按 final_score 降序排列的记忆列表
        """
        # Step 1: Vector recall
        results = self.chroma.search(query, top_k=top_k * 2)  # 多取一些做排序
        if not results:
            return []

        # Step 2: 统一排序
        scored = []
        now = datetime.now(timezone.utc)

        for r in results:
            # 相关性：cosine distance → similarity
            distance = r.get("distance", 0.0)
            relevance = 1.0 - min(distance, 1.0)  # cosine distance ∈ [0, 2]

            # 时近性：指数衰减
            created_at_str = r.get("metadata", {}).get("created_at", "")
            recency = 0.0
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                    age_days = (now - created_at).total_seconds() / 86400.0
                    recency = math.exp(-age_days / decay_days)
                except (ValueError, TypeError):
                    recency = 0.5  # 默认中等新鲜度

            # 重要性：从 metadata 读取
            importance = float(r.get("metadata", {}).get("importance", 0.5))

            # 综合评分
            final_score = (
                w_relevance * relevance
                + w_recency * recency
                + w_importance * importance
            )

            scored.append(
                {
                    "id": r["id"],
                    "content": r["content"],
                    "metadata": r.get("metadata", {}),
                    "relevance": round(relevance, 3),
                    "recency": round(recency, 3),
                    "importance": round(importance, 3),
                    "final_score": round(final_score, 3),
                }
            )

        # 按 final_score 降序排列
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored[:top_k]


