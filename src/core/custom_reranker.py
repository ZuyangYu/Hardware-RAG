# src/core/custom_reranker.py
import requests
from typing import List, Optional
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from src.core.logger import error, warn
from llama_index.core.bridge.pydantic import Field


class OllamaReranker(BaseNodePostprocessor):
    """Ollama Reranker - 使用 embedding 相似度模拟 rerank"""

    def __init__(self, model: str, base_url: str, top_n: int = 5):
        super().__init__()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.top_n = top_n

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """获取文本的 embedding 向量"""
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("embedding")
            else:
                error(f"Ollama Embedding 失败: {response.text}")
                return None
        except Exception as e:
            error(f"Ollama Embedding 错误: {e}")
            return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        import math
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        if not query_bundle or not nodes:
            return nodes[:self.top_n]

        try:
            query = query_bundle.query_str

            # 获取查询的 embedding
            query_embedding = self._get_embedding(query)
            if not query_embedding:
                warn("无法获取查询 embedding,返回原始排序")
                return nodes[:self.top_n]

            # 计算每个文档与查询的相似度
            scored_nodes = []
            for node in nodes:
                doc_text = node.node.get_content()
                doc_embedding = self._get_embedding(doc_text)

                if doc_embedding:
                    similarity = self._cosine_similarity(query_embedding, doc_embedding)
                    node.score = float(similarity)
                    scored_nodes.append(node)
                else:
                    # 如果无法获取 embedding,保留原始分数
                    scored_nodes.append(node)

            # 按相似度排序
            scored_nodes = sorted(scored_nodes, key=lambda x: x.score, reverse=True)

            return scored_nodes[:self.top_n]

        except Exception as e:
            error(f"Ollama Reranker 错误: {e}")
            return nodes[:self.top_n]


class APIReranker(BaseNodePostprocessor):
    """API Reranker(OpenAI 兼容格式)"""

    def __init__(self, model: str, api_key: str, api_base: str, top_n: int = 5):
        super().__init__()
        self.model = model
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.top_n = top_n

    def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        if not query_bundle or not nodes:
            return nodes[:self.top_n]

        try:
            query = query_bundle.query_str
            documents = [node.node.get_content() for node in nodes]

            # 调用 Rerank API
            response = requests.post(
                f"{self.api_base}/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": self.top_n
                },
                timeout=60
            )

            if response.status_code != 200:
                error(f"API Reranker 失败: {response.text}")
                return nodes[:self.top_n]

            results = response.json().get("results", [])

            # 重新排序
            if results:
                reranked_nodes = []
                for result in results:
                    idx = result.get("index")
                    score = result.get("relevance_score", 0)
                    if idx < len(nodes):
                        node = nodes[idx]
                        node.score = float(score)
                        reranked_nodes.append(node)

                return reranked_nodes[:self.top_n]

            return nodes[:self.top_n]

        except Exception as e:
            error(f"API Reranker 错误: {e}")
            return nodes[:self.top_n]


class NoReranker(BaseNodePostprocessor):
    """不使用 Reranker,直接返回 Top-N 结果"""

    top_n: int = Field(default=5, description="返回的节点数量")
    def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        # 直接截取前 N 个
        return nodes[:self.top_n]