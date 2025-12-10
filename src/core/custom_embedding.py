# src/core/custom_embedding.py
from typing import List, Optional
from openai import OpenAI
from llama_index.core.embeddings import BaseEmbedding
from pydantic import Field, PrivateAttr
import os


class OpenRouterEmbedding(BaseEmbedding):
    """
    支持 OpenRouter 等 OpenAI 兼容 Embedding API 的封装
    """
    model: str = Field(default="baai/bge-base-en-v1.5", description="Embedding 模型名称")
    api_key: Optional[str] = Field(default=None, description="API 密钥")
    api_base: Optional[str] = Field(default="https://openrouter.ai/api/v1", description="API 基地址")

    # 使用 PrivateAttr 存储非字段属性（不会被序列化）
    _client: OpenAI = PrivateAttr()

    def __init__(self, **data):
        # 先让 Pydantic 处理字段
        super().__init__(**data)

        # 再初始化私有属性
        api_key = self.api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("CUSTOM_API_KEY")
        api_base = self.api_base or "https://openrouter.ai/api/v1"

        if not api_key:
            raise ValueError("未提供 API Key，请设置 OPENROUTER_API_KEY 或传入 api_key 参数")

        self._client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )

    @classmethod
    def class_name(cls) -> str:
        return "OpenRouterEmbedding"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
        )
        return [data.embedding for data in response.data]