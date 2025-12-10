# src/core/custom_llm.py
from typing import Any, Sequence, List
from llama_index.core.llms import (
    CustomLLM,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback
from pydantic import Field
import requests


class GenericOpenAILLM(CustomLLM):
    """
    通用 OpenAI 兼容 API 封装（支持 OpenRouter、DeepSeek 等）
    已修复装饰器错误并移除冗余代码
    """
    model: str = Field(description="模型名称")
    api_key: str = Field(description="API 密钥")
    api_base: str = Field(default="https://openrouter.ai/api/v1", description="API 基础地址")
    context_window: int = Field(default=128000, description="上下文窗口大小")
    max_tokens: int = Field(default=4096, description="最大输出 token 数")
    temperature: float = Field(default=0.7, description="采样温度")
    timeout: int = Field(default=120, description="请求超时时间（秒）")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model,
            is_chat_model=True,
        )

    # ✅ complete 返回 CompletionResponse -> 使用 llm_completion_callback
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = [{"role": "user", "content": prompt}]
        response_text = self._call_api(messages, **kwargs)
        return CompletionResponse(text=response_text, raw={"model": self.model})

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("流式补全暂不支持")

    # ✅ chat 返回 ChatResponse -> 必须使用 llm_chat_callback
    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # 将 LlamaIndex 消息对象转为 API 字典
        api_messages = [
            {"role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
             "content": msg.content}
            for msg in messages
        ]

        response_text = self._call_api(api_messages, **kwargs)

        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=response_text),
            raw={"model": self.model, "usage": {}}
        )

    # ✅ stream_chat 也应该对应 chat callback (虽然这里未实现)
    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("流式聊天暂不支持")

    def _call_api(self, messages: List[dict], **kwargs) -> str:
        url = f"{self.api_base.rstrip('/')}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        if "extra_headers" in kwargs:
            headers.update(kwargs["extra_headers"])

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                if not content or not content.strip():
                    raise ValueError("API 返回空响应")
                return content
            else:
                raise ValueError(f"API 响应格式错误: {data}")

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                raise RuntimeError("API 请求频率过高 (429)，请稍后重试。")
            else:
                raise RuntimeError(f"API 调用失败: {e}")
        except Exception as e:
            raise RuntimeError(f"请求异常: {e}")
