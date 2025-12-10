# src/core/model_factory.py
"""
模型工厂 - 统一的模型初始化模块
支持两种模式：
1. ollama: 本地 Ollama 服务
2. custom: 所有第三方 API（OpenAI、OpenRouter、DeepSeek 等）
"""
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from src.core.custom_embedding import OpenRouterEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from config.settings import *
from src.core.logger import log, error, warn
from src.core.custom_reranker import OllamaReranker, APIReranker, NoReranker
from src.core.custom_llm import GenericOpenAILLM


def init_global_models():
    """
    初始化全局模型配置
    这是整个 RAG 系统的入口函数，负责初始化：
    - LLM（大语言模型）
    - Embedding（文本向量化模型）
    - Reranker（重排序模型）
    """
    try:
        log("=" * 60)
        log(f"🚀 初始化模型 Provider: {PROVIDER.name.upper()}")
        log("=" * 60)

        # 1. 初始化 LLM
        _init_llm()

        # 2. 初始化 Embedding
        _init_embedding()

        # 3. 初始化 Reranker
        _init_reranker()

        log("=" * 60)
        log("✅ 全局模型初始化完成")
        log("=" * 60)

    except Exception as e:
        error(f"❌ 模型初始化失败: {e}")
        raise


def _init_llm():
    """
    初始化 LLM 模型

    根据 PROVIDER 配置选择：
    - ollama: 本地 Ollama 服务
    - custom: 第三方 API 服务
    """
    try:
        if PROVIDER == Provider.OLLAMA:
            # ==================== Ollama 本地模型 ====================
            Settings.llm = Ollama(
                model=OLLAMA_LLM_MODEL,
                base_url=OLLAMA_BASE_URL,
                request_timeout=360
            )
            log(f"📝 LLM: Ollama")
            log(f"   ├─ 模型: {OLLAMA_LLM_MODEL}")
            log(f"   └─ 地址: {OLLAMA_BASE_URL}")

        elif PROVIDER == Provider.CUSTOM:
            # ==================== 自定义 API =========================
            # 验证必要参数
            if not CUSTOM_API_KEY or not CUSTOM_BASE_URL or not CUSTOM_LLM_MODEL:
                error("❌ CUSTOM 模式需要设置以下参数:")
                error("   - CUSTOM_API_KEY")
                error("   - CUSTOM_BASE_URL")
                error("   - CUSTOM_LLM_MODEL")
                raise ValueError("CUSTOM 模式配置不完整")

            # 使用自定义LLM封装
            Settings.llm = GenericOpenAILLM(
                model=CUSTOM_LLM_MODEL,
                api_key=CUSTOM_API_KEY,
                api_base=CUSTOM_BASE_URL,
                context_window=CUSTOM_CONTEXT_WINDOW,
                max_tokens=CUSTOM_MAX_TOKENS
            )
            log(f"📝 LLM: 自定义 API")
            log(f"   ├─ 模型: {CUSTOM_LLM_MODEL}")
            log(f"   ├─ 地址: {CUSTOM_BASE_URL}")
            log(f"   ├─ 上下文: {CUSTOM_CONTEXT_WINDOW}")
            log(f"   └─ 最大Token: {CUSTOM_MAX_TOKENS}")

        else:
            raise ValueError(f"❌ 未知的 Provider: {PROVIDER}")

    except Exception as e:
        error(f"❌ LLM 初始化失败: {e}")
        raise


def _init_embedding():
    """
    初始化 Embedding 模型

    根据配置选择：
    - Ollama Embedding（本地，免费）
    - 第三方 API Embedding（可能需要付费）
    注意：很多第三方 API 不支持 Embedding，推荐使用本地 Ollama
    """
    try:
        if PROVIDER == Provider.OLLAMA or USE_OLLAMA_EMBEDDING:
            # ==================== Ollama Embedding ====================
            # 两种情况会使用 Ollama：
            # 1. Provider 本身就是 ollama
            # 2. 强制设置 USE_OLLAMA_EMBEDDING=true（混合模式）

            Settings.embed_model = OllamaEmbedding(
                model_name=OLLAMA_EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL
            )

            if USE_OLLAMA_EMBEDDING and PROVIDER == Provider.CUSTOM: # 判断是不是混合模式
                log(f"🔢 Embedding: Ollama（混合模式）")
                log(f"   ℹ️LLM 用第三方 API，Embedding 用本地 Ollama")
            else:
                log(f"🔢 Embedding: Ollama")

            log(f"   ├─ 模型: {OLLAMA_EMBEDDING_MODEL}")
            log(f"   └─ 地址: {OLLAMA_BASE_URL}")

        elif PROVIDER == Provider.CUSTOM:
            # ==================== 第三方 API Embedding ====================
            # 只在明确不使用 Ollama 时才会走到这里
            if not CUSTOM_EMBEDDING_MODEL:
                warn("⚠️未设置 CUSTOM_EMBEDDING_MODEL")
                Settings.embed_model = None
            else:
                embedding_model = CUSTOM_EMBEDDING_MODEL

            warn("⚠️注意：很多第三方 API 不支持 Embedding")
            warn("⚠️如遇到错误，请设置 USE_OLLAMA_EMBEDDING=true")

            Settings.embed_model = OpenRouterEmbedding(
                model=CUSTOM_EMBEDDING_MODEL,
                api_key=CUSTOM_API_KEY,
                api_base=CUSTOM_BASE_URL
            )
            log(f"🔢 Embedding: 自定义 API")
            log(f"   ├─ 模型: {embedding_model}")
            log(f"   └─ 地址: {CUSTOM_BASE_URL}")

        else:
            raise ValueError(f"❌ 未知的 Provider: {PROVIDER}")

    except Exception as e:
        error(f"❌ Embedding 初始化失败: {e}")
        raise


def _init_reranker():
    """
    初始化 Reranker 模型

    Reranker 用于对检索结果进行二次排序，提高精度

    支持四种模式：
    - none: 不使用 Reranker（最快，精度一般）
    - local: 本地 Sentence Transformer 模型（精度高，需要下载模型）
    - ollama: 使用 Ollama 模拟 Reranker（兼容性好）
    - api: 使用 API 服务（需要付费）
    """
    try:
        if RERANKER_TYPE == RerankerType.NONE:
            # ==================== 不使用 Reranker ====================
            Settings.node_postprocessors = [NoReranker(top_n=FINAL_TOP_K)]
            log(f"🎯 Reranker: 不使用（直接返回 Top {FINAL_TOP_K}）")

        elif RERANKER_TYPE == RerankerType.LOCAL:
            # ==================== 本地 Reranker ====================
            # 设置模型缓存目录
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = RERANKER_CACHE

            Settings.node_postprocessors = [
                SentenceTransformerRerank(
                    top_n=FINAL_TOP_K,
                    model=RERANKER_MODEL
                )
            ]
            log(f"🎯 Reranker: 本地模型")
            log(f"   ├─ 模型: {RERANKER_MODEL}")
            log(f"   ├─ Top-N: {FINAL_TOP_K}")
            log(f"   └─ 缓存: {RERANKER_CACHE}")

        elif RERANKER_TYPE == RerankerType.OLLAMA:
            # ==================== Ollama Reranker ====================
            Settings.node_postprocessors = [
                OllamaReranker(
                    model=OLLAMA_RERANKER_MODEL,
                    base_url=OLLAMA_BASE_URL,
                    top_n=FINAL_TOP_K
                )
            ]
            log(f"🎯 Reranker: Ollama")
            log(f"   ├─ 模型: {OLLAMA_RERANKER_MODEL}")
            log(f"   ├─ 地址: {OLLAMA_BASE_URL}")
            log(f"   └─ Top-N: {FINAL_TOP_K}")

        elif RERANKER_TYPE == RerankerType.API:
            # ==================== API Reranker ====================
            api_key = RERANKER_API_KEY or CUSTOM_API_KEY

            if not api_key:
                warn("⚠️  Reranker API Key 未设置")

            Settings.node_postprocessors = [
                APIReranker(
                    model=RERANKER_MODEL,
                    api_key=api_key,
                    api_base=RERANKER_API_BASE,
                    top_n=FINAL_TOP_K
                )
            ]
            log(f"🎯 Reranker: API")
            log(f"   ├─ 模型: {RERANKER_MODEL}")
            log(f"   ├─ 地址: {RERANKER_API_BASE}")
            log(f"   └─ Top-N: {FINAL_TOP_K}")

        else:
            warn(f"⚠️  未知的 Reranker 类型: {RERANKER_TYPE}")
            Settings.node_postprocessors = [NoReranker(top_n=FINAL_TOP_K)]

    except Exception as e:
        error(f"❌ Reranker 初始化失败: {e}")
        warn("⚠️  降级到不使用 Reranker")
        Settings.node_postprocessors = [NoReranker(top_n=FINAL_TOP_K)]


def get_current_config() -> dict:
    """
    获取当前配置信息（用于调试和监控）

    Returns:
        dict: 当前配置的字典
    """
    config = {
        "provider": PROVIDER.name,
        "llm_model": OLLAMA_LLM_MODEL if PROVIDER == Provider.OLLAMA else CUSTOM_LLM_MODEL,
        "embedding_model": OLLAMA_EMBEDDING_MODEL if (
                PROVIDER == Provider.OLLAMA or USE_OLLAMA_EMBEDDING) else CUSTOM_EMBEDDING_MODEL,
        "use_ollama_embedding": USE_OLLAMA_EMBEDDING,
        "reranker_type": RERANKER_TYPE.name,
        "reranker_model": RERANKER_MODEL if RERANKER_TYPE != RerankerType.NONE else None,
        "final_top_k": FINAL_TOP_K,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    return config


def print_config():
    """
    打印当前配置（用于调试）

    在启动时调用此函数可以查看完整的配置信息
    """
    config = get_current_config()
    log("\n" + "=" * 60)
    log("📋 当前配置")
    log("=" * 60)
    for key, value in config.items():
        log(f"   {key}: {value}")
    log("=" * 60 + "\n")


def validate_config() -> tuple[bool, list[str]]:
    """
    验证配置是否完整

    Returns:
        tuple: (是否有效, 错误信息列表)
    """
    errors = []

    # 验证 Provider
    if PROVIDER == Provider.CUSTOM:
        if not CUSTOM_API_KEY:
            errors.append("缺少 CUSTOM_API_KEY")
        if not CUSTOM_BASE_URL:
            errors.append("缺少 CUSTOM_BASE_URL")
        if not CUSTOM_LLM_MODEL:
            errors.append("缺少 CUSTOM_LLM_MODEL")

        # 如果不使用 Ollama Embedding，检查 Embedding 配置
        if not USE_OLLAMA_EMBEDDING and not CUSTOM_EMBEDDING_MODEL:
            errors.append("缺少 CUSTOM_EMBEDDING_MODEL（或设置 USE_OLLAMA_EMBEDDING=true）")

    elif PROVIDER == Provider.OLLAMA:
        if not OLLAMA_BASE_URL:
            errors.append("缺少 OLLAMA_BASE_URL")
        if not OLLAMA_LLM_MODEL:
            errors.append("缺少 OLLAMA_LLM_MODEL")
        if not OLLAMA_EMBEDDING_MODEL:
            errors.append("缺少 OLLAMA_EMBEDDING_MODEL")

    return len(errors) == 0, errors


if __name__ != "__main__":
    is_valid, errors = validate_config()
    if not is_valid:
        warn("⚠️  配置验证失败:")
        for err in errors:
            warn(f"   - {err}")
