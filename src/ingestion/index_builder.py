# src/ingestion/index_builder.py
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from src.ingestion.data_loader import load_documents
from src.core.logger import log, error, warn
import threading
from typing import Optional, Dict


# ==================== 索引缓存 ====================
class _IndexCache:
    """简单的索引缓存（内部使用）"""

    def __init__(self):
        self._cache: Dict[str, VectorStoreIndex] = {}
        self._lock = threading.RLock()

    def get(self, kb_name: str) -> Optional[VectorStoreIndex]:
        with self._lock:
            return self._cache.get(kb_name)

    def set(self, kb_name: str, index: VectorStoreIndex):
        with self._lock:
            self._cache[kb_name] = index
            log(f"💾 索引已缓存: {kb_name}")

    def invalidate(self, kb_name: str):
        with self._lock:
            if kb_name in self._cache:
                del self._cache[kb_name]
                log(f"🗑️ 索引缓存已清除: {kb_name}")


# 全局缓存实例
_index_cache = _IndexCache()


# ==================== 主要函数 ====================
def get_or_build_index(kb_name: str, chroma_client, use_cache: bool = True) -> VectorStoreIndex:
    """
    获取或构建知识库索引（带缓存优化）

    Args:
        kb_name: 知识库名称
        chroma_client: ChromaDB 客户端
        use_cache: 是否使用缓存（默认 True）

    Returns:
        VectorStoreIndex 实例
    """
    # 1. 尝试从缓存获取
    if use_cache:
        cached_index = _index_cache.get(kb_name)
        if cached_index is not None:
            log(f"⚡ 使用缓存的索引: {kb_name}")
            return cached_index

    # 2. 从 ChromaDB 加载或构建新索引
    try:
        coll_name = f"kb_{kb_name}"
        collection = chroma_client.get_or_create_collection(coll_name)

        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 检查 ChromaDB 中是否已有数据
        vector_count = collection.count()

        if vector_count > 0:
            # ✅ 从已有的向量存储加载索引（不重建）
            log(f"📂 从 ChromaDB 加载索引: {kb_name} ({vector_count} 个向量)")
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context
            )
        else:
            # ⚠️ 没有数据,需要构建新索引
            log(f"🔨 构建新索引: {kb_name}")
            docs = load_documents(kb_name)

            if docs:
                index = VectorStoreIndex.from_documents(
                    docs,
                    storage_context=storage_context,
                    show_progress=True
                )
                log(f"✅ 索引构建完成: {kb_name} ({len(docs)} 个文档)")
            else:
                log(f"⚠️ 知识库为空,创建空索引: {kb_name}")
                # ✅ 关键修复：显式传入空列表，初始化 DocStore 和 VectorStore 的连接
                # 否则后续增量插入会报错
                index = VectorStoreIndex.from_documents(
                    [],
                    storage_context=storage_context
                )

        # 3. 缓存索引
        if use_cache:
            _index_cache.set(kb_name, index)

        return index

    except Exception as e:
        error(f"❌ 索引加载/构建失败: {kb_name} - {e}")
        raise


def rebuild_index(kb_name: str, chroma_client) -> VectorStoreIndex:
    """
    强制重建索引（清除缓存并重新构建）
    注意：在增量更新模式下，通常不再需要调用此函数，除非需要彻底重置
    """
    log(f"🔄 强制重建索引: {kb_name}")

    # 1. 清除索引缓存
    invalidate_index_cache(kb_name)

    # 2. 清除 ChromaDB 集合
    try:
        coll_name = f"kb_{kb_name}"
        chroma_client.delete_collection(coll_name)
        log(f"🗑️ 已删除旧集合: {coll_name}")
    except Exception as e:
        warn(f"删除集合时出错（可能不存在）: {e}")

    # 3. 重新构建索引（不使用缓存）
    return get_or_build_index(kb_name, chroma_client, use_cache=False)


def invalidate_index_cache(kb_name: str):
    """
    使索引缓存失效（文件变更后调用）

    Args:
        kb_name: 知识库名称
    """
    _index_cache.invalidate(kb_name)
