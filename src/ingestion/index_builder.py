# src/ingestion/index_builder.py
import os
import threading
from typing import Optional, Dict
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.schema import TextNode, Document
from src.core.logger import log, error, warn
from config.settings import STORAGE_DIR


class _IndexCache:
    def __init__(self):
        self._cache: Dict[str, VectorStoreIndex] = {}
        self._lock = threading.RLock()

    def get(self, kb_name: str) -> Optional[VectorStoreIndex]:
        with self._lock:
            return self._cache.get(kb_name)

    def set(self, kb_name: str, index: VectorStoreIndex):
        with self._lock:
            self._cache[kb_name] = index

    def invalidate(self, kb_name: str):
        with self._lock:
            if kb_name in self._cache:
                del self._cache[kb_name]


_index_cache = _IndexCache()


def get_or_build_index(kb_name: str, chroma_client, use_cache: bool = True) -> VectorStoreIndex:
    # 1. 缓存层
    if use_cache:
        cached_index = _index_cache.get(kb_name)
        if cached_index is not None:
            return cached_index

    try:
        coll_name = f"kb_{kb_name}"
        collection = chroma_client.get_or_create_collection(coll_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)

        # 持久化目录
        kb_persist_dir = os.path.join(STORAGE_DIR, f"docstore_{kb_name}")
        os.makedirs(kb_persist_dir, exist_ok=True)

        vector_count = collection.count()

        # 尝试从磁盘恢复 StorageContext
        try:
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=kb_persist_dir
            )

            if vector_count > 0:
                log(f"从磁盘加载完整索引: {kb_name}")
                index = load_index_from_storage(storage_context, vector_store=vector_store)

                # ✅ 验证 DocStore 完整性
                if _validate_docstore(index, collection):
                    log(f"✅ DocStore 验证通过: {kb_name}")
                else:
                    # DocStore 不完整，需要重建
                    warn(f"检测到 DocStore 与 ChromaDB 不一致,正在修复...")
                    index = _rebuild_docstore_from_chroma(vector_store, kb_persist_dir, collection)
            else:
                # Chroma 空，初始化空索引
                index = VectorStoreIndex.from_documents([], storage_context=storage_context)

        except Exception as e:
            warn(f"DocStore 加载失败 ({e})，正在重建...")

            if vector_count > 0:
                # 从 ChromaDB 重建
                index = _rebuild_docstore_from_chroma(vector_store, kb_persist_dir, collection)
            else:
                # 初始化空索引
                docstore = SimpleDocumentStore()
                index_store = SimpleIndexStore()
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                    docstore=docstore,
                    index_store=index_store
                )
                log(f"🆕 初始化空索引: {kb_name}")
                index = VectorStoreIndex.from_documents([], storage_context=storage_context)
                index.storage_context.persist(persist_dir=kb_persist_dir)

        # 3. 缓存
        if use_cache:
            _index_cache.set(kb_name, index)
        return index

    except Exception as e:
        error(f"❌ 索引构建严重失败: {kb_name} - {e}")
        raise


def _validate_docstore(index: VectorStoreIndex, collection) -> bool:
    """验证 DocStore 是否完整"""
    try:
        docstore = index.docstore
        chroma_count = collection.count()

        # 获取几个 ID 测试
        results = collection.get(limit=min(10, chroma_count), include=["metadatas"])
        test_ids = results.get("ids", [])

        missing_count = 0
        for node_id in test_ids:
            try:
                docstore.get_node(node_id)
            except:
                missing_count += 1

        if missing_count > 0:
            warn(f"DocStore 缺失率过高: {missing_count}/{len(test_ids)}")
            return False

        return True

    except Exception as e:
        warn(f"验证 DocStore 失败: {e}")
        return False


def _rebuild_docstore_from_chroma(
        vector_store: ChromaVectorStore,
        kb_persist_dir: str,
        collection
) -> VectorStoreIndex:
    """
    从 ChromaDB 重建 DocStore
    正确创建 TextNode，不直接设置 ref_doc_id
    """
    log("从 ChromaDB 重建 DocStore...")

    # 创建新的存储组件
    docstore = SimpleDocumentStore()
    index_store = SimpleIndexStore()

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
        index_store=index_store
    )

    try:
        # 获取所有数据
        results = collection.get(include=["documents", "metadatas", "embeddings"])

        node_count = len(results["ids"])
        log(f"从 ChromaDB 获取到 {node_count} 个节点")

        success_count = 0

        # ✅ 关键修复：正确创建节点
        for idx, node_id in enumerate(results["ids"]):
            try:
                text = results["documents"][idx]
                metadata = results["metadatas"][idx]

                # 方案1: 如果有 doc_id，创建 Document
                doc_id = metadata.get("doc_id") or metadata.get("ref_doc_id")

                if doc_id:
                    # 创建 Document（会自动设置 doc_id）
                    doc = Document(
                        text=text,
                        id_=doc_id,
                        metadata=metadata,
                        excluded_embed_metadata_keys=["file_name", "file_path"],
                        excluded_llm_metadata_keys=["file_name", "file_path"]
                    )
                    docstore.add_documents([doc])

                    # 再创建对应的 TextNode
                    node = TextNode(
                        text=text,
                        id_=node_id,
                        metadata=metadata,
                        excluded_embed_metadata_keys=["file_name", "file_path"],
                        excluded_llm_metadata_keys=["file_name", "file_path"]
                    )
                    # ✅ 通过 relationships 关联 Document
                    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
                    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                        node_id=doc_id,
                        metadata={}
                    )
                    docstore.add_documents([node])
                else:
                    # 如果没有 doc_id，直接创建独立 TextNode
                    node = TextNode(
                        text=text,
                        id_=node_id,
                        metadata=metadata,
                        excluded_embed_metadata_keys=["file_name", "file_path"],
                        excluded_llm_metadata_keys=["file_name", "file_path"]
                    )
                    docstore.add_documents([node])

                success_count += 1

            except Exception as e:
                warn(f"重建节点失败 {node_id}: {e}")
                continue

        log(f"✅ DocStore 重建完成,成功 {success_count}/{node_count} 个节点")

        # 验证重建结果
        if success_count == 0:
            error("❌ DocStore 验证失败: 没有节点被正确保存!")
            log(f"DocStore 内容: {len(docstore.docs)} 个文档")

        # 持久化
        log("💾 正在持久化 DocStore...")
        storage_context.persist(persist_dir=kb_persist_dir)

        # 验证文件大小
        docstore_path = os.path.join(kb_persist_dir, "docstore.json")
        if os.path.exists(docstore_path):
            size = os.path.getsize(docstore_path)
            if size < 100:  # 小于 100 字节说明基本是空的
                warn(f"DocStore 文件过小: {size} bytes")

        log("已保存重建的 DocStore")

        # 从重建的 storage_context 创建索引
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )

        return index

    except Exception as e:
        error(f"❌ 重建 DocStore 失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def invalidate_index_cache(kb_name: str):
    """清除索引缓存"""
    _index_cache.invalidate(kb_name)