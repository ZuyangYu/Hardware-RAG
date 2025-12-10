# src/core/hybrid_retriever.py
from typing import List, Optional, Dict
import jieba
from rank_bm25 import BM25Okapi
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.vector_stores.chroma import ChromaVectorStore
from config.settings import VECTOR_TOP_K, BM25_TOP_K, RRF_K, FINAL_TOP_K
from src.core.bm25_cache import BM25Cache
from src.core.logger import log, warn, error

_bm25_node_map: Dict[str, List[str]] = {}


def build_bm25_index(kb_name: str, index: VectorStoreIndex, force_rebuild: bool = False) -> Optional[BM25Okapi]:
    """构建或获取 BM25 索引"""
    cache = BM25Cache()

    vector_store = index._vector_store
    if not isinstance(vector_store, ChromaVectorStore):
        warn("BM25 仅支持 ChromaVectorStore")
        return None

    collection = vector_store._collection
    current_doc_count = collection.count()

    # 检查缓存是否过期
    if not force_rebuild:
        bm25 = cache.get(kb_name)
        cached_node_map = _bm25_node_map.get(kb_name, [])

        if bm25 is not None and len(cached_node_map) == current_doc_count:
            log(f"✅ 使用缓存的 BM25 索引: {kb_name} ({current_doc_count} 个文档)")
            return bm25
        else:
            if bm25 is not None:
                log(f"⚠️ BM25 缓存过期，重建索引")

    try:
        # 获取所有文档
        results = collection.get(include=["documents", "metadatas"])
        documents = results["documents"]
        ids = results["ids"]

        if not documents:
            log(f"⚠️ 知识库为空，跳过 BM25 索引构建: {kb_name}")
            return None

        log(f"🔨 构建 BM25 索引: {kb_name} ({len(documents)} 个文档)")

        # ========================================================
        # [核心修复] 使用 Jieba 分词，而不是 split()
        corpus = [jieba.lcut(doc) for doc in documents]
        # ========================================================

        bm25 = BM25Okapi(corpus)

        # 更新映射关系
        _bm25_node_map[kb_name] = ids

        # 持久化缓存
        if cache.set(kb_name, bm25):
            log(f"✅ BM25 索引构建并缓存成功")
        else:
            warn(f"⚠️ BM25 索引构建成功但缓存失败")

        return bm25

    except Exception as e:
        error(f"❌ 构建 BM25 索引失败: {kb_name} - {e}")
        import traceback
        traceback.print_exc()
        return None


def hybrid_retrieve(
        query: str,
        index: VectorStoreIndex,
        kb_name: str,
        top_k: int = 20,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
) -> List[NodeWithScore]:
    """混合检索：向量检索 + BM25 + RRF 融合 + Reranker"""

    # 1. 向量检索
    log(f"🔍 向量检索: {query[:50]}...")
    vector_retriever = index.as_retriever(similarity_top_k=VECTOR_TOP_K)
    vector_nodes = vector_retriever.retrieve(query)
    log(f"   └─ 向量检索返回: {len(vector_nodes)} 个结果")

    # 2. BM25 检索
    log(f"🔍 BM25 检索: {query[:50]}...")
    bm25_nodes = []
    # 尝试获取或构建索引
    bm25 = build_bm25_index(kb_name, index)
    node_ids_map = _bm25_node_map.get(kb_name, [])

    if bm25 is not None and node_ids_map:
        try:
            # 检索词也必须用 Jieba 分词
            query_tokens = jieba.lcut(query)
            bm25_scores = bm25.get_scores(query_tokens)

            # 获取分数最高的 Top K
            top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True
            )[:BM25_TOP_K]

            for i in top_indices:
                if i >= len(node_ids_map):
                    continue

                node_id = node_ids_map[i]
                score = float(bm25_scores[i])

                # 过滤掉分数极低的结果 (噪音)
                if score <= 0.0:
                    continue

                try:
                    node = index.docstore.get_node(node_id)
                    bm25_nodes.append(NodeWithScore(node=node, score=score))
                except Exception:
                    continue

            log(f"   └─ BM25 检索返回: {len(bm25_nodes)} 个结果")
        except Exception as e:
            error(f"❌ BM25 检索计算失败: {e}")

    # 3. RRF 融合
    if bm25_nodes:
        log(f"🔀 RRF 融合: 向量({len(vector_nodes)}) + BM25({len(bm25_nodes)})")
        fused_nodes = rrf_fusion(
            vector_nodes,
            bm25_nodes,
            top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight
        )
        log(f"   └─ 融合后返回: {len(fused_nodes)} 个结果")
    else:
        log("⚠️ 仅使用向量检索结果 (BM25 未返回或失败)")
        fused_nodes = vector_nodes[:top_k]

    # 4. Reranker 重排序
    if Settings.node_postprocessors:
        log("🎯 执行 Reranker 重排序...")
        query_bundle = QueryBundle(query_str=query)
        reranked_nodes = fused_nodes

        for processor in Settings.node_postprocessors:
            try:
                reranked_nodes = processor.postprocess_nodes(
                    reranked_nodes,
                    query_bundle=query_bundle
                )
            except Exception as e:
                error(f"❌ Reranker 执行失败: {e}")

        log(f"   └─ Reranker 后保留: {len(reranked_nodes)} 个结果")
        return reranked_nodes

    return fused_nodes[:FINAL_TOP_K]


def rrf_fusion(
        vector_nodes: List[NodeWithScore],
        bm25_nodes: List[NodeWithScore],
        top_k: int,
        k: int = RRF_K,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
) -> List[NodeWithScore]:
    """RRF (Reciprocal Rank Fusion) 融合算法"""
    scores = {}
    node_map = {}

    for rank, node in enumerate(vector_nodes, 1):
        node_id = node.node.node_id
        # 加权 RRF
        scores[node_id] = vector_weight / (k + rank)
        node_map[node_id] = node

    for rank, node in enumerate(bm25_nodes, 1):
        node_id = node.node.node_id
        if node_id in scores:
            scores[node_id] += bm25_weight / (k + rank)
        else:
            scores[node_id] = bm25_weight / (k + rank)
            node_map[node_id] = node

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]

    result = []
    for node_id in sorted_ids:
        node = node_map[node_id]
        result.append(NodeWithScore(node=node.node, score=scores[node_id]))

    return result


def invalidate_bm25_cache(kb_name: str) -> bool:
    """使 BM25 缓存失效"""
    cache = BM25Cache()
    success = cache.delete(kb_name)

    if kb_name in _bm25_node_map:
        del _bm25_node_map[kb_name]

    if success:
        log(f"✅ 已清除 BM25 缓存: {kb_name}")
    else:
        error(f"❌ 清除 BM25 缓存失败: {kb_name}")

    return success
