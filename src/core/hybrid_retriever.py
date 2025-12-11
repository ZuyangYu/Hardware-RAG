# src/core/hybrid_retriever.py
from typing import List, Optional, Tuple
import jieba
from rank_bm25 import BM25Okapi
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from config.settings import VECTOR_TOP_K, BM25_TOP_K, RRF_K, FINAL_TOP_K
from src.core.bm25_cache import BM25Cache
from src.core.logger import log, error


def aggressive_tokenize(text: str) -> List[str]:
    """强力分词"""
    if not text:
        return []
    text = text.lower()
    tokens = jieba.lcut(text)
    return [t for t in tokens if t.strip()]


def build_bm25_index(kb_name: str, index: VectorStoreIndex, force_rebuild: bool = False) -> Optional[
    Tuple[BM25Okapi, List[str]]]:
    """
    构建或获取 BM25 索引
    """
    cache = BM25Cache()
    vector_store = index._vector_store
    if not isinstance(vector_store, ChromaVectorStore):
        return None

    collection = vector_store._collection
    current_doc_count = collection.count()

    if current_doc_count == 0:
        return None

    if not force_rebuild:
        cached_data = cache.get(kb_name)
        if cached_data is not None:
            bm25, cached_ids = cached_data

            # ✅ 优化 1: 先比对数量，数量不对直接重建，省去昂贵的 ID 比对
            if len(cached_ids) != current_doc_count:
                log(f"文档数量变更 (缓存:{len(cached_ids)} vs DB:{current_doc_count}) -> 触发重建")
            else:
                # 数量一致，进一步校验 ID (确保不是删一个加一个的情况)
                # 虽然这一步稍慢，但为了严谨性保留，因为此时我们已经知道数量是对的
                current_chroma_ids = collection.get(include=[])["ids"]
                if set(cached_ids) == set(current_chroma_ids):
                    # log(f"✅ 使用缓存的 BM25 索引: {kb_name}")
                    return bm25, cached_ids
                else:
                    log("文档 ID 列表不匹配 -> 触发重建")

    try:
        log(f"构建 BM25 索引: {kb_name} ({current_doc_count} docs)")

        # 获取所有文档
        results = collection.get(include=["documents", "metadatas"])
        documents = results["documents"]
        ids = results["ids"]

        valid_docs = []
        valid_ids = []

        # 尝试优先从 DocStore 同步内容，保证一致性
        for i, doc_id in enumerate(ids):
            try:
                node = index.docstore.get_node(doc_id)
                text = node.get_content()
                valid_docs.append(text)
                valid_ids.append(doc_id)
            except:
                # 容错：DocStore 没有就用 Chroma 的
                if documents[i]:
                    valid_docs.append(documents[i])
                    valid_ids.append(doc_id)

        if not valid_ids:
            return None

        corpus = [aggressive_tokenize(doc) for doc in valid_docs]
        bm25 = BM25Okapi(corpus)

        # 存入缓存
        cache.set(kb_name, (bm25, valid_ids))
        return bm25, valid_ids

    except Exception as e:
        error(f"❌ 构建 BM25 索引失败: {e}")
        return None


def hybrid_retrieve(
        query: str,
        index: VectorStoreIndex,
        kb_name: str,
        top_k: int = 20,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
) -> List[NodeWithScore]:
    """混合检索 (带自动修复功能)"""

    # 1. 向量检索
    log(f"向量检索: {query[:20]}...")
    try:
        vector_retriever = index.as_retriever(similarity_top_k=VECTOR_TOP_K)
        vector_nodes = vector_retriever.retrieve(query)
        log(f"   └─ 向量检索返回: {len(vector_nodes)} 个结果")
    except Exception as e:
        error(f"向量检索失败: {e}")
        vector_nodes = []

    # 2. BM25 检索
    bm25_nodes = []
    bm25_data = build_bm25_index(kb_name, index, force_rebuild=False)

    if bm25_data:
        bm25, node_ids = bm25_data
        try:
            query_tokens = aggressive_tokenize(query)
            if query_tokens:
                bm25_scores = bm25.get_scores(query_tokens)
                top_indices = sorted(
                    range(len(bm25_scores)),
                    key=lambda i: bm25_scores[i],
                    reverse=True
                )[:BM25_TOP_K]

                for i in top_indices:
                    if i >= len(node_ids): continue
                    score = float(bm25_scores[i])
                    if score <= 0.0: continue

                    node_id = node_ids[i]

                    # ✅ 增加兜底机制，防止 DocStore 丢失导致结果为空
                    try:
                        node = index.docstore.get_node(node_id)
                        bm25_nodes.append(NodeWithScore(node=node, score=score))
                    except Exception:
                        # Fallback: 尝试直接从 ChromaDB 恢复节点
                        try:
                            if isinstance(index.vector_store, ChromaVectorStore):
                                # 直接查库
                                res = index.vector_store._collection.get(ids=[node_id],
                                                                         include=["documents", "metadatas"])
                                if res["documents"] and len(res["documents"]) > 0:
                                    # 现场重建节点
                                    recovered_node = TextNode(
                                        text=res["documents"][0],
                                        id_=node_id,
                                        metadata=res["metadatas"][0] if res["metadatas"] else {}
                                    )
                                    bm25_nodes.append(NodeWithScore(node=recovered_node, score=score))
                                else:
                                    pass
                        except:
                            pass  # 恢复也失败，彻底放弃

                log(f"   └─ BM25 检索返回: {len(bm25_nodes)} 个结果")
        except Exception as e:
            error(f"BM25 计算出错: {e}")

    # 3. RRF 融合
    if bm25_nodes:
        fused_nodes = rrf_fusion(vector_nodes, bm25_nodes, top_k, vector_weight, bm25_weight)
    else:
        fused_nodes = vector_nodes[:top_k]

    # 4. Reranker
    if Settings.node_postprocessors:
        try:
            query_bundle = QueryBundle(query_str=query)
            reranked_nodes = fused_nodes
            for processor in Settings.node_postprocessors:
                reranked_nodes = processor.postprocess_nodes(reranked_nodes, query_bundle=query_bundle)
            return reranked_nodes
        except Exception as e:
            error(f"Reranker 失败: {e}")
            return fused_nodes[:FINAL_TOP_K]

    return fused_nodes[:FINAL_TOP_K]


def rrf_fusion(
        vector_nodes: List[NodeWithScore],
        bm25_nodes: List[NodeWithScore],
        top_k: int,
        k: int = RRF_K,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
) -> List[NodeWithScore]:
    """RRF 融合算法 (保持不变)"""
    scores = {}
    node_map = {}

    for rank, node in enumerate(vector_nodes, 1):
        node_id = node.node.node_id
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
    return [NodeWithScore(node=node_map[nid].node, score=scores[nid]) for nid in sorted_ids]


def invalidate_bm25_cache(kb_name: str) -> bool:
    """清除 BM25 缓存"""
    cache = BM25Cache()
    return cache.delete(kb_name)
