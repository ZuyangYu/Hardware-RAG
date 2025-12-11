# src/core/rag_pipeline.py
import os
import shutil
import time
from typing import List, Tuple
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from config.settings import DEFAULT_KB_NAME, DATA_ROOT
from config.settings import STORAGE_DIR
from src.ingestion.index_builder import (
    get_or_build_index,
    invalidate_index_cache
)
from src.ingestion.data_loader import get_kb_path, list_knowledge_bases
from src.core.resource_manager import resource_manager
from src.core.hybrid_retriever import invalidate_bm25_cache
from src.core.logger import log, warn, error
from src.core.custom_rag_chat import CustomRAGChat


class RAGPipeline:
    """RAG 核心逻辑"""
    SUPPORTED_FORMATS = {
        '.pdf', '.txt', '.md', '.docx', '.doc',
        '.html', '.htm', '.csv', '.json'
    }

    def __init__(self):
        """初始化仅负责资源检查，不再绑定特定 KB"""
        try:
            # 这里调用资源管理器，进行第一次全面初始化
            if not resource_manager.initialize():
                raise RuntimeError("资源初始化失败")
        except Exception as e:
            error(f"❌ 资源初始化异常: {e}")
            raise

        # 确保存储目录存在
        os.makedirs(DATA_ROOT, exist_ok=True)

    def get_index(self, kb_name: str):
        """获取指定知识库的索引"""
        return get_or_build_index(
            kb_name,
            resource_manager.chroma_client,
            use_cache=True
        )

    def list_knowledge_bases(self) -> List[str]:
        return list_knowledge_bases()

    def query(self, msg: str, kb_name: str, history: List[Tuple[str, str]]) -> str:
        """
        处理查询
        Args:
            msg: 用户问题
            kb_name: 目标知识库
            history: 对话历史 [[q, a], [q, a]]
        """
        if not msg.strip():
            return "请输入有效问题"

        if not kb_name:
            return "❌ 未选择知识库"

        try:
            # 获取对应知识库的索引
            index = self.get_index(kb_name)

            # 实例化聊天引擎（轻量级）
            chat_engine = CustomRAGChat(kb_name, index)

            # 生成回复
            response = chat_engine.chat(msg, history)
            return response

        except Exception as e:
            error(f"查询出错: {e}")
            return f"❌ 系统错误: {str(e)}"

    def upload_files(self, files, target_kb: str) -> str:
        if not files:
            return "未选择文件"
        if not target_kb:
            return "❌ 未选择目标知识库"

        results = []
        success_count = 0

        for file in files:
            file_path = file if isinstance(file, str) else file.name
            try:
                result = self.add_document(file_path, target_kb)
                results.append(result)
                if "✅" in result:
                    success_count += 1
            except Exception as e:
                error(f"上传文件失败 {file_path}: {e}")
                results.append(f"❌ {os.path.basename(file_path)}: {str(e)}")

        # 清除相关缓存
        if success_count > 0:
            invalidate_bm25_cache(target_kb)
            # 注意：Index Cache 不需要清除，因为我们是直接操作内存中的 Index 对象

        return f"✅ 成功上传 {success_count}/{len(files)} 个文件\n" + "\n".join(results)

    def add_document(self, temp_file_path: str, kb_name: str) -> str:
        """增量添加文档 """
        try:
            if not os.path.exists(temp_file_path):
                return "❌ 文件不存在"

            filename = os.path.basename(temp_file_path)
            _, ext = os.path.splitext(filename)

            if ext.lower() not in self.SUPPORTED_FORMATS:
                return f"❌ 不支持的文件格式: {ext}"

            # 1. 移动文件到知识库目录
            target_dir = get_kb_path(kb_name)
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, filename)

            if os.path.exists(target_path):
                base, ext = os.path.splitext(filename)
                filename = f"{base}_{int(time.time())}{ext}"
                target_path = os.path.join(target_dir, filename)
                log(f"文件名冲突，重命名为: {filename}")

            shutil.copy2(temp_file_path, target_path)

            # 2. 获取当前索引
            index = self.get_index(kb_name)

            # 3. 增量更新
            log(f"正在增量索引: {filename}")
            new_docs = SimpleDirectoryReader(input_files=[target_path]).load_data()

            # 此时 Settings.node_parser 已经是我们在 model_factory 里配置好的了
            nodes = Settings.node_parser.get_nodes_from_documents(new_docs)
            index.insert_nodes(nodes)

            # ✅ 持久化到指定目录 (DocStore)
            kb_persist_dir = os.path.join(STORAGE_DIR, f"docstore_{kb_name}")
            os.makedirs(kb_persist_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=kb_persist_dir)

            # ✅ 让 BM25 缓存失效，以便下次查询时包含新文件
            invalidate_bm25_cache(kb_name)
            invalidate_index_cache(kb_name)

            log(f"✅ 增量索引完成并保存: {filename}")
            return f"✅ 索引成功: {filename}"

        except Exception as e:
            error(f"❌ 上传文档失败: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ 上传失败: {str(e)}"

    def create_kb(self, name: str) -> Tuple[bool, str]:
        try:
            name = name.strip().replace(" ", "_")
            if not name: return False, "❌ 名称不能为空"

            path = get_kb_path(name)
            if os.path.exists(path): return False, "❌ 知识库已存在"

            os.makedirs(path, exist_ok=True)
            # 初始化一个空索引
            get_or_build_index(name, resource_manager.chroma_client, use_cache=False)
            return True, f"✅ 知识库 '{name}' 创建成功"
        except Exception as e:
            return False, str(e)

    def delete_document(self, filename: str, kb_name: str) -> str:
        """删除文档"""
        if not filename or not filename.strip():
            return "❌ 文件名不能为空"

        try:
            # 1. 删除物理文件
            path = os.path.join(get_kb_path(kb_name), filename)
            if os.path.exists(path):
                os.remove(path)
                log(f"🗑已删除文件: {filename}")

            # 2. 获取索引
            index = self.get_index(kb_name)

            # 首先需要找到该文件对应的所有 ref_doc_id
            try:
                vector_store = index._vector_store
                if isinstance(vector_store, ChromaVectorStore):
                    collection = vector_store._collection

                    # 查询所有包含该文件名的文档
                    results = collection.get(
                        where={"file_name": filename},
                        include=["metadatas"]
                    )

                    doc_ids_to_delete = set()
                    for metadata in results.get("metadatas", []):
                        # 提取 ref_doc_id
                        ref_doc_id = metadata.get("ref_doc_id") or metadata.get("doc_id")
                        if ref_doc_id:
                            doc_ids_to_delete.add(ref_doc_id)

                    # 通过 ref_doc_id 删除
                    for doc_id in doc_ids_to_delete:
                        try:
                            index.delete_ref_doc(doc_id, delete_from_docstore=True)
                            log(f"已删除文档向量: {doc_id}")
                        except Exception as e:
                            warn(f"删除文档向量失败 {doc_id}: {e}")

                    if not doc_ids_to_delete:
                        collection.delete(where={"file_name": filename})
                        log(f"通过 metadata 删除向量: {filename}")

            except Exception as e:
                error(f"向量清理失败: {e}")
            invalidate_index_cache(kb_name)
            invalidate_bm25_cache(kb_name)
            return f"✅ 已删除: {filename}"

        except Exception as e:
            error(f"删除文档失败: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ 删除失败: {str(e)}"

    def list_files(self, kb_name: str) -> List[str]:
        try:
            if not kb_name: return []
            kb_path = get_kb_path(kb_name)
            if not os.path.exists(kb_path): return []
            return sorted([f for f in os.listdir(kb_path) if os.path.isfile(os.path.join(kb_path, f))])
        except:
            return []

    def delete_knowledge_base(self, kb_name: str) -> Tuple[bool, str]:
        # (保持原有逻辑，增加缓存清理)
        if kb_name == DEFAULT_KB_NAME: return False, "不可删除默认库"
        try:
            coll_name = f"kb_{kb_name}"
            try:
                resource_manager.chroma_client.delete_collection(coll_name)
            except:
                pass

            kb_path = get_kb_path(kb_name)
            if os.path.exists(kb_path): shutil.rmtree(kb_path)

            invalidate_index_cache(kb_name)
            invalidate_bm25_cache(kb_name)
            return True, "已删除"
        except Exception as e:
            return False, str(e)
