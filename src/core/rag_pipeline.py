# src/core/rag_pipeline.py
import os
import shutil
import time
from typing import List, Tuple, Generator
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from config.settings import DEFAULT_KB_NAME, DATA_ROOT, STORAGE_DIR
from src.ingestion.index_builder import get_or_build_index, invalidate_index_cache
from src.ingestion.data_loader import get_kb_path, list_knowledge_bases
from src.core.resource_manager import resource_manager
from src.core.hybrid_retriever import invalidate_bm25_cache
from src.core.logger import log, warn, error
from src.core.custom_rag_chat import CustomRAGChat


class RAGPipeline:
    """RAG 核心逻辑"""
    SUPPORTED_FORMATS = {'.pdf', '.txt', '.md', '.docx', '.doc', '.html', '.htm', '.csv', '.json'}

    def __init__(self):
        try:
            if not resource_manager.initialize():
                raise RuntimeError("资源初始化失败")
        except Exception as e:
            error(f"❌ 资源初始化异常: {e}")
            raise
        os.makedirs(DATA_ROOT, exist_ok=True)

    def get_index(self, kb_name: str):
        return get_or_build_index(kb_name, resource_manager.chroma_client, use_cache=True)

    def list_knowledge_bases(self) -> List[str]:
        return list_knowledge_bases()

    def query(self, msg: str, kb_name: str, history: List[Tuple[str, str]]) -> Generator[str, None, None]:
        """处理查询 - 返回一个流式响应的生成器"""
        if not msg.strip():
            def empty_gen():
                yield "请输入有效问题"
            return empty_gen()

        if not kb_name:
            def empty_gen():
                yield "❌ 未选择知识库"
            return empty_gen()

        try:
            index = self.get_index(kb_name)
            chat_engine = CustomRAGChat(kb_name, index)
            return chat_engine.chat(msg, history)
        except Exception as e:
            error(f"查询出错: {e}")
            def error_gen():
                yield f"❌ 系统错误: {str(e)}"
            return error_gen()

    def upload_files(self, files, target_kb: str) -> str:
        if not files: return "未选择文件"
        if not target_kb: return "❌ 未选择目标知识库"
        results, success_count = [], 0
        for file in files:
            file_path = file if isinstance(file, str) else file.name
            try:
                result = self.add_document(file_path, target_kb)
                results.append(result)
                if "✅" in result: success_count += 1
            except Exception as e:
                error(f"上传文件失败 {file_path}: {e}")
                results.append(f"❌ {os.path.basename(file_path)}: {str(e)}")
        if success_count > 0:
            invalidate_bm25_cache(target_kb)
        return f"✅ 成功上传 {success_count}/{len(files)} 个文件\n" + "\n".join(results)

    def add_document(self, temp_file_path: str, kb_name: str) -> str:
        try:
            if not os.path.exists(temp_file_path): return "❌ 文件不存在"
            filename = os.path.basename(temp_file_path)
            _, ext = os.path.splitext(filename)
            if ext.lower() not in self.SUPPORTED_FORMATS: return f"❌ 不支持的文件格式: {ext}"
            target_dir = get_kb_path(kb_name)
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, filename)
            if os.path.exists(target_path):
                base, ext = os.path.splitext(filename)
                filename = f"{base}_{int(time.time())}{ext}"
                target_path = os.path.join(target_dir, filename)
            shutil.copy2(temp_file_path, target_path)
            index = self.get_index(kb_name)
            new_docs = SimpleDirectoryReader(input_files=[target_path]).load_data()
            nodes = Settings.node_parser.get_nodes_from_documents(new_docs)
            index.insert_nodes(nodes)
            kb_persist_dir = os.path.join(STORAGE_DIR, f"docstore_{kb_name}")
            os.makedirs(kb_persist_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=kb_persist_dir)
            invalidate_bm25_cache(kb_name)
            invalidate_index_cache(kb_name)
            return f"✅ 索引成功: {filename}"
        except Exception as e:
            error(f"❌ 上传文档处理失败: {e}")
            if 'target_path' in locals() and os.path.exists(target_path):
                os.remove(target_path)
            raise e

    def create_kb(self, name: str) -> Tuple[bool, str]:
        try:
            name = name.strip().replace(" ", "_")
            if not name: return False, "❌ 名称不能为空"
            path = get_kb_path(name)
            if os.path.exists(path): return False, "❌ 知识库已存在"
            os.makedirs(path, exist_ok=True)
            get_or_build_index(name, resource_manager.chroma_client, use_cache=False)
            return True, f"✅ 知识库 '{name}' 创建成功"
        except Exception as e:
            return False, str(e)

    def delete_document(self, filename: str, kb_name: str) -> str:
        if not filename or not filename.strip(): return "❌ 文件名不能为空"
        try:
            path = os.path.join(get_kb_path(kb_name), filename)
            if os.path.exists(path): os.remove(path)
            index = self.get_index(kb_name)
            vector_store = index._vector_store
            if isinstance(vector_store, ChromaVectorStore):
                collection = vector_store._collection
                results = collection.get(where={"file_name": filename}, include=["metadatas"])
                doc_ids_to_delete = {meta.get("ref_doc_id") or meta.get("doc_id") for meta in results.get("metadatas", [])}
                for doc_id in doc_ids_to_delete:
                    if doc_id: index.delete_ref_doc(doc_id, delete_from_docstore=True)
                if not doc_ids_to_delete:
                    collection.delete(where={"file_name": filename})
            invalidate_index_cache(kb_name)
            invalidate_bm25_cache(kb_name)
            return f"✅ 已删除: {filename}"
        except Exception as e:
            error(f"删除文档失败: {e}")
            return f"❌ 删除失败: {str(e)}"

    def list_files(self, kb_name: str) -> List[str]:
        if not kb_name: return []
        kb_path = get_kb_path(kb_name)
        if not os.path.exists(kb_path): return []
        return sorted([f for f in os.listdir(kb_path) if os.path.isfile(os.path.join(kb_path, f))])

    def delete_knowledge_base(self, kb_name: str) -> Tuple[bool, str]:
        if kb_name == DEFAULT_KB_NAME: return False, "不可删除默认库"
        try:
            coll_name = f"kb_{kb_name}"
            try:
                resource_manager.chroma_client.delete_collection(coll_name)
            except: pass
            kb_path = get_kb_path(kb_name)
            if os.path.exists(kb_path): shutil.rmtree(kb_path)
            invalidate_index_cache(kb_name)
            invalidate_bm25_cache(kb_name)
            return True, "已删除"
        except Exception as e:
            return False, str(e)
