# src/core/bm25_cache.py
import os
import pickle
import threading
from typing import Optional
from rank_bm25 import BM25Okapi
from config.settings import STORAGE_DIR
from src.core.logger import log, error, warn


class BM25Cache:
    """
    BM25 索引缓存管理器

    特性:
    - 线程安全的单例实现
    - 自动持久化到磁盘
    - 支持并发读写
    - 异常容错机制
    """

    _instance: Optional['BM25Cache'] = None
    _init_lock = threading.Lock()  # 用于初始化的锁

    def __new__(cls):
        """双重检查锁定的单例模式"""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
                    instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化实例变量"""
        self.path = os.path.join(STORAGE_DIR, "bm25_cache.pkl")
        self.rw_lock = threading.RLock()  # 可重入锁，用于读写操作
        self.cache = self._load()
        log(f"BM25 缓存管理器初始化完成: {self.path}")

    def _load(self) -> dict:
        """
        从磁盘加载缓存

        Returns:
            dict: 缓存字典 {kb_name: bm25_index}
        """
        if not os.path.exists(self.path):
            log("BM25 缓存文件不存在，创建新缓存")
            return {}

        try:
            with open(self.path, "rb") as f:
                cache_data = pickle.load(f)
                log(f"成功加载 BM25 缓存: {len(cache_data)} 个知识库")
                return cache_data
        except (pickle.UnpicklingError, EOFError) as e:
            error(f"BM25 缓存文件损坏，创建新缓存: {e}")
            # 备份损坏的文件
            try:
                backup_path = f"{self.path}.corrupt.{os.getpid()}"
                os.rename(self.path, backup_path)
                warn(f"已备份损坏的缓存文件到: {backup_path}")
            except Exception:
                pass
            return {}
        except Exception as e:
            error(f"加载 BM25 缓存失败: {e}")
            return {}

    def _save(self):
        """
        保存缓存到磁盘（内部方法，需要在锁内调用）

        Raises:
            Exception: 保存失败时抛出异常
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

            # 先写入临时文件，然后原子性替换
            temp_path = f"{self.path}.tmp.{os.getpid()}"
            with open(temp_path, "wb") as f:
                pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 原子性替换（防止写入过程中断导致文件损坏）
            if os.path.exists(self.path):
                os.replace(temp_path, self.path)
            else:
                os.rename(temp_path, self.path)

            log(f"BM25 缓存已保存: {len(self.cache)} 个知识库")
        except Exception as e:
            error(f"保存 BM25 缓存失败: {e}")
            # 清理临时文件
            temp_path = f"{self.path}.tmp.{os.getpid()}"
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise

    def get(self, kb_name: str) -> Optional[BM25Okapi]:
        """
        获取知识库的 BM25 索引

        Args:
            kb_name: 知识库名称

        Returns:
            BM25Okapi 实例或 None
        """
        with self.rw_lock:
            bm25_index = self.cache.get(kb_name)
            if bm25_index is not None:
                log(f"从缓存获取 BM25 索引: {kb_name}")
            return bm25_index

    def set(self, kb_name: str, bm25_index: BM25Okapi) -> bool:
        """
        设置知识库的 BM25 索引并持久化

        Args:
            kb_name: 知识库名称
            bm25_index: BM25 索引实例

        Returns:
            bool: 是否成功保存
        """
        with self.rw_lock:
            try:
                self.cache[kb_name] = bm25_index
                self._save()
                log(f"✅ 保存 BM25 索引成功: {kb_name}")
                return True
            except Exception as e:
                error(f"❌ 保存 BM25 索引失败: {kb_name} - {e}")
                # 回滚操作
                if kb_name in self.cache:
                    del self.cache[kb_name]
                return False

    def delete(self, kb_name: str) -> bool:
        """
        删除知识库的 BM25 索引

        Args:
            kb_name: 知识库名称

        Returns:
            bool: 是否成功删除
        """
        with self.rw_lock:
            if kb_name not in self.cache:
                warn(f"BM25 索引不存在: {kb_name}")
                return True

            try:
                del self.cache[kb_name]
                self._save()
                log(f"✅ 删除 BM25 索引成功: {kb_name}")
                return True
            except Exception as e:
                error(f"❌ 删除 BM25 索引失败: {kb_name} - {e}")
                return False

    def exists(self, kb_name: str) -> bool:
        """
        检查知识库的 BM25 索引是否存在

        Args:
            kb_name: 知识库名称

        Returns:
            bool: 是否存在
        """
        with self.rw_lock:
            return kb_name in self.cache

    def list_all(self) -> list[str]:
        """
        列出所有已缓存的知识库

        Returns:
            list: 知识库名称列表
        """
        with self.rw_lock:
            return list(self.cache.keys())

    def clear(self) -> bool:
        """
        清空所有缓存

        Returns:
            bool: 是否成功清空
        """
        with self.rw_lock:
            try:
                self.cache.clear()
                self._save()
                log("✅ 已清空所有 BM25 缓存")
                return True
            except Exception as e:
                error(f"❌ 清空 BM25 缓存失败: {e}")
                return False

    def get_cache_info(self) -> dict:
        """
        获取缓存信息（用于调试和监控）

        Returns:
            dict: 缓存统计信息
        """
        with self.rw_lock:
            import os.path

            file_size = 0
            if os.path.exists(self.path):
                file_size = os.path.getsize(self.path)

            return {
                "total_kbs": len(self.cache),
                "kb_names": list(self.cache.keys()),
                "cache_file": self.path,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "file_exists": os.path.exists(self.path)
            }

    def __repr__(self) -> str:
        """字符串表示"""
        with self.rw_lock:
            return f"<BM25Cache: {len(self.cache)} knowledge bases>"


# 便捷访问函数
def get_cache() -> BM25Cache:
    """获取 BM25Cache 单例实例"""
    return BM25Cache()
