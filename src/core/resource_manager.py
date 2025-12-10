# src/core/resource_manager.py
import chromadb
import threading
import time
import atexit
from typing import Optional
from config.settings import CHROMA_PATH
from src.core.model_factory import init_global_models
from src.core.logger import log, error, warn


class ResourceManager:
    """
    全局资源管理器（线程安全的单例模式 + 上下文管理器）

    功能:
    - 管理 ChromaDB 客户端连接
    - 管理全局模型（LLM、Embedding、Reranker）
    - 提供连接重试机制
    - 健康检查和资源监控
    - 自动资源清理
    """
    _instance: Optional['ResourceManager'] = None
    _init_lock = threading.Lock()

    # 配置参数
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 2
    CONNECTION_TIMEOUT = 10

    def __new__(cls):
        """双重检查锁定的单例模式"""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self):
        """初始化（只执行一次）"""
        if self._initialized:
            return

        with self._init_lock:
            if self._initialized:
                return

            self._chroma_client: Optional[chromadb.PersistentClient] = None
            self._models_initialized = False
            self._chroma_lock = threading.RLock()
            self._model_lock = threading.RLock()
            self._health_status = {
                "chroma": False,
                "models": False,
                "last_check": None
            }
            self._is_shutdown = False  # ✅ 新增：标记是否已关闭
            self._initialized = True

            # ✅ 自动注册清理函数
            atexit.register(self._atexit_cleanup)

            log("📦 资源管理器已创建")

    def _atexit_cleanup(self):
        """程序退出时自动清理"""
        if not self._is_shutdown:
            log("🔔 检测到程序退出，执行资源清理...")
            self.shutdown()

    # ==================== 上下文管理器协议 ====================
    def __enter__(self):
        """进入上下文：确保资源已初始化"""
        if not self.initialize():
            raise RuntimeError("资源初始化失败")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文：自动清理资源"""
        self.shutdown()
        # 返回 False 表示不抑制异常
        return False

    # ==================== 初始化方法（保持不变）====================
    def initialize(self, force: bool = False) -> bool:
        """初始化所有全局资源"""
        if self._is_shutdown:
            warn("⚠️ 资源管理器已关闭，无法重新初始化")
            return False

        if not force and self._models_initialized and self._chroma_client is not None:
            log("✅ 资源已初始化，跳过")
            return True

        log("=" * 70)
        log("🚀 开始初始化全局资源")
        log("=" * 70)

        success = True

        # 1. 初始化模型
        if not self._initialize_models(force):
            success = False

        # 2. 初始化 ChromaDB
        if not self._initialize_chroma(force):
            success = False

        log("=" * 70)
        if success:
            log("✅ 全局资源初始化完成")
            self._health_status["last_check"] = time.time()
        else:
            error("❌ 全局资源初始化部分失败")
        log("=" * 70)

        return success

    def _initialize_models(self, force: bool = False) -> bool:
        """初始化全局模型"""
        with self._model_lock:
            if self._is_shutdown:
                return False

            if not force and self._models_initialized:
                log("✅ 模型已初始化，跳过")
                return True

            try:
                log("🤖 初始化全局模型...")
                init_global_models()
                self._models_initialized = True
                self._health_status["models"] = True
                log("✅ 模型初始化成功")
                return True

            except Exception as e:
                error(f"❌ 模型初始化失败: {e}")
                self._health_status["models"] = False
                import traceback
                traceback.print_exc()
                return False

    def _initialize_chroma(self, force: bool = False) -> bool:
        """初始化 ChromaDB 客户端"""
        with self._chroma_lock:
            if self._is_shutdown:
                return False

            if not force and self._chroma_client is not None:
                if self._test_chroma_connection():
                    log("✅ ChromaDB 已连接，跳过")
                    return True
                else:
                    warn("⚠️ ChromaDB 连接失效，尝试重新连接")
                    self._chroma_client = None

            log(f"🗄️ 连接 ChromaDB: {CHROMA_PATH}")

            for attempt in range(1, self.MAX_RETRIES + 1):
                try:
                    log(f"   尝试 {attempt}/{self.MAX_RETRIES}...")

                    self._chroma_client = chromadb.PersistentClient(
                        path=CHROMA_PATH,
                        settings=chromadb.Settings(
                            allow_reset=True,
                            anonymized_telemetry=False,
                            is_persistent=True
                        )
                    )

                    if self._test_chroma_connection():
                        log(f"✅ ChromaDB 连接成功")
                        self._health_status["chroma"] = True
                        return True
                    else:
                        raise Exception("连接测试失败")

                except Exception as e:
                    error(f"❌ ChromaDB 连接失败 (尝试 {attempt}/{self.MAX_RETRIES}): {e}")
                    self._chroma_client = None
                    self._health_status["chroma"] = False

                    if attempt < self.MAX_RETRIES:
                        delay = self.RETRY_DELAY_BASE ** attempt
                        log(f"⏳ 等待 {delay} 秒后重试...")
                        time.sleep(delay)
                    else:
                        error("❌ ChromaDB 连接失败，已达最大重试次数")
                        import traceback
                        traceback.print_exc()
                        return False

            return False

    def _test_chroma_connection(self) -> bool:
        """测试 ChromaDB 连接是否有效"""
        if self._chroma_client is None:
            return False

        try:
            self._chroma_client.list_collections()
            return True
        except Exception as e:
            warn(f"ChromaDB 连接测试失败: {e}")
            return False

    @property
    def chroma_client(self) -> chromadb.PersistentClient:
        """获取 ChromaDB 客户端（带自动重连）"""
        if self._is_shutdown:
            raise RuntimeError("资源管理器已关闭，无法访问 ChromaDB 客户端")

        with self._chroma_lock:
            if self._chroma_client is None or not self._test_chroma_connection():
                log("⚠️ ChromaDB 连接不可用，尝试重新连接...")
                if not self._initialize_chroma(force=True):
                    raise RuntimeError(
                        "无法连接到 ChromaDB。请检查:\n"
                        f"1. 数据库路径是否正确: {CHROMA_PATH}\n"
                        "2. 是否有足够的磁盘空间\n"
                        "3. 是否有文件访问权限\n"
                        "4. ChromaDB 进程是否正常"
                    )

            return self._chroma_client

    def health_check(self) -> dict:
        """执行健康检查"""
        if self._is_shutdown:
            return {
                "overall": False,
                "status": "已关闭",
                "last_check": time.time()
            }

        log("🏥 执行健康检查...")

        models_ok = self._models_initialized
        chroma_ok = self._test_chroma_connection()

        chroma_stats = {}
        if chroma_ok and self._chroma_client:
            try:
                collections = self._chroma_client.list_collections()
                total_vectors = sum(col.count() for col in collections)
                chroma_stats = {
                    "collections": len(collections),
                    "total_vectors": total_vectors,
                    "collection_names": [col.name for col in collections]
                }
            except Exception as e:
                error(f"获取 ChromaDB 统计信息失败: {e}")

        status = {
            "overall": models_ok and chroma_ok,
            "models": {
                "status": "✅ 正常" if models_ok else "❌ 异常",
                "initialized": models_ok
            },
            "chroma": {
                "status": "✅ 正常" if chroma_ok else "❌ 异常",
                "connected": chroma_ok,
                "path": CHROMA_PATH,
                **chroma_stats
            },
            "last_check": time.time()
        }

        self._health_status.update({
            "models": models_ok,
            "chroma": chroma_ok,
            "last_check": time.time()
        })

        log(f"   模型: {status['models']['status']}")
        log(f"   ChromaDB: {status['chroma']['status']}")
        if chroma_stats:
            log(f"   - 集合数: {chroma_stats.get('collections', 0)}")
            log(f"   - 总向量数: {chroma_stats.get('total_vectors', 0)}")

        return status

    def get_status(self) -> dict:
        """获取当前状态（不执行检查）"""
        return {
            "models_initialized": self._models_initialized,
            "chroma_connected": self._chroma_client is not None,
            "is_shutdown": self._is_shutdown,
            "health_status": self._health_status.copy(),
            "chroma_path": CHROMA_PATH
        }

    def reset_chroma(self) -> bool:
        """重置 ChromaDB 连接"""
        if self._is_shutdown:
            warn("⚠️ 资源管理器已关闭")
            return False

        with self._chroma_lock:
            log("🔄 重置 ChromaDB 连接...")

            if self._chroma_client is not None:
                try:
                    self._chroma_client = None
                    log("✅ 已关闭旧连接")
                except Exception as e:
                    warn(f"关闭旧连接时出错: {e}")

            return self._initialize_chroma(force=True)

    def reset_models(self) -> bool:
        """重置模型"""
        if self._is_shutdown:
            warn("⚠️ 资源管理器已关闭")
            return False

        with self._model_lock:
            log("🔄 重置模型...")
            self._models_initialized = False
            return self._initialize_models(force=True)

    def reset_all(self) -> bool:
        """重置所有资源"""
        if self._is_shutdown:
            warn("⚠️ 资源管理器已关闭")
            return False

        log("🔄 重置所有资源...")
        models_ok = self.reset_models()
        chroma_ok = self.reset_chroma()
        return models_ok and chroma_ok

    # ✅ 优化后的 shutdown 方法
    def shutdown(self):
        """优雅关闭资源（幂等操作）"""
        # 防止重复关闭
        if self._is_shutdown:
            return

        log("🛑 关闭资源管理器...")
        self._is_shutdown = True

        # 1. 关闭 ChromaDB 连接
        with self._chroma_lock:
            if self._chroma_client is not None:
                try:
                    # ChromaDB 的 PersistentClient 会自动处理清理
                    log("🗄️ 正在关闭 ChromaDB 连接...")
                    self._chroma_client = None
                    self._health_status["chroma"] = False
                    log("✅ ChromaDB 连接已关闭")
                except Exception as e:
                    error(f"关闭 ChromaDB 连接时出错: {e}")

        # 2. 清理模型状态
        with self._model_lock:
            try:
                log("🤖 正在清理模型状态...")
                self._models_initialized = False
                self._health_status["models"] = False
                log("✅ 模型状态已清理")
            except Exception as e:
                error(f"清理模型状态时出错: {e}")

        log("✅ 资源管理器已完全关闭")

    def __repr__(self) -> str:
        """字符串表示"""
        status = "已关闭" if self._is_shutdown else "运行中"
        return (
            f"<ResourceManager[{status}]: "
            f"models={'✅' if self._models_initialized else '❌'}, "
            f"chroma={'✅' if self._chroma_client else '❌'}>"
        )

    def __del__(self):
        """析构函数：确保资源被清理"""
        if not self._is_shutdown:
            warn("⚠️ ResourceManager 未正常关闭，执行紧急清理")
            try:
                self.shutdown()
            except:
                pass


# ============================================================
# 全局访问点
# ============================================================

resource_manager = ResourceManager()


# ============================================================
# 便捷函数
# ============================================================

def get_chroma_client() -> chromadb.PersistentClient:
    """获取 ChromaDB 客户端"""
    return resource_manager.chroma_client


def ensure_resources_initialized() -> bool:
    """确保资源已初始化"""
    return resource_manager.initialize()


def get_resource_status() -> dict:
    """获取资源状态"""
    return resource_manager.get_status()


def perform_health_check() -> dict:
    """执行健康检查"""
    return resource_manager.health_check()