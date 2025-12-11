# src/streamlit_app.py
import os
import tempfile
import streamlit as st
import time
from src.core.rag_pipeline import RAGPipeline
from src.core.resource_manager import resource_manager
from config.settings import DEFAULT_KB_NAME

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="HardWare RAG",
    page_icon="😺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS 样式 ====================
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ========== 侧边栏样式自定义 ========== */
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span {
        font-size: 16px !important;
        line-height: 1.8 !important;
    }
    section[data-testid="stSidebar"] .stRadio label p {
        font-size: 18px !important;
        font-weight: 500 !important;
        padding-bottom: 8px !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-size: 20px !important;
        padding-top: 5px !important;
        padding-bottom: 30px !important;
    }
    section[data-testid="stSidebar"] hr {
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
    }

    /* 聊天气泡样式 */
    .user-msg-container {
        background-color: #e3f2fd;
        padding: 15px 20px;
        border-radius: 15px 15px 0 15px;
        margin: 10px 0;
        float: right;
        max-width: 85%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        text-align: left;
    }

    .assistant-msg-container {
        background-color: #f5f5f5;
        padding: 15px 20px;
        border-radius: 15px 15px 15px 0;
        margin: 10px 0;
        float: left;
        max-width: 85%;
        border: 1px solid #e0e0e0;
        text-align: left;
    }

    /* 状态指示灯 */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    .status-ok { background-color: #4caf50; }
    .status-error { background-color: #f44336; }

    .stButton button {
        transition: all 0.2s;
    }
</style>
""", unsafe_allow_html=True)


# ==================== 初始化逻辑 ====================
@st.cache_resource  # 这个装饰器保证 pipeline 只初始化一次，不会每点一下按钮就重启
def init_pipeline():
    """初始化 RAG Pipeline"""
    try:
        pipeline = RAGPipeline() # 初始化总的管道
        # 确保默认库存在
        pipeline.create_kb(DEFAULT_KB_NAME)
        return pipeline, None
    except Exception as e:
        return None, str(e)


def init_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_kb" not in st.session_state:
        st.session_state.current_kb = DEFAULT_KB_NAME
    if "kb_list" not in st.session_state:
        st.session_state.kb_list = []
    if "show_create_kb" not in st.session_state:
        st.session_state.show_create_kb = False

    if "confirm_delete_file" not in st.session_state:
        st.session_state.confirm_delete_file = None
    if "confirm_delete_kb" not in st.session_state:
        st.session_state.confirm_delete_kb = None

    if "toast_msg" not in st.session_state:
        st.session_state.toast_msg = None
    if "error_msg" not in st.session_state:
        st.session_state.error_msg = None


# ==================== 逻辑处理回调函数 ====================
def create_kb_callback(pipeline):
    """创建知识库回调"""
    name = st.session_state.get("new_kb_name_input", "").strip()

    if not name:
        st.session_state.error_msg = "❌ 名称不能为空"
        return

    ok, msg = pipeline.create_kb(name)
    if ok:
        st.session_state.kb_list = pipeline.list_knowledge_bases()
        st.session_state.current_kb = name
        st.session_state.kb_selector = name
        st.session_state.show_create_kb = False
        st.session_state.toast_msg = msg
    else:
        st.session_state.error_msg = msg


def delete_kb_confirmed(pipeline, kb_name):
    """执行已确认的知识库删除"""
    pipeline.delete_knowledge_base(kb_name)

    if st.session_state.current_kb == kb_name:
        st.session_state.current_kb = DEFAULT_KB_NAME
        st.session_state.kb_selector = DEFAULT_KB_NAME
        st.session_state.messages = []

    st.session_state.kb_list = pipeline.list_knowledge_bases()
    st.session_state.confirm_delete_kb = None
    st.session_state.toast_msg = f"已删除知识库: {kb_name}"


def switch_kb_callback(kb_name):
    """切换知识库回调"""
    st.session_state.current_kb = kb_name
    st.session_state.kb_selector = kb_name
    st.session_state.messages = []
    st.session_state.confirm_delete_file = None
    st.session_state.confirm_delete_kb = None


def refresh_kb_list(pipeline):
    st.session_state.kb_list = pipeline.list_knowledge_bases()


# ==================== 主界面 ====================
def main():
    init_session_state()        # 初始化一些变量（比如聊天记录）
    pipeline, error = init_pipeline() # 初始化

    if error:
        st.error(f"❌ 系统初始化失败: {error}")
        st.stop()

    if st.session_state.toast_msg:
        st.toast(st.session_state.toast_msg)
        st.session_state.toast_msg = None
        time.sleep(0.5)

    if st.session_state.error_msg:
        st.error(st.session_state.error_msg)
        st.session_state.error_msg = None

    if not st.session_state.kb_list:
        refresh_kb_list(pipeline)

    # ------------------ 顶部栏 ------------------
    col_header, col_status = st.columns([4, 1])
    with col_header:
        st.title("😺 HardWare RAG")

    with col_status:
        # 状态显示
        status = resource_manager.get_status()
        st.markdown(f"""
            <div style="text-align:right; padding-top:10px;">
                <span class="status-indicator {'status-ok' if status.get('models_initialized') else 'status-error'}"></span> AI模型<br>
                <span class="status-indicator {'status-ok' if status.get('chroma_connected') else 'status-error'}"></span> 向量库</div>
        """, unsafe_allow_html=True)

    # ------------------ 侧边栏 ------------------
    with st.sidebar:
        st.subheader("😼 Hardware RAG导航")

        selected_tab = st.radio(
            "功能切换",
            ["💬 智能对话", "📚 知识库管理"],
            label_visibility="collapsed"
        )

        st.divider()

        st.markdown(f"**📍 当前知识库:**")
        if st.session_state.current_kb not in st.session_state.kb_list:
            st.session_state.current_kb = DEFAULT_KB_NAME
            if DEFAULT_KB_NAME not in st.session_state.kb_list:
                st.session_state.kb_list.append(DEFAULT_KB_NAME)

        selected_kb = st.selectbox(
            "选择知识库",
            options=st.session_state.kb_list,
            key="kb_selector"
        )

        if selected_kb != st.session_state.current_kb:
            st.session_state.current_kb = selected_kb
            st.session_state.messages = []
            st.session_state.confirm_delete_file = None
            st.rerun()

        st.info(f"当前库包含 {len(pipeline.list_files(st.session_state.current_kb))} 个文件")

        st.divider()
        st.markdown("### 🐱‍👓️ 说明与注意事项")
        st.warning(
            """
            **1. 文件支持:**
            支持 PDF, TXT, MD, DOCX, CSV, HTML 格式文档。

            **2. 知识库管理:**
            - **新建**: 点击"知识库管理"页面的"➕ 新建"。
            - **切换**: 切换知识库会**清空当前对话**。

            **3. 数据安全:**
            - 删除文件或知识库的操作是**不可恢复**的，请谨慎操作。
            - 默认库 `source_documents` 不可被删除。
            """
        )
        st.caption("© 2025 HardWare RAG Assistant")

    # ------------------ 页面内容分发 ------------------
    if selected_tab == "💬 智能对话":
        render_chat_tab(pipeline)
    elif selected_tab == "📚 知识库管理":
        render_kb_management_tab(pipeline)


# ==================== Tab 1: 对话界面 ====================
def render_chat_tab(pipeline):
    st.caption(f"正在使用知识库: `{st.session_state.current_kb}`")

    chat_container = st.container(height=650, border=True)

    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                "<div style='text-align:center; color:gray; padding-top:200px;'>👋 你好！我是硬件检索助手，请问有什么可以帮你？</div>",
                unsafe_allow_html=True)

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                safe_content = msg["content"].replace("\n", "<br>")

                html_code = f"""
<div style="overflow: hidden;">
<div class="user-msg-container">
<strong>🐱‍👤 :</strong><br>{safe_content}
</div>
</div>
"""
                st.markdown(html_code, unsafe_allow_html=True)

            else:
                content = msg["content"]
                source_display = ""
                if "🔍 检索到的上下文" in content:
                    main_text, source_text = content.split("🔍 检索到的上下文", 1)
                    content = main_text.strip()
                    safe_source = source_text.replace("\n", "<br>")

                    source_display = f"""
<details style="margin-top:10px; border-top:1px solid #ddd; padding-top:5px;">
<summary style="cursor:pointer; color:#2196f3;">📚 参考来源 (点击展开)</summary>
<div style="font-size:0.9em; color:#666; margin-top:5px;">
{safe_source}
</div>
</details>
"""

                # 内容替换换行符
                safe_content = content.replace("\n", "<br>")

                # 助手消息 HTML 顶格
                html_code = f"""
<div style="overflow: hidden;">
<div class="assistant-msg-container">
<strong>😽 :</strong><br>{safe_content}
{source_display}
</div>
</div>
"""
                st.markdown(html_code, unsafe_allow_html=True)

    st.markdown("---")
    col_input, col_btn = st.columns([6, 1])

    with col_input:
        user_input = st.chat_input("请输入您的问题...", key="chat_input")

    with col_btn:
        if st.button("🗑️ 清空", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_container:
            with st.spinner("😻 正在检索与思考..."):
                history = [(m["content"], "") for m in st.session_state.messages if m["role"] == "user"]
                response = pipeline.query(user_input, st.session_state.current_kb, history[-5:])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


# ==================== Tab 2: 管理界面 ====================
def render_kb_management_tab(pipeline):
    st.subheader("📚 知识库管理")

    # --- 1. 上传区建立索引区 ---
    with st.container(border=True):
        st.markdown("##### 📤 当前知识库上传文档")
        files = st.file_uploader(
            "拖拽文件到此处",
            accept_multiple_files=True,
            type=["pdf", "txt", "md", "docx", "html", "csv"]
        )

        if files and st.button("开始上传", type="primary"):
            with st.status("处理中...", expanded=True) as status:
                st.write("保存临时文件...")
                temp_paths = []
                temp_dir = tempfile.gettempdir()
                for f in files:
                    path = os.path.join(temp_dir, f.name)
                    with open(path, "wb") as wb:
                        wb.write(f.getbuffer())
                    temp_paths.append(path)

                st.write("正在建立索引...")
                res = pipeline.upload_files(temp_paths, st.session_state.current_kb)

                for p in temp_paths:
                    try:
                        os.remove(p)
                    except:
                        pass

                status.update(label="✅ 完成", state="complete", expanded=False)

            st.success(res.split('\n')[0])
            time.sleep(1)
            st.rerun()
    st.divider()

    # --- 2. 列表与切换区 ---
    st.markdown("##### 📁 知识库列表")

    col_kbs, col_new = st.columns([9, 1])
    with col_kbs:
        st.caption(f"共有 {len(st.session_state.kb_list)} 个知识库")
    with col_new:
        if st.button("➕ 新建"):
            st.session_state.show_create_kb = True

    if st.session_state.show_create_kb:
        with st.container(border=True):
            st.markdown("###### 新建知识库")
            with st.form("new_kb_form"):
                st.text_input("输入新知识库名称", placeholder="例如: project_alpha", key="new_kb_name_input")
                st.form_submit_button("确认创建", on_click=create_kb_callback, args=(pipeline,))

            if st.button("取消", key="cancel_create_kb"):
                st.session_state.show_create_kb = False
                st.rerun()

    # --- 知识库列表展示 ---
    for kb in st.session_state.kb_list:
        files = pipeline.list_files(kb)
        is_current = (kb == st.session_state.current_kb)

        with st.expander(f"{'🟢' if is_current else '⚪'} {kb} ({len(files)} 文件)", expanded=is_current):

            # --- 文件列表 ---
            if files:
                st.markdown("**📄 文件列表:**")
                container_kwargs = {"border": True}
                if len(files) > 5:
                    container_kwargs["height"] = 300

                with st.container(**container_kwargs):
                    for f in files:
                        c1, c2 = st.columns([0.80, 0.20])
                        with c1:
                            st.text(f)
                        with c2:
                            current_confirm = st.session_state.confirm_delete_file
                            is_confirming = (current_confirm == (kb, f))

                            if is_confirming:
                                sub_c1, sub_c2 = st.columns([1, 1])
                                with sub_c1:
                                    if st.button("✓", key=f"yes_f_{kb}_{f}", help="确认删除"):
                                        with st.spinner("删除中..."):
                                            res = pipeline.delete_document(f, kb)
                                            st.session_state.confirm_delete_file = None
                                            if "✅" in res:
                                                st.session_state.toast_msg = f"已删除: {f}"
                                            else:
                                                st.session_state.error_msg = res
                                            st.rerun()
                                with sub_c2:
                                    if st.button("✗", key=f"no_f_{kb}_{f}", help="取消"):
                                        st.session_state.confirm_delete_file = None
                                        st.rerun()
                            else:
                                if st.button("🗑️", key=f"del_f_{kb}_{f}", help="删除文件"):
                                    st.session_state.confirm_delete_file = (kb, f)
                                    st.rerun()
            else:
                st.caption("暂无文件")

            st.divider()

            # --- 底部按钮 ---
            col_switch, col_del = st.columns([1, 1])
            with col_switch:
                if not is_current:
                    st.button(
                        "🔄 切换到此知识库",
                        key=f"btn_switch_{kb}",
                        on_click=switch_kb_callback,
                        args=(kb,)
                    )
                else:
                    st.button("✅ 当前使用中", disabled=True, key=f"btn_cur_{kb}")

            with col_del:
                if kb != DEFAULT_KB_NAME:
                    if st.session_state.confirm_delete_kb == kb:
                        st.markdown("**确认删除?**")
                        sub_c1, sub_c2 = st.columns([1, 1])
                        with sub_c1:
                            st.button(
                                "✅ 是",
                                key=f"yes_kb_{kb}",
                                on_click=delete_kb_confirmed,
                                args=(pipeline, kb)
                            )
                        with sub_c2:
                            if st.button("❌ 否", key=f"no_kb_{kb}"):
                                st.session_state.confirm_delete_kb = None
                                st.rerun()
                    else:
                        if st.button("🗑️ 删除整个库", key=f"del_kb_{kb}"):
                            st.session_state.confirm_delete_kb = kb
                            st.rerun()


if __name__ == "__main__":
    main()
