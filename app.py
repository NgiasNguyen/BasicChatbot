#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Giao diện Streamlit cho Chatbot hỏi-đáp
Hỗ trợ 2 loại: TF-IDF (chatbot.py) và Semantic Search (chatbot_pro.py)
"""

import streamlit as st
from chatbot import Chatbot
from chatbot_pro import ChatbotPro

# Cấu hình trang
st.set_page_config(
    page_title="Chatbot Hỏi-Đáp",
    page_icon="💬",
    layout="centered"
)

# Tùy biến giao diện Streamlit
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
    :root {
        --primary-color: #d946ef;
        --primary-soft: #fdf4ff;
        --border-color: #f5d0fe;
        --text-main: #4a044e;
        --text-muted: #a21caf;
    }

    .stApp {
        background: linear-gradient(180deg, #fff7fb 0%, #f5f3ff 55%, #fdf2f8 100%);
        color: var(--text-main);
        font-family: 'Nunito', 'Segoe UI', sans-serif;
    }

    .stApp, .stApp p, .stApp span, .stApp label, .stApp li,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stMarkdown, .stText {
        color: var(--text-main) !important;
    }

    /* Thanh trên cùng của Streamlit */
    [data-testid="stHeader"] {
        background: #fff9fd !important;
        border-bottom: 1px solid var(--border-color);
    }

    /* Khu vực toolbar góc trên phải */
    [data-testid="stToolbar"] {
        background: transparent !important;
    }

    [data-testid="stSidebar"] {
        background: #fff9fd;
        border-right: 1px solid var(--border-color);
    }

    .block-container {
        padding-top: 1.2rem;
    }

    /* Card trong sidebar */
    [data-testid="stSidebar"] .stAlert {
        border-radius: 14px;
        border: 1px solid var(--border-color);
    }

    /* Bong chat */
    [data-testid="stChatMessage"] {
        background: #fff7fb;
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 6px 16px rgba(217, 70, 239, 0.12);
    }

    /* Chat input */
    [data-testid="stChatInputTextArea"] {
        border-radius: 14px !important;
        border: 1px solid #f0abfc !important;
        background: #ffffff !important;
        color: var(--text-main) !important;
    }

    [data-testid="stChatInputTextArea"] textarea,
    [data-testid="stTextInputRootElement"] input {
        color: var(--text-main) !important;
        caret-color: var(--primary-color) !important;
    }

    [data-testid="stChatInputTextArea"] textarea::placeholder,
    [data-testid="stTextInputRootElement"] input::placeholder {
        color: var(--text-muted) !important;
    }

    /* Nút chung */
    .stButton > button, .stDownloadButton > button {
        border-radius: 12px !important;
        border: 1px solid transparent !important;
        background: var(--primary-color) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        filter: brightness(0.95);
    }

    /* Radio option */
    [data-testid="stRadio"] > div {
        background: #fff7fb;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 0.4rem;
    }

    /* Caption dịu màu hơn */
    .stCaption {
        color: var(--text-muted) !important;
    }

    .stSelectbox label, .stRadio label, .stTextInput label, .stCheckbox label {
        color: #6b217d !important;
        font-weight: 700 !important;
    }

    /* Thanh dưới cùng / footer */
    footer, .stApp footer, [data-testid="stDecoration"] {
        background: #fff9fd !important;
        color: var(--text-muted) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Khởi tạo chatbot TF-IDF trong session state
@st.cache_resource
def load_chatbot_tfidf():
    """Load và train chatbot TF-IDF (chỉ chạy một lần)"""
    chatbot = Chatbot(csv_file='data_converted.csv', similarity_threshold=0.1)
    if chatbot.load_data():
        chatbot.train()
        return chatbot
    return None

# Khởi tạo chatbot Semantic Search trong session state
@st.cache_resource
def load_chatbot_pro():
    """Load và khởi tạo chatbot Semantic Search (chỉ chạy một lần)"""
    chatbot_pro = ChatbotPro(csv_file='data_converted.csv')
    if chatbot_pro.initialize():
        return chatbot_pro
    return None

# Sidebar để chọn loại chatbot
with st.sidebar:
    st.header("⚙️ Cấu hình")
    
    # Lựa chọn loại chatbot
    chatbot_type = st.radio(
        "Chọn loại Chatbot:",
        ["TF-IDF (Nhanh)", "Semantic Search (Chính xác)"],
        help="TF-IDF: Nhanh, dựa trên từ khóa\nSemantic Search: Chính xác hơn, hiểu ngữ nghĩa"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Thông tin")
    
    if chatbot_type == "TF-IDF (Nhanh)":
        st.info("**TF-IDF + Cosine Similarity**\n\n- Nhanh, hiệu quả\n- Dựa trên từ khóa\n- Phù hợp cho FAQ đơn giản")
    else:
        st.info("**Semantic Search**\n\n- Hiểu ngữ nghĩa\n- Chính xác hơn\n- Hỗ trợ đa ngôn ngữ")

# Load chatbot dựa trên lựa chọn
if chatbot_type == "TF-IDF (Nhanh)":
    chatbot = load_chatbot_tfidf()
    chatbot_pro = None
    
    if chatbot is None:
        st.error("❌ Không thể tải dữ liệu chatbot TF-IDF. Vui lòng kiểm tra file data_converted.csv")
        st.stop()
    
    st.sidebar.markdown(f"**Số lượng câu hỏi:** {len(chatbot.questions)}")
else:
    chatbot_pro = load_chatbot_pro()
    chatbot = None
    
    if chatbot_pro is None:
        st.error("❌ Không thể tải chatbot Semantic Search. Vui lòng kiểm tra file data_converted.csv")
        st.stop()
    
    st.sidebar.markdown(f"**Số lượng câu hỏi:** {len(chatbot_pro.questions)}")
    st.sidebar.markdown(f"**Mô hình:** paraphrase-multilingual-MiniLM-L12-v2")

# Tiêu đề
st.title("💬 Cùng trò chuyện với mình nhé")

# Hiển thị loại chatbot đang dùng
if chatbot_type == "TF-IDF (Nhanh)":
    st.caption("🔍 Đang sử dụng: **TF-IDF + Cosine Similarity**")
else:
    st.caption("🧠 Đang sử dụng: **Semantic Search (sentence-transformers)**")

# Khởi tạo lịch sử chat trong session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Reset lịch sử khi đổi loại chatbot
if "last_chatbot_type" not in st.session_state:
    st.session_state.last_chatbot_type = chatbot_type

if st.session_state.last_chatbot_type != chatbot_type:
    st.session_state.messages = []
    st.session_state.last_chatbot_type = chatbot_type
    st.rerun()

# Vùng hiển thị câu trả lời (lịch sử chat)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ô nhập câu hỏi
user_question = st.chat_input("Nhập câu hỏi của bạn...")

if user_question:
    # Thêm câu hỏi vào lịch sử
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    # Hiển thị câu hỏi
    with st.chat_message("user"):
        st.write(user_question)
    
    # Lấy và hiển thị câu trả lời
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            if chatbot_type == "TF-IDF (Nhanh)":
                # Sử dụng chatbot TF-IDF
                answer = chatbot.answer(user_question)
                st.write(answer)
            else:
                # Sử dụng chatbot Semantic Search
                answer, score, matched = chatbot_pro.answer(user_question)
                st.markdown(answer)
    
    # Thêm câu trả lời vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Nút xóa lịch sử
if st.session_state.messages:
    if st.button("🗑️ Xóa lịch sử", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
