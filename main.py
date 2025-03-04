import os
import torch
import numpy as np
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc, setup_logger
from lightrag import LightRAG, QueryParam
from sentence_transformers import SentenceTransformer
import streamlit as st
from datetime import datetime
from unstructured.partition.auto import partition

torch.classes.__path__ = []

# =======================SETTING UP===============================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WORKING_DIR = "./skypos_data"

setup_logger("lightrag", level="INFO")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in environment variable or .env file.")


os.mkdir(WORKING_DIR)


# System Prompt (for internal context only — not shown to user)
SYSTEM_PROMPT = """Bạn là một nhân viên chăm sóc khách hàng của SkyPOS. 
Bạn sẽ nhận được các câu hỏi từ khách hàng và cần nghiên cứu tài liệu sản phẩm rồi trả lời đầy đủ. Lưu ý không được trả lời những câu hỏi không liên quan tới sản phẩm. Nếu không tìm thấy câu trả lời trong tài liệu sản phẩm, hãy nói người dùng liên hệ hotline để có thêm chi tiết."""


#LIGHTRAG SETUP
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    # 1. Initialize the GenAI Client with your Gemini API Key
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash-exp")

    # 2. Combine prompts: system prompt, history, and user prompt
    if history_messages is None:
        history_messages = []

    combined_prompt = ""
    if system_prompt:
        combined_prompt += f"{system_prompt}\n"

    for msg in history_messages:
        # Each msg is expected to be a dict: {"role": "...", "content": "..."}
        combined_prompt += f"{msg['role']}: {msg['content']}\n"

    # Finally, add the new user prompt
    combined_prompt += f"user: {prompt}"

    # 3. Call the Gemini model
    response = model.generate_content([combined_prompt])

    # 4. Return the response text
    return response.text
async def embedding_func(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=8192,
        func=embedding_func,
    ),
)

elements = partition(filename="SkyPos_FAQ.docx")
text = "\n".join([el.text for el in elements])
print("======> Unstructured Text:",text)
rag.insert(text)

# ========================STREAMLIT===============================

# ====== Chat History Management ======
def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Initialize the Gemini conversation object and store it in session state
    if "gemini_conversation" not in st.session_state:
        st.session_state.gemini_conversation = model.start_chat(history=[])
        # Add system prompt to initialize the conversation
        st.session_state.gemini_conversation.send_message(SYSTEM_PROMPT)

def add_message(role, content):
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": role, "content": content, "timestamp": timestamp})
    print("Added to state:",st.session_state.messages)

def clear_input():
    st.session_state.user_input = ""

# ====== Gemini Query with Retrieval and History ======
def call_gemini_with_history(user_query):
    # Using LightRAG to retrieve relevant context
    retrieved_docs = rag.query(query=user_query, param=QueryParam(mode="hybrid", top_k=5, response_type="single line"))
    
    # Format the retrieved documents into a context string
    context = ""
    if retrieved_docs:
        context = "\n\n".join([doc for doc in retrieved_docs])
    
    # Prepare the prompt with context and user query
    final_prompt = f"""
Thông tin từ tài liệu: {context}

Lịch sử chat với khách hàng: {st.session_state.messages}

Câu hỏi của khách hàng: {user_query}

Hãy trả lời câu hỏi dựa trên thông tin từ tài liệu. Nếu không có thông tin, hãy đề nghị khách hàng liên hệ hotline.
"""
    
    print("===========================Final Prompt:",final_prompt)
    # Get response from the persistent Gemini conversation
    response = st.session_state.gemini_conversation.send_message(final_prompt)
    
    
    return response.text.strip()

# ====== Main App ======
def main():
    st.title("💬 Hỗ trợ SkyPOS - LightRAG x Gemini 2.0 Flash")

    init_chat_history()

    # Chat interface container
    chat_container = st.container()

    # Display existing messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    st.write(message["content"])
                with col2:
                    st.write(f"_{message['timestamp']}_")

    # User input box
    user_input = st.chat_input("Type your message...", key="user_input")

    if user_input:
        
        add_message("user", user_input)
        assistant_reply = call_gemini_with_history(user_input)
        add_message("assistant", assistant_reply)
        st.rerun()

if __name__ == "__main__":
    st.set_page_config(page_title="Chat App", page_icon="💬", layout="wide")
    main()