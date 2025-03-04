import os
import numpy as np
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG, QueryParam
from sentence_transformers import SentenceTransformer
import streamlit as st
from datetime import datetime
import tempfile
from docling.document_converter import DocumentConverter
import shutil

# =======================SETTING UP===============================

load_dotenv()
torch.classes.__path__ = []

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print(GOOGLE_API_KEY)
WORKING_DIR = "./skypos_data"

if not GOOGLE_API_KEY:
    raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in environment variable or .env file.")

# Initialize Gemini model directly
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

WORKING_DIR = "./skypos_data"

if os.path.exists(WORKING_DIR):
    import shutil

    shutil.rmtree(WORKING_DIR)

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


rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=8192,
        func=embedding_func,
    ),
)

# ===============Chat backend=================
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




# ==================================================================User Chat Page===================================
def user_chat_page():
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

# ================================================================Admin Page=====================
def convert_to_markdown(file_path):
    """Convert document to markdown using the DocumentConverter."""
    try:
        converter = DocumentConverter()
        result = converter.convert(file_path)
        return result.document.export_to_markdown()
    except Exception as e:
        return f"Error converting document: {str(e)}"

def admin_page():
    st.set_page_config(page_title="Document to Markdown Converter", layout="wide")
    
    # Sidebar for authentication
    with st.sidebar:
        st.title("Authentication")
        
        # Simple authentication - in production, use a more secure method
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.button("Login")
        
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
            
        if login_button:
            # Replace with your actual authentication logic
            if username == "admin" and password == "password":
                st.session_state.authenticated = True
                st.success("Login successful!")
            else:
                st.error("Invalid username or password")
    
    # Main content area
    st.title("Document to Markdown Converter")
    
    if st.session_state.authenticated:
        st.write("Upload documents to convert them to markdown format.")
        
        uploaded_files = st.file_uploader("Choose document files", 
                                         accept_multiple_files=True,
                                         type=["pdf", "docx", "txt", "html","jpeg","xlsx","pptx"])
        
        if uploaded_files and st.button("Convert to Markdown"):
            st.write("### Conversion Results")
            
            for uploaded_file in uploaded_files:
                st.write(f"Processing: **{uploaded_file.name}**")
                
                # Create a temporary file to save the uploaded content
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name
                
                try:
                    # Convert the file to markdown
                    markdown_content = convert_to_markdown(temp_file_path)
                    
                    # Display the markdown content
                    st.markdown("#### Preview:")
                    st.markdown(markdown_content[:1000] + "..." if len(markdown_content) > 1000 else markdown_content)
                    
                    # Provide download option for the full markdown
                    st.download_button(
                        label="Download Markdown",
                        data=markdown_content,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}.md",
                        mime="text/markdown"
                    )
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                
                st.markdown("---")
    else:
        st.info("Please log in to use the document converter.")

# Main app with navigation
def main():
    st.set_page_config(page_title="SkyPOS Support System", page_icon="💬", layout="wide")
    
    # Navigation in sidebar
    with st.sidebar:
        st.title("SkyPOS Navigation")
        page = st.radio("Go to:", ["User Chat", "Admin Panel"])
    
    # Display the selected page
    if page == "User Chat":
        user_chat_page()
    else:
        admin_page()

if __name__ == "__main__":
    main()