import streamlit as st
import os
import tempfile
from docling.document_converter import DocumentConverter

def convert_to_markdown(file_path):
    """Convert document to markdown using the DocumentConverter."""
    try:
        converter = DocumentConverter()
        result = converter.convert(file_path)
        return result.document.export_to_markdown()
    except Exception as e:
        return f"Error converting document: {str(e)}"

def main():
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

if __name__ == "__main__":
    main()