import os
import tempfile
import streamlit as st
from embedchain import App
import base64
from streamlit_chat import message

def embedchain_bot(db_path):
    return App.from_config(config={
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "llama2:latest",
                "max_tokens": 250,
                "temperature": 0.5,
                "stream": True,
                "base_url": 'http://localhost:11434'
            }
        },
        "vectordb": {
            "provider": "chroma",
            "config": {"dir": db_path}
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "llama2:latest",
                "base_url": 'http://localhost:11434'
            }
        }
    })

def display_pdf(file):
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

st.title("Chat with PDF using Llama 2")
st.caption("This app allows you to chat with a PDF using Llama 2 running locally with Ollama")

# Initialize session state variables
if 'db_path' not in st.session_state:
    st.session_state.db_path = tempfile.mkdtemp()
if 'app' not in st.session_state:
    st.session_state.app = embedchain_bot(st.session_state.db_path)
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'pdf_added' not in st.session_state:
    st.session_state.pdf_added = False

# PDF upload section
st.header("PDF Upload")
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

if pdf_file is not None:
    st.subheader("PDF Preview")
    display_pdf(pdf_file)
    
    if not st.session_state.pdf_added and st.button("Add to Knowledge Base"):
        with st.spinner("Adding PDF to knowledge base..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(pdf_file.getvalue())
                st.session_state.app.add(f.name, data_type="pdf_file")
            os.remove(f.name)
        st.session_state.pdf_added = True
        st.success(f"Added {pdf_file.name} to knowledge base!")

# Chat interface
st.header("Chat")
for i, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=msg["role"] == "user", key=str(i))

if prompt := st.chat_input("Ask a question about the PDF"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    message(prompt, is_user=True)

    with st.spinner("Thinking..."):
        response = st.session_state.app.chat(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        message(response)

if st.button("Clear Chat History"):
    st.session_state.messages = []

# Option to upload a new PDF
if st.button("Upload a new PDF"):
    st.session_state.pdf_added = False
    st.experimental_rerun()