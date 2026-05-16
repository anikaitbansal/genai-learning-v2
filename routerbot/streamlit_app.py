import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="GenAI RouterBot UI", layout="wide")
st.title("GenAI RouterBot")
st.caption("Upload PDF and chat with your RAG-enabled backend")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = "demo-session-1"

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

if "use_rag" not in st.session_state:
    st.session_state.use_rag = True

if "debug" not in st.session_state:
    st.session_state.debug = False


with st.sidebar:
    st.header("Settings")

    st.session_state.session_id = st.text_input(
        "Session ID",
        value=st.session_state.session_id
    )

    st.session_state.use_rag = st.checkbox(
        "Use RAG",
        value=st.session_state.use_rag
    )

    st.session_state.debug = st.checkbox(
        "Debug mode",
        value=st.session_state.debug
    )

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat cleared.")

    st.markdown("---")
    st.subheader("Upload PDF")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if st.button("Upload PDF"):
        if uploaded_file is None:
            st.warning("Please select a PDF file first.")
        elif uploaded_file.name == st.session_state.last_uploaded_file:
            st.warning("This file is already uploaded.")
        else:
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf"
                    )
                }

                with st.spinner("Processing PDF..."):
                    response = requests.post(
                        f"{BACKEND_URL}/upload-pdf",
                        files=files,
                        timeout=120
                    )

                if response.status_code == 200:
                    response_data = response.json()
                    st.session_state.last_uploaded_file = uploaded_file.name
                    st.success("PDF uploaded and processed successfully.")

                    with st.expander("Upload details"):
                        st.json(response_data)
                else:
                    st.error(f"Upload failed: {response.status_code}")
                    st.text(response.text)

            except Exception as error:
                st.error(f"Error while uploading PDF: {error}")


st.subheader("Chat with your document")
st.caption("Ask questions about uploaded PDFs or general queries")

if st.session_state.last_uploaded_file:
    st.info(f"Last uploaded PDF: {st.session_state.last_uploaded_file}")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question about the uploaded PDF or chat normally...")

if user_input:
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        payload = {
            "message": user_input,
            "session_id": st.session_state.session_id,
            "use_rag": st.session_state.use_rag,
            "debug": st.session_state.debug
        }

        with st.spinner("Thinking..."):
            response = requests.post(
                f"{BACKEND_URL}/chat",
                json=payload,
                timeout=120
            )

        if response.status_code == 200:
            response_data = response.json()
            bot_reply = response_data.get("bot_reply", "No bot_reply found in response.")

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": bot_reply
            })

            with st.chat_message("assistant"):
                st.markdown(bot_reply)

                if st.session_state.debug:
                    st.caption(f"RAG used: {response_data.get('rag_used')}")

            if st.session_state.debug:
                with st.expander("Debug response"):
                    st.json(response_data)

        else:
            error_message = f"Chat request failed: {response.status_code}"
            try:
                error_details = response.json()
                error_message += f"\n{error_details}"
            except Exception:
                error_message += f"\n{response.text}"

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_message
            })

            with st.chat_message("assistant"):
                st.error(error_message)

    except Exception as error:
        error_message = f"Error while calling /chat API: {error}"

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": error_message
        })

        with st.chat_message("assistant"):
            st.error(error_message)