
from pathlib import Path
from uuid import uuid4

import streamlit as st

from src.pipeline_factory import create_rag_components
from src.document_ingestion import ingest_document


st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Document Assistant")
st.caption("Upload your documents and ask questions using local AI.")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@st.cache_resource
def load_components():
    return create_rag_components()


try:
    components = load_components()

    rag = components["rag"]
    embedding_generator = components["embedding_generator"]
    vector_store = components["vector_store"]

    st.success("RAG pipeline connected successfully!")

    # Document upload section
    st.sidebar.header("📁 Upload Documents")

    uploaded_files = st.sidebar.file_uploader(
        "Choose your documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.sidebar.button(
        "Process Documents",
        disabled=not uploaded_files,
        use_container_width=True,
    ):
        for uploaded_file in uploaded_files:
            extension = Path(uploaded_file.name).suffix.lower().lstrip(".")

            if extension not in ALLOWED_EXTENSIONS:
                st.sidebar.error(
                    f"Unsupported file: {uploaded_file.name}"
                )
                continue

            if uploaded_file.size > MAX_FILE_SIZE:
                st.sidebar.error(
                    f"{uploaded_file.name} exceeds the 20 MB limit."
                )
                continue

            # Use a unique filename to avoid overwriting other uploads
            safe_name = Path(uploaded_file.name).name
            unique_name = f"{uuid4().hex}_{safe_name}"
            file_path = UPLOAD_DIR / unique_name

            try:
                file_path.write_bytes(uploaded_file.getvalue())

                with st.spinner(
                    f"Processing {uploaded_file.name}..."
                ):
                    result = ingest_document(
                        file_path=file_path,
                        embedding_generator=embedding_generator,
                        vector_store=vector_store,
                    )

                if result["chunks_added"] > 0:
                    st.sidebar.success(
                        f"{uploaded_file.name}: "
                        f"{result['chunks_added']} chunks indexed."
                    )
                else:
                    st.sidebar.warning(
                        f"{uploaded_file.name}: {result['message']}"
                    )

            except Exception as error:
                st.sidebar.error(
                    f"Could not process {uploaded_file.name}: {error}"
                )

    st.sidebar.divider()
    st.sidebar.metric(
        "Indexed document chunks",
        vector_store.count(),
    )

    # Chat section
    st.subheader("💬 Ask Your Documents")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("View sources"):
                    for source in message["sources"]:
                        st.write(
                            f"**{source['reference']}. "
                            f"{source['source']}**"
                        )
                        st.write(
                            f"Chunk: {source['chunk_index']}"
                        )

                        if source["page_number"] is not None:
                            st.write(
                                f"Page: {source['page_number']}"
                            )

    question = st.chat_input(
        "Ask a question about your uploaded documents..."
    )

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                result = rag.ask(question)

            st.markdown(result["answer"])

            if result["sources"]:
                with st.expander("View sources"):
                    for source in result["sources"]:
                        st.write(
                            f"**{source['reference']}. "
                            f"{source['source']}**"
                        )
                        st.write(
                            f"Chunk: {source['chunk_index']}"
                        )

                        if source["page_number"] is not None:
                            st.write(
                                f"Page: {source['page_number']}"
                            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })

except Exception as error:
    st.error(f"Application error: {error}")
    st.info(
        "Check that Ollama is running and the required "
        "Python dependencies are installed."
    )
