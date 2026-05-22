import os
import re
import tempfile
import streamlit as st
from openai import OpenAI
from supabase import create_client
from dotenv import load_dotenv
from pathlib import Path
from pypdf import PdfReader
from PIL import Image
import base64

# Load secrets
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Password protection
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown('<div class="main-title">💜 Health Knowledge Assistant</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Please enter your password to continue</div>', unsafe_allow_html=True)
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if password == os.getenv("APP_PASSWORD"):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        st.stop()

check_password()

# Connect to OpenAI and Supabase
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Page config
st.set_page_config(page_title="Health Assistant", page_icon="💜", layout="centered")

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #1a1040, #24243e);
        min-height: 100vh;
    }

    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }

    .source-card {
        background: rgba(168, 85, 247, 0.1);
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 12px;
        font-size: 0.82rem;
        color: #c4b5fd;
        overflow: hidden;
        word-wrap: break-word;
    }

    .source-card strong {
        color: #a855f7;
    }

    .source-tag {
        display: inline-block;
        background: rgba(168, 85, 247, 0.2);
        border: 1px solid rgba(168, 85, 247, 0.4);
        border-radius: 20px;
        padding: 2px 10px;
        margin: 3px;
        font-size: 0.78rem;
        color: #d8b4fe;
        white-space: normal;
        word-break: break-word;
    }

    .stChatMessage {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        padding: 12px !important;
        margin-bottom: 8px !important;
    }

    .stChatInputContainer {
        border-top: 1px solid rgba(168, 85, 247, 0.2) !important;
        padding-top: 1rem !important;
    }

    .history-item {
        background: rgba(168, 85, 247, 0.1);
        border: 1px solid rgba(168, 85, 247, 0.2);
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.8rem;
        color: #c4b5fd;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def ask_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def save_message(role, content, sources=""):
    supabase.table("conversations").insert({
        "role": role,
        "content": content,
        "sources": sources
    }).execute()

def load_history():
    result = supabase.table("conversations")\
        .select("*")\
        .order("created_at", desc=False)\
        .execute()
    return result.data

def moderate_content(text):
    response = client.moderations.create(input=text)
    return response.results[0].flagged

def chunk_text(text):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + 500
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - 50
    return chunks

def embed_and_store(chunks, source_name):
    valid_chunks = [(i, c) for i, c in enumerate(chunks) if len(c.strip()) >= 50]
    batch_size = 20
    progress = st.progress(0)

    for batch_start in range(0, len(valid_chunks), batch_size):
        batch = valid_chunks[batch_start:batch_start + batch_size]
        batch_texts = [c for _, c in batch]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch_texts
        )

        rows = []
        for j, (i, chunk) in enumerate(batch):
            rows.append({
                "content": chunk,
                "embedding": response.data[j].embedding,
                "metadata": {"source": source_name, "chunk": i}
            })

        supabase.table("documents").insert(rows).execute()
        progress.progress((batch_start + len(batch)) / len(valid_chunks))

def extract_text_from_image(image_file):
    image_data = base64.b64encode(image_file.read()).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                },
                {
                    "type": "text",
                    "text": """Extract ALL text from this image completely. 
If the image has multiple columns, read every column fully from top to bottom.
Do not skip or summarize anything.
Return only the raw extracted text, nothing else."""
                }
            ]
        }],
        max_tokens=4096
    )
    return response.choices[0].message.content

def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    os.unlink(tmp_path)
    return transcript.text

def check_duplicate(filename):
    existing = supabase.table("documents")\
        .select("id")\
        .eq("metadata->>source", filename)\
        .limit(1)\
        .execute()
    return len(existing.data) > 0

# Sidebar
with st.sidebar:
    st.markdown("### 📚 Add New Knowledge")
    st.caption("Upload PDFs, images, or audio files")

    file_type = st.selectbox("File type", ["PDF", "Image", "Audio"])

    if file_type == "PDF":
        uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    elif file_type == "Image":
        uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])
    else:
        uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "mp4", "wav", "m4a"])

    if uploaded_file is not None:
        if st.button("⚡ Process File", use_container_width=True, disabled=st.session_state.get("processing", False)):
            if check_duplicate(uploaded_file.name):
                st.warning(f"⚠️ '{uploaded_file.name}' has already been uploaded.")
            else:
                st.session_state.processing = True
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        if file_type == "PDF":
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(uploaded_file.read())
                                tmp_path = tmp.name
                            reader = PdfReader(tmp_path)
                            text = ""
                            for page in reader.pages:
                                text += page.extract_text() + "\n"
                            os.unlink(tmp_path)

                        elif file_type == "Image":
                            text = extract_text_from_image(uploaded_file)

                        else:
                            text = transcribe_audio(uploaded_file)

                        if moderate_content(text[:2000]):
                            st.error("⚠️ This file contains inappropriate content and cannot be added.")
                            st.session_state.processing = False
                            st.stop()

                        chunks = chunk_text(text)
                        embed_and_store(chunks, uploaded_file.name)
                        st.success(f"✅ {uploaded_file.name} added successfully!")
                        st.session_state.processing = False

                    except Exception as e:
                        st.error(f"Something went wrong: {e}")
                        st.session_state.processing = False

    st.markdown("---")

    # Chat History
    st.markdown("### 💬 Chat History")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        supabase.table("conversations").delete().neq("id", 0).execute()
        st.rerun()

    history = load_history()
    if history:
        chat_text = ""
        for msg in history:
            role = "You" if msg["role"] == "user" else "Assistant"
            chat_text += f"{role}:\n{msg['content']}\n\n"
            if msg.get("sources"):
                clean = re.sub('<[^<]+?>', '', msg["sources"])
                chat_text += f"Sources: {clean}\n\n"
            chat_text += "-" * 40 + "\n\n"

        st.download_button(
            label="📥 Download Chat",
            data=chat_text,
            file_name="health_consultation.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.markdown("**Recent questions:**")
        user_messages = [m for m in history if m["role"] == "user"][-20:]
        for msg in reversed(user_messages):
            st.markdown(f'<div class="history-item">💬 {msg["content"][:50]}...</div>', unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">💜 Health Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by your course material — ask anything</div>', unsafe_allow_html=True)

# Load chat from Supabase on first load
if "messages" not in st.session_state:
    st.session_state.messages = []
    history = load_history()
    for msg in history:
        st.session_state.messages.append({
            "role": msg["role"],
            "content": msg["content"],
            "sources": msg.get("sources", "")
        })

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.markdown(message["sources"], unsafe_allow_html=True)

# Handle new question
if question := st.chat_input("Ask a question from the course material..."):

    st.session_state.messages.append({"role": "user", "content": question})
    save_message("user", question)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):

            # Step 1: Embed
            q_embedding = get_embedding(question)

            # Step 2: Search Supabase
            results = supabase.rpc("match_documents", {
                "query_embedding": q_embedding,
                "match_threshold": 0.2,
                "match_count": 5
            }).execute()

            # Step 3: Build context
            context = "\n\n".join([r["content"] for r in results.data])

            # Step 4: Ask GPT
            prompt = f"""You are a helpful health and nutrition assistant.
Use ONLY the following information from the course material to answer the question.
If the answer isn't in the material, say so honestly.

Course material:
{context}

Question: {question}

Answer:"""

            answer = ask_gpt(prompt)
            st.markdown(answer)

            # Step 5: Sources
            seen = set()
            source_tags = ""
            for r in results.data:
                meta = r["metadata"]
                source = meta.get("source", "Unknown")
                chunk = meta.get("chunk", "?")
                key = f"{source}-{chunk}"
                if key not in seen:
                    seen.add(key)
                    source_tags += f'<span class="source-tag">📄 {source} — section {chunk}</span>'

            sources_html = f'<div class="source-card"><strong>📚 Sources used:</strong><br>{source_tags}</div>'
            st.markdown(sources_html, unsafe_allow_html=True)

            save_message("assistant", answer, sources_html)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources_html
            })
