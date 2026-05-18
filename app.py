import os
import streamlit as st
from google import genai
from supabase import create_client
from dotenv import load_dotenv
from pathlib import Path

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

# Connect to Google and Supabase
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
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
</style>
""", unsafe_allow_html=True)

# Sidebar for PDF upload
with st.sidebar:
    st.markdown("### 📚 Add New Knowledge")
    st.caption("Upload a new PDF to add it to the knowledge base")

    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file is not None:
        if st.button("⚡ Process PDF", use_container_width=True):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    # Read PDF
                    import tempfile
                    from pypdf import PdfReader
                    import time

                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    # Extract text
                    reader = PdfReader(tmp_path)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"

                    # Chunk it
                    words = text.split()
                    chunks = []
                    start = 0
                    while start < len(words):
                        end = start + 500
                        chunk = " ".join(words[start:end])
                        chunks.append(chunk)
                        start = end - 50

                    # Embed and save each chunk
                        # Filter out tiny chunks
                        valid_chunks = [(i, c) for i, c in enumerate(chunks) if len(c.strip()) >= 50]

                        # Process in batches of 20
                        batch_size = 20
                        progress = st.progress(0)

                        for batch_start in range(0, len(valid_chunks), batch_size):
                            batch = valid_chunks[batch_start:batch_start + batch_size]
                            batch_texts = [c for _, c in batch]

                            # Embed entire batch in one API call
                            result = client.models.embed_content(
                                model="gemini-embedding-2-flash-001",
                                contents=batch_texts,
                                config={"output_dimensionality": 1536}
                            )

                            # Save all chunks in this batch
                            rows = []
                            for j, (i, chunk) in enumerate(batch):
                                rows.append({
                                    "content": chunk,
                                    "embedding": result.embeddings[j].values,
                                    "metadata": {"source": uploaded_file.name, "chunk": i}
                                })

                            supabase.table("documents").insert(rows).execute()
                            progress.progress((batch_start + len(batch)) / len(valid_chunks))
                            time.sleep(2)

                    st.success(f"✅ {uploaded_file.name} added successfully!")
                    os.unlink(tmp_path)

                except Exception as e:
                    st.error(f"Something went wrong: {e}")

# Header
st.markdown('<div class="main-title">💜 Health Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by your course material — ask anything</div>', unsafe_allow_html=True)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            st.markdown(message["sources"], unsafe_allow_html=True)

# Handle new question
if question := st.chat_input("Ask a question from the course material..."):

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):

            # Step 0: Clean up the question
            cleaned = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"Fix any typos and rephrase this health question clearly, return only the fixed question, nothing else: {question}"
            )
            question = cleaned.text.strip()

            # Step 1: Embed the question
            q_embedding = client.models.embed_content(
                model="gemini-embedding-001",
                contents=question,
                config={"output_dimensionality": 1536}
            ).embeddings[0].values

            # Step 2: Search Supabase
            results = supabase.rpc("match_documents", {
                "query_embedding": q_embedding,
                "match_threshold": 0.5,
                "match_count": 5
            }).execute()

            # Step 3: Build context
            context = "\n\n".join([r["content"] for r in results.data])

            # Step 4: Ask Gemini
            prompt = f"""You are a helpful health and nutrition assistant.
Use ONLY the following information from the course material to answer the question.
If the answer isn't in the material, say so honestly.

Course material:
{context}

Question: {question}

Answer:"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            answer = response.text
            st.markdown(answer)

            # Step 5: Show sources
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

                    # Step 6: Generate follow-up questions
                    followup_prompt = f"""Based on this question: "{question}"
            And this answer: "{answer}"
            Generate exactly 3 short follow-up questions the user might want to ask next.
            Return ONLY the 3 questions, one per line, no numbering, no extra text."""

                    followup_response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=followup_prompt
                    )

                    followups = [q.strip() for q in followup_response.text.strip().split("\n") if q.strip()][:3]

                    st.markdown("**💡 You might also want to ask:**")
                    for fq in followups:
                        if st.button(fq, key=fq):
                            st.session_state.messages.append({"role": "user", "content": fq})
                            st.rerun()

            sources_html = f'<div class="source-card"><strong>📚 Sources used:</strong><br>{source_tags}</div>'
            st.markdown(sources_html, unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources_html
            })
