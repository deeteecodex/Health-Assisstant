import os
from google import genai
from google.genai import types
from supabase import create_client
from pypdf import PdfReader
from dotenv import load_dotenv
import time

# Load our secret keys from .env file
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
import os



# Connect to Google AI
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Connect to Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# -----------------------------------
# STEP 1: Extract text from a PDF
# -----------------------------------
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# -----------------------------------
# STEP 2: Split text into chunks
# -----------------------------------
def split_into_chunks(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks

# -----------------------------------
# STEP 3: Convert chunk to embedding
# -----------------------------------
from google.genai.errors import ClientError
from tenacity import retry, stop_after_attempt, wait_random_exponential

# This decorator will automatically retry if a ClientError (like 429) happens
@retry(
    wait=wait_random_exponential(min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
def embed_batch_with_retry(client, texts):
    return client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config={"output_dimensionality": 1536}
    )
    return result.embeddings[0].values

# -----------------------------------
# STEP 4: Save chunk to Supabase
# -----------------------------------
def save_to_supabase(content, embedding, metadata):
    supabase.table("documents").insert({
        "content": content,
        "embedding": embedding,
        "metadata": metadata
    }).execute()

# -----------------------------------
# MAIN: Process all PDFs in a folder
# -----------------------------------
def process_all_pdfs(folder_path):
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDFs total in folder.")

    for pdf_file in pdf_files:
        print(f"\nChecking status for: {pdf_file}")

        # 1. CHECK IF ALREADY IN SUPABASE
        # This checks if ANY rows exist where metadata->source equals the current filename
        existing_check = supabase.table("documents") \
            .select("id") \
            .eq("metadata->>source", pdf_file) \
            .limit(1) \
            .execute()

        if existing_check.data:
            print(f"⏩ Skipping {pdf_file} (Already processed in Supabase)")
            continue

        pdf_path = os.path.join(folder_path, pdf_file)

        # Extract text
        text = extract_text_from_pdf(pdf_path)

        # Split into chunks
        chunks = split_into_chunks(text)
        print(f"  Split into {len(chunks)} chunks")

        # Filter tiny chunks
        valid_chunks = [(i, c) for i, c in enumerate(chunks) if len(c.strip()) >= 50]

        # 2. OPTIMIZED BATCHING (Increased to 100 to reduce total requests)
        batch_size = 100
        for batch_start in range(0, len(valid_chunks), batch_size):
            batch = valid_chunks[batch_start:batch_start + batch_size]
            batch_texts = [c for _, c in batch]

            # 3. CALL RETRY HELPER
            try:
                result = embed_batch_with_retry(client, batch_texts)
            except Exception as e:
                print(f"\n❌ Permanent failure embedding batch after 5 attempts: {e}")
                raise e

            # Save all chunks in this batch
            rows = []
            for j, (i, chunk) in enumerate(batch):
                rows.append({
                    "content": chunk,
                    "embedding": result.embeddings[j].values,
                    "metadata": {"source": pdf_file, "chunk": i}
                })

            supabase.table("documents").insert(rows).execute()
            print(f"  Saved chunks {batch_start + 1} to {batch_start + len(batch)} of {len(valid_chunks)}")

            # Tiny safety buffer sleep
            time.sleep(3)

    print("\n✅ Processing run complete! All files caught up.")

# Run it
process_all_pdfs("pdfs")
