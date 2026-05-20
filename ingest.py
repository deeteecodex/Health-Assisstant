import os
from openai import OpenAI
from supabase import create_client
from pypdf import PdfReader
from dotenv import load_dotenv
from pathlib import Path

# Load secrets
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Connect to OpenAI and Supabase
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

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
# MAIN: Process all PDFs in a folder
# -----------------------------------
def process_all_pdfs(folder_path):
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDFs to process")

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file}")
        pdf_path = os.path.join(folder_path, pdf_file)

        # Extract text
        text = extract_text_from_pdf(pdf_path)

        # Split into chunks
        chunks = split_into_chunks(text)

        # Filter tiny chunks
        valid_chunks = [(i, c) for i, c in enumerate(chunks) if len(c.strip()) >= 50]
        print(f"  Split into {len(valid_chunks)} chunks")

        # Process in batches of 20
        batch_size = 20
        for batch_start in range(0, len(valid_chunks), batch_size):
            batch = valid_chunks[batch_start:batch_start + batch_size]
            batch_texts = [c for _, c in batch]

            # Embed entire batch in one call
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch_texts
            )

            # Save all chunks in this batch
            rows = []
            for j, (i, chunk) in enumerate(batch):
                rows.append({
                    "content": chunk,
                    "embedding": response.data[j].embedding,
                    "metadata": {"source": pdf_file, "chunk": i}
                })

            supabase.table("documents").insert(rows).execute()
            print(f"  Saved chunks {batch_start+1} to {batch_start+len(batch)} of {len(valid_chunks)}")

    print("\n✅ All PDFs processed successfully!")

# Run it
process_all_pdfs("pdfs")