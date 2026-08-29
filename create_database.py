import os
import sys
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# Muneeb Portfolio RAG - Chroma Database Builder
# ============================================================
# This script:
# 1. Loads your resume PDF
# 2. Splits it into chunks
# 3. Creates Gemini embeddings
# 4. Stores them in ChromaDB
#
# IMPORTANT:
# The same embedding model/settings MUST also be used by
# your Streamlit app when performing similarity search.
# ============================================================


# -----------------------------
# Project paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "data" / "resume.pdf"
DB_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "muneeb_portfolio"

# Stable Gemini Embedding 2 model.
# Google deprecated text-embedding-004 on January 14, 2026.
EMBEDDING_MODEL = "gemini-embedding-2"

# 768 is a good balance between quality and vector storage size.
# Google recommends 768, 1536, or 3072 dimensions.
EMBEDDING_DIMENSION = 768


# -----------------------------
# Load .env
# -----------------------------
load_dotenv(BASE_DIR / ".env")


# -----------------------------
# API key
# -----------------------------
GOOGLE_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)

if not GOOGLE_API_KEY:
    print("\nERROR: Gemini API key was not found.")
    print("Create a .env file in the project root and add:")
    print("GOOGLE_API_KEY=YOUR_GEMINI_API_KEY")
    sys.exit(1)


# Prevent accidental Vertex AI/OAuth authentication.
# This project uses the Gemini Developer API with an API key.
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"


# -----------------------------
# Check PDF
# -----------------------------
if not PDF_PATH.exists():
    print("\nERROR: Resume PDF was not found.")
    print(f"Expected location: {PDF_PATH}")
    print("\nPlease make sure your project looks like this:")
    print("muneeb-portfolio-rag/")
    print("├── create_database.py")
    print("├── .env")
    print("├── data/")
    print("│   └── resume.pdf")
    print("└── ...")
    sys.exit(1)


# -----------------------------
# Load PDF
# -----------------------------
print("\n" + "=" * 55)
print("MUNEEB PORTFOLIO RAG - DATABASE BUILDER")
print("=" * 55)

print("\n[1/5] Loading PDF...")

loader = PyMuPDFLoader(str(PDF_PATH))
documents = loader.load()

if not documents:
    print("ERROR: The PDF contains no readable pages/text.")
    sys.exit(1)

print(f"PDF loaded successfully: {len(documents)} page(s)")


# -----------------------------
# Split text
# -----------------------------
print("\n[2/5] Splitting text into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = splitter.split_documents(documents)

# Remove empty chunks
chunks = [
    chunk
    for chunk in chunks
    if chunk.page_content and chunk.page_content.strip()
]

if not chunks:
    print("ERROR: No text chunks were created from the PDF.")
    sys.exit(1)

# Add useful metadata
for i, chunk in enumerate(chunks):
    chunk.metadata["source"] = str(PDF_PATH.name)
    chunk.metadata["chunk_id"] = i

print(f"Created {len(chunks)} chunks.")


# -----------------------------
# Create embeddings
# -----------------------------
print("\n[3/5] Loading Gemini Embedding 2...")

try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        vertexai=False,
        output_dimensionality=EMBEDDING_DIMENSION,
    )
except Exception as e:
    print("\nERROR: Could not initialize Gemini embeddings.")
    print(f"Details: {e}")
    sys.exit(1)


# -----------------------------
# Test API before Chroma
# -----------------------------
print("\n[4/5] Testing Gemini API connection...")

try:
    test_vector = embeddings.embed_query(
        "Test query for Muneeb portfolio RAG."
    )

    if not test_vector:
        raise RuntimeError("Gemini returned an empty embedding.")

    print(
        f"Gemini API connection successful. "
        f"Vector size: {len(test_vector)}"
    )

except Exception as e:
    print("\nERROR: Gemini embedding test failed.")
    print("This is an API-key/model configuration problem,")
    print("not a ChromaDB problem.")
    print(f"\nDetails: {e}")
    print("\nCheck:")
    print("1. GOOGLE_API_KEY in .env")
    print("2. API key is a Gemini API key from Google AI Studio")
    print("3. GOOGLE_GENAI_USE_VERTEXAI is not forcing Vertex AI")
    print("4. langchain-google-genai and google-genai are updated")
    sys.exit(1)


# -----------------------------
# Rebuild Chroma database
# -----------------------------
print("\n[5/5] Creating Chroma database...")

# A database made with another embedding model can have a different
# vector dimension. Rebuilding prevents old vectors from causing
# dimension/mixed-model errors.
if DB_PATH.exists():
    print(f"Removing old Chroma database: {DB_PATH}")
    shutil.rmtree(DB_PATH)

try:
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_PATH),
    )

    # Force a real retrieval operation so we know Chroma can read
    # the newly-created vectors.
    test_results = db.similarity_search(
        "Who is Muneeb Ullah?",
        k=1,
    )

except Exception as e:
    print("\nERROR: Chroma database creation failed.")
    print(f"Details: {e}")
    sys.exit(1)


# -----------------------------
# Success
# -----------------------------
print("\n" + "=" * 55)
print("DATABASE CREATED SUCCESSFULLY!")
print("=" * 55)
print(f"PDF:       {PDF_PATH}")
print(f"Chunks:    {len(chunks)}")
print(f"Embedding: {EMBEDDING_MODEL}")
print(f"Dimension: {EMBEDDING_DIMENSION}")
print(f"Database:  {DB_PATH}")
print(f"Collection:{COLLECTION_NAME}")
print("=" * 55)
print("\nYour Chroma database is ready for the Streamlit RAG app.")
