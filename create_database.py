import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Settings

PDF_PATH = 'data/resume.pdf'
DB_PATH = 'chroma_db'

EMBEDDING_MODEL = 'embeddinggemma'
COLLECTION_NAME = 'muneeb_portfolio'

# Check PDF

if not os.path.exists(PDF_PATH):
    print("Error: PDF file not found!")
    print('Please put your resume inside the folder.')
    exit()

# Load PDF

print("Loading PDF...")
loader = PyMuPDFLoader(PDF_PATH)
documents = loader.load()

print('PDF Loaded successfully!')

# Split text into chunks
print('Splitting text into chunks...')

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)
chunks = splitter.split_documents(documents)

print(f'Created {len(chunks)} chunks from the PDF.')

# Create embeddings
print('Loading Ollama embeddings model...')

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

# Create Chroma database
print('Creating Chroma database...')
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=DB_PATH
)

print()
print('==================================')
print('Database created successfully!')
print('==================================')
print()
print(f'Chunks stored: {len(chunks)}')
print(f'Chroma database path: {DB_PATH}')