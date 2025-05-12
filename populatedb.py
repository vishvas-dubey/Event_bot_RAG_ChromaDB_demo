import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import shutil

# Load environment variables
load_dotenv()

# --- Configuration ---
DOCUMENTS_DIR = "documents"
CHROMA_PERSIST_DIR = "chroma"
EMBEDDING_MODEL = "models/embedding-001"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Validation ---
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file.")
    exit()

if not os.path.exists(DOCUMENTS_DIR):
    print(f"Error: Documents directory '{DOCUMENTS_DIR}' not found.")
    print("Please create a folder named 'documents' and place your PDF files inside.")
    exit()

# --- Functions ---
def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() or "" # Add empty string if page is empty
            return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return None

def process_documents(documents_dir):
    """Reads all PDFs in the directory and extracts text."""
    all_texts = []
    print(f"Scanning directory: {documents_dir}")
    for filename in os.listdir(documents_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(documents_dir, filename)
            print(f"Processing {filename}...")
            text = extract_text_from_pdf(pdf_path)
            if text:
                # Store as Langchain Document objects, including source metadata
                all_texts.append(Document(page_content=text, metadata={"source": filename}))
                print(f"Extracted text from {filename}")
            else:
                print(f"Could not extract text from {filename}")
    return all_texts

def split_text_into_chunks(documents):
    """Splits list of documents into smaller chunks."""
    # You can adjust chunk_size and chunk_overlap as needed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=800)
    print(f"Splitting documents into chunks...")
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def create_vector_database(chunks, persist_directory, api_key, embedding_model):
    """Creates or updates a Chroma vector database from text chunks."""
    print(f"Initializing Google Generative AI Embeddings using model: {embedding_model}")
    embedding_function = GoogleGenerativeAIEmbeddings(
        model=embedding_model,
        google_api_key=api_key
    )

    # Option: Clear existing database before creating a new one
    if os.path.exists(persist_directory):
        print(f"Removing existing Chroma database at {persist_directory}...")
        shutil.rmtree(persist_directory)
        print("Existing database removed.")

    print(f"Creating new Chroma database at {persist_directory}...")
    db = Chroma.from_documents(
        chunks,
        embedding_function,
        persist_directory=persist_directory
    )
    db.persist() # Ensure data is written to disk
    print(f"Chroma database created successfully with {len(chunks)} chunks.")
    print(f"Database stored in: {os.path.abspath(persist_directory)}")

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Process PDF documents
    documents = process_documents(DOCUMENTS_DIR)

    if not documents:
        print("No text extracted from any PDF files. Exiting.")
        exit()

    # 2. Split text into chunks
    text_chunks = split_text_into_chunks(documents)

    if not text_chunks:
        print("No text chunks created after splitting. Exiting.")
        exit()

    # 3. Create/Update the vector database
    create_vector_database(
        text_chunks,
        CHROMA_PERSIST_DIR,
        GEMINI_API_KEY,
        EMBEDDING_MODEL
    )

    print("\nVector database creation process finished.")
    print("You can now run your Streamlit app.")