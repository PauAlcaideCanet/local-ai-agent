from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
DATA_PATH = "data/"
CHROMA_PATH = "chroma_db"

def sync_database():
    # 1. Load Documents
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    
    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    # 3. Create/Update Vector Store
    print(f"Indexing {len(chunks)} chunks into {CHROMA_PATH}...")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory=CHROMA_PATH
    )
    print("Ingestion complete.")
    return vectorstore

if __name__ == "__main__":
    sync_database()