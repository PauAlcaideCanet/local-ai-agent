from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def sync_database():
    # 1. Define how to load different file types
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
        ".md": TextLoader,
    }

    def create_directory_loader(extension, loader_cls):
        return DirectoryLoader(
            path="data/",
            glob=f"**/*{extension}",
            loader_cls=loader_cls,
            show_progress=True
        )

    # 2. Load all documents from the data folder
    docs = []
    for ext, loader_cls in loaders.items():
        loader = create_directory_loader(ext, loader_cls)
        docs.extend(loader.load())

    if not docs:
        print("No documents found in /data.")
        return

    # 3. Split and Index (Same as before)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory="chroma_db"
    )
    print(f"Ingested {len(docs)} files ({len(chunks)} chunks).")