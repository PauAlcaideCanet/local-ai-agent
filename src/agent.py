from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

CHROMA_PATH = "chroma_db"

class LocalAgent:
    def __init__(self):
        # Initialize Embeddings (must match ingest.py)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Load the existing database
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=self.embeddings
        )
        
        # Initialize LLM
        self.llm = ChatOllama(model="llama3.2")
        
        # Create the Retrieval Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            self.llm, 
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
        )

    def ask(self, query):
        return self.qa_chain.invoke(query)