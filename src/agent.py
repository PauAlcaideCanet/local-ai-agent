from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

class LocalAgent:
    def __init__(self):
        # 1. Configuration
        self.model_name = "llama3.2"
        self.embeddings_model = "nomic-embed-text"
        self.db_path = "chroma_db"
        
        # 2. Initialize Embeddings and Vector Store
        # We check if the DB exists to avoid errors on the first run
        self.embeddings = OllamaEmbeddings(model=self.embeddings_model)
        
        if os.path.exists(self.db_path):
            self.vectorstore = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
        else:
            self.vectorstore = None
            print(f"Warning: {self.db_path} not found. Please run ingestion first.")

        # 3. Initialize the LLM (Hardware-aware)
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=0,  # Keep it deterministic for engineering tasks
            num_gpu=1       # Hint to use NVIDIA if available
        )

        # 4. Define the Prompt Template
        # This tells the AI how to behave and how to use the retrieved context
        template = """
        You are a professional Engineering Assistant. Use the following pieces of 
        retrieved context to answer the question. If you don't know the answer 
        based on the context, just say that you don't know, don't try to 
        make up an answer.

        Context:
        {context}

        Question: 
        {question}

        Answer:
        """
        self.prompt = ChatPromptTemplate.from_template(template)

        # 5. Build the RAG Chain (The "2026" Pipe Syntax)
        if self.vectorstore:
            self.chain = (
                {"context": self.vectorstore.as_retriever(  search_type="mmr", search_kwargs={ "k": 6,              # Number of chunks to send to the AI
                                                                                                "fetch_k": 20,       # Number of chunks to initially pull from DB to choose from
                                                                                                "lambda_mult": 0.5    # 0.5 balances relevance vs. diversity
                                                                                            }), 
                 "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            self.chain = None

    def ask(self, query: str):
        """Processes the query through the RAG pipeline."""
        if not self.chain:
            return "Error: Vector database not initialized. Add documents to /data and sync."
        
        try:
            # invoke() is the modern standard for executing chains
            return self.chain.invoke(query)
        except Exception as e:
            return f"An error occurred: {str(e)}"

# Self-test block: This only runs if you execute agent.py directly
if __name__ == "__main__":
    agent = LocalAgent()
    if agent.chain:
        print("Agent initialized successfully.")