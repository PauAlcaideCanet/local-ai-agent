# Local RAG Intelligence Agent

A privacy-focused, local-first Retrieval-Augmented Generation (RAG) system. This project leverages **Ollama** and **LangChain** to provide an intelligent interface for querying heterogeneous document sets (PDF, DOCX, TXT, MD) without data ever leaving the local machine.

---

## System Architecture

The system is built with a decoupled architecture to ensure scalability and maintainability:

*   **UI Layer:** Streamlit-based web interface providing real-time chat interactions and session state management.
*   **Orchestration:** LangChain Expression Language (LCEL) managing the data flow between the vector store and the LLM.
*   **Vector Engine:** ChromaDB utilizing `nomic-embed-text` for high-dimensional semantic search and persistence.
*   **Inference Engine:** Ollama running `llama3.2` locally.

---

## Prerequisites

*   **Python:** 3.10 or higher
*   **Ollama:** Installed and running on the local host.
*   **Models:**
    *   `ollama pull llama3.2`
    *   `ollama pull nomic-embed-text`

---

## Getting Started
Clone the repository:
```
  git clone https://github.com/PauAlcaideCanet/local-ai-agent
```

### Environment Setup
Initialize a virtual environment:
```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
Put the files you want to inspect into the `/data` folder

### Run the streamlit application
To run the app, from the project main folder run:
```
  streamlit run app.py
```

