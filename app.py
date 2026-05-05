import streamlit as st
import os
from src.agent import LocalAgent
from src.ingest import sync_database

st.set_page_config(page_title="Local AI Agent", layout="wide")
st.title("🤖 Privacy-First Local AI Agent")

# --- 1. INITIALIZATION LOGIC ---
# This block runs automatically when the app opens
if "initialized" not in st.session_state:
    st.info("Checking system status...")
    
    # Check if the vector database exists
    if not os.path.exists("chroma_db"):
        with st.spinner("First-time setup: Indexing your /data folder..."):
            sync_database()
            st.success("Indexing complete!")
    
    # Load the agent into memory
    st.session_state.agent = LocalAgent()
    st.session_state.initialized = True
    st.rerun() # Refresh to clear the "Initializing" messages

# --- 2. SIDEBAR (Manual Sync) ---
with st.sidebar:
    st.header("Settings")
    if st.button("Re-index Data"):
        with st.spinner("Updating database..."):
            sync_database()
            # We need to reload the agent to see the new data
            st.session_state.agent = LocalAgent()
        st.success("Database Updated!")

# --- 3. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Since we initialized above, st.session_state.agent is guaranteed to exist
            response = st.session_state.agent.ask(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})