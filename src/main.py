import sys
from ingest import sync_database
from agent import LocalAgent

def main():
    print("--- Local AI Agent Shell ---")
    
    # Optional: Sync database on startup
    choice = input("Do you want to re-index your 'data' folder? (y/n): ")
    if choice.lower() == 'y':
        sync_database()

    # Initialize the agent
    print("Loading Agent...")
    agent = LocalAgent()
    
    print("\nReady! Type 'exit' to quit.")
    
    while True:
        user_input = input("\nQuestion: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        print("\nThinking...")
        response = agent.ask(user_input)
        print(f"\nAI: {response['result']}")

if __name__ == "__main__":
    main()