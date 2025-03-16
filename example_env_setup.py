"""
Example script to set up Azure AI Inference environment variables.
Run this script before using the RAG system.
"""
import getpass
import os
import sys
import platform

def setup_azure_env():
    """Set up Azure AI Inference environment variables interactively."""
    print("Setting up Azure AI Inference environment variables...")
    
    # Get GitHub token which serves as API key
    if not os.environ.get("GITHUB_TOKEN"):
        api_key = getpass.getpass("Enter your GitHub token (used as API key): ")
        if not api_key:
            print("Error: GitHub token is required to use Azure AI Inference services.")
            print("Please run this script again and provide a valid token.")
            sys.exit(1)
        os.environ["GITHUB_TOKEN"] = api_key
        # Also explicitly set it for the current process
        print(f"GitHub token set (length: {len(api_key)})")
    else:
        print(f"GitHub token already set in environment (length: {len(os.environ.get('GITHUB_TOKEN'))})")
    
    # Get Azure AI Inference endpoint
    if not os.environ.get("AZURE_INFERENCE_ENDPOINT"):
        endpoint = input("Enter Azure AI Inference endpoint (default: https://models.inference.ai.azure.com): ") or "https://models.inference.ai.azure.com"
        os.environ["AZURE_INFERENCE_ENDPOINT"] = endpoint
        print(f"Azure Inference endpoint set to: {endpoint}")
    else:
        print(f"Azure Inference endpoint already set to: {os.environ.get('AZURE_INFERENCE_ENDPOINT')}")
    
    # Get chat model name
    if not os.environ.get("AZURE_INFERENCE_CHAT_MODEL"):
        chat_model = input("Enter chat model name (default: gpt-4o): ") or "gpt-4o"
        os.environ["AZURE_INFERENCE_CHAT_MODEL"] = chat_model
        print(f"Chat model set to: {chat_model}")
    else:
        print(f"Chat model already set to: {os.environ.get('AZURE_INFERENCE_CHAT_MODEL')}")
    
    # Get embeddings model name
    if not os.environ.get("AZURE_INFERENCE_EMBEDDING_MODEL"):
        embedding_model = input("Enter embedding model name (default: text-embedding-3-small): ") or "text-embedding-3-small"
        os.environ["AZURE_INFERENCE_EMBEDDING_MODEL"] = embedding_model
        print(f"Embedding model set to: {embedding_model}")
    else:
        print(f"Embedding model already set to: {os.environ.get('AZURE_INFERENCE_EMBEDDING_MODEL')}")
    
    # Save to .env file for future sessions
    save_to_env_file = input("\nWould you like to save these variables to a .env file for future use? (y/n): ").lower()
    if save_to_env_file == 'y':
        with open(".env", "w") as f:
            f.write(f"GITHUB_TOKEN={os.environ.get('GITHUB_TOKEN')}\n")
            f.write(f"AZURE_INFERENCE_ENDPOINT={os.environ.get('AZURE_INFERENCE_ENDPOINT')}\n")
            f.write(f"AZURE_INFERENCE_CHAT_MODEL={os.environ.get('AZURE_INFERENCE_CHAT_MODEL')}\n")
            f.write(f"AZURE_INFERENCE_EMBEDDING_MODEL={os.environ.get('AZURE_INFERENCE_EMBEDDING_MODEL')}\n")
        print("\n.env file created successfully.")
        
        # Add guidance for system-specific .env loading
        system = platform.system()
        if system == "Windows":
            print("\nTo load variables in the current terminal session:")
            print("  for /f \"tokens=*\" %i in (.env) do set %i")
        else:
            print("\nTo load variables in the current terminal session:")
            print("  export $(grep -v '^#' .env | xargs)")
        
        print("\nAlternatively, you can now use the wrapper script to run commands:")
        print("  python run.py index")
        print("  python run.py ask \"Your question here\"")
    
    print("\nEnvironment variables set successfully.")
    print("You can now run the RAG system with Azure AI Inference integration.")
    
    # Verify setup
    verify = input("\nWould you like to verify your environment setup? (y/n): ").lower()
    if verify == 'y':
        # Print current environment variables (masked for security)
        print("\nCurrent environment variables:")
        token = os.environ.get("GITHUB_TOKEN", "")
        masked_token = token[:4] + "****" + token[-4:] if len(token) > 8 else "Not set"
        print(f"GITHUB_TOKEN: {masked_token}")
        print(f"AZURE_INFERENCE_ENDPOINT: {os.environ.get('AZURE_INFERENCE_ENDPOINT', 'Not set')}")
        print(f"AZURE_INFERENCE_CHAT_MODEL: {os.environ.get('AZURE_INFERENCE_CHAT_MODEL', 'Not set')}")
        print(f"AZURE_INFERENCE_EMBEDDING_MODEL: {os.environ.get('AZURE_INFERENCE_EMBEDDING_MODEL', 'Not set')}")

if __name__ == "__main__":
    setup_azure_env()
