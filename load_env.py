"""
Utility script to load environment variables from a .env file.
Run this script before using the RAG system if you've saved variables to a .env file.
"""
import os
import sys
from dotenv import load_dotenv

def load_env_file(env_file=".env", verbose=True):
    """
    Load environment variables from a .env file.
    
    Args:
        env_file: Path to the .env file (default: ".env")
        verbose: Whether to print status messages
        
    Returns:
        Dictionary containing the loaded environment variables
    """
    loaded_vars = {}
    
    if os.path.exists(env_file):
        # Load environment variables from .env file
        load_dotenv(env_file)
        
        # Explicitly set them in the current process and capture them
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    # Set in current environment
                    os.environ[key] = value
                    loaded_vars[key] = value
        
        # Check if key variables were loaded
        if verbose:
            token = os.environ.get("GITHUB_TOKEN")
            if token:
                print(f"GitHub token loaded (length: {len(token)})")
            else:
                print("Warning: GITHUB_TOKEN not found in .env file")
            
            endpoint = os.environ.get("AZURE_INFERENCE_ENDPOINT")
            if endpoint:
                print(f"Azure Inference endpoint loaded: {endpoint}")
            
            chat_model = os.environ.get("AZURE_INFERENCE_CHAT_MODEL")
            if chat_model:
                print(f"Chat model loaded: {chat_model}")
            
            embedding_model = os.environ.get("AZURE_INFERENCE_EMBEDDING_MODEL")
            if embedding_model:
                print(f"Embedding model loaded: {embedding_model}")
        
        return loaded_vars
    else:
        if verbose:
            print(f"Error: {env_file} file not found.")
            print("Run 'python example_env_setup.py' to create a .env file.")
        return {}

if __name__ == "__main__":
    vars_loaded = load_env_file()
    if vars_loaded:
        print("\nEnvironment variables loaded successfully.")
        print("You can now run the RAG system.")
    else:
        print("\nFailed to load environment variables.")
        sys.exit(1)
