"""
Environment setup script for the RAG system.
This script helps install dependencies and set up environment variables.
"""
import subprocess
import sys
import os
import getpass
import platform

def check_pip():
    """Check if pip is installed."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"])
        return True
    except subprocess.CalledProcessError:
        return False

def install_package(package_name):
    """Install a package using pip."""
    print(f"Installing {package_name}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    print(f"Successfully installed {package_name}")

def install_dependencies(mode="all"):
    """
    Install dependencies for the RAG system.
    
    Args:
        mode: Installation mode - "base", "azure_inference", "azure_openai", or "all"
    """
    if not check_pip():
        print("Error: pip is not installed or not working correctly.")
        return False
    
    try:
        # Install base requirements first
        if mode in ["base", "all"]:
            print("Installing base dependencies...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "langchain>=0.0.325",
                "langchain-community>=0.0.13",
                "sentence-transformers>=2.2.2",
                "chromadb>=0.4.6",
                "PyYAML>=6.0",
                "transformers>=4.30.1",
                "torch>=2.0.0",
                "numpy>=1.24.3",
                "scikit-learn>=1.2.2",
                "rouge>=1.0.1",
                "pypdf>=3.12.1",
                "ctransformers>=0.2.24",
                "flask==2.3.3",
                "requests>=2.31.0"
            ])
            print("Base dependencies installed successfully.")
        
        # Install Azure AI Inference dependencies
        if mode in ["azure_inference", "all"]:
            print("Installing Azure AI Inference dependencies...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "azure-ai-inference>=1.0.0",
                "azure-core>=1.29.4"
            ])
            print("Azure AI Inference dependencies installed successfully.")
        
        # Install Azure OpenAI dependencies
        if mode in ["azure_openai", "all"]:
            print("Installing Azure OpenAI dependencies...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "langchain-openai>=0.0.2",
                "openai>=1.3.0"
            ])
            print("Azure OpenAI dependencies installed successfully.")
        
        print("\nAll required dependencies installed successfully!")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def setup_azure_inference_env():
    """Set up environment variables for Azure AI Inference."""
    print("\nSetting up Azure AI Inference environment variables...")
    
    # Get GitHub token which serves as API key
    if not os.environ.get("GITHUB_TOKEN"):
        api_key = getpass.getpass("Enter your GitHub token (used as API key): ")
        os.environ["GITHUB_TOKEN"] = api_key
    
    # Get Azure AI Inference endpoint
    if not os.environ.get("AZURE_INFERENCE_ENDPOINT"):
        endpoint = input("Enter Azure AI Inference endpoint (default: https://models.inference.ai.azure.com): ") or "https://models.inference.ai.azure.com"
        os.environ["AZURE_INFERENCE_ENDPOINT"] = endpoint
    
    # Get chat model name
    if not os.environ.get("AZURE_INFERENCE_CHAT_MODEL"):
        chat_model = input("Enter chat model name (default: gpt-4o): ") or "gpt-4o"
        os.environ["AZURE_INFERENCE_CHAT_MODEL"] = chat_model
    
    # Get embeddings model name
    if not os.environ.get("AZURE_INFERENCE_EMBEDDING_MODEL"):
        embedding_model = input("Enter embedding model name (default: text-embedding-3-small): ") or "text-embedding-3-small"
        os.environ["AZURE_INFERENCE_EMBEDDING_MODEL"] = embedding_model
    
    print("Environment variables set successfully for current session.")
    
    # Option to save environment variables permanently
    save_env = input("Would you like to save these environment variables permanently? (y/n): ").lower()
    if save_env == 'y':
        save_environment_variables({
            "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN"),
            "AZURE_INFERENCE_ENDPOINT": os.environ.get("AZURE_INFERENCE_ENDPOINT"),
            "AZURE_INFERENCE_CHAT_MODEL": os.environ.get("AZURE_INFERENCE_CHAT_MODEL"),
            "AZURE_INFERENCE_EMBEDDING_MODEL": os.environ.get("AZURE_INFERENCE_EMBEDDING_MODEL")
        })

def save_environment_variables(env_vars):
    """Save environment variables permanently based on the operating system."""
    system = platform.system()
    
    if system == "Windows":
        # Create a batch file for Windows
        with open("set_env_vars.bat", "w") as f:
            for key, value in env_vars.items():
                if value:  # Only write non-empty values
                    f.write(f'setx {key} "{value}"\n')
        
        print("Environment variables saved to set_env_vars.bat")
        print("Run this file as administrator to set variables permanently.")
    
    elif system == "Linux" or system == "Darwin":  # Darwin is macOS
        # Create a shell script for Linux/macOS
        with open("set_env_vars.sh", "w") as f:
            f.write("#!/bin/bash\n")
            for key, value in env_vars.items():
                if value:  # Only write non-empty values
                    f.write(f'export {key}="{value}"\n')
            f.write('\necho "To make these variables permanent, add them to your .bashrc or .bash_profile file."\n')
        
        os.chmod("set_env_vars.sh", 0o755)  # Make the script executable
        print("Environment variables saved to set_env_vars.sh")
        print("Run 'source set_env_vars.sh' to set variables for current session.")
    
    else:
        print(f"Unsupported operating system: {system}")
        print("You'll need to set the environment variables manually.")

def main():
    """Main function to run the setup script."""
    print("========================================")
    print("RAG System Environment Setup")
    print("========================================")
    
    print("\nThis script will help you set up your environment for the RAG system.")
    print("It will install required dependencies and set up environment variables.")
    
    # Ask user which components to install
    print("\nWhich components would you like to set up?")
    print("1. Base system only (LangChain, vector store, etc.)")
    print("2. Base + Azure AI Inference integration")
    print("3. Base + Azure OpenAI integration")
    print("4. Complete system (all components)")
    
    choice = input("\nEnter your choice (1-4): ")
    
    mode_map = {
        "1": "base",
        "2": "azure_inference",
        "3": "azure_openai", 
        "4": "all"
    }
    
    mode = mode_map.get(choice, "base")
    
    # Install dependencies
    success = install_dependencies(mode)
    
    if success and (mode in ["azure_inference", "all"]):
        setup_azure_inference_env()
    
    if success:
        print("\n========================================")
        print("Setup completed successfully!")
        print("========================================")
        
        print("\nYou can now use the RAG system. To get started, try:")
        print("1. Add PDF documents to the 'data' directory")
        print("2. Run 'python cli.py index' to index your documents")
        print("3. Run 'python cli.py ask \"Your question here\"' to ask a question")
    else:
        print("\n========================================")
        print("Setup encountered some errors.")
        print("Please check the error messages above and try again.")
        print("========================================")

if __name__ == "__main__":
    main()
