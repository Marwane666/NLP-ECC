"""
Utility script to install missing packages required by the RAG system.
"""
import subprocess
import sys
import os

def install_packages():
    """Install missing packages for the RAG system."""
    print("Installing required packages...")
    
    # First uninstall any existing pydantic to avoid conflicts
    print("Removing existing pydantic installation...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "pydantic"])
        print("Successfully removed existing pydantic")
    except subprocess.CalledProcessError as e:
        print(f"Error removing pydantic: {e}")
    
    # Base packages
    base_packages = [
        "langchain>=0.0.325",
        "langchain-core>=0.1.17", 
        "langchain-community>=0.0.13",
        "langchain-huggingface>=0.0.10",
        "langchain-chroma>=0.0.10",
        "pydantic>=2.4.0",  # Use Pydantic v2 which is compatible with latest langchain
        "pypdf>=3.12.1",
        "azure-ai-inference>=1.0.0b9",
        "python-dotenv>=1.0.0",
        "colorama>=0.4.6"
    ]
    
    # Install base packages first
    for package in base_packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
    
    print("\nInstallation complete. Please run your command again.")

if __name__ == "__main__":
    install_packages()
