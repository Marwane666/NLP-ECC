"""
Create and set up a virtual environment for the RAG system.
This helps isolate dependencies and avoid conflicts.
"""
import subprocess
import sys
import os
import platform

def create_venv():
    """Create a virtual environment and install dependencies."""
    venv_name = "rag_env"
    
    print(f"Creating virtual environment '{venv_name}'...")
    
    # Determine the system
    is_windows = platform.system() == "Windows"
    
    # Create the virtual environment
    try:
        subprocess.check_call([sys.executable, "-m", "venv", venv_name])
        print(f"Virtual environment '{venv_name}' created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        return False
    
    # Determine path to pip in the virtual environment
    if is_windows:
        pip_path = os.path.join(venv_name, "Scripts", "pip")
        python_path = os.path.join(venv_name, "Scripts", "python")
        activate_path = os.path.join(venv_name, "Scripts", "activate")
    else:
        pip_path = os.path.join(venv_name, "bin", "pip")
        python_path = os.path.join(venv_name, "bin", "python")
        activate_path = os.path.join(venv_name, "bin", "activate")
    
    # Upgrade pip in the virtual environment
    try:
        subprocess.check_call([pip_path, "install", "--upgrade", "pip"])
        print("Pip upgraded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error upgrading pip: {e}")
    
    # Install dependencies in the virtual environment
    packages = [
        "langchain>=0.0.325",
        "langchain-core>=0.1.17",
        "langchain-community>=0.0.13",
        "langchain-huggingface>=0.0.10",
        "langchain-chroma>=0.0.10",
        "pydantic>=2.4.0",
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
        "requests>=2.31.0",
        "azure-ai-inference>=1.0.0b9",
        "azure-core>=1.29.4",
        "colorama>=0.4.6",
        "python-dotenv>=1.0.0"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([pip_path, "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
    
    # Create activation scripts
    if is_windows:
        with open("activate.bat", "w") as f:
            f.write(f"@echo off\n"
                   f"echo Activating virtual environment {venv_name}...\n"
                   f"call {activate_path}\n"
                   f"echo Environment activated. Run your commands now.\n")
        print("Created activate.bat for activating the virtual environment.")
    else:
        with open("activate.sh", "w") as f:
            f.write(f"#!/bin/bash\n"
                   f"echo Activating virtual environment {venv_name}...\n"
                   f"source {activate_path}\n"
                   f"echo Environment activated. Run your commands now.\n")
        os.chmod("activate.sh", 0o755)
        print("Created activate.sh for activating the virtual environment.")
    
    print("\nVirtual environment setup complete!")
    print("\nTo activate the virtual environment:")
    if is_windows:
        print("  Run: activate.bat")
    else:
        print("  Run: source ./activate.sh")
    
    print("\nAfter activation, you can run commands like:")
    print("  python cli.py chat")
    
    return True

if __name__ == "__main__":
    create_venv()
