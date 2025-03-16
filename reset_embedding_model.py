"""
Utility script to help recover from embedding dimension mismatch errors.
This script completely rebuilds the vector database using the current embedding model.
"""
import os
import sys
import shutil
import time
import yaml
import argparse
from load_env import load_env_file
from src.document_indexer import DocumentIndexer

def reset_embedding_model(config_path: str = "config.yaml"):
    """
    Reset the vector database completely to fix dimension mismatch issues.
    
    Args:
        config_path: Path to the configuration file
    """
    # First load environment variables
    load_env_file(verbose=False)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    vector_db_dir = config['vector_db_directory']
    data_dir = config['data_directory']
    
    print(f"Checking data directory {data_dir}...")
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print("No documents found to index. Please add documents to the data directory first.")
        return False
    
    print(f"Backing up vector database at {vector_db_dir}...")
    if os.path.exists(vector_db_dir):
        # Create backup
        backup_dir = f"{vector_db_dir}_backup_{int(time.time())}"
        try:
            shutil.copytree(vector_db_dir, backup_dir)
            print(f"Created backup at {backup_dir}")
        except Exception as e:
            print(f"Error creating backup: {e}")
    
    print("Removing existing vector database...")
    if os.path.exists(vector_db_dir):
        try:
            shutil.rmtree(vector_db_dir)
            print("Vector database removed successfully.")
        except Exception as e:
            print(f"Error removing vector database: {e}")
            print("The database might be in use by another process.")
            print("Please close any applications that might be using it and try again.")
            return False
    
    # Create the directory again
    os.makedirs(vector_db_dir, exist_ok=True)
    
    print("Creating new indexer...")
    indexer = DocumentIndexer(config_path, reset_db=True)
    
    print("Loading documents...")
    docs = indexer.load_documents()
    
    if not docs:
        print("No documents were loaded. Please check your data directory.")
        return False
    
    print(f"Creating vector store with {len(docs)} document chunks...")
    try:
        indexer.create_vector_store(docs)
        print("Vector store created successfully!")
        return True
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reset embedding model and rebuild vector database')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EMBEDDING MODEL RESET UTILITY")
    print("=" * 80)
    print("\nThis utility will reset your vector database and rebuild it using the current embedding model.")
    print("It's useful when you encounter dimension mismatch errors.")
    print("\nWARNING: This will delete your existing vector database. A backup will be created, but proceed with caution.")
    
    confirm = input("\nAre you sure you want to continue? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        sys.exit(0)
    
    success = reset_embedding_model(args.config)
    
    if success:
        print("\nVector database has been successfully rebuilt!")
    else:
        print("\nFailed to rebuild vector database. Check the error messages above.")
