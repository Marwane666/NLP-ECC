"""
Utility script to reset the vector database and reindex documents.
This is useful when encountering dimension mismatch errors.
"""
import os
import sys
import shutil
import argparse
from load_env import load_env_file
from src.document_indexer import DocumentIndexer

def reset_vector_db(config_path: str = "config.yaml"):
    """
    Reset the vector database and reindex all documents.
    
    Args:
        config_path: Path to the configuration file
    """
    # First load environment variables
    load_env_file(verbose=False)
    
    # Create a document indexer with reset_db=True
    indexer = DocumentIndexer(config_path, reset_db=True)
    
    # Run the indexing process
    indexer.index()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reset vector database and reindex documents')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    reset_vector_db(args.config)
