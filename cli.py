import argparse
import yaml
import os
import json
from typing import List, Dict, Any
import sys

# Check for required environment variables at startup
def check_env_vars():
    """Check for required environment variables and provide guidance if missing."""
    if os.environ.get('GITHUB_TOKEN') is None:
        print("ERROR: GITHUB_TOKEN environment variable is not set.")
        print("Please run one of the following:")
        print("  1. python example_env_setup.py   # to set up all variables interactively")
        print("  2. python load_env.py            # to load variables from .env file")
        print("  3. python run.py <command>       # to run a command with automatic env loading")
        return False
    return True

# Import project modules after environment check
from src.document_indexer import DocumentIndexer
from src.query_engine import QueryEngine
from src.qa_system import QASystem
from src.evaluation import Evaluator
from src.chatbot import Chatbot

def index_documents(config_path: str, reset_db: bool = False, file_path: str = None) -> None:
    """
    Index documents in the data directory.
    
    Args:
        config_path: Path to configuration file
        reset_db: Whether to reset the vector database
        file_path: Optional specific file to index
    """
    indexer = DocumentIndexer(config_path)
    
    if file_path:
        # Check if the file exists
        if not os.path.isfile(file_path):
            print(f"Error: File '{file_path}' does not exist.")
            return
            
        # Check if the file is in the data directory or copy it there
        filename = os.path.basename(file_path)
        target_path = os.path.join(indexer.data_dir, filename)
        
        if file_path != target_path:
            print(f"Copying file to data directory: {file_path} -> {target_path}")
            import shutil
            shutil.copy2(file_path, target_path)
            file_path = target_path
        
        # Process single file
        print(f"Indexing specific file: {file_path}")
        docs = indexer.load_and_process_file(file_path)
        if docs:
            indexer.create_vector_store(docs)
            print(f"Successfully indexed file: {file_path}")
        else:
            print(f"No documents extracted from {file_path}")
    else:
        # Index all documents in the data directory
        indexer.index(reset_db=reset_db)

def list_indexed_files(config_path: str) -> None:
    """List all indexed files and their statistics."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = config['data_directory']
    vector_db_dir = config['vector_db_directory']
    
    # Check if the vector database exists
    if not os.path.exists(vector_db_dir):
        print("No vector database found. Please index documents first.")
        return
    
    print("\n===== Indexed Files =====\n")
    
    # Load the vector store to get metadata
    try:
        query_engine = QueryEngine(config_path)
        metadata = query_engine.get_document_metadata()
        
        if not metadata:
            print("No documents found in the vector store.")
            return
        
        # Group by source file
        files_stats = {}
        for meta in metadata:
            source = meta.get('source', 'Unknown')
            file_type = meta.get('file_type', 'Unknown')
            
            if source not in files_stats:
                files_stats[source] = {
                    'count': 0,
                    'file_type': file_type
                }
            
            files_stats[source]['count'] += 1
        
        # Display statistics
        for filename, stats in sorted(files_stats.items()):
            print(f"- {filename} ({stats['file_type']}): {stats['count']} chunks")
        
        print(f"\nTotal indexed documents: {len(metadata)} chunks")
        
    except Exception as e:
        print(f"Error accessing vector store: {str(e)}")

def query_documents(config_path: str, query: str) -> None:
    """Query the vector store and display results."""
    query_engine = QueryEngine(config_path)
    results = query_engine.query_vector_store(query)
    
    print(f"\n===== Query Results for: '{query}' =====\n")
    for i, (doc, score) in enumerate(results, 1):
        print(f"Result {i} (Similarity: {score:.4f}):")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}")
        
        # Show file type information if available
        if 'file_type' in doc.metadata:
            print(f"File type: {doc.metadata['file_type']}")
            
        print(f"Content: {doc.page_content[:200]}...\n")

def ask_question(config_path: str, query: str) -> None:
    """Generate a response to a question."""
    qa_system = QASystem(config_path)
    result = qa_system.generate_response(query)
    
    print(f"\n===== Question: '{query}' =====\n")
    print(f"Answer: {result['response']}\n")

def evaluate_system(config_path: str, test_file: str = None) -> None:
    """Evaluate the QA system using test cases."""
    evaluator = Evaluator(config_path)
    
    if test_file and os.path.exists(test_file):
        with open(test_file, 'r') as f:
            test_cases = json.load(f)
    else:
        # Default test cases if no file is provided
        test_cases = [
            {
                "query": "What is retrieval augmented generation?",
                "expected_answer": "Retrieval Augmented Generation (RAG) is a technique that enhances language models by retrieving relevant information from external sources before generating responses."
            },
            {
                "query": "What are embedding models used for?",
                "expected_answer": "Embedding models are used to convert text into numerical vectors, enabling semantic search and similarity comparisons between documents."
            }
        ]
    
    evaluator.evaluate_test_set(test_cases)

def start_chat(config_path: str) -> None:
    """Start an interactive chat session with the chatbot."""
    # First create the QA system to get the LLM
    qa_system = QASystem(config_path)
    
    # Then create the chatbot using the same LLM
    chatbot = Chatbot(config_path, qa_system.llm)
    
    print("\n===== Welcome to the RAG Chatbot =====")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Type 'reset' to clear conversation history.\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        elif user_input.lower() == 'reset':
            chatbot.reset()
            print("Conversation history has been reset.")
            continue
        
        response = chatbot.chat(user_input)
        print(f"Assistant: {response}\n")

def show_config(config_path: str) -> None:
    """Display the current configuration."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("\n===== Current Configuration =====\n")
        
        # Function to recursively print nested dicts with indentation
        def print_dict(d, indent=0):
            for key, value in d.items():
                if isinstance(value, dict):
                    print(" " * indent + f"{key}:")
                    print_dict(value, indent + 2)
                else:
                    print(" " * indent + f"{key}: {value}")
        
        print_dict(config)
        
    except Exception as e:
        print(f"Error reading configuration: {str(e)}")

def main():
    """Main function to handle CLI arguments and execute commands."""
    # Check environment variables first
    if not check_env_vars():
        sys.exit(1)
        
    parser = argparse.ArgumentParser(description='RAG System CLI')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index documents')
    index_parser.add_argument('--reset-db', action='store_true', help='Reset the vector database')
    index_parser.add_argument('--file', type=str, help='Specific file to index')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List indexed documents')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the vector store')
    query_parser.add_argument('query_text', type=str, help='Query text')
    
    # Ask command
    ask_parser = subparsers.add_parser('ask', help='Ask a question')
    ask_parser.add_argument('query_text', type=str, help='Question text')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the QA system')
    eval_parser.add_argument('--test-file', type=str, help='Path to test cases JSON file')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Start an interactive chat session')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show current configuration')
    
    args = parser.parse_args()
    
    if args.command == 'index':
        index_documents(args.config, args.reset_db, args.file)
    elif args.command == 'list':
        list_indexed_files(args.config)
    elif args.command == 'query':
        query_documents(args.config, args.query_text)
    elif args.command == 'ask':
        ask_question(args.config, args.query_text)
    elif args.command == 'evaluate':
        evaluate_system(args.config, args.test_file)
    elif args.command == 'chat':
        start_chat(args.config)
    elif args.command == 'config':
        show_config(args.config)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
