import argparse
import yaml
import os
import json
from typing import List, Dict, Any

from src.document_indexer import DocumentIndexer
from src.query_engine import QueryEngine
from src.qa_system import QASystem
from src.evaluation import Evaluator
from src.chatbot import Chatbot

def index_documents(config_path: str) -> None:
    """Index documents in the data directory."""
    indexer = DocumentIndexer(config_path)
    indexer.index()

def query_documents(config_path: str, query: str) -> None:
    """Query the vector store and display results."""
    query_engine = QueryEngine(config_path)
    results = query_engine.query_vector_store(query)
    
    print(f"\n===== Query Results for: '{query}' =====\n")
    for i, (doc, score) in enumerate(results, 1):
        print(f"Result {i} (Similarity: {score:.4f}):")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}")
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

def main():
    """Main function to handle CLI arguments and execute commands."""
    parser = argparse.ArgumentParser(description='RAG System CLI')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index documents')
    
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
    
    args = parser.parse_args()
    
    if args.command == 'index':
        index_documents(args.config)
    elif args.command == 'query':
        query_documents(args.config, args.query_text)
    elif args.command == 'ask':
        ask_question(args.config, args.query_text)
    elif args.command == 'evaluate':
        evaluate_system(args.config, args.test_file)
    elif args.command == 'chat':
        start_chat(args.config)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
