# Retrieval Augmented Generation (RAG) System

This project implements a Retrieval Augmented Generation (RAG) system, which combines document retrieval with language model generation to provide accurate, context-aware answers to user queries.

## Features

- Document indexing with proper chunking and metadata preservation
- Vector storage using ChromaDB
- Semantic search for query-document matching
- Question answering using LLMs
- Evaluation mechanisms for response quality
- Interactive chatbot with conversation history

## Project Structure

## How to Use the RAG System

first of all you have to install the requirements:

```bash
pip install -r requirements.txt
```

### 1. Index Your Documents

This process:
- Loads PDFs from the data directory
- Splits them into manageable chunks
- Computes vector embeddings
- Stores them in a vector database

```bash
python cli.py index
```

### 2. Using the RAG System

#### Querying the Vector Store
To find relevant document sections:
- Output will show the most relevant document chunks with similarity scores.

```bash
python cli.py  query "Your search query here"
```

#### Asking Questions
To generate a complete answer using the LLM:
- The system will:
  - Find relevant document chunks
  - Use them as context for the LLM
  - Generate a response grounded in your documents

```bash
python cli.py  ask "Your question here?"
```

### 3. Evaluating System Performance
Test case format:
- [Additional evaluation details would go here]

```bash
python cli.py  evaluate --test-file test_cases.json
```

### 4. Interactive Chatbot
For a conversational experience:

```bash
python cli.py  chat
```

In chat mode:
- The system maintains conversation history
- Type 'reset' to clear history
- Type 'exit' or 'quit' to end the session

