---
noteId: "0d6eafb0021211f08616c98447463cc4"
tags: []

---

# Retrieval Augmented Generation (RAG) System

This project implements a Retrieval Augmented Generation (RAG) system, which combines document retrieval with language model generation to provide accurate, context-aware answers to user queries.

## Features

- Document indexing with proper chunking and metadata preservation
- Vector storage using ChromaDB
- Semantic search for query-document matching
- Question answering using LLMs
- Evaluation mechanisms for response quality
- Interactive chatbot with conversation history
- Web interface using Flask

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

## Web Interface (Flask Application)

The system also includes a web interface built with Flask, providing a more user-friendly way to interact with the RAG system.

### Running the Web Application

To start the web interface:

```bash
python app.py
```

Then open your browser and navigate to: `http://127.0.0.1:5000`

### Features of the Web Interface

The web application provides the following functionality:

1. **Document Indexing**: Index documents through the web interface
2. **Document Querying**: Search for relevant document chunks
3. **Question Answering**: Get answers to your questions based on the indexed documents
4. **Interactive Chat**: Have a conversation with the RAG chatbot with persistent session history

The interface is intuitive and provides real-time feedback for all operations.

