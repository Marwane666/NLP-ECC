from flask import Flask, render_template, request, jsonify, session
import os
import sys
import subprocess
from load_env import load_env_file
from src.document_indexer import DocumentIndexer
from src.query_engine import QueryEngine
from src.qa_system import QASystem
from src.chatbot import Chatbot

app = Flask(__name__)
app.secret_key = "rag_system_secret_key"  # For session management
config_path = 'config.yaml'

# Load environment variables first
env_vars = load_env_file(verbose=False)
if not env_vars:
    print("Warning: Environment variables not loaded. Some features may not work correctly.")

# Initialize components
try:
    qa_system = QASystem(config_path)
    query_engine = QueryEngine(config_path)
    chatbot = Chatbot(config_path, qa_system.llm)
    components_initialized = True
except Exception as e:
    print(f"Error initializing components: {str(e)}")
    components_initialized = False

@app.route('/')
def home():
    # Check if environment variables are set and components are initialized
    env_check = {
        "github_token": os.environ.get("GITHUB_TOKEN") is not None,
        "azure_endpoint": os.environ.get("AZURE_INFERENCE_ENDPOINT") is not None,
        "chat_model": os.environ.get("AZURE_INFERENCE_CHAT_MODEL") is not None,
        "embedding_model": os.environ.get("AZURE_INFERENCE_EMBEDDING_MODEL") is not None,
        "components_initialized": components_initialized
    }
    return render_template('index.html', env_check=env_check)

@app.route('/index_documents', methods=['POST'])
def index_documents():
    try:
        # Run through run.py to ensure environment variables are loaded
        result = subprocess.run(
            [sys.executable, 'run.py', 'index'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return jsonify({
                "success": True, 
                "message": "Documents indexed successfully",
                "details": result.stdout
            })
        else:
            return jsonify({
                "success": False,
                "message": "Error indexing documents",
                "details": result.stderr
            })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/query', methods=['POST'])
def query_documents():
    query = request.form.get('query', '')
    if not query:
        return jsonify({"success": False, "message": "Query cannot be empty"})
    
    try:
        # Initialize QueryEngine each time to ensure we have updated settings
        query_engine = QueryEngine(config_path)
        results = query_engine.query_vector_store(query)
        formatted_results = []
        
        for i, (doc, score) in enumerate(results, 1):
            formatted_results.append({
                "id": i,
                "similarity": round(score, 4),
                "source": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 'Unknown'),
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            })
        
        return jsonify({
            "success": True,
            "query": query,
            "results": formatted_results
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/ask', methods=['POST'])
def ask_question():
    query = request.form.get('query', '')
    if not query:
        return jsonify({"success": False, "message": "Question cannot be empty"})
    
    try:
        # Use run.py to ensure environment variables are loaded
        result = subprocess.run(
            [sys.executable, 'run.py', 'ask', query],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # Parse the response from the command line output
            response_lines = result.stdout.strip().split("\n")
            answer = ""
            capture = False
            
            for line in response_lines:
                if line.startswith("Answer:"):
                    capture = True
                    answer = line[7:].strip()  # Remove "Answer: " prefix
                elif capture:
                    answer += " " + line.strip()
            
            return jsonify({
                "success": True,
                "question": query,
                "answer": answer if answer else "No answer found in the response."
            })
        else:
            return jsonify({
                "success": False,
                "message": "Error processing question",
                "details": result.stderr
            })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    message = request.form.get('message', '')
    if not message:
        return jsonify({"success": False, "message": "Message cannot be empty"})
    
    if message.lower() == 'reset':
        # Initialize with fresh components to reset
        try:
            global chatbot
            qa_system = QASystem(config_path)
            chatbot = Chatbot(config_path, qa_system.llm)
            session['chat_history'] = []
            return jsonify({
                "success": True,
                "response": "Conversation history has been reset."
            })
        except Exception as e:
            return jsonify({"success": False, "message": f"Error resetting chat: {str(e)}"})
    
    try:
        if not components_initialized:
            raise Exception("Components not properly initialized. Check environment variables.")
            
        response = chatbot.chat(message)
        session['chat_history'].append({"user": message, "assistant": response})
        return jsonify({
            "success": True,
            "response": response
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/chat_history')
def get_chat_history():
    return jsonify(session.get('chat_history', []))

@app.route('/check_env', methods=['GET'])
def check_environment():
    """Check if environment variables are properly set"""
    status = {
        "github_token": bool(os.environ.get("GITHUB_TOKEN")),
        "azure_endpoint": bool(os.environ.get("AZURE_INFERENCE_ENDPOINT")),
        "chat_model": bool(os.environ.get("AZURE_INFERENCE_CHAT_MODEL")),
        "embedding_model": bool(os.environ.get("AZURE_INFERENCE_EMBEDDING_MODEL")),
        "all_set": all([
            os.environ.get("GITHUB_TOKEN"),
            os.environ.get("AZURE_INFERENCE_ENDPOINT"),
            os.environ.get("AZURE_INFERENCE_CHAT_MODEL"),
            os.environ.get("AZURE_INFERENCE_EMBEDDING_MODEL")
        ])
    }
    return jsonify(status)

@app.route('/setup_env', methods=['GET'])
def setup_env():
    """Redirect to env setup page"""
    return render_template('setup_env.html')

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True)
