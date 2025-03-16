from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import sys
import subprocess
import yaml
from load_env import load_env_file
from src.document_indexer import DocumentIndexer
from src.query_engine import QueryEngine
from src.qa_system import QASystem
from src.chatbot import Chatbot
import traceback
import shutil

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
    indexer = DocumentIndexer(config_path)
    components_initialized = True
except Exception as e:
    print(f"Error initializing components: {str(e)}")
    traceback.print_exc()
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
    
    # Get document processing settings
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        file_processors = config.get('file_processors', {})
        parallel_config = config.get('parallel_processing', {})
    except Exception:
        file_processors = {}
        parallel_config = {}
    
    return render_template('index.html', 
                          env_check=env_check, 
                          file_processors=file_processors,
                          parallel_config=parallel_config)

@app.route('/index_documents', methods=['POST'])
def index_documents():
    try:
        reset_db = request.form.get('reset_db', 'false').lower() == 'true'
        
        # Check if a file was uploaded
        if 'file' in request.files and request.files['file'].filename:
            uploaded_file = request.files['file']
            
            # Save the file to the data directory
            file_path = os.path.join(indexer.data_dir, uploaded_file.filename)
            uploaded_file.save(file_path)
            
            # Process the single file
            docs = indexer.load_and_process_file(file_path)
            if docs:
                indexer.create_vector_store(docs)
                return jsonify({
                    "success": True,
                    "message": f"Successfully indexed file: {uploaded_file.filename}",
                    "details": f"Processed {len(docs)} chunks from {uploaded_file.filename}"
                })
            else:
                return jsonify({
                    "success": False,
                    "message": f"No documents extracted from {uploaded_file.filename}",
                    "details": "The file might be empty or in an unsupported format"
                })
        else:
            # Run through run.py to ensure environment variables are loaded
            # Pass reset_db flag if needed
            cmd = [sys.executable, 'run.py', 'index']
            if reset_db:
                cmd.append('--reset-db')
                
            result = subprocess.run(
                cmd,
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
        return jsonify({
            "success": False, 
            "message": str(e),
            "details": traceback.format_exc()
        })

@app.route('/list_documents', methods=['GET'])
def list_documents():
    try:
        # Run the list command through the CLI
        result = subprocess.run(
            [sys.executable, 'run.py', 'list'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # Parse the output to extract file information
            output_lines = result.stdout.strip().split("\n")
            files_info = []
            
            # Skip the header lines
            started = False
            for line in output_lines:
                if line.startswith("===== Indexed Files ====="):
                    started = True
                    continue
                
                if started and line.startswith("-"):
                    # Parse line like: "- example.pdf (pdf): 25 chunks"
                    parts = line.strip("- ").split(":")
                    file_info = parts[0].strip()
                    chunks = parts[1].strip().split(" ")[0]
                    
                    file_name, file_type = file_info.rsplit(" (", 1)
                    file_type = file_type.rstrip(")")
                    
                    files_info.append({
                        "name": file_name,
                        "type": file_type,
                        "chunks": chunks
                    })
            
            return jsonify({
                "success": True,
                "files": files_info
            })
        else:
            return jsonify({
                "success": False,
                "message": "Error listing documents",
                "details": result.stderr
            })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e),
            "details": traceback.format_exc()
        })

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
                "file_type": doc.metadata.get('file_type', 'Unknown'),
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            })
        
        return jsonify({
            "success": True,
            "query": query,
            "results": formatted_results
        })
    except Exception as e:
        return jsonify({
            "success": False, 
            "message": str(e),
            "details": traceback.format_exc()
        })

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
        return jsonify({
            "success": False, 
            "message": str(e),
            "details": traceback.format_exc()
        })

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
            return jsonify({
                "success": False, 
                "message": f"Error resetting chat: {str(e)}",
                "details": traceback.format_exc()
            })
    
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
        return jsonify({
            "success": False, 
            "message": str(e),
            "details": traceback.format_exc()
        })

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

@app.route('/config', methods=['GET'])
def show_config():
    """Show current configuration in the browser"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return render_template('config.html', config=config)
    except Exception as e:
        return jsonify({
            "success": False, 
            "message": str(e),
            "details": traceback.format_exc()
        })

@app.route('/delete_document', methods=['POST'])
def delete_document():
    try:
        data = request.json
        filename = data.get('filename')
        
        if not filename:
            return jsonify({
                "success": False,
                "message": "No filename provided"
            })
        
        # Check if the file exists in the data directory
        file_path = os.path.join(indexer.data_dir, filename)
        file_exists = os.path.isfile(file_path)
        
        # Try to remove the file
        if file_exists:
            os.remove(file_path)
            file_deleted = True
        else:
            file_deleted = False
        
        # Force a re-index without the deleted file
        # We need to rebuild the vector DB to completely remove the file
        try:
            # Create a temporary indexer with reset_db=False to avoid clearing everything
            temp_indexer = DocumentIndexer(config_path)
            # Process all remaining files
            docs = temp_indexer.load_documents()
            
            # If there are documents, recreate the vector store
            if docs:
                # First remove the old vector store
                shutil.rmtree(temp_indexer.db_dir)
                os.makedirs(temp_indexer.db_dir, exist_ok=True)
                
                # Then create a new one with the remaining documents
                temp_indexer.create_vector_store(docs)
                
                return jsonify({
                    "success": True,
                    "message": f"Document '{filename}' was deleted and index was updated",
                    "file_deleted": file_deleted
                })
            else:
                # No documents left, just clear the vector store
                shutil.rmtree(temp_indexer.db_dir)
                os.makedirs(temp_indexer.db_dir, exist_ok=True)
                
                return jsonify({
                    "success": True,
                    "message": f"Document '{filename}' was deleted. No documents left in the index.",
                    "file_deleted": file_deleted
                })
                
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Error updating index: {str(e)}",
                "details": traceback.format_exc(),
                "file_deleted": file_deleted
            })
            
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e),
            "details": traceback.format_exc()
        })

@app.route('/update_config', methods=['POST'])
def update_config():
    """Update the configuration file with new settings"""
    try:
        # Get the new configuration from the request
        new_config = request.json
        
        if not new_config:
            return jsonify({
                "success": False,
                "message": "No configuration data provided"
            })
        
        # Create a backup of the current config file
        backup_path = config_path + '.backup'
        try:
            shutil.copy2(config_path, backup_path)
        except Exception as e:
            print(f"Warning: Could not create config backup: {e}")
        
        # Write the new configuration to the file
        with open(config_path, 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
        
        # Reinitialize components with the new configuration
        try:
            global qa_system, query_engine, chatbot, indexer, components_initialized
            
            # Reset components using new config
            qa_system = QASystem(config_path)
            query_engine = QueryEngine(config_path)
            chatbot = Chatbot(config_path, qa_system.llm)
            indexer = DocumentIndexer(config_path)
            components_initialized = True
            
            return jsonify({
                "success": True,
                "message": "Configuration updated successfully"
            })
        except Exception as e:
            # If there's an error with the new config, restore the backup
            print(f"Error reinitializing components with new config: {e}")
            
            try:
                if os.path.exists(backup_path):
                    shutil.copy2(backup_path, config_path)
                    # Attempt to reinitialize with the old config
                    qa_system = QASystem(config_path)
                    query_engine = QueryEngine(config_path)
                    chatbot = Chatbot(config_path, qa_system.llm)
                    indexer = DocumentIndexer(config_path)
            except Exception as restore_error:
                print(f"Error restoring config: {restore_error}")
            
            return jsonify({
                "success": False,
                "message": f"Error applying configuration: {str(e)}",
                "details": traceback.format_exc()
            })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error saving configuration: {str(e)}",
            "details": traceback.format_exc()
        })

@app.route('/restore_config_defaults', methods=['POST'])
def restore_config_defaults():
    """Restore the configuration file to default settings"""
    try:
        # Path to default config file
        default_config_path = os.path.join(os.path.dirname(__file__), 'default_config.yaml')
        
        # If default config doesn't exist, create it with current settings
        if not os.path.exists(default_config_path):
            # Define default configuration
            default_config = {
                "data_directory": "data",
                "vector_db_directory": "vector_db",
                "embedding_model": {
                    "type": "huggingface",
                    "name": "sentence-transformers/all-mpnet-base-v2",
                    "kwargs": {
                        "device": "cpu"
                    }
                },
                "text_splitter": {
                    "chunk_size": 1000,
                    "chunk_overlap": 200
                },
                "file_processors": {
                    "pdf": {
                        "processor": "chunk_semantic",
                        "text_splitter": {
                            "chunk_size": 1000,
                            "chunk_overlap": 200
                        }
                    },
                    "docx": {
                        "processor": "chunk_semantic",
                        "text_splitter": {
                            "chunk_size": 800,
                            "chunk_overlap": 150
                        }
                    },
                    "txt": {
                        "processor": "chunk_semantic",
                        "text_splitter": {
                            "chunk_size": 1200,
                            "chunk_overlap": 250
                        }
                    },
                    "csv": {
                        "processor": "structured",
                        "separator": ","
                    },
                    "md": {
                        "processor": "chunk_semantic",
                        "text_splitter": {
                            "chunk_size": 500,
                            "chunk_overlap": 100
                        }
                    }
                },
                "parallel_processing": {
                    "enabled": True,
                    "max_workers": 4
                },
                "retrieval": {
                    "top_k": 5
                },
                "llm": {
                    "type": "azure_inference",
                    "endpoint": "https://models.inference.ai.azure.com",
                    "model_name": "gpt-4o",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "top_p": 1.0,
                    "system_message": "You are a helpful assistant that provides accurate, factual information based on the provided context."
                },
                "max_chat_history": 10
            }
            
            # Write default config to a file
            with open(default_config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
        
        # Make a backup of the current config
        backup_path = config_path + '.user'
        try:
            shutil.copy2(config_path, backup_path)
        except Exception as e:
            print(f"Warning: Could not create user config backup: {e}")
        
        # Copy default config to main config
        shutil.copy2(default_config_path, config_path)
        
        # Reinitialize components with the default configuration
        try:
            global qa_system, query_engine, chatbot, indexer, components_initialized
            
            # Reset components using new config
            qa_system = QASystem(config_path)
            query_engine = QueryEngine(config_path)
            chatbot = Chatbot(config_path, qa_system.llm)
            indexer = DocumentIndexer(config_path)
            components_initialized = True
            
            return jsonify({
                "success": True,
                "message": "Configuration restored to defaults"
            })
        except Exception as e:
            # If there's an error with the default config, restore the user backup
            print(f"Error reinitializing components with default config: {e}")
            
            try:
                if os.path.exists(backup_path):
                    shutil.copy2(backup_path, config_path)
                    # Attempt to reinitialize with the old config
                    qa_system = QASystem(config_path)
                    query_engine = QueryEngine(config_path)
                    chatbot = Chatbot(config_path, qa_system.llm)
                    indexer = DocumentIndexer(config_path)
            except Exception as restore_error:
                print(f"Error restoring user config: {restore_error}")
            
            return jsonify({
                "success": False,
                "message": f"Error applying default configuration: {str(e)}",
                "details": traceback.format_exc()
            })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error restoring default configuration: {str(e)}",
            "details": traceback.format_exc()
        })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True)
