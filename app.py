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
import time  # Add missing import for timestamp generation

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

@app.route('/index')
def index_page():
    """Redirect to the index tab on the home page"""
    return redirect('/#index')

@app.route('/index_documents', methods=['POST'])
def index_documents():
    try:
        reset_db = request.form.get('reset_db', 'false').lower() == 'true'
        
        # Check if a file was uploaded
        if 'file' in request.files and request.files['file'].filename:
            uploaded_file = request.files['file']
            
            if not uploaded_file.filename:
                return jsonify({
                    "success": False,
                    "message": "No file selected"
                })
                
            # Ensure valid file extension
            _, file_extension = os.path.splitext(uploaded_file.filename.lower())
            if file_extension not in ['.pdf', '.txt', '.docx', '.doc', '.csv', '.md']:
                return jsonify({
                    "success": False,
                    "message": f"Unsupported file type: {file_extension}. Supported formats: PDF, TXT, DOCX, DOC, CSV, MD."
                })
            
            # Save the file to the data directory
            file_path = os.path.join(indexer.data_dir, uploaded_file.filename)
            uploaded_file.save(file_path)
            
            # Check if this is an upload-only request or upload-and-index request
            index_after_upload = request.form.get('index_after_upload', 'false').lower() == 'true'
            
            if not index_after_upload:
                # Return success without indexing
                return jsonify({
                    "success": True,
                    "message": f"File {uploaded_file.filename} uploaded successfully to data directory",
                    "indexed": False
                })
            
            # If we get here, the user wants to index the file after uploading
            # Create a new indexer instance to avoid file locking problems
            new_indexer = DocumentIndexer(config_path)
            
            # Process the single file
            docs = new_indexer.load_and_process_file(file_path)
            if docs and len(docs) > 0:
                # Add to vector store
                new_indexer.create_vector_store(docs)
                
                return jsonify({
                    "success": True,
                    "message": f"Successfully indexed file: {uploaded_file.filename}",
                    "details": f"Processed {len(docs)} chunks from {uploaded_file.filename}",
                    "indexed": True
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
                    answer = line[7:].trip()  # Remove "Answer: " prefix
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

@app.route('/delete_data_file', methods=['POST'])
def delete_data_file():
    """Delete a file from the data directory without affecting the vector store"""
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
        if not os.path.isfile(file_path):
            return jsonify({
                "success": False,
                "message": f"File '{filename}' not found in data directory"
            })
        
        # Try to remove the file
        try:
            os.remove(file_path)
            return jsonify({
                "success": True,
                "message": f"Successfully deleted file '{filename}' from data directory"
            })
        except PermissionError:
            return jsonify({
                "success": False,
                "message": f"Could not delete file '{filename}' because it's being used by another process"
            })
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Error deleting file: {str(e)}"
            })
            
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
        delete_from_data = data.get('delete_from_data', False)
        
        if not filename:
            return jsonify({
                "success": False,
                "message": "No filename provided"
            })
        
        # Check if the file exists in the data directory
        file_path = os.path.join(indexer.data_dir, filename)
        file_exists = os.path.isfile(file_path)
        file_deleted = False
        
        # Delete the file from data directory if requested
        if delete_from_data and file_exists:
            try:
                os.remove(file_path)
                file_deleted = True
            except Exception as e:
                return jsonify({
                    "success": False,
                    "message": f"Error deleting file from data directory: {str(e)}",
                    "details": traceback.format_exc()
                })
        
        # Delete documents from vector store without rebuilding
        try:
            # Create a new query engine instance to avoid stale connections
            fresh_query_engine = QueryEngine(config_path)
            
            # Create filter to find documents from this file
            filter_dict = {"source": filename}
            
            # Delete documents that match the filter
            deleted = fresh_query_engine.delete_documents(filter_dict)
            
            if deleted:
                return jsonify({
                    "success": True,
                    "message": f"Document '{filename}' was removed from the index",
                    "file_deleted": file_deleted,
                    "docs_deleted": deleted
                })
            else:
                return jsonify({
                    "success": False,
                    "message": f"No documents found for '{filename}' in the vector store",
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

@app.route('/list_data_files', methods=['GET'])
def list_data_files():
    """List all files in the data directory (before indexing)"""
    try:
        # Create a new indexer instance to get data directory path
        temp_indexer = DocumentIndexer(config_path)
        data_dir = temp_indexer.data_dir
        
        if not os.path.exists(data_dir):
            return jsonify({
                "success": False,
                "message": f"Data directory {data_dir} does not exist"
            })
            
        # Get all files in the data directory
        files = []
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path):
                # Get file size and last modified time
                file_stat = os.stat(file_path)
                size_kb = file_stat.st_size / 1024  # Convert to KB
                modified_time = file_stat.st_mtime
                
                # Get file extension
                _, file_extension = os.path.splitext(filename.lower())
                file_type = file_extension[1:] if file_extension.startswith('.') else file_extension
                
                # Determine if file is indexable
                indexable = file_extension.lower() in ['.pdf', '.txt', '.docx', '.doc', '.csv', '.md']
                
                files.append({
                    "name": filename,
                    "type": file_type,
                    "size_kb": round(size_kb, 2),
                    "modified": modified_time,
                    "indexable": indexable
                })
        
        return jsonify({
            "success": True,
            "data_dir": data_dir,
            "files": files
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e),
            "details": traceback.format_exc()
        })

@app.route('/empty_vector_db', methods=['POST'])
def empty_vector_db():
    """Empty the vector database completely"""
    try:
        # Create a new indexer instance to get vector DB path
        temp_indexer = DocumentIndexer(config_path)
        vector_db_dir = temp_indexer.db_dir
        
        # Always ensure we're using the exact name from config, with no suffixes
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            vector_db_dir = config_data.get('vector_db_directory', 'vector_db')
            
        # Make sure the path is absolute
        if not os.path.isabs(vector_db_dir):
            vector_db_dir = os.path.join(os.path.dirname(__file__), vector_db_dir)
        
        print(f"Using vector database directory: {vector_db_dir}")
        
        if not os.path.exists(vector_db_dir):
            return jsonify({
                "success": True,
                "message": "Vector database does not exist or is already empty."
            })
        
        try:
            # Release all global references to vector store
            global qa_system, query_engine, chatbot, indexer
            
            print("Releasing all references to the database...")
            # Set to None to release connections
            qa_system = None
            query_engine = None
            chatbot = None
            indexer = None
            
            # Force garbage collection to release file handles
            import gc
            gc.collect()
            
            # Wait a moment for resources to be released
            time.sleep(2)
            
            # Delete the vector DB directory completely
            print(f"Deleting vector database directory: {vector_db_dir}")
            
            # Delete directory and all its contents
            if os.path.exists(vector_db_dir):
                # First approach: use shutil.rmtree with error handler
                try:
                    def onerror(func, path, exc_info):
                        """Error handler for shutil.rmtree that attempts to handle permission errors"""
                        import stat
                        os.chmod(path, stat.S_IWRITE)
                        func(path)
                    
                    shutil.rmtree(vector_db_dir, onerror=onerror)
                except Exception as e:
                    print(f"Standard deletion failed: {e}")
                    
                    # Second approach: use system commands
                    try:
                        if sys.platform == 'win32':
                            os.system(f'rd /s /q "{vector_db_dir}"')
                        else:
                            os.system(f'rm -rf "{vector_db_dir}"')
                    except Exception as e2:
                        print(f"System command failed: {e2}")
            
            # Create the directory with original name from config
            os.makedirs(vector_db_dir, exist_ok=True)
            
            # Let Python release any handles before proceeding
            gc.collect()
            time.sleep(1)
            
            # Create an empty Chroma database
            print("Initializing empty vector database...")
            
            # Initialize with a HuggingFace embedding model for compatibility
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={"device": "cpu"}
            )
            
            # Create a new empty Chroma instance
            try:
                # Try to import the newer package first
                from langchain_chroma import Chroma
                Chroma(persist_directory=vector_db_dir, embedding_function=embedding_model)
            except ImportError:
                # Fall back to community version
                from langchain_community.vectorstores import Chroma
                Chroma(persist_directory=vector_db_dir, embedding_function=embedding_model)
            
            # Reinitialize components
            print("Reinitializing components...")
            try:
                qa_system = QASystem(config_path)
                query_engine = QueryEngine(config_path)
                chatbot = Chatbot(config_path, qa_system.llm)
                indexer = DocumentIndexer(config_path)
            except Exception as reinit_err:
                print(f"Warning: Error reinitializing components: {reinit_err}")
            
            return jsonify({
                "success": True,
                "message": "Vector database has been completely emptied."
            })
            
        except Exception as e:
            # Attempt to reinitialize components even if there was an error
            try:
                print(f"Error during vector DB emptying: {e}")
                qa_system = QASystem(config_path)
                query_engine = QueryEngine(config_path)
                chatbot = Chatbot(config_path, qa_system.llm)
                indexer = DocumentIndexer(config_path)
            except Exception as reinit_error:
                print(f"Error reinitializing components: {reinit_error}")
                
            return jsonify({
                "success": False,
                "message": f"Error emptying vector database: {str(e)}",
                "details": traceback.format_exc()
            })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}",
            "details": traceback.format_exc()
        })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True)
