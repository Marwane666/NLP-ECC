from flask import Flask, render_template, request, jsonify, session
import os
from src.document_indexer import DocumentIndexer
from src.query_engine import QueryEngine
from src.qa_system import QASystem
from src.chatbot import Chatbot

app = Flask(__name__)
app.secret_key = "rag_system_secret_key"  # For session management
config_path = 'config.yaml'

# Initialize components
qa_system = QASystem(config_path)
query_engine = QueryEngine(config_path)
chatbot = Chatbot(config_path, qa_system.llm)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index_documents', methods=['POST'])
def index_documents():
    try:
        indexer = DocumentIndexer(config_path)
        indexer.index()
        return jsonify({"success": True, "message": "Documents indexed successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/query', methods=['POST'])
def query_documents():
    query = request.form.get('query', '')
    if not query:
        return jsonify({"success": False, "message": "Query cannot be empty"})
    
    try:
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
        result = qa_system.generate_response(query)
        return jsonify({
            "success": True,
            "question": query,
            "answer": result['response']
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
        chatbot.reset()
        session['chat_history'] = []
        return jsonify({
            "success": True,
            "response": "Conversation history has been reset."
        })
    
    try:
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

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True)
