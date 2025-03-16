from langchain.prompts import PromptTemplate

def get_qa_prompt() -> PromptTemplate:
    """
    Get the prompt template for the question-answering system.
    
    Returns:
        PromptTemplate configured for RAG
    """
    template = """
    You are a helpful AI assistant that answers questions based on provided context.
    
    Context:
    {context}
    
    Question: {query}
    
    Instructions:
    1. Answer the question based ONLY on the provided context.
    2. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question."
    3. Keep your answer concise, clear, and directly addressing the question.
    4. Do not make up information or use knowledge outside of the provided context.
    5. If the question asks for an opinion, provide an objective analysis based on the context.
    
    Answer:
    """
    
    return PromptTemplate(
        input_variables=["context", "query"],
        template=template
    )

def get_chat_prompt() -> PromptTemplate:
    """
    Get the prompt template for the chatbot system.
    
    Returns:
        PromptTemplate configured for conversational RAG
    """
    template = """
    You are a helpful AI assistant that engages in a conversation with the user.
    You answer questions based on provided context and maintain a coherent conversation.
    
    Previous conversation:
    {history}
    
    Context for the current question:
    {context}
    
    Current question: {query}
    
    Instructions:
    1. Answer the question based primarily on the provided context.
    2. If the context doesn't contain enough information, but the conversation history does, use that information.
    3. If neither the context nor the history contains enough information, say "I don't have enough information to answer this question."
    4. Keep your answer conversational but concise.
    5. Do not make up information or use knowledge outside of the provided context and conversation history.
    
    Answer:
    """
    
    return PromptTemplate(
        input_variables=["history", "context", "query"],
        template=template
    )
