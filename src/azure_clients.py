"""
Azure AI Inference SDK client integrations for the RAG system.
"""
import os
from typing import List, Dict, Any, Optional, Union, ClassVar
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
import numpy as np
import warnings
import sys
from pydantic import root_validator, Field, PrivateAttr

# Try to import Azure packages - provide meaningful warnings if not available
try:
    from azure.ai.inference import EmbeddingsClient, ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential
    AZURE_INFERENCE_AVAILABLE = True
except ImportError:
    AZURE_INFERENCE_AVAILABLE = False
    warnings.warn(
        "Azure AI Inference SDK is not installed. "
        "Run 'pip install azure-ai-inference' to use Azure AI Inference features."
    )

try:
    from langchain_openai import AzureChatOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False
    warnings.warn(
        "LangChain OpenAI package is not installed. "
        "Run 'pip install langchain-openai' to use Azure OpenAI features."
    )

# Only define AzureInferenceEmbeddings if the dependencies are available
if AZURE_INFERENCE_AVAILABLE:
    class AzureInferenceEmbeddings(Embeddings):
        """
        Azure AI Inference SDK Embeddings wrapper for LangChain compatibility.
        """
        
        def __init__(self, 
                    endpoint: str = None, 
                    model_name: str = None,
                    api_key: str = None):
            """
            Initialize the Azure AI Inference embeddings client.
            
            Args:
                endpoint: Azure AI Inference endpoint
                model_name: Model name for embeddings
                api_key: API key (taken from GITHUB_TOKEN environment variable if not provided)
            """
            # Try to get values from environment if not provided
            self.endpoint = endpoint or os.environ.get("AZURE_INFERENCE_ENDPOINT")
            self.model_name = model_name or os.environ.get("AZURE_INFERENCE_EMBEDDING_MODEL")
            self.api_key = api_key or os.environ.get("GITHUB_TOKEN")
            
            # Detailed validation with helpful error messages
            if not self.endpoint:
                raise ValueError(
                    "Azure AI Inference endpoint is required. Please set it via:\n"
                    "1. The 'endpoint' parameter in config.yaml, or\n"
                    "2. The AZURE_INFERENCE_ENDPOINT environment variable\n"
                    "You can set environment variables by running 'python example_env_setup.py'"
                )
            
            if not self.model_name:
                raise ValueError(
                    "Model name is required. Please set it via:\n"
                    "1. The 'model_name' parameter in config.yaml, or\n"
                    "2. The AZURE_INFERENCE_EMBEDDING_MODEL environment variable\n"
                    "You can set environment variables by running 'python example_env_setup.py'"
                )
            
            if not self.api_key:
                raise ValueError(
                    "API key (GITHUB_TOKEN) is required. Please set it via:\n"
                    "1. The GITHUB_TOKEN environment variable\n"
                    "You can set environment variables by running 'python example_env_setup.py'\n\n"
                    "Note: The GitHub token should have appropriate permissions to access the Azure AI Inference API."
                )
            
            try:
                # Initialize the client
                self.client = EmbeddingsClient(
                    endpoint=self.endpoint,
                    credential=AzureKeyCredential(self.api_key)
                )
                # Test connection
                # self._test_connection()
            except Exception as e:
                raise ConnectionError(f"Failed to initialize Azure AI Inference client: {str(e)}")
            
        def embed_query(self, text: str) -> List[float]:
            """Get embedding for a single query text."""
            embeddings = self._get_embeddings_batch([text])
            return embeddings[0]
        
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """Get embeddings for a batch of documents."""
            batch_size = 16  # Adjust based on your requirements
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_embeddings = self._get_embeddings_batch(batch)
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
            """Helper method to get embeddings for a batch of texts."""
            try:
                # Make the API request
                response = self.client.embed(
                    input=texts,
                    model=self.model_name
                )
                
                # Extract embeddings from the response
                embeddings = [data.embedding for data in response.data]
                
                return embeddings
            except Exception as e:
                print(f"Error getting embeddings: {e}")
                # Return simple embeddings as a fallback
                return [[0.0] * 768 for _ in range(len(texts))]
else:
    # Placeholder class if Azure AI Inference is not available
    class AzureInferenceEmbeddings:
        def __init__(self, **kwargs):
            raise ImportError("Azure AI Inference SDK is not installed. Run 'pip install azure-ai-inference' to use this feature.")

# Only define AzureInferenceChatLLM if the dependencies are available
if AZURE_INFERENCE_AVAILABLE:
    class AzureInferenceChatLLM(LLM):
        """
        Azure AI Inference SDK Chat LLM wrapper for LangChain compatibility.
        """
        
        # Use Field() for Pydantic fields or ClassVar for class variables
        endpoint: str = Field(default=None, description="Azure AI Inference endpoint")
        model_name: str = Field(default=None, description="Model name for chat completions")
        temperature: float = Field(default=0.7, description="Temperature parameter")
        max_tokens: int = Field(default=1000, description="Maximum tokens to generate")
        top_p: float = Field(default=1.0, description="Top-p sampling parameter")
        system_message: str = Field(
            default="You are a helpful assistant that provides accurate, factual information.",
            description="System message for the assistant"
        )
        
        # Use PrivateAttr for non-Pydantic attributes 
        _api_key: str = PrivateAttr(default=None)
        _client: Any = PrivateAttr(default=None)
        
        def __init__(self, 
                    endpoint: Optional[str] = None, 
                    model_name: Optional[str] = None,
                    api_key: Optional[str] = None,
                    temperature: float = 0.7,
                    max_tokens: int = 1000,
                    top_p: float = 1.0,
                    system_message: str = "You are a helpful assistant that provides accurate, factual information."):
            """Initialize the Azure AI Inference chat client."""
            # Set attributes that should be part of the model
            base_attrs = {
                "endpoint": endpoint or os.environ.get("AZURE_INFERENCE_ENDPOINT"),
                "model_name": model_name or os.environ.get("AZURE_INFERENCE_CHAT_MODEL"),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "system_message": system_message,
            }
            
            # Initialize the base class
            super().__init__(**base_attrs)
            
            # Set private attributes
            self._api_key = api_key or os.environ.get("GITHUB_TOKEN")
            
            # Validate required parameters
            if not self.endpoint:
                raise ValueError("Azure AI Inference endpoint is required. Check environment variable AZURE_INFERENCE_ENDPOINT.")
            if not self.model_name:
                raise ValueError("Model name is required. Check environment variable AZURE_INFERENCE_CHAT_MODEL.")
            if not self._api_key:
                raise ValueError("API key (GITHUB_TOKEN) is required. Check environment variable GITHUB_TOKEN.")
            
            # Initialize the client outside of Pydantic
            self._init_client()
        
        def _init_client(self):
            """Initialize the Azure AI Inference chat client."""
            try:
                self._client = ChatCompletionsClient(
                    endpoint=self.endpoint,
                    credential=AzureKeyCredential(self._api_key)
                )
                print(f"Successfully initialized Azure AI Inference Chat client with model: {self.model_name}")
            except Exception as e:
                print(f"Error initializing Azure AI Inference Chat client: {str(e)}")
                raise
        
        def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
            """Call the chat model to generate a response."""
            try:
                # Print debug information
                print(f"Calling Azure AI Inference with prompt length: {len(prompt)}")
                
                # Prepare messages
                messages = [
                    SystemMessage(self.system_message),
                    UserMessage(prompt)
                ]
                
                # Make the API request
                response = self._client.complete(
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    model=self.model_name
                )
                
                # Extract the response content
                content = response.choices[0].message.content
                
                # Handle stop strings if provided
                if stop:
                    for stop_str in stop:
                        if stop_str in content:
                            content = content[:content.index(stop_str)]
                            
                print(f"Generated response of length: {len(content)}")
                return content
                
            except Exception as e:
                print(f"Error in Azure AI Inference call: {str(e)}")
                # Return a helpful error message instead of failing
                return f"I encountered an error: {str(e)}. Please try again."
        
        @property
        def _llm_type(self) -> str:
            """Return the type of LLM."""
            return "azure_inference_chat"
        
        @property
        def _identifying_params(self) -> Dict[str, Any]:
            """Return identifying parameters."""
            return {
                "endpoint": self.endpoint,
                "model_name": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p
            }
else:
    # Placeholder class if Azure AI Inference is not available
    class AzureInferenceChatLLM:
        def __init__(self, **kwargs):
            raise ImportError("Azure AI Inference SDK is not installed. Run 'pip install azure-ai-inference' to use this feature.")

# Only define AzureOpenAIEmbeddings if the dependencies are available
if AZURE_OPENAI_AVAILABLE:
    class AzureOpenAIEmbeddings(Embeddings):
        """
        Azure OpenAI Embeddings wrapper for LangChain compatibility.
        """
        
        def __init__(self, 
                    azure_endpoint: str = None, 
                    azure_deployment: str = None,
                    api_version: str = "2023-05-15",
                    api_key: str = None):
            """
            Initialize the Azure OpenAI embeddings client.
            
            Args:
                azure_endpoint: Azure OpenAI endpoint
                azure_deployment: Deployment name for embeddings
                api_version: API version
                api_key: API key (taken from AZURE_OPENAI_API_KEY environment variable if not provided)
            """
            # Try to get values from environment if not provided
            self.azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
            self.azure_deployment = azure_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
            self.api_version = api_version
            self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
            
            # Validate required parameters
            if not self.azure_endpoint:
                raise ValueError("Azure OpenAI endpoint is required")
            if not self.azure_deployment:
                raise ValueError("Deployment name is required")
            if not self.api_key:
                raise ValueError("API key (AZURE_OPENAI_API_KEY) is required")
            
            # Initialize the client
            self.client = AzureChatOpenAI(
                azure_endpoint=self.azure_endpoint,
                azure_deployment=self.azure_deployment,
                openai_api_version=self.api_version,
                api_key=self.api_key
            )
            
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """
            Get embeddings for a batch of documents.
            
            Args:
                texts: List of text strings to embed
                
            Returns:
                List of embedding vectors
            """
            # Azure OpenAI embeddings has limits on batch size, process in smaller batches if needed
            batch_size = 16  # Adjust based on your requirements
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_embeddings = self._get_embeddings_batch(batch)
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        def embed_query(self, text: str) -> List[float]:
            """
            Get embedding for a single query text.
            
            Args:
                text: Text string to embed
                
            Returns:
                Embedding vector
            """
            embeddings = self._get_embeddings_batch([text])
            return embeddings[0]
        
        def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
            """
            Helper method to get embeddings for a batch of texts.
            
            Args:
                texts: Batch of texts to embed
                
            Returns:
                List of embedding vectors
            """
            # Make the API request
            response = self.client.embed(
                input=texts,
                model=self.azure_deployment
            )
            
            # Extract embeddings from the response
            embeddings = [data.embedding for data in response.data]
            
            return embeddings
else:
    # Placeholder class if Azure OpenAI is not available
    class AzureOpenAIEmbeddings:
        def __init__(self, **kwargs):
            raise ImportError("LangChain OpenAI package is not installed. Run 'pip install langchain-openai' to use this feature.")

def get_azure_inference_chat_llm(config: Dict[str, Any]):
    """
    Create an AzureInferenceChatLLM instance based on the provided configuration.
    
    Args:
        config: Configuration dictionary with LLM settings
        
    Returns:
        Configured AzureInferenceChatLLM instance
    """
    if not AZURE_INFERENCE_AVAILABLE:
        raise ImportError("Azure AI Inference SDK is not installed. Run 'pip install azure-ai-inference' to use this feature.")
    
    # Get configuration values
    endpoint = config.get("endpoint") or os.environ.get("AZURE_INFERENCE_ENDPOINT")
    model_name = config.get("model_name") or os.environ.get("AZURE_INFERENCE_CHAT_MODEL")
    api_key = os.environ.get("GITHUB_TOKEN")
    
    # Print debug info
    print(f"Initializing Azure AI Inference Chat LLM:")
    print(f"Endpoint: {endpoint}")
    print(f"Model: {model_name}")
    print(f"API key present: {'Yes' if api_key else 'No'}")
    
    # Create and return the AzureInferenceChatLLM instance
    return AzureInferenceChatLLM(
        endpoint=endpoint,
        model_name=model_name,
        api_key=api_key,
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 1000),
        top_p=config.get("top_p", 1.0)
    )

def get_azure_chat_openai(config: Dict[str, Any]):
    """
    Create an AzureChatOpenAI instance based on the provided configuration.
    
    Args:
        config: Configuration dictionary with LLM settings
        
    Returns:
        Configured AzureChatOpenAI instance
    """
    if not AZURE_OPENAI_AVAILABLE:
        raise ImportError("LangChain OpenAI package is not installed. Run 'pip install langchain-openai' to use this feature.")
    
    # Get configuration values, with fallback to environment variables
    azure_endpoint = config.get("azure_endpoint") or os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_deployment = config.get("azure_deployment") or os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = config.get("api_version") or os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    
    # Create and return the AzureChatOpenAI instance
    return AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        openai_api_version=api_version,
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 512)
    )
