"""
Debug utility for Azure AI Inference integration.
This script checks the connection to Azure AI Inference and tests basic functionality.
"""
import os
import sys
from load_env import load_env_file
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

def test_azure_inference():
    """Test the Azure AI Inference connection and API."""
    # First load environment variables
    env_vars = load_env_file()
    
    token = os.environ.get("GITHUB_TOKEN")
    endpoint = os.environ.get("AZURE_INFERENCE_ENDPOINT")
    model = os.environ.get("AZURE_INFERENCE_CHAT_MODEL")
    
    if not token:
        print("ERROR: GITHUB_TOKEN environment variable is not set.")
        sys.exit(1)
    
    if not endpoint:
        print("ERROR: AZURE_INFERENCE_ENDPOINT environment variable is not set.")
        sys.exit(1)
    
    if not model:
        print("ERROR: AZURE_INFERENCE_CHAT_MODEL environment variable is not set.")
        sys.exit(1)
    
    print("\n===== Azure AI Inference Configuration =====")
    print(f"Endpoint: {endpoint}")
    print(f"Model: {model}")
    print(f"Token is set: {'Yes' if token else 'No'}")
    
    try:
        print("\n===== Testing Azure AI Inference Chat API =====")
        
        # Initialize the client
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(token)
        )
        print("Client initialized successfully.")
        
        # Simple message
        messages = [
            SystemMessage("You are a helpful assistant."),
            UserMessage("Hello, how are you?")
        ]
        
        print("Sending test message to API...")
        response = client.complete(
            messages=messages,
            temperature=0.7,
            top_p=1.0,
            max_tokens=100,
            model=model
        )
        
        # Print the response
        print("\n===== API Response =====")
        print(response.choices[0].message.content)
        
        print("\nAPI TEST SUCCESSFUL!")
        return True
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nAPI TEST FAILED.")
        return False

if __name__ == "__main__":
    success = test_azure_inference()
    sys.exit(0 if success else 1)
