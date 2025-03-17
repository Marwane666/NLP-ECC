"""
Enhanced Environment validation utility for the RAG system.
This script checks if all required environment variables are set correctly and logs the results.
"""
import os
import sys
import logging
import argparse
import requests
from colorama import init, Fore, Style
from azure.core.credentials import AzureKeyCredential

# Initialize colorama for colored terminal output with auto-reset
init(autoreset=True)

# Configure logging
logging.basicConfig(filename="environment_check.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def log_and_print(message, level="info"):
    """Log and print messages simultaneously."""
    print(message)
    if level == "info":
        logging.info(message)
    elif level == "error":
        logging.error(message)
    elif level == "warning":
        logging.warning(message)


def print_success(message):
    """Print a success message in green."""
    log_and_print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")


def print_error(message):
    """Print an error message in red."""
    log_and_print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}", "error")


def print_warning(message):
    """Print a warning message in yellow."""
    log_and_print(f"{Fore.YELLOW}! {message}{Style.RESET_ALL}", "warning")


def print_info(message):
    """Print an info message in cyan."""
    log_and_print(f"{Fore.CYAN}ℹ {message}{Style.RESET_ALL}")


def check_python_version():
    """Ensure Python version is at least 3.8."""
    if sys.version_info < (3, 8):
        print_error("Python 3.8 or higher is required. Please update your Python version.")
        sys.exit(1)


def check_variable(name, required=True):
    """
    Check if an environment variable is set.

    Args:
        name: Name of the environment variable
        required: Whether the variable is required

    Returns:
        True if the variable is set, False otherwise
    """
    value = os.environ.get(name)
    if value:
        masked_value = value[:4] + "****" + value[-4:] if "TOKEN" in name or "KEY" in name else value
        print_success(f"{name} is set: {masked_value}")
        return True
    else:
        if required:
            print_error(f"{name} is not set but is required")
        else:
            print_warning(f"{name} is not set (optional)")
        return False


def test_azure_inference_connection():
    """
    Test the connection to Azure AI Inference API.

    Returns:
        True if the connection is successful, False otherwise
    """
    token = os.environ.get("GITHUB_TOKEN")
    endpoint = os.environ.get("AZURE_INFERENCE_ENDPOINT")
    model = os.environ.get("AZURE_INFERENCE_EMBEDDING_MODEL")

    if not (token and endpoint and model):
        print_error("Cannot test connection: required environment variables are missing")
        return False

    try:
        from azure.ai.inference import EmbeddingsClient

        client = EmbeddingsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(token)
        )

        response = client.embed(
            input=["Test connection"],
            model=model,
            timeout=5  # Add timeout to prevent hanging
        )

        if response and response.data and len(response.data) > 0:
            print_success("Successfully connected to Azure AI Inference API")
            print_info(f"Embedding length: {len(response.data[0].embedding)}")
            return True
        else:
            print_error("API response was empty or invalid")
            return False

    except requests.exceptions.Timeout:
        print_error("Connection to Azure AI Inference API timed out")
    except requests.exceptions.RequestException as e:
        print_error(f"Network error: {str(e)}")
    except ImportError:
        print_error("Azure AI Inference SDK is not installed")
        print_info("Run 'pip install azure-ai-inference' to install it")
    except Exception as e:
        print_error(f"Failed to connect to Azure AI Inference API: {str(e)}")

    return False


def check_environment():
    """Check the environment setup for the RAG system."""
    print("\n===== RAG System Environment Check =====\n")

    # Check required environment variables
    github_token_set = check_variable("GITHUB_TOKEN")
    endpoint_set = check_variable("AZURE_INFERENCE_ENDPOINT")
    chat_model_set = check_variable("AZURE_INFERENCE_CHAT_MODEL")
    embedding_model_set = check_variable("AZURE_INFERENCE_EMBEDDING_MODEL")

    # Check optional environment variables
    check_variable("AZURE_OPENAI_API_KEY", required=False)
    check_variable("AZURE_OPENAI_ENDPOINT", required=False)
    check_variable("AZURE_OPENAI_DEPLOYMENT_NAME", required=False)

    # Check installed Python packages
    try:
        import azure.ai.inference
        print_success("Azure AI Inference SDK is installed")
    except ImportError:
        print_error("Azure AI Inference SDK is not installed")

    try:
        import langchain
        print_success(f"LangChain is installed (version: {langchain.__version__})")
    except (ImportError, AttributeError):
        print_error("LangChain is not installed or version info not available")

    # Test connection if all required variables are set
    if github_token_set and endpoint_set and embedding_model_set:
        print("\n===== Testing Azure AI Inference Connection =====\n")
        test_azure_inference_connection()

    print("\n===== Environment Check Complete =====\n")


if __name__ == "__main__":
    check_python_version()
    parser = argparse.ArgumentParser(description="Check environment setup for the RAG system.")
    parser.add_argument("--azure", action="store_true", help="Check Azure AI Inference connection")
    parser.add_argument("--env", action="store_true", help="Check required environment variables")
    args = parser.parse_args()

    if args.env:
        check_environment()
    if args.azure:
        test_azure_inference_connection()
