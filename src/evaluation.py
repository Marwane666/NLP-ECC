import yaml
import warnings
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge

from src.qa_system import QASystem

# Use updated imports to fix deprecation warnings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = False
    warnings.warn(
        "Using deprecated HuggingFaceEmbeddings from langchain_community. "
        "Install langchain-huggingface for the updated version."
    )

# Conditionally import Azure clients
try:
    from src.azure_clients import AzureOpenAIEmbeddings, AzureInferenceEmbeddings
except ImportError:
    # Create placeholder classes to avoid errors when Azure modules are not available
    class AzureOpenAIEmbeddings:
        def __init__(self, **kwargs):
            raise ImportError("Azure OpenAI modules not available. Please install with 'pip install langchain-openai'")
    
    class AzureInferenceEmbeddings:
        def __init__(self, **kwargs):
            raise ImportError("Azure AI Inference modules not available. Please install with 'pip install azure-ai-inference'")

class Evaluator:
    """Evaluates the quality of QA system responses."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration from YAML file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize QA system
        self.qa_system = QASystem(config_path)
        
        # Initialize embedding model for semantic similarity
        self._initialize_embedding_model()
        
        # Initialize ROUGE for text comparison
        self.rouge = Rouge()
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model based on configuration."""
        embedding_config = self.config['embedding_model']
        
        # Check embedding model type
        embedding_type = embedding_config.get('type', 'huggingface')
        
        if embedding_type == 'azure_inference':
            # Use Azure AI Inference SDK embeddings
            self.embedding_model = AzureInferenceEmbeddings(
                endpoint=embedding_config.get('endpoint'),
                model_name=embedding_config.get('model_name')
            )
        elif embedding_type == 'azure_openai':
            # Use Azure OpenAI embeddings
            self.embedding_model = AzureOpenAIEmbeddings(
                azure_endpoint=embedding_config.get('azure_endpoint'),
                azure_deployment=embedding_config.get('azure_deployment'),
                api_version=embedding_config.get('api_version', '2023-05-15')
            )
        else:
            # Default to HuggingFace embeddings
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_config['name'],
                model_kwargs=embedding_config.get('kwargs', {})
            )
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts using embeddings."""
        embedding1 = self.embedding_model.embed_query(text1)
        embedding2 = self.embedding_model.embed_query(text2)
        
        # Reshape embeddings for cosine_similarity
        embedding1 = np.array(embedding1).reshape(1, -1)
        embedding2 = np.array(embedding2).reshape(1, -1)
        
        # Compute cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        return similarity
    
    def compute_rouge_scores(self, predicted: str, reference: str) -> Dict[str, Any]:
        """Compute ROUGE scores between predicted and reference texts."""
        try:
            scores = self.rouge.get_scores(predicted, reference)[0]
            return scores
        except Exception as e:
            print(f"Error computing ROUGE scores: {e}")
            return {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    
    def evaluate_response(self, query: str, expected_answer: str) -> Dict[str, Any]:
        """
        Evaluate a response to a query compared to an expected answer.
        
        Args:
            query: The user's question
            expected_answer: The ground truth answer
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Generate response using the QA system
        result = self.qa_system.generate_response(query)
        generated_answer = result['response']
        
        # Compute semantic similarity
        semantic_similarity = self.compute_semantic_similarity(generated_answer, expected_answer)
        
        # Compute ROUGE scores
        rouge_scores = self.compute_rouge_scores(generated_answer, expected_answer)
        
        return {
            'query': query,
            'expected_answer': expected_answer,
            'generated_answer': generated_answer,
            'semantic_similarity': semantic_similarity,
            'rouge_1_f': rouge_scores['rouge-1']['f'],
            'rouge_2_f': rouge_scores['rouge-2']['f'],
            'rouge_l_f': rouge_scores['rouge-l']['f']
        }
    
    def evaluate_test_set(self, test_cases: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Evaluate a set of test cases.
        
        Args:
            test_cases: List of dictionaries with 'query' and 'expected_answer' keys
            
        Returns:
            List of evaluation results
        """
        results = []
        for case in test_cases:
            result = self.evaluate_response(case['query'], case['expected_answer'])
            results.append(result)
        
        # Calculate average scores
        avg_semantic_similarity = np.mean([res['semantic_similarity'] for res in results])
        avg_rouge_1 = np.mean([res['rouge_1_f'] for res in results])
        avg_rouge_2 = np.mean([res['rouge_2_f'] for res in results])
        avg_rouge_l = np.mean([res['rouge_l_f'] for res in results])
        
        print(f"===== Evaluation Results =====")
        print(f"Average Semantic Similarity: {avg_semantic_similarity:.4f}")
        print(f"Average ROUGE-1 F1: {avg_rouge_1:.4f}")
        print(f"Average ROUGE-2 F1: {avg_rouge_2:.4f}")
        print(f"Average ROUGE-L F1: {avg_rouge_l:.4f}")
        
        return results
