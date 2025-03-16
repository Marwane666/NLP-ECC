import yaml
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge

from src.qa_system import QASystem
from langchain.embeddings import HuggingFaceEmbeddings

class Evaluator:
    """Evaluates the quality of QA system responses."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration from YAML file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize QA system
        self.qa_system = QASystem(config_path)
        
        # Initialize embedding model for semantic similarity
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config['embedding_model']['name'],
            model_kwargs=self.config['embedding_model'].get('kwargs', {})
        )
        
        # Initialize ROUGE for text comparison
        self.rouge = Rouge()
    
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
