"""
Model configuration for embeddings and reasoning
"""

class ModelConfig:
    """Configuration for AI models and processing"""
    
    # Sentence Transformer Models (choose based on performance needs)
    EMBEDDING_MODELS = {
        'fast': 'all-MiniLM-L6-v2',  # Fastest, good quality
        'balanced': 'all-mpnet-base-v2',  # Balanced speed/quality
        'accurate': 'all-roberta-large-v1'  # Most accurate, slower
    }
    
    # Text processing parameters
    TEXT_PROCESSING = {
        'min_chunk_length': 100,
        'max_chunk_length': 1000,
        'sentence_overlap': 2,
        'language': 'english'
    }
    
    # Reasoning parameters
    REASONING_CONFIG = {
        'max_context_length': 4000,
        'temperature': 0.3,  # For deterministic outputs
        'top_k_retrieval': 10,
        'rerank_top_k': 5
    }
    
    # Local LLM options (for advanced reasoning)
    LOCAL_LLM_OPTIONS = {
        'ollama_models': ['llama2:7b', 'mistral:7b'],
        'gpt4all_models': ['orca-mini-3b.ggmlv3.q4_0.bin'],
        'use_local_llm': False  # Set to True if local LLM available
    }