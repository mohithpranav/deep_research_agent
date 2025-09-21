"""
Embedding Engine - Handle local embedding generation using sentence-transformers
"""
import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from sentence_transformers import SentenceTransformer
import torch

from config.settings import Settings
from config.model_config import ModelConfig
from utils.logger import logger
from .text_processor import TextChunk


class EmbeddingEngine:
    """Handles local embedding generation and caching"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding engine
        
        Args:
            model_name: Optional model name override
        """
        self.settings = Settings()
        self.model_config = ModelConfig()
        
        # Use provided model or default
        self.model_name = model_name or self.settings.EMBEDDING_MODEL
        
        # Initialize model (will download on first use)
        self.model = None
        self.device = self._get_best_device()
        
        # Embedding cache
        self.cache_file = self.settings.MODELS_DIR / f"embedding_cache_{self.model_name.replace('/', '_')}.pkl"
        self.embedding_cache = self._load_cache()
        
        # Initialize model
        self._initialize_model()
    
    def _get_best_device(self) -> str:
        """Determine the best device to use for inference"""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("Using CUDA for embeddings")
        elif torch.backends.mps.is_available():  # Apple Silicon
            device = "mps"
            logger.info("Using MPS for embeddings")
        else:
            device = "cpu"
            logger.info("Using CPU for embeddings")
        
        return device
    
    def _initialize_model(self) -> None:
        """Initialize the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Create models directory if it doesn't exist
            self.settings.ensure_directories()
            
            # Load model
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.settings.MODELS_DIR),
                device=self.device
            )
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Could not initialize embedding model: {e}")
    
    def _load_cache(self) -> Dict:
        """Load embedding cache from disk"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(f"Loaded embedding cache with {len(cache)} entries")
                return cache
        except Exception as e:
            logger.warning(f"Could not load embedding cache: {e}")
        
        return {}
    
    def _save_cache(self) -> None:
        """Save embedding cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.debug(f"Saved embedding cache with {len(self.embedding_cache)} entries")
        except Exception as e:
            logger.error(f"Could not save embedding cache: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def encode_single(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to encode
            use_cache: Whether to use/update cache
            
        Returns:
            Embedding vector as numpy array
        """
        if not text or not text.strip():
            # Return zero embedding for empty text
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        if use_cache and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalize for cosine similarity
            )
            
            # Cache the result
            if use_cache:
                self.embedding_cache[cache_key] = embedding
                
                # Save cache periodically (every 100 new embeddings)
                if len(self.embedding_cache) % 100 == 0:
                    self._save_cache()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero embedding on error
            return np.zeros(self.model.get_sentence_embedding_dimension())
    
    def encode_batch(self, texts: List[str], batch_size: int = 32, use_cache: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            use_cache: Whether to use/update cache
            
        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])
        
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for existing embeddings
        for i, text in enumerate(texts):
            if not text or not text.strip():
                # Zero embedding for empty text
                embeddings.append(np.zeros(self.model.get_sentence_embedding_dimension()))
            else:
                cache_key = self._get_cache_key(text)
                if use_cache and cache_key in self.embedding_cache:
                    embeddings.append(self.embedding_cache[cache_key])
                else:
                    embeddings.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                logger.info(f"Generating embeddings for {len(uncached_texts)} texts")
                
                new_embeddings = self.model.encode(
                    uncached_texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=len(uncached_texts) > 10
                )
                
                # Update cache and results
                for i, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                    result_index = uncached_indices[i]
                    embeddings[result_index] = embedding
                    
                    if use_cache:
                        cache_key = self._get_cache_key(text)
                        self.embedding_cache[cache_key] = embedding
                
                # Save cache
                if use_cache:
                    self._save_cache()
                    
            except Exception as e:
                logger.error(f"Error in batch embedding generation: {e}")
                # Fill remaining with zero embeddings
                for i in uncached_indices:
                    if embeddings[i] is None:
                        embeddings[i] = np.zeros(self.model.get_sentence_embedding_dimension())
        
        return np.array(embeddings)
    
    def encode_chunks(self, chunks: List[TextChunk], batch_size: int = 32) -> Tuple[np.ndarray, List[str]]:
        """
        Generate embeddings for text chunks
        
        Args:
            chunks: List of TextChunk objects
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (embeddings array, chunk_ids list)
        """
        if not chunks:
            return np.array([]), []
        
        # Extract text content
        texts = [chunk.content for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.encode_batch(texts, batch_size=batch_size, use_cache=True)
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return embeddings, chunk_ids
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding
        """
        return self.encode_single(query, use_cache=False)  # Don't cache queries
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model"""
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return self.settings.EMBEDDING_DIMENSION
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        info = {
            "model_name": self.model_name,
            "embedding_dimension": self.get_embedding_dimension(),
            "device": self.device,
            "cache_size": len(self.embedding_cache),
            "cache_file": str(self.cache_file)
        }
        
        if self.model:
            info["max_seq_length"] = getattr(self.model, 'max_seq_length', 'Unknown')
        
        return info
    
    def clear_cache(self) -> None:
        """Clear the embedding cache"""
        self.embedding_cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        cache_size_mb = 0
        if self.cache_file.exists():
            cache_size_mb = self.cache_file.stat().st_size / (1024 * 1024)
        
        return {
            "cache_entries": len(self.embedding_cache),
            "cache_size_mb": round(cache_size_mb, 2),
            "cache_file": str(self.cache_file),
            "cache_exists": self.cache_file.exists()
        }
    
    def warmup(self, sample_texts: List[str] = None) -> None:
        """
        Warm up the model with sample texts
        
        Args:
            sample_texts: Optional list of sample texts for warmup
        """
        if not sample_texts:
            sample_texts = [
                "This is a sample text for warming up the embedding model.",
                "Another example sentence to test the model performance.",
                "Machine learning and natural language processing are fascinating fields."
            ]
        
        logger.info("Warming up embedding model...")
        start_time = datetime.now()
        
        # Run a small batch to warm up
        _ = self.encode_batch(sample_texts, use_cache=False)
        
        warmup_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Model warmup completed in {warmup_time:.2f} seconds")
    
    def benchmark(self, test_texts: List[str] = None, num_iterations: int = 3) -> Dict:
        """
        Benchmark the embedding generation performance
        
        Args:
            test_texts: Optional test texts
            num_iterations: Number of benchmark iterations
            
        Returns:
            Performance statistics
        """
        if not test_texts:
            test_texts = [
                "This is a benchmark test for the embedding model performance evaluation.",
                "Natural language processing involves understanding and generating human language.",
                "Machine learning algorithms can learn patterns from data automatically.",
                "Deep learning uses neural networks with multiple layers for complex tasks.",
                "Information retrieval systems help find relevant documents from large collections."
            ]
        
        import time
        times = []
        
        logger.info(f"Running benchmark with {len(test_texts)} texts, {num_iterations} iterations")
        
        for i in range(num_iterations):
            start_time = time.time()
            _ = self.encode_batch(test_texts, use_cache=False)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        texts_per_second = len(test_texts) / avg_time
        
        stats = {
            "num_texts": len(test_texts),
            "num_iterations": num_iterations,
            "avg_time_seconds": round(avg_time, 3),
            "texts_per_second": round(texts_per_second, 1),
            "model_name": self.model_name,
            "device": self.device
        }
        
        logger.info(f"Benchmark results: {texts_per_second:.1f} texts/second")
        return stats