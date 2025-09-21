"""
Vector Store - FAISS-based vector database for similarity search
"""
import os
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

from config.settings import Settings
from utils.logger import logger
from core.text_processor import TextChunk


class VectorStore:
    """FAISS-based vector database for efficient similarity search"""
    
    def __init__(self, embedding_dimension: int = None):
        """
        Initialize vector store
        
        Args:
            embedding_dimension: Dimension of embeddings (auto-detected if None)
        """
        self.settings = Settings()
        self.embedding_dimension = embedding_dimension or self.settings.EMBEDDING_DIMENSION
        
        # FAISS index
        self.index = None
        self.index_type = "IVFFlat"  # Default index type
        
        # Metadata storage
        self.chunk_metadata = {}  # chunk_id -> metadata
        self.id_to_chunk_id = {}  # FAISS ID -> chunk_id
        self.chunk_id_to_id = {}  # chunk_id -> FAISS ID
        self.next_id = 0
        
        # File paths
        self.index_file = self.settings.VECTOR_DB_PATH
        self.metadata_file = self.settings.MODELS_DIR / "vector_metadata.json"
        
        # Initialize
        self._initialize_index()
        self._load_metadata()
    
    def _initialize_index(self) -> None:
        """Initialize or load FAISS index"""
        try:
            # Try to load existing index
            if self.index_file.exists():
                self._load_index()
            else:
                self._create_new_index()
                
        except Exception as e:
            logger.error(f"Error initializing index: {e}")
            self._create_new_index()
    
    def _create_new_index(self) -> None:
        """Create a new FAISS index"""
        logger.info(f"Creating new FAISS index with dimension {self.embedding_dimension}")
        
        # Create index based on expected data size
        # For small datasets (< 10k vectors): use Flat index
        # For medium datasets (10k-100k): use IVFFlat
        # For large datasets (>100k): use HNSW or more complex index
        
        # Start with Flat index (exact search, good for small datasets)
        self.index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner Product (cosine with normalized vectors)
        self.index_type = "Flat"
        
        logger.info(f"Created {self.index_type} index")
    
    def _create_ivf_index(self, nlist: int = 100) -> None:
        """Create IVF (Inverted File) index for larger datasets"""
        quantizer = faiss.IndexFlatIP(self.embedding_dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist)
        self.index_type = "IVFFlat"
        logger.info(f"Created IVFFlat index with {nlist} clusters")
    
    def _load_index(self) -> None:
        """Load existing FAISS index from disk"""
        try:
            self.index = faiss.read_index(str(self.index_file))
            logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            self._create_new_index()
    
    def _save_index(self) -> None:
        """Save FAISS index to disk"""
        try:
            # Ensure directory exists
            self.index_file.parent.mkdir(parents=True, exist_ok=True)
            
            faiss.write_index(self.index, str(self.index_file))
            logger.debug(f"Saved FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def _load_metadata(self) -> None:
        """Load metadata from disk"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                self.chunk_metadata = data.get('chunk_metadata', {})
                self.id_to_chunk_id = {int(k): v for k, v in data.get('id_to_chunk_id', {}).items()}
                self.chunk_id_to_id = {v: int(k) for k, v in self.id_to_chunk_id.items()}
                self.next_id = data.get('next_id', 0)
                self.embedding_dimension = data.get('embedding_dimension', self.embedding_dimension)
                
                logger.info(f"Loaded metadata for {len(self.chunk_metadata)} chunks")
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self._reset_metadata()
    
    def _save_metadata(self) -> None:
        """Save metadata to disk"""
        try:
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'chunk_metadata': self.chunk_metadata,
                'id_to_chunk_id': {str(k): v for k, v in self.id_to_chunk_id.items()},
                'next_id': self.next_id,
                'embedding_dimension': self.embedding_dimension,
                'index_type': self.index_type,
                'created_at': datetime.now().isoformat(),
                'total_vectors': self.index.ntotal if self.index else 0
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
                
            logger.debug("Saved vector store metadata")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def _reset_metadata(self) -> None:
        """Reset all metadata"""
        self.chunk_metadata = {}
        self.id_to_chunk_id = {}
        self.chunk_id_to_id = {}
        self.next_id = 0
    
    def add_chunks(self, chunks: List[TextChunk], embeddings: np.ndarray) -> bool:
        """
        Add chunks and their embeddings to the vector store
        
        Args:
            chunks: List of TextChunk objects
            embeddings: Corresponding embeddings array
            
        Returns:
            Success status
        """
        if len(chunks) != len(embeddings):
            logger.error(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")
            return False
        
        if len(embeddings) == 0:
            return True
        
        try:
            # Validate embedding dimension
            if embeddings.shape[1] != self.embedding_dimension:
                logger.error(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {embeddings.shape[1]}")
                return False
            
            # Check if we need to upgrade to IVF index for better performance
            if self.index_type == "Flat" and self.index.ntotal + len(chunks) > 10000:
                logger.info("Upgrading to IVF index for better performance")
                self._upgrade_to_ivf()
            
            # Prepare data
            chunk_ids = []
            faiss_ids = []
            
            for i, chunk in enumerate(chunks):
                # Check for duplicates
                if chunk.chunk_id in self.chunk_id_to_id:
                    logger.warning(f"Chunk {chunk.chunk_id} already exists, skipping")
                    continue
                
                # Assign FAISS ID
                faiss_id = self.next_id
                self.next_id += 1
                
                # Store mappings
                self.id_to_chunk_id[faiss_id] = chunk.chunk_id
                self.chunk_id_to_id[chunk.chunk_id] = faiss_id
                
                # Store metadata
                self.chunk_metadata[chunk.chunk_id] = {
                    'doc_id': chunk.doc_id,
                    'content': chunk.content,
                    'chunk_index': chunk.chunk_index,
                    'word_count': chunk.word_count,
                    'char_count': chunk.char_count,
                    'page_number': chunk.page_number,
                    'added_at': datetime.now().isoformat()
                }
                
                chunk_ids.append(chunk.chunk_id)
                faiss_ids.append(faiss_id)
            
            # Add to FAISS index
            if chunk_ids:  # Only add if there are new chunks
                valid_embeddings = embeddings[:len(chunk_ids)]  # In case some were skipped
                
                # Train index if needed (for IVF)
                if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                    if len(valid_embeddings) >= 100:  # Need minimum samples for training
                        self.index.train(valid_embeddings.astype(np.float32))
                    else:
                        logger.warning("Not enough samples to train IVF index, using existing training")
                
                # Add vectors
                self.index.add(valid_embeddings.astype(np.float32))
                
                logger.info(f"Added {len(chunk_ids)} new chunks to vector store")
            
            # Save to disk
            self._save_index()
            self._save_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            return False
    
    def _upgrade_to_ivf(self) -> None:
        """Upgrade from Flat to IVF index for better performance"""
        try:
            if self.index.ntotal == 0:
                return
            
            # Extract all vectors
            all_vectors = np.zeros((self.index.ntotal, self.embedding_dimension), dtype=np.float32)
            for i in range(self.index.ntotal):
                all_vectors[i] = self.index.reconstruct(i)
            
            # Create new IVF index
            nlist = min(int(np.sqrt(self.index.ntotal)), 1000)  # Reasonable number of clusters
            self._create_ivf_index(nlist)
            
            # Train and add vectors
            self.index.train(all_vectors)
            self.index.add(all_vectors)
            
            logger.info(f"Successfully upgraded to IVF index with {nlist} clusters")
            
        except Exception as e:
            logger.error(f"Error upgrading to IVF index: {e}")
    
    def search(self, query_embedding: np.ndarray, k: int = None, threshold: float = None) -> List[Dict]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results with metadata
        """
        k = k or self.settings.MAX_RESULTS
        threshold = threshold or self.settings.SIMILARITY_THRESHOLD
        
        if self.index.ntotal == 0:
            return []
        
        try:
            # Ensure query is the right shape and type
            query = query_embedding.reshape(1, -1).astype(np.float32)
            
            # Search
            scores, indices = self.index.search(query, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                # Skip invalid indices
                if idx == -1 or idx not in self.id_to_chunk_id:
                    continue
                
                # Apply threshold filter
                if score < threshold:
                    continue
                
                chunk_id = self.id_to_chunk_id[idx]
                metadata = self.chunk_metadata.get(chunk_id, {})
                
                result = {
                    'chunk_id': chunk_id,
                    'score': float(score),
                    'content': metadata.get('content', ''),
                    'doc_id': metadata.get('doc_id', ''),
                    'chunk_index': metadata.get('chunk_index', 0),
                    'word_count': metadata.get('word_count', 0),
                    'page_number': metadata.get('page_number'),
                    'metadata': metadata
                }
                results.append(result)
            
            logger.debug(f"Found {len(results)} results above threshold {threshold}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Get chunk metadata by chunk ID"""
        return self.chunk_metadata.get(chunk_id)
    
    def remove_chunk(self, chunk_id: str) -> bool:
        """
        Remove a chunk from the vector store
        Note: FAISS doesn't support efficient removal, so we mark as removed
        """
        if chunk_id not in self.chunk_id_to_id:
            return False
        
        try:
            # Mark as removed in metadata
            if chunk_id in self.chunk_metadata:
                self.chunk_metadata[chunk_id]['removed'] = True
                self.chunk_metadata[chunk_id]['removed_at'] = datetime.now().isoformat()
            
            logger.info(f"Marked chunk {chunk_id} as removed")
            self._save_metadata()
            return True
            
        except Exception as e:
            logger.error(f"Error removing chunk {chunk_id}: {e}")
            return False
    
    def remove_document(self, doc_id: str) -> int:
        """
        Remove all chunks belonging to a document
        
        Args:
            doc_id: Document ID
            
        Returns:
            Number of chunks removed
        """
        removed_count = 0
        
        for chunk_id, metadata in self.chunk_metadata.items():
            if metadata.get('doc_id') == doc_id and not metadata.get('removed', False):
                if self.remove_chunk(chunk_id):
                    removed_count += 1
        
        logger.info(f"Removed {removed_count} chunks for document {doc_id}")
        return removed_count
    
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        total_chunks = len(self.chunk_metadata)
        active_chunks = len([m for m in self.chunk_metadata.values() if not m.get('removed', False)])
        removed_chunks = total_chunks - active_chunks
        
        stats = {
            'total_vectors': self.index.ntotal if self.index else 0,
            'total_chunks': total_chunks,
            'active_chunks': active_chunks,
            'removed_chunks': removed_chunks,
            'embedding_dimension': self.embedding_dimension,
            'index_type': self.index_type,
            'next_id': self.next_id
        }
        
        # Add file sizes
        if self.index_file.exists():
            stats['index_size_mb'] = round(self.index_file.stat().st_size / (1024 * 1024), 2)
        
        if self.metadata_file.exists():
            stats['metadata_size_mb'] = round(self.metadata_file.stat().st_size / (1024 * 1024), 2)
        
        return stats
    
    def rebuild_index(self) -> bool:
        """Rebuild the index to remove deleted entries"""
        try:
            logger.info("Rebuilding vector store index...")
            
            # Get all active chunks
            active_chunks = []
            active_embeddings = []
            
            for chunk_id, metadata in self.chunk_metadata.items():
                if not metadata.get('removed', False):
                    faiss_id = self.chunk_id_to_id.get(chunk_id)
                    if faiss_id is not None and faiss_id < self.index.ntotal:
                        # Reconstruct embedding
                        embedding = self.index.reconstruct(faiss_id)
                        active_embeddings.append(embedding)
                        active_chunks.append(chunk_id)
            
            if not active_chunks:
                # Empty index
                self._create_new_index()
                self._reset_metadata()
                self._save_index()
                self._save_metadata()
                return True
            
            # Create new index
            old_index = self.index
            self._create_new_index()
            
            # Re-add active chunks
            embeddings_array = np.array(active_embeddings)
            
            # Reset ID mappings
            self._reset_metadata()
            
            # Create dummy chunks for re-adding
            from core.text_processor import TextChunk
            dummy_chunks = []
            for i, chunk_id in enumerate(active_chunks):
                old_metadata = self.chunk_metadata.get(chunk_id, {})
                chunk = TextChunk(
                    content=old_metadata.get('content', ''),
                    chunk_id=chunk_id,
                    doc_id=old_metadata.get('doc_id', ''),
                    chunk_index=old_metadata.get('chunk_index', 0)
                )
                dummy_chunks.append(chunk)
            
            # Re-add chunks
            success = self.add_chunks(dummy_chunks, embeddings_array)
            
            if success:
                logger.info(f"Successfully rebuilt index with {len(active_chunks)} active chunks")
            else:
                # Rollback
                self.index = old_index
                logger.error("Failed to rebuild index, rolled back")
                
            return success
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all data from vector store"""
        self._create_new_index()
        self._reset_metadata()
        self._save_index()
        self._save_metadata()
        logger.info("Vector store cleared")