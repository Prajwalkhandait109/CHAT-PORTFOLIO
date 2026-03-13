from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from langchain.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
    SentenceTransformerEmbeddings,
)
from langchain.schema import Document as LangchainDocument
import logging
import hashlib
import json

logger = logging.getLogger(__name__)

class EmbeddingProvider(Enum):
    """Available embedding providers"""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    LOCAL = "local"

class EmbeddingModel(Enum):
    """Available embedding models"""
    # OpenAI models
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    
    # HuggingFace models
    ALL_MPNET_BASE_V2 = "sentence-transformers/all-mpnet-base-v2"
    ALL_MINILM_L6_V2 = "sentence-transformers/all-MiniLM-L6-v2"
    ALL_MINILM_L12_V2 = "sentence-transformers/all-MiniLM-L12-v2"
    PARAPHRASE_MINILM_L6_V2 = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    
    # Local models
    WORD2VEC = "word2vec"
    FASTTEXT = "fasttext"

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    provider: EmbeddingProvider
    model: EmbeddingModel
    model_name: str
    dimension: int
    max_tokens: int
    batch_size: int = 32
    normalize: bool = True
    cache_embeddings: bool = True
    cache_size: int = 10000

@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    embeddings: np.ndarray
    metadata: Dict[str, Any]
    processing_time: float
    cache_hit: bool = False
    embedding_id: str = ""

class EmbeddingCache:
    """LRU cache for embeddings to avoid recomputation"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, embedding: np.ndarray):
        """Put embedding in cache"""
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        self.cache[key] = embedding
        self.access_order.append(key)
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)

class AdvancedEmbeddingEngine:
    """Advanced embedding engine with multiple providers and techniques"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.embedding_model = None
        self.cache = EmbeddingCache(config.cache_size) if config.cache_embeddings else None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model based on configuration"""
        try:
            if self.config.provider == EmbeddingProvider.OPENAI:
                self.embedding_model = OpenAIEmbeddings(
                    model=self.config.model_name,
                    max_retries=3,
                    request_timeout=30
                )
            elif self.config.provider == EmbeddingProvider.HUGGINGFACE:
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=self.config.model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': self.config.normalize}
                )
            elif self.config.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
                self.embedding_model = SentenceTransformerEmbeddings(
                    model_name=self.config.model_name,
                    normalize_embeddings=self.config.normalize
                )
            else:
                raise ValueError(f"Unsupported embedding provider: {self.config.provider}")
            
            logger.info(f"Initialized embedding model: {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise
    
    def generate_embeddings(self, 
                          texts: List[str],
                          batch_size: Optional[int] = None,
                          show_progress: bool = False) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing (overrides config)
            show_progress: Show progress bar
        
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        import time
        
        if not texts:
            return EmbeddingResult(
                embeddings=np.array([]),
                metadata={"text_count": 0},
                processing_time=0.0
            )
        
        batch_size = batch_size or self.config.batch_size
        start_time = time.time()
        
        # Check cache for embeddings
        cached_embeddings = []
        texts_to_process = []
        cache_indices = []
        
        if self.cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                cached_embedding = self.cache.get(cache_key)
                
                if cached_embedding is not None:
                    cached_embeddings.append((i, cached_embedding))
                else:
                    texts_to_process.append(text)
                    cache_indices.append(i)
        else:
            texts_to_process = texts
            cache_indices = list(range(len(texts)))
        
        # Generate embeddings for non-cached texts
        new_embeddings = []
        if texts_to_process:
            try:
                if show_progress:
                    logger.info(f"Generating embeddings for {len(texts_to_process)} texts")
                
                # Process in batches
                for i in range(0, len(texts_to_process), batch_size):
                    batch_texts = texts_to_process[i:i + batch_size]
                    batch_embeddings = self.embedding_model.embed_documents(batch_texts)
                    new_embeddings.extend(batch_embeddings)
                
                # Cache new embeddings
                if self.cache:
                    for text, embedding in zip(texts_to_process, new_embeddings):
                        cache_key = self._get_cache_key(text)
                        self.cache.put(cache_key, np.array(embedding))
                
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                raise
        
        # Combine cached and new embeddings
        all_embeddings = [None] * len(texts)
        
        # Place cached embeddings
        for idx, embedding in cached_embeddings:
            all_embeddings[idx] = embedding
        
        # Place new embeddings
        for idx, embedding in zip(cache_indices, new_embeddings):
            all_embeddings[idx] = np.array(embedding)
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings)
        
        processing_time = time.time() - start_time
        
        metadata = {
            "text_count": len(texts),
            "cached_count": len(cached_embeddings),
            "processed_count": len(texts_to_process),
            "cache_hit_rate": len(cached_embeddings) / len(texts) if texts else 0,
            "batch_size": batch_size,
            "embedding_dimension": embeddings_array.shape[1] if embeddings_array.size > 0 else 0
        }
        
        return EmbeddingResult(
            embeddings=embeddings_array,
            metadata=metadata,
            processing_time=processing_time,
            cache_hit=len(cached_embeddings) > 0
        )
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        result = self.generate_embeddings([text])
        return result.embeddings[0] if result.embeddings.size > 0 else np.array([])
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Use hash of normalized text
        normalized_text = text.strip().lower()
        return hashlib.md5(normalized_text.encode()).hexdigest()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if self.config.dimension:
            return self.config.dimension
        
        # Test with a sample text to get dimension
        try:
            sample_embedding = self.generate_single_embedding("test")
            return len(sample_embedding)
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {e}")
            return 0
    
    def similarity_search(self, 
                         query_embedding: np.ndarray,
                         document_embeddings: np.ndarray,
                         top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Perform similarity search between query and document embeddings
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: Document embedding matrix
            top_k: Number of top results to return
        
        Returns:
            List of (index, similarity_score) tuples
        """
        if query_embedding.size == 0 or document_embeddings.size == 0:
            return []
        
        try:
            # Normalize embeddings
            if self.config.normalize:
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
                document_embeddings = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
            
            # Calculate similarities (cosine similarity)
            similarities = np.dot(document_embeddings, query_embedding)
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Create result list
            results = [(int(idx), float(similarities[idx])) for idx in top_indices]
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def batch_similarity_search(self,
                               query_embeddings: np.ndarray,
                               document_embeddings: np.ndarray,
                               top_k: int = 5) -> List[List[Tuple[int, float]]]:
        """
        Batch similarity search for multiple queries
        
        Args:
            query_embeddings: Query embedding matrix
            document_embeddings: Document embedding matrix
            top_k: Number of top results per query
        
        Returns:
            List of similarity results for each query
        """
        results = []
        
        for query_emb in query_embeddings:
            query_result = self.similarity_search(query_emb, document_embeddings, top_k)
            results.append(query_result)
        
        return results

class MultiModalEmbeddingEngine:
    """Multi-modal embedding engine for different content types"""
    
    def __init__(self, configs: Dict[str, EmbeddingConfig]):
        """
        Initialize multi-modal embedding engine
        
        Args:
            configs: Dictionary mapping content types to embedding configurations
        """
        self.engines = {}
        
        for content_type, config in configs.items():
            self.engines[content_type] = AdvancedEmbeddingEngine(config)
    
    def generate_embeddings(self, 
                          content_type: str,
                          texts: List[str],
                          **kwargs) -> EmbeddingResult:
        """Generate embeddings for specific content type"""
        if content_type not in self.engines:
            raise ValueError(f"No embedding engine configured for content type: {content_type}")
        
        return self.engines[content_type].generate_embeddings(texts, **kwargs)
    
    def generate_multi_modal_embeddings(self, 
                                      content_dict: Dict[str, List[str]]) -> Dict[str, EmbeddingResult]:
        """
        Generate embeddings for multiple content types
        
        Args:
            content_dict: Dictionary mapping content types to text lists
        
        Returns:
            Dictionary mapping content types to embedding results
        """
        results = {}
        
        for content_type, texts in content_dict.items():
            if content_type in self.engines:
                try:
                    result = self.generate_embeddings(content_type, texts)
                    results[content_type] = result
                except Exception as e:
                    logger.error(f"Error generating embeddings for {content_type}: {e}")
                    continue
        
        return results
    
    def get_available_content_types(self) -> List[str]:
        """Get list of available content types"""
        return list(self.engines.keys())

# Predefined configurations for common use cases
EMBEDDING_CONFIGS = {
    "general_text": EmbeddingConfig(
        provider=EmbeddingProvider.HUGGINGFACE,
        model=EmbeddingModel.ALL_MPNET_BASE_V2,
        model_name="sentence-transformers/all-mpnet-base-v2",
        dimension=768,
        max_tokens=512,
        normalize=True
    ),
    "technical_code": EmbeddingConfig(
        provider=EmbeddingProvider.HUGGINGFACE,
        model=EmbeddingModel.ALL_MINILM_L6_V2,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        max_tokens=256,
        normalize=True
    ),
    "multilingual": EmbeddingConfig(
        provider=EmbeddingProvider.HUGGINGFACE,
        model=EmbeddingModel.PARAPHRASE_MINILM_L6_V2,
        model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
        dimension=384,
        max_tokens=512,
        normalize=True
    ),
    "high_quality": EmbeddingConfig(
        provider=EmbeddingProvider.OPENAI,
        model=EmbeddingModel.TEXT_EMBEDDING_3_LARGE,
        model_name="text-embedding-3-large",
        dimension=3072,
        max_tokens=8191,
        normalize=True
    ),
    "fast_processing": EmbeddingConfig(
        provider=EmbeddingProvider.HUGGINGFACE,
        model=EmbeddingModel.ALL_MINILM_L6_V2,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        max_tokens=512,
        normalize=True,
        batch_size=64
    )
}

def create_embedding_engine(config_name: str = "general_text") -> AdvancedEmbeddingEngine:
    """
    Create embedding engine with predefined configuration
    
    Args:
        config_name: Name of predefined configuration
    
    Returns:
        AdvancedEmbeddingEngine instance
    """
    if config_name not in EMBEDDING_CONFIGS:
        raise ValueError(f"Unknown configuration: {config_name}")
    
    config = EMBEDDING_CONFIGS[config_name]
    return AdvancedEmbeddingEngine(config)