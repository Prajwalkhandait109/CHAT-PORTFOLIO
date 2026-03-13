import os
import pickle
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain.schema import Document as LangchainDocument
from langchain_huggingface import HuggingFaceEmbeddings

# Import our custom modules
from .src.rag.document_processor import DocumentProcessor, Document
from .src.rag.chunking import IntelligentChunker, Chunk
from .src.rag.embedding import AdvancedEmbeddingEngine, create_embedding_engine

logger = logging.getLogger(__name__)

class EnhancedVectorStore:
    """Enhanced vector store with multi-modal document support"""
    
    def __init__(self, 
                 embedding_config: str = "general_text",
                 vector_store_path: str = "db",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize enhanced vector store
        
        Args:
            embedding_config: Name of embedding configuration
            vector_store_path: Path to save/load vector store
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.embedding_config = embedding_config
        self.vector_store_path = vector_store_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.chunker = IntelligentChunker(
            default_chunk_size=chunk_size,
            default_chunk_overlap=chunk_overlap
        )
        self.embedding_engine = create_embedding_engine(embedding_config)
        self.vector_store = None
        
        # Load existing vector store if available
        self._load_vector_store()
    
    def _load_vector_store(self):
        """Load existing vector store"""
        try:
            if os.path.exists(self.vector_store_path) and os.listdir(self.vector_store_path):
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embedding_engine.embedding_model,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Loaded existing vector store from {self.vector_store_path}")
            else:
                logger.info("No existing vector store found, will create new one")
        except Exception as e:
            logger.warning(f"Error loading vector store: {e}")
            self.vector_store = None
    
    def add_documents_from_files(self, file_paths: List[Union[str, Path]], 
                                metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add documents from multiple file paths
        
        Args:
            file_paths: List of file paths to process
            metadata: Additional metadata to attach to documents
        
        Returns:
            Processing statistics
        """
        stats = {
            "total_files": len(file_paths),
            "processed_files": 0,
            "failed_files": 0,
            "total_documents": 0,
            "total_chunks": 0,
            "errors": []
        }
        
        all_chunks = []
        
        for file_path in file_paths:
            try:
                # Process file
                documents = self.document_processor.process_file(file_path)
                
                # Chunk documents
                file_chunks = []
                for doc in documents:
                    chunks = self.chunker.chunk_document(
                        content=doc.content,
                        doc_type=doc.doc_type,
                        metadata={**doc.metadata, **(metadata or {})}
                    )
                    file_chunks.extend(chunks)
                
                all_chunks.extend(file_chunks)
                
                stats["processed_files"] += 1
                stats["total_documents"] += len(documents)
                stats["total_chunks"] += len(file_chunks)
                
                logger.info(f"Processed file {file_path}: {len(documents)} docs, {len(file_chunks)} chunks")
                
            except Exception as e:
                stats["failed_files"] += 1
                stats["errors"].append(f"Error processing {file_path}: {str(e)}")
                logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        # Add chunks to vector store
        if all_chunks:
            self._add_chunks_to_vector_store(all_chunks)
            logger.info(f"Added {len(all_chunks)} chunks to vector store")
        
        return stats
    
    def add_documents_from_web(self, urls: List[str], 
                               include_metadata: bool = True,
                               additional_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add documents from web URLs
        
        Args:
            urls: List of URLs to process
            include_metadata: Whether to include web metadata
            additional_metadata: Additional metadata to attach
        
        Returns:
            Processing statistics
        """
        stats = {
            "total_urls": len(urls),
            "processed_urls": 0,
            "failed_urls": 0,
            "total_documents": 0,
            "total_chunks": 0,
            "errors": []
        }
        
        all_chunks = []
        
        for url in urls:
            try:
                # Process URL
                documents = self.document_processor.process_url(url, include_metadata)
                
                # Chunk documents
                url_chunks = []
                for doc in documents:
                    chunks = self.chunker.chunk_document(
                        content=doc.content,
                        doc_type=doc.doc_type,
                        metadata={**doc.metadata, **(additional_metadata or {})}
                    )
                    url_chunks.extend(chunks)
                
                all_chunks.extend(url_chunks)
                
                stats["processed_urls"] += 1
                stats["total_documents"] += len(documents)
                stats["total_chunks"] += len(url_chunks)
                
                logger.info(f"Processed URL {url}: {len(documents)} docs, {len(url_chunks)} chunks")
                
            except Exception as e:
                stats["failed_urls"] += 1
                stats["errors"].append(f"Error processing {url}: {str(e)}")
                logger.error(f"Error processing URL {url}: {e}")
                continue
        
        # Add chunks to vector store
        if all_chunks:
            self._add_chunks_to_vector_store(all_chunks)
            logger.info(f"Added {len(all_chunks)} chunks to vector store")
        
        return stats
    
    def add_text_documents(self, texts: List[str], 
                          metadata_list: Optional[List[Dict[str, Any]]] = None,
                          doc_type: str = "text") -> int:
        """
        Add text documents directly
        
        Args:
            texts: List of text strings
            metadata_list: List of metadata dictionaries (one per text)
            doc_type: Document type
        
        Returns:
            Number of chunks added
        """
        if metadata_list is None:
            metadata_list = [{} for _ in texts]
        
        all_chunks = []
        
        for text, metadata in zip(texts, metadata_list):
            try:
                # Create document
                doc = Document(
                    content=text,
                    metadata=metadata,
                    doc_type=doc_type,
                    source="direct_input"
                )
                
                # Chunk document
                chunks = self.chunker.chunk_document(
                    content=doc.content,
                    doc_type=doc.doc_type,
                    metadata=doc.metadata
                )
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error processing text: {e}")
                continue
        
        # Add chunks to vector store
        if all_chunks:
            self._add_chunks_to_vector_store(all_chunks)
            logger.info(f"Added {len(all_chunks)} text chunks to vector store")
        
        return len(all_chunks)
    
    def _add_chunks_to_vector_store(self, chunks: List[Chunk]):
        """Add chunks to vector store"""
        if not chunks:
            return
        
        # Convert chunks to langchain documents
        lc_documents = []
        
        for chunk in chunks:
            # Create langchain document
            doc = LangchainDocument(
                page_content=chunk.content,
                metadata={
                    **chunk.metadata,
                    'chunk_id': chunk.chunk_id,
                    'word_count': chunk.word_count,
                    'char_count': chunk.char_count,
                    'chunking_strategy': chunk.metadata.get('chunk_strategy', 'unknown'),
                    'processing_timestamp': datetime.now().isoformat()
                }
            )
            lc_documents.append(doc)
        
        # Generate embeddings
        texts = [doc.page_content for doc in lc_documents]
        embedding_result = self.embedding_engine.generate_embeddings(texts)
        
        if embedding_result.embeddings.size == 0:
            logger.error("No embeddings generated")
            return
        
        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(
                lc_documents,
                self.embedding_engine.embedding_model
            )
        else:
            # Add to existing vector store
            self.vector_store.add_documents(lc_documents)
        
        # Save vector store
        self.save_vector_store()
    
    def search(self, query: str, k: int = 4, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search vector store
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional filter dictionary
        
        Returns:
            List of search results
        """
        if self.vector_store is None:
            logger.warning("Vector store is empty")
            return []
        
        try:
            if filter_dict:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k
                )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(score),
                    'source': doc.metadata.get('source', 'unknown'),
                    'doc_type': doc.metadata.get('doc_type', 'unknown')
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def similarity_search_with_embeddings(self, 
                                         query_embedding: List[float], 
                                         k: int = 4) -> List[Dict[str, Any]]:
        """
        Search using query embedding directly
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results
        
        Returns:
            List of search results
        """
        if self.vector_store is None:
            logger.warning("Vector store is empty")
            return []
        
        try:
            results = self.vector_store.similarity_search_by_vector(
                embedding=query_embedding,
                k=k
            )
            
            # Format results
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source': doc.metadata.get('source', 'unknown'),
                    'doc_type': doc.metadata.get('doc_type', 'unknown')
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching with embeddings: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if self.vector_store is None:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "embedding_dimension": self.embedding_engine.get_embedding_dimension(),
                "embedding_config": self.embedding_config,
                "vector_store_path": self.vector_store_path,
                "status": "empty"
            }
        
        try:
            # Get document count
            docstore = self.vector_store.docstore
            total_docs = len(docstore._dict) if hasattr(docstore, '_dict') else 0
            
            # Analyze document types
            doc_types = {}
            sources = set()
            
            if hasattr(docstore, '_dict'):
                for doc_id, doc in docstore._dict.items():
                    doc_type = doc.metadata.get('doc_type', 'unknown')
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    
                    source = doc.metadata.get('source', 'unknown')
                    sources.add(source)
            
            return {
                "total_documents": total_docs,
                "total_chunks": total_docs,  # Assuming one chunk per document for now
                "embedding_dimension": self.embedding_engine.get_embedding_dimension(),
                "embedding_config": self.embedding_config,
                "vector_store_path": self.vector_store_path,
                "document_types": doc_types,
                "unique_sources": len(sources),
                "status": "loaded"
            }
            
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "embedding_dimension": self.embedding_engine.get_embedding_dimension(),
                "embedding_config": self.embedding_config,
                "vector_store_path": self.vector_store_path,
                "status": "error",
                "error": str(e)
            }
    
    def save_vector_store(self):
        """Save vector store to disk"""
        if self.vector_store is None:
            logger.warning("No vector store to save")
            return
        
        try:
            # Ensure directory exists
            os.makedirs(self.vector_store_path, exist_ok=True)
            
            # Save vector store
            self.vector_store.save_local(self.vector_store_path)
            
            # Save metadata
            metadata = {
                "embedding_config": self.embedding_config,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(self.vector_store_path, "metadata.json")
            with open(metadata_path, 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Vector store saved to {self.vector_store_path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
    
    def clear_vector_store(self):
        """Clear all documents from vector store"""
        self.vector_store = None
        
        # Remove vector store files
        if os.path.exists(self.vector_store_path):
            import shutil
            shutil.rmtree(self.vector_store_path)
        
        logger.info("Vector store cleared")
    
    def delete_vector_store(self):
        """Delete vector store completely"""
        self.clear_vector_store()

# Convenience functions for backward compatibility
def create_enhanced_vector_store(embedding_config: str = "general_text", **kwargs) -> EnhancedVectorStore:
    """Create enhanced vector store with specified configuration"""
    return EnhancedVectorStore(embedding_config=embedding_config, **kwargs)

def process_and_add_documents(file_paths: List[Union[str, Path]], **kwargs) -> Dict[str, Any]:
    """Process and add documents to vector store"""
    vector_store = create_enhanced_vector_store(**kwargs)
    return vector_store.add_documents_from_files(file_paths)

def search_vector_store(query: str, k: int = 4, **kwargs) -> List[Dict[str, Any]]:
    """Search vector store"""
    vector_store = create_enhanced_vector_store(**kwargs)
    return vector_store.search(query, k)