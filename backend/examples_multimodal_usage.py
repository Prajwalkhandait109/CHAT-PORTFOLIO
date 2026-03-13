#!/usr/bin/env python3
"""
Usage example for multi-modal document processing system
Demonstrates how to process different document types and integrate with the chatbot
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.src.rag.document_processor import DocumentProcessor
from backend.src.rag.chunking import IntelligentChunker, ChunkingStrategy
from backend.src.rag.embedding import create_embedding_engine
from backend.enhanced_vector_store import EnhancedVectorStore
from backend.agentic_ai_chatbot import AgenticAIChatbot

def example_basic_document_processing():
    """Example: Basic document processing"""
    print("=== Basic Document Processing Example ===")
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Example: Process a text file
    text_content = """
    Machine Learning is a subset of artificial intelligence that enables systems to learn from data. 
    It focuses on developing algorithms that can access data and use it to learn for themselves.
    
    Deep Learning is a subset of machine learning that uses neural networks with multiple layers. 
    These networks can learn complex patterns in data and are particularly effective for tasks like 
    image recognition, natural language processing, and speech recognition.
    """
    
    # Process text content
    documents = processor.process_text_content(
        text_content,
        source="example_text",
        metadata={"topic": "AI/ML", "author": "Example"}
    )
    
    print(f"Processed {len(documents)} documents")
    for i, doc in enumerate(documents):
        print(f"Document {i+1}: {doc.content[:100]}...")
        print(f"  Metadata: {doc.metadata}")

def example_intelligent_chunking():
    """Example: Intelligent chunking strategies"""
    print("\n=== Intelligent Chunking Example ===")
    
    # Sample technical document
    technical_doc = """
    # Machine Learning Algorithms
    
    ## Supervised Learning
    Supervised learning algorithms learn from labeled training data. Common algorithms include:
    
    ### Linear Regression
    Linear regression models the relationship between a dependent variable and one or more independent variables.
    
    ### Decision Trees  
    Decision trees create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.
    
    ## Unsupervised Learning
    Unsupervised learning finds hidden patterns in data without pre-existing labels.
    
    ### K-Means Clustering
    K-means clustering partitions data into k clusters where each data point belongs to the cluster with the nearest mean.
    
    ### Principal Component Analysis
    PCA reduces the dimensionality of data while preserving as much variance as possible.
    """
    
    chunker = IntelligentChunker(chunk_size=300, chunk_overlap=50)
    
    # Test different strategies
    strategies = [
        ChunkingStrategy.RECURSIVE,
        ChunkingStrategy.SEMANTIC,
        ChunkingStrategy.STRUCTURAL
    ]
    
    for strategy in strategies:
        print(f"\nUsing {strategy.value} strategy:")
        
        chunks = chunker.chunk_document(
            content=technical_doc,
            strategy=strategy,
            metadata={"doc_type": "markdown"}
        )
        
        print(f"Generated {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
            print(f"Chunk {i+1} ({chunk.word_count} words):")
            print(f"  {chunk.content[:150]}...")

def example_enhanced_vector_store():
    """Example: Enhanced vector store with multi-modal support"""
    print("\n=== Enhanced Vector Store Example ===")
    
    # Create vector store
    vector_store = EnhancedVectorStore(
        embedding_config="general_text",
        vector_store_path="example_vector_store",
        chunk_size=400,
        chunk_overlap=100
    )
    
    # Example 1: Add text documents directly
    print("Adding text documents...")
    
    texts = [
        "Machine learning enables computers to learn from data without explicit programming.",
        "Deep learning uses neural networks with multiple layers to solve complex problems.",
        "Natural language processing allows computers to understand and generate human language.",
        "Computer vision enables machines to interpret and understand visual information."
    ]
    
    metadata_list = [
        {"topic": "machine_learning", "category": "AI/ML"},
        {"topic": "deep_learning", "category": "AI/ML"},
        {"topic": "nlp", "category": "AI/ML"},
        {"topic": "computer_vision", "category": "AI/ML"}
    ]
    
    chunks_added = vector_store.add_text_documents(texts, metadata_list)
    print(f"Added {chunks_added} text chunks to vector store")
    
    # Example 2: Search the vector store
    print("\nSearching vector store...")
    
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Tell me about AI technologies"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        results = vector_store.search(query, k=2)
        
        for i, result in enumerate(results):
            print(f"Result {i+1} (score: {result['score']:.4f}):")
            print(f"  {result['content']}")
            print(f"  Source: {result['source']}, Type: {result['doc_type']}")

def example_integration_with_chatbot():
    """Example: Integration with Agentic AI Chatbot"""
    print("\n=== Integration with Agentic AI Chatbot ===")
    
    try:
        # Initialize chatbot with enhanced vector store
        chatbot = AgenticAIChatbot(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            vector_store_path="example_vector_store",
            use_advanced_rag=True,
            enable_agent=True
        )
        
        # Example queries that would benefit from multi-modal processing
        test_queries = [
            "What is machine learning and how does it work?",
            "Explain the difference between AI and machine learning",
            "What are the main applications of computer vision?"
        ]
        
        for query in test_queries:
            print(f"\nUser: {query}")
            
            response = chatbot.ask(query)
            
            print(f"Assistant: {response['answer']}")
            
            if response.get('sources'):
                print("Sources:")
                for source in response['sources']:
                    print(f"  - {source}")
            
            if response.get('confidence_score'):
                print(f"Confidence: {response['confidence_score']:.2f}")
    
    except Exception as e:
        print(f"Note: Chatbot integration requires GROQ_API_KEY. Error: {e}")
        print("The multi-modal document processing system is ready for integration.")

def example_processing_pipeline():
    """Example: Complete document processing pipeline"""
    print("\n=== Complete Document Processing Pipeline ===")
    
    # Step 1: Document Processing
    print("Step 1: Processing documents...")
    processor = DocumentProcessor()
    
    # Simulate processing multiple document types
    documents = []
    
    # Text document
    text_docs = processor.process_text_content(
        "Machine learning is a subset of artificial intelligence.",
        source="text_input",
        metadata={"type": "definition"}
    )
    documents.extend(text_docs)
    
    # JSON document
    json_data = [
        {"title": "AI Basics", "content": "Artificial Intelligence overview"},
        {"title": "ML Fundamentals", "content": "Machine Learning principles"}
    ]
    json_content = str(json_data)  # Convert to string for processing
    json_docs = processor.process_text_content(
        json_content,
        source="json_data",
        metadata={"type": "structured_data"}
    )
    documents.extend(json_docs)
    
    print(f"Processed {len(documents)} documents")
    
    # Step 2: Intelligent Chunking
    print("Step 2: Intelligent chunking...")
    chunker = IntelligentChunker(chunk_size=200, chunk_overlap=30)
    
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(
            content=doc.content,
            doc_type=doc.doc_type,
            metadata=doc.metadata
        )
        all_chunks.extend(chunks)
    
    print(f"Generated {len(all_chunks)} chunks")
    
    # Step 3: Embedding Generation
    print("Step 3: Generating embeddings...")
    engine = create_embedding_engine("general_text")
    
    chunk_texts = [chunk.content for chunk in all_chunks]
    embedding_result = engine.generate_embeddings(chunk_texts)
    
    print(f"Generated {len(embedding_result.embeddings)} embeddings")
    print(f"Embedding dimension: {embedding_result.embeddings.shape[1]}")
    print(f"Processing time: {embedding_result.processing_time:.2f}s")
    
    # Step 4: Vector Store Integration
    print("Step 4: Adding to vector store...")
    vector_store = EnhancedVectorStore(
        embedding_config="general_text",
        vector_store_path="pipeline_vector_store"
    )
    
    # Add chunks as documents
    chunk_texts = [chunk.content for chunk in all_chunks]
    chunk_metadata = [chunk.metadata for chunk in all_chunks]
    
    chunks_added = vector_store.add_text_documents(chunk_texts, chunk_metadata)
    print(f"Added {chunks_added} chunks to vector store")
    
    # Step 5: Search and Retrieval
    print("Step 5: Testing search functionality...")
    
    search_query = "artificial intelligence machine learning"
    results = vector_store.search(search_query, k=3)
    
    print(f"Search results for '{search_query}':")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['content'][:100]}... (score: {result['score']:.4f})")

def main():
    """Main function to run all examples"""
    print("🚀 Multi-Modal Document Processing Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_document_processing()
        example_intelligent_chunking()
        example_enhanced_vector_store()
        example_integration_with_chatbot()
        example_processing_pipeline()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("The multi-modal document processing system is ready for use.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    main()