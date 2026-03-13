#!/usr/bin/env python3
"""
Test script for multi-modal document processing system
Tests document processing, chunking, embedding, and vector store functionality
"""

import os
import sys
import tempfile
import json
import csv
from pathlib import Path
from typing import Dict, Any, List

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.src.rag.document_processor import DocumentProcessor, Document
from backend.src.rag.chunking import IntelligentChunker, ChunkingStrategy
from backend.src.rag.embedding import create_embedding_engine
from backend.enhanced_vector_store import EnhancedVectorStore

def create_test_files(temp_dir: str) -> Dict[str, str]:
    """Create test files of different formats"""
    test_files = {}
    
    # Create text file
    text_content = """
    Artificial Intelligence and Machine Learning
    
    AI and ML are transforming industries across the globe. Machine learning algorithms 
    can analyze vast amounts of data to identify patterns and make predictions. 
    Deep learning, a subset of ML, uses neural networks to solve complex problems.
    
    Natural Language Processing
    
    NLP enables computers to understand and generate human language. Applications 
    include chatbots, language translation, and sentiment analysis. Modern NLP uses 
    transformer architectures like BERT and GPT.
    
    Computer Vision
    
    Computer vision allows machines to interpret and understand visual information 
    from the world. Applications include facial recognition, medical imaging, and 
    autonomous vehicles.
    """
    
    text_file = os.path.join(temp_dir, "test_document.txt")
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(text_content)
    test_files['text'] = text_file
    
    # Create JSON file
    json_data = [
        {
            "title": "Machine Learning Basics",
            "category": "AI/ML",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "tags": ["AI", "ML", "basics"],
            "difficulty": "beginner"
        },
        {
            "title": "Deep Learning Fundamentals", 
            "category": "AI/ML",
            "content": "Deep learning uses neural networks with multiple layers to solve complex problems.",
            "tags": ["AI", "deep learning", "neural networks"],
            "difficulty": "intermediate"
        },
        {
            "title": "Natural Language Processing",
            "category": "AI/ML", 
            "content": "NLP combines computational linguistics with machine learning to process human language.",
            "tags": ["AI", "NLP", "language processing"],
            "difficulty": "intermediate"
        }
    ]
    
    json_file = os.path.join(temp_dir, "test_data.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    test_files['json'] = json_file
    
    # Create CSV file
    csv_data = [
        ["Technology", "Description", "Use Case", "Industry"],
        ["Machine Learning", "Algorithms that learn from data", "Predictive analytics", "Finance"],
        ["Computer Vision", "Visual data interpretation", "Quality control", "Manufacturing"],
        ["Natural Language Processing", "Human language understanding", "Chatbots", "Customer Service"],
        ["Robotics", "Automated physical tasks", "Assembly lines", "Automotive"],
        ["Blockchain", "Distributed ledger technology", "Supply chain tracking", "Logistics"]
    ]
    
    csv_file = os.path.join(temp_dir, "test_data.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    test_files['csv'] = csv_file
    
    # Create Markdown file
    md_content = """
# AI Technology Overview

## Machine Learning
Machine learning enables computers to learn from data without explicit programming. Key concepts include:

- **Supervised Learning**: Learning from labeled data
- **Unsupervised Learning**: Finding patterns in unlabeled data  
- **Reinforcement Learning**: Learning through interaction and rewards

## Deep Learning
Deep learning uses artificial neural networks with multiple layers.

### Neural Network Architecture
```
Input Layer -> Hidden Layers -> Output Layer
```

### Applications
- Image recognition
- Speech processing
- Natural language understanding

## Computer Vision
Computer vision enables machines to interpret visual information.

#### Key Techniques
1. **Image Classification**: Categorizing images
2. **Object Detection**: Locating objects in images
3. **Image Segmentation**: Partitioning images into regions

> "The future of AI is not about replacing humans, but augmenting human capabilities."

---
*Last updated: 2024*
"""
    
    md_file = os.path.join(temp_dir, "test_document.md")
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    test_files['markdown'] = md_file
    
    # Create HTML file
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Technologies Guide</title>
    <meta name="description" content="Comprehensive guide to artificial intelligence technologies">
</head>
<body>
    <h1>Artificial Intelligence Technologies</h1>
    
    <h2>Machine Learning</h2>
    <p>Machine learning is a method of data analysis that automates analytical model building.</p>
    
    <h3>Types of Machine Learning</h3>
    <ul>
        <li>Supervised Learning</li>
        <li>Unsupervised Learning</li>
        <li>Reinforcement Learning</li>
    </ul>
    
    <h2>Deep Learning</h2>
    <p>Deep learning is a subset of machine learning that uses neural networks with multiple layers.</p>
    
    <h3>Neural Network Types</h3>
    <ul>
        <li>CNNs for image processing</li>
        <li>RNNs for sequential data</li>
        <li>Transformers for language tasks</li>
    </ul>
    
    <h2>Computer Vision</h2>
    <p>Computer vision enables machines to interpret and understand visual information from the world.</p>
</body>
</html>
"""
    
    html_file = os.path.join(temp_dir, "test_page.html")
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    test_files['html'] = html_file
    
    return test_files

def test_document_processor():
    """Test document processor with different file types"""
    print("=== Testing Document Processor ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_files = create_test_files(temp_dir)
        processor = DocumentProcessor()
        
        results = {}
        
        for file_type, file_path in test_files.items():
            try:
                print(f"\nProcessing {file_type} file: {file_path}")
                
                documents = processor.process_file(file_path)
                
                print(f"  Generated {len(documents)} documents")
                
                for i, doc in enumerate(documents[:2]):  # Show first 2 documents
                    print(f"  Document {i+1}:")
                    print(f"    Type: {doc.doc_type}")
                    print(f"    Content length: {len(doc.content)} chars")
                    print(f"    Metadata keys: {list(doc.metadata.keys())}")
                    if len(doc.content) < 200:
                        print(f"    Content preview: {doc.content[:100]}...")
                
                results[file_type] = {
                    "document_count": len(documents),
                    "success": True
                }
                
            except Exception as e:
                print(f"  Error processing {file_type}: {e}")
                results[file_type] = {
                    "document_count": 0,
                    "success": False,
                    "error": str(e)
                }
        
        return results

def test_chunking_strategies():
    """Test different chunking strategies"""
    print("\n=== Testing Chunking Strategies ===")
    
    # Create sample text
    sample_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.

    Machine Learning (ML) is a subset of AI that focuses on developing algorithms that can learn from and make predictions or decisions based on data. ML algorithms build models based on training data to make predictions or decisions without being explicitly programmed to perform the task.

    Deep Learning is a subset of machine learning that uses neural networks with multiple layers (hence "deep") to progressively extract higher-level features from raw input. For example, in image processing, lower layers may identify edges, while higher layers may identify human-relevant concepts like letters or faces.

    Natural Language Processing (NLP) is another crucial area of AI that deals with the interaction between computers and human language. It involves programming computers to process and analyze large amounts of natural language data. The goal is to enable computers to understand, interpret, and generate human language in a valuable way.

    Computer Vision enables machines to interpret and understand visual information from the world. It involves developing algorithms that can automatically perform visual tasks that the human visual system can do, such as object recognition, image classification, and scene understanding.
    """
    
    chunker = IntelligentChunker(chunk_size=200, chunk_overlap=50)
    
    strategies = [
        ChunkingStrategy.RECURSIVE,
        ChunkingStrategy.CHARACTER,
        ChunkingStrategy.SEMANTIC,
        ChunkingStrategy.STRUCTURAL
    ]
    
    results = {}
    
    for strategy in strategies:
        try:
            print(f"\nTesting {strategy.value} chunking:")
            
            chunks = chunker.chunk_document(
                content=sample_text,
                strategy=strategy
            )
            
            print(f"  Generated {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                print(f"  Chunk {i+1}:")
                print(f"    Word count: {chunk.word_count}")
                print(f"    Char count: {chunk.char_count}")
                print(f"    Content preview: {chunk.content[:100]}...")
            
            results[strategy.value] = {
                "chunk_count": len(chunks),
                "avg_chunk_size": sum(c.char_count for c in chunks) / len(chunks) if chunks else 0,
                "success": True
            }
            
        except Exception as e:
            print(f"  Error with {strategy.value}: {e}")
            results[strategy.value] = {
                "chunk_count": 0,
                "success": False,
                "error": str(e)
            }
    
    return results

def test_embedding_engine():
    """Test embedding engine"""
    print("\n=== Testing Embedding Engine ===")
    
    try:
        # Create embedding engine
        engine = create_embedding_engine("general_text")
        
        # Test texts
        texts = [
            "Artificial Intelligence is transforming the world.",
            "Machine learning algorithms can learn from data.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing enables computers to understand human language.",
            "Computer vision allows machines to interpret visual information."
        ]
        
        print("Generating embeddings for test texts...")
        
        result = engine.generate_embeddings(texts)
        
        print(f"Generated {len(result.embeddings)} embeddings")
        print(f"Embedding dimension: {result.embeddings.shape[1]}")
        print(f"Processing time: {result.processing_time:.2f} seconds")
        print(f"Cache hit: {result.cache_hit}")
        
        # Test similarity search
        print("\nTesting similarity search:")
        
        query_embedding = engine.generate_single_embedding("AI and machine learning")
        
        similarities = engine.similarity_search(
            query_embedding=query_embedding,
            document_embeddings=result.embeddings,
            top_k=3
        )
        
        print("Top 3 similar texts:")
        for idx, score in similarities:
            print(f"  Text {idx+1}: '{texts[idx]}' (score: {score:.4f})")
        
        return {
            "embedding_count": len(result.embeddings),
            "embedding_dimension": result.embeddings.shape[1],
            "processing_time": result.processing_time,
            "similarity_results": len(similarities),
            "success": True
        }
        
    except Exception as e:
        print(f"Error testing embedding engine: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def test_enhanced_vector_store():
    """Test enhanced vector store with multi-modal documents"""
    print("\n=== Testing Enhanced Vector Store ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_files = create_test_files(temp_dir)
        
        try:
            # Create vector store
            vector_store = EnhancedVectorStore(
                embedding_config="general_text",
                vector_store_path=os.path.join(temp_dir, "test_vector_store"),
                chunk_size=300,
                chunk_overlap=50
            )
            
            # Add documents from files
            print("Adding documents from files...")
            file_paths = list(test_files.values())
            
            stats = vector_store.add_documents_from_files(file_paths)
            
            print(f"Processing statistics:")
            print(f"  Total files: {stats['total_files']}")
            print(f"  Processed files: {stats['processed_files']}")
            print(f"  Failed files: {stats['failed_files']}")
            print(f"  Total documents: {stats['total_documents']}")
            print(f"  Total chunks: {stats['total_chunks']}")
            
            if stats['errors']:
                print(f"  Errors: {stats['errors']}")
            
            # Test search functionality
            print("\nTesting search functionality:")
            
            test_queries = [
                "machine learning algorithms",
                "neural networks and deep learning",
                "computer vision applications"
            ]
            
            search_results = {}
            
            for query in test_queries:
                print(f"\nSearching for: '{query}'")
                
                results = vector_store.search(query, k=3)
                
                print(f"  Found {len(results)} results")
                
                for i, result in enumerate(results):
                    print(f"  Result {i+1}:")
                    print(f"    Source: {result['source']}")
                    print(f"    Type: {result['doc_type']}")
                    print(f"    Score: {result['score']:.4f}")
                    print(f"    Content preview: {result['content'][:100]}...")
                
                search_results[query] = len(results)
            
            # Get vector store stats
            print("\nVector store statistics:")
            store_stats = vector_store.get_stats()
            
            print(f"  Total documents: {store_stats['total_documents']}")
            print(f"  Total chunks: {store_stats['total_chunks']}")
            print(f"  Embedding dimension: {store_stats['embedding_dimension']}")
            print(f"  Status: {store_stats['status']}")
            
            if 'document_types' in store_stats:
                print(f"  Document types: {store_stats['document_types']}")
            
            return {
                "processing_stats": stats,
                "search_results": search_results,
                "store_stats": store_stats,
                "success": True
            }
            
        except Exception as e:
            print(f"Error testing vector store: {e}")
            return {
                "success": False,
                "error": str(e)
            }

def test_web_content_processing():
    """Test web content processing"""
    print("\n=== Testing Web Content Processing ===")
    
    try:
        processor = DocumentProcessor()
        
        # Test with a simple web page (using a data URL for testing)
        test_url = "https://httpbin.org/html"  # Simple HTML endpoint for testing
        
        print(f"Processing web content from: {test_url}")
        
        documents = processor.process_url(test_url, include_metadata=True)
        
        print(f"Generated {len(documents)} documents from web content")
        
        for i, doc in enumerate(documents):
            print(f"  Document {i+1}:")
            print(f"    Type: {doc.doc_type}")
            print(f"    Content length: {len(doc.content)} chars")
            print(f"    Metadata: {doc.metadata}")
        
        return {
            "document_count": len(documents),
            "success": True
        }
        
    except Exception as e:
        print(f"Error testing web content processing: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def run_all_tests():
    """Run all tests and generate summary report"""
    print("🚀 Starting Multi-Modal Document Processing Tests")
    print("=" * 60)
    
    all_results = {}
    
    # Run tests
    all_results['document_processor'] = test_document_processor()
    all_results['chunking_strategies'] = test_chunking_strategies()
    all_results['embedding_engine'] = test_embedding_engine()
    all_results['enhanced_vector_store'] = test_enhanced_vector_store()
    all_results['web_content_processing'] = test_web_content_processing()
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY REPORT")
    print("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    
    for test_name, results in all_results.items():
        print(f"\n{test_name.upper()}:")
        
        if isinstance(results, dict):
            if results.get('success', False):
                print("  ✅ PASSED")
                passed_tests += 1
                
                # Show key metrics
                if 'document_processor' in test_name:
                    for file_type, file_results in results.items():
                        if isinstance(file_results, dict) and 'document_count' in file_results:
                            status = "✅" if file_results.get('success') else "❌"
                            print(f"    {status} {file_type}: {file_results['document_count']} documents")
                
                elif 'chunking_strategies' in test_name:
                    for strategy, strategy_results in results.items():
                        if isinstance(strategy_results, dict) and 'chunk_count' in strategy_results:
                            status = "✅" if strategy_results.get('success') else "❌"
                            print(f"    {status} {strategy}: {strategy_results['chunk_count']} chunks")
                
                else:
                    # Show general success metrics
                    for key, value in results.items():
                        if key not in ['success', 'error'] and isinstance(value, (int, float, str)):
                            print(f"    {key}: {value}")
            
            else:
                print("  ❌ FAILED")
                if 'error' in results:
                    print(f"    Error: {results['error']}")
        
        total_tests += 1
    
    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Multi-modal document processing system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    print("=" * 60)
    
    return all_results

if __name__ == "__main__":
    # Run all tests
    results = run_all_tests()
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if all(r.get('success', False) for r in results.values()) else 1)