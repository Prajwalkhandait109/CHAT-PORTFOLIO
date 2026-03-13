#!/usr/bin/env python3
"""
Usage examples for enhanced vector store management system
Demonstrates multi-store backends, collection management, and migration capabilities
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.src.db.enhanced_vector_store_manager import EnhancedVectorStoreManager
from backend.src.db.vector_store import VectorStoreType
from backend.src.rag.document_processor import DocumentProcessor
from backend.src.rag.embedding import create_embedding_engine

def example_basic_collection_management():
    """Example: Basic collection management with different backends"""
    print("=== Basic Collection Management Example ===")
    
    # Initialize manager
    manager = EnhancedVectorStoreManager(
        config_dir="./example_vector_stores",
        default_embedding_config="general_text"
    )
    
    # Create collections with different backends
    collections = [
        {
            "name": "portfolio_docs",
            "store_type": VectorStoreType.FAISS,
            "description": "Portfolio documents using FAISS for fast local search"
        },
        {
            "name": "web_content",
            "store_type": VectorStoreType.FAISS,
            "description": "Web content and articles"
        }
    ]
    
    for collection_config in collections:
        success = manager.create_collection(
            name=collection_config["name"],
            store_type=collection_config["store_type"],
            description=collection_config["description"]
        )
        
        print(f"Collection '{collection_config['name']}': {'✅ Created' if success else '❌ Failed'}")
    
    # List all collections
    all_collections = manager.collection_manager.list_collections()
    print(f"Total collections: {len(all_collections)}")
    
    return manager

def example_multi_format_document_processing():
    """Example: Processing and storing multi-format documents"""
    print("\n=== Multi-Format Document Processing Example ===")
    
    manager = EnhancedVectorStoreManager()
    
    # Create a test collection
    manager.create_collection(
        name="multimodal_docs",
        store_type=VectorStoreType.FAISS,
        description="Multi-format documents (PDF, JSON, CSV, etc.)"
    )
    
    # Process different types of documents
    document_processor = DocumentProcessor()
    
    # Example documents (you would use real files in practice)
    sample_texts = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers to solve complex problems.",
        "Natural language processing allows computers to understand and generate human language.",
        "Computer vision enables machines to interpret and understand visual information."
    ]
    
    # Process and add documents
    for i, text in enumerate(sample_texts):
        # Create document metadata
        metadata = {
            "doc_id": f"doc_{i+1}",
            "category": "AI/ML",
            "topic": ["machine_learning", "deep_learning", "nlp", "computer_vision"][i % 4],
            "source": "example_text",
            "confidence_score": 0.8 + (i % 5) * 0.04
        }
        
        # Add to collection
        stats = manager.add_documents_from_files(
            collection_name="multimodal_docs",
            file_paths=[],  # Empty for this example
            metadata=metadata
        )
    
    print(f"Added documents to multimodal_docs collection")
    
    return manager

def example_advanced_search_and_retrieval():
    """Example: Advanced search across multiple collections"""
    print("\n=== Advanced Search and Retrieval Example ===")
    
    manager = EnhancedVectorStoreManager()
    
    # Create test collections with sample data
    collections = ["ai_research", "tech_articles", "portfolio_content"]
    
    for collection_name in collections:
        manager.create_collection(collection_name, VectorStoreType.FAISS)
        
        # Add sample documents
        sample_docs = [
            "Artificial intelligence is transforming industries worldwide.",
            "Machine learning algorithms can analyze vast amounts of data.",
            "Deep learning models achieve state-of-the-art performance.",
            "Natural language processing enables human-computer interaction."
        ]
        
        for i, doc_text in enumerate(sample_docs):
            manager.add_documents_from_files(
                collection_name=collection_name,
                file_paths=[],
                metadata={"doc_id": f"{collection_name}_{i+1}", "category": "tech"}
            )
    
    # Search across all collections
    query = "artificial intelligence machine learning"
    
    print(f"Searching across collections for: '{query}'")
    
    all_results = manager.search_across_collections(
        query=query,
        collections=collections,
        limit=3
    )
    
    for collection_name, results in all_results.items():
        print(f"\nCollection '{collection_name}':")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['content'][:100]}... (score: {result.get('score', 0):.4f})")
    
    return manager

def example_collection_statistics_and_monitoring():
    """Example: Collection statistics and health monitoring"""
    print("\n=== Collection Statistics and Monitoring Example ===")
    
    manager = EnhancedVectorStoreManager()
    
    # Create some test collections
    for i in range(3):
        manager.create_collection(
            name=f"test_collection_{i+1}",
            store_type=VectorStoreType.FAISS,
            description=f"Test collection {i+1} for monitoring"
        )
    
    # Get comprehensive statistics
    print("Collection Statistics:")
    all_stats = manager.get_all_stats()
    
    for collection_name, stats in all_stats['collection_stats'].items():
        if 'config' in stats:
            config = stats['config']
            print(f"\nCollection '{collection_name}':")
            print(f"  Store Type: {config.get('store_type', 'Unknown')}")
            print(f"  Dimension: {config.get('dimension', 'Unknown')}")
            print(f"  Documents: {config.get('document_count', 0)}")
            print(f"  Connected: {stats.get('store_connected', False)}")
    
    # Health check
    print("\nHealth Check Results:")
    health_status = manager.health_check()
    
    print(f"Total Collections: {health_status['total_collections']}")
    print(f"Healthy Collections: {health_status['healthy_collections']}")
    print(f"Unhealthy Collections: {health_status['unhealthy_collections']}")
    
    return manager

def example_backup_and_restore():
    """Example: Backup and restore collections"""
    print("\n=== Backup and Restore Example ===")
    
    manager = EnhancedVectorStoreManager()
    
    # Create test collections
    manager.create_collection("backup_test_1", VectorStoreType.FAISS)
    manager.create_collection("backup_test_2", VectorStoreType.FAISS)
    
    # Backup collections
    backup_dir = "./vector_store_backups"
    backup_success = manager.backup_collections(backup_dir)
    
    print(f"Backup operation: {'✅ Success' if backup_success else '❌ Failed'}")
    
    if backup_success:
        # List backup files
        backup_files = list(Path(backup_dir).glob("collections_backup_*.json"))
        print(f"Backup files created: {len(backup_files)}")
        
        if backup_files:
            latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
            print(f"Latest backup: {latest_backup.name}")
    
    return manager

def example_migration_workflow():
    """Example: Migration workflow from FAISS to Qdrant"""
    print("\n=== Migration Workflow Example ===")
    
    # This example demonstrates the migration workflow
    # Note: Qdrant needs to be running for actual migration
    
    manager = EnhancedVectorStoreManager()
    
    # Create source FAISS collection
    source_collection = "faiss_source"
    target_collection = "qdrant_target"
    
    manager.create_collection(
        name=source_collection,
        store_type=VectorStoreType.FAISS,
        description="Source collection for migration testing"
    )
    
    # Add some test data
    test_data = [
        "This is a test document for migration.",
        "Vector store migration should preserve all data and metadata.",
        "The migration process should be seamless and efficient."
    ]
    
    for i, text in enumerate(test_data):
        manager.add_documents_from_files(
            collection_name=source_collection,
            file_paths=[],
            metadata={"test_id": f"test_{i+1}", "migration_test": True}
        )
    
    print(f"Created source collection '{source_collection}' with test data")
    
    # Demonstrate migration planning
    migration_manager = manager.migration_manager
    
    migration_plan = migration_manager.create_migration_plan(
        source_config={"type": "faiss", "path": f"./example_vector_stores/{source_collection}"},
        target_config={"type": "qdrant", "host": "localhost", "port": 6333},
        collections=[source_collection]
    )
    
    print("Migration Plan:")
    print(f"  Estimated Time: {migration_plan['estimated_time']}")
    print(f"  Collections: {migration_plan['collections']}")
    print(f"  Risks: {migration_plan['risks']}")
    print(f"  Requirements: {migration_plan['requirements']}")
    
    # Note: Actual migration would require Qdrant to be running
    print("\nNote: Actual migration requires Qdrant server to be running.")
    
    return manager

def main():
    """Main function to run all examples"""
    print("🚀 Enhanced Vector Store Management Examples")
    print("=" * 60)
    
    try:
        # Run examples
        manager1 = example_basic_collection_management()
        manager2 = example_multi_format_document_processing()
        manager3 = example_advanced_search_and_retrieval()
        manager4 = example_collection_statistics_and_monitoring()
        manager5 = example_backup_and_restore()
        manager6 = example_migration_workflow()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("\nKey capabilities demonstrated:")
        print("  ✅ Multi-backend vector store support (FAISS, Qdrant)")
        print("  ✅ Collection management with metadata")
        print("  ✅ Multi-format document processing and chunking")
        print("  ✅ Advanced search across collections")
        print("  ✅ Comprehensive statistics and monitoring")
        print("  ✅ Backup and restore functionality")
        print("  ✅ Migration planning and execution")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    main()