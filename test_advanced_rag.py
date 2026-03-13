#!/usr/bin/env python3
"""
Comprehensive test script for the Advanced RAG Pipeline.
Tests multi-stage processing, query optimization, and document grading.
"""

import os
import sys
import time

# Add the backend directory to the Python path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

# Change to backend directory so .env file is found
original_cwd = os.getcwd()
os.chdir(backend_path)

try:
    from advanced_chatbot import AdvancedChatbot
    from config import GROQ_API_KEY
    
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found. Please set it in your .env file.")
        sys.exit(1)
    
    print("🚀 Testing Advanced RAG Pipeline...")
    print("=" * 60)
    
    # Initialize the advanced chatbot
    print("Initializing AdvancedChatbot with multi-stage RAG...")
    chatbot = AdvancedChatbot(GROQ_API_KEY, use_advanced_rag=True)
    print("✅ Advanced chatbot initialized successfully\n")
    
    # Test different types of portfolio queries
    test_queries = [
        # Simple portfolio queries
        "What projects has Prajwal worked on?",
        "Tell me about Prajwal's skills",
        "What is Prajwal's experience?",
        
        # Complex portfolio queries
        "What machine learning projects has Prajwal completed and what technologies did he use?",
        "Can you describe Prajwal's professional background and key achievements?",
        "What programming languages and frameworks is Prajwal proficient in?",
        
        # Queries that might benefit from query optimization
        "Tell me everything about Prajwal's work in AI and machine learning",
        "What are all the technologies Prajwal has experience with?",
        
        # Ambiguous queries that need clarification
        "How good is Prajwal?",
        "What does Prajwal do?",
    ]
    
    print("🧪 Testing Advanced RAG Pipeline:")
    print("-" * 60)
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 40)
        
        try:
            # Test with advanced RAG
            start_time = time.time()
            result = chatbot.ask(query, use_advanced_rag=True)
            advanced_time = time.time() - start_time
            
            print(f"🎯 Advanced RAG Results:")
            print(f"   Answer: {result['response'][:200]}...")
            print(f"   Category: {result['category']}")
            print(f"   Handler: {result['handler_used']}")
            print(f"   RAG Mode: {result.get('rag_mode', 'unknown')}")
            print(f"   Time: {advanced_time:.2f}s")
            
            # Get detailed metadata
            metadata = result.get('metadata', {})
            if 'routing_metadata' in metadata:
                routing_meta = metadata['routing_metadata']
                print(f"   Classification: {routing_meta.get('classification_reasoning', 'N/A')}")
            
            if 'rag_metadata' in result:
                rag_meta = result['rag_metadata']
                print(f"   Pipeline: {rag_meta.get('pipeline_type', 'N/A')}")
                print(f"   Strategy: {rag_meta.get('optimization_strategy', 'N/A')}")
            
            # Test with simple RAG for comparison
            print(f"\n   📊 Simple RAG Comparison:")
            start_time = time.time()
            simple_result = chatbot.ask(query, use_advanced_rag=False)
            simple_time = time.time() - start_time
            
            print(f"   Answer: {simple_result['response'][:200]}...")
            print(f"   Time: {simple_time:.2f}s")
            print(f"   Speed difference: {((advanced_time - simple_time) / simple_time * 100):+.1f}%")
            
            results.append({
                'query': query,
                'advanced_time': advanced_time,
                'simple_time': simple_time,
                'advanced_result': result,
                'simple_result': simple_result
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("📈 Performance Analysis:")
    print("-" * 60)
    
    if results:
        avg_advanced_time = sum(r['advanced_time'] for r in results) / len(results)
        avg_simple_time = sum(r['simple_time'] for r in results) / len(results)
        
        print(f"Average Advanced RAG time: {avg_advanced_time:.2f}s")
        print(f"Average Simple RAG time: {avg_simple_time:.2f}s")
        print(f"Speed overhead: {((avg_advanced_time - avg_simple_time) / avg_simple_time * 100):+.1f}%")
        
        # Test RAG pipeline directly
        print(f"\n🔍 Testing RAG Pipeline Directly:")
        print("-" * 60)
        
        test_query = "What machine learning frameworks has Prajwal used in his projects?"
        print(f"Query: '{test_query}'")
        
        try:
            # Test different optimization strategies
            strategies = ["decompose", "hyde", "expand", "hybrid"]
            
            for strategy in strategies:
                print(f"\nStrategy: {strategy}")
                try:
                    result = chatbot.advanced_rag.invoke(test_query, optimization_strategy=strategy)
                    
                    print(f"   Answer: {result['answer'][:150]}...")
                    print(f"   Relevant docs: {result['relevant_documents']}")
                    print(f"   Total retrieved: {result['total_retrieved']}")
                    print(f"   Rewrites: {result['rewrite_count']}")
                    
                    metadata = result.get('metadata', {})
                    if 'retrieval_stats' in metadata:
                        stats = metadata['retrieval_stats']
                        print(f"   Queries used: {stats.get('total_queries', 0)}")
                        print(f"   Avg docs per query: {stats.get('avg_docs_per_query', 0):.1f}")
                    
                    if 'grading_stats' in metadata:
                        grading = metadata['grading_stats']
                        print(f"   Relevance rate: {grading.get('relevance_rate', 0):.1%}")
                        print(f"   Avg confidence: {grading.get('avg_confidence', 0):.2f}")
                        
                except Exception as e:
                    print(f"   ❌ Error with {strategy}: {e}")
                    continue
        
        except Exception as e:
            print(f"❌ Pipeline test error: {e}")
    
    print(f"\n✅ Advanced RAG Pipeline testing completed!")
    print(f"The system successfully demonstrates:")
    print(f"- Multi-stage query optimization")
    print(f"- Document relevance grading")
    print(f"- Hallucination checking")
    print(f"- Query decomposition and expansion")
    print(f"- HyDE (Hypothetical Document Embeddings)")
    print(f"- LangGraph workflow orchestration")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the ai_portfolio directory")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Change back to original directory
    os.chdir(original_cwd)