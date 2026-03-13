#!/usr/bin/env python3
"""
Test script for the Intelligent Query Routing System.
Tests various query types and verifies correct classification and routing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.enhanced_chatbot import EnhancedChatbot
from backend.config import GROQ_API_KEY

def test_query_routing():
    """Test the query routing system with various query types."""
    
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found. Please set it in your .env file.")
        return
    
    print("🚀 Initializing Enhanced Chatbot with Intelligent Routing...")
    
    try:
        chatbot = EnhancedChatbot(GROQ_API_KEY)
        print("✅ Enhanced chatbot initialized successfully\n")
        
        # Test queries organized by expected category
        test_queries = [
            # Greeting queries
            ("Hello!", "GREETING"),
            ("Hi there", "GREETING"),
            ("Good morning", "GREETING"),
            
            # Portfolio queries
            ("What projects has Prajwal worked on?", "PORTFOLIO"),
            ("Tell me about Prajwal's skills", "PORTFOLIO"),
            ("What experience does Prajwal have?", "PORTFOLIO"),
            ("Show me Prajwal's certifications", "PORTFOLIO"),
            
            # Technical queries
            ("What is React.js?", "TECHNICAL"),
            ("How does machine learning work?", "TECHNICAL"),
            ("Explain Python decorators", "TECHNICAL"),
            
            # General queries
            ("What is the capital of France?", "GENERAL"),
            ("How tall is Mount Everest?", "GENERAL"),
            
            # Goodbye queries
            ("Bye!", "GOODBYE"),
            ("Thank you", "GOODBYE"),
            ("See you later", "GOODBYE"),
            
            # Unclear queries
            ("asdf", "UNCLEAR"),
            ("", "UNCLEAR"),
            ("???", "UNCLEAR"),
            
            # Out of scope queries
            ("What's the weather like?", "OUT_OF_SCOPE"),
            ("Tell me a joke", "OUT_OF_SCOPE"),
            ("What should I eat for dinner?", "OUT_OF_SCOPE"),
        ]
        
        print("🧪 Testing Query Classification and Routing:\n")
        
        passed_tests = 0
        total_tests = len(test_queries)
        
        for query, expected_category in test_queries:
            print(f"Query: '{query}'")
            print(f"Expected: {expected_category}")
            
            try:
                # Get classification
                classification = chatbot.router.classifier.classify(query)
                actual_category = classification.category.value
                confidence = classification.confidence
                
                print(f"Classified: {actual_category} (confidence: {confidence:.2f})")
                print(f"Reasoning: {classification.reasoning}")
                
                # Test routing
                route_decision = chatbot.router.route(query)
                print(f"Routed to: {route_decision.handler_name}")
                print(f"Use RAG: {route_decision.should_use_rag}")
                
                # Check if classification matches expected
                if actual_category == expected_category:
                    print("✅ PASS")
                    passed_tests += 1
                else:
                    print("❌ FAIL - Classification mismatch")
                
                print("-" * 50)
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
                print("-" * 50)
        
        # Test full responses for a few queries
        print("\n🎯 Testing Full Responses:\n")
        
        sample_queries = [
            "Hello!",
            "What projects has Prajwal worked on?", 
            "What is Python?",
            "What's the weather?",
            "Bye!"
        ]
        
        for query in sample_queries:
            print(f"Query: '{query}'")
            try:
                result = chatbot.ask(query)
                print(f"Response: {result['response'][:100]}...")
                print(f"Category: {result['category']}")
                print(f"Handler: {result['handler_used']}")
                print(f"Used RAG: {result.get('used_rag', False)}")
                print("-" * 50)
            except Exception as e:
                print(f"❌ ERROR: {e}")
                print("-" * 50)
        
        # Summary
        print(f"\n📊 Test Summary:")
        print(f"Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! The routing system is working correctly.")
        else:
            print(f"⚠️  {total_tests - passed_tests} tests failed. Review the classifications above.")
            
    except Exception as e:
        print(f"❌ Failed to initialize enhanced chatbot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_query_routing()