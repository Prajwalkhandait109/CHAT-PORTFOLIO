#!/usr/bin/env python3
"""
Simple test script for the Intelligent Query Routing System.
"""

import os
import sys

# Add the backend directory to the Python path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

# Change to backend directory so .env file is found
original_cwd = os.getcwd()
os.chdir(backend_path)

try:
    from enhanced_chatbot import EnhancedChatbot
    from config import GROQ_API_KEY
    
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found. Please set it in your .env file.")
        sys.exit(1)
    
    print("🚀 Testing Enhanced Chatbot with Intelligent Routing...")
    
    # Initialize the chatbot
    chatbot = EnhancedChatbot(GROQ_API_KEY)
    print("✅ Enhanced chatbot initialized successfully\n")
    
    # Test a few sample queries
    test_queries = [
        "Hello!",
        "What projects has Prajwal worked on?",
        "What is Python?",
        "What's the weather like?",
        "Bye!"
    ]
    
    print("🧪 Testing Query Classification and Responses:\n")
    
    for query in test_queries:
        print(f"Query: '{query}'")
        try:
            result = chatbot.ask(query)
            print(f"Response: {result['response'][:100]}...")
            print(f"Category: {result['category']}")
            print(f"Handler: {result['handler_used']}")
            print(f"Used RAG: {result.get('used_rag', False)}")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Error: {e}")
            print("-" * 50)
    
    print("✅ Test completed successfully!")
    
    # Change back to original directory
    os.chdir(original_cwd)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the ai_portfolio directory")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()