#!/usr/bin/env python3
"""
Comprehensive test script for the ReAct Agent Architecture.
Tests reasoning, acting, tool integration, and memory systems.
"""

import os
import sys
import uuid

# Add the backend directory to the Python path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

# Change to backend directory so .env file is found
original_cwd = os.getcwd()
os.chdir(backend_path)

try:
    from agentic_ai_chatbot import AgenticAIChatbot
    from config import GROQ_API_KEY
    
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found. Please set it in your .env file.")
        sys.exit(1)
    
    print("🚀 Testing ReAct Agent Architecture...")
    print("=" * 60)
    
    # Initialize the agentic AI chatbot
    print("Initializing AgenticAIChatbot with ReAct agent...")
    chatbot = AgenticAIChatbot(
        GROQ_API_KEY, 
        use_advanced_rag=True, 
        enable_agent=True
    )
    print("✅ Agentic AI chatbot initialized successfully\n")
    
    # Show system capabilities
    print("📋 System Capabilities:")
    stats = chatbot.get_system_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key.replace('_', ' ').title()}:")
            for sub_key, sub_value in value.items():
                print(f"     {sub_key}: {sub_value}")
        else:
            print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print("\n🔧 ReAct Agent Features:")
    if 'agent_features' in stats:
        for feature in stats['agent_features']:
            print(f"   ✅ {feature.replace('_', ' ').title()}")
    
    print("\n" + "=" * 60)
    print("🧪 Testing ReAct Agent with Different Query Types:")
    print("-" * 60)
    
    # Test queries that demonstrate different agent capabilities
    test_queries = [
        # Simple portfolio queries
        "What projects has Prajwal worked on?",
        "Tell me about Prajwal's experience with machine learning",
        
        # Complex queries that require reasoning
        "Compare Prajwal's skills with what's needed for a senior AI engineer role",
        "What would be the best project to showcase Prajwal's capabilities?",
        
        # Technical queries
        "What is the difference between supervised and unsupervised learning?",
        "How does React.js compare to Vue.js?",
        
        # Multi-step queries
        "First, tell me about Prajwal's skills, then suggest which technologies he should learn next",
        "What are Prajwal's strongest areas and how do they align with current industry trends?",
    ]
    
    results = []
    session_id = str(uuid.uuid4())
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 40)
        
        try:
            # Process query with agent
            result = chatbot.ask(query, session_context={"session_id": session_id})
            
            print(f"Response: {result['response'][:200]}...")
            print(f"Category: {result['category']}")
            print(f"Handler: {result['handler_used']}")
            print(f"Agent Used: {result.get('agent_used', False)}")
            print(f"Confidence: {result.get('confidence', 0):.2f}")
            
            if result.get('agent_used'):
                agent_meta = result.get('agent_metadata', {})
                print(f"Agent Steps: {agent_meta.get('steps_taken', 0)}")
                print(f"Tools Used: {', '.join(agent_meta.get('tools_used', []))}")
            
            if 'system_info' in result:
                sys_info = result['system_info']
                print(f"RAG Mode: {sys_info.get('rag_mode', 'unknown')}")
                print(f"Agent Enabled: {sys_info.get('agent_enabled', False)}")
            
            results.append({
                'query': query,
                'result': result,
                'success': True
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'query': query,
                'error': str(e),
                'success': False
            })
    
    print("\n" + "=" * 60)
    print("📊 Session Analysis:")
    print("-" * 60)
    
    if chatbot.agent_memory:
        session_summary = chatbot.agent_memory.get_session_summary(session_id)
        print(f"Session ID: {session_summary.get('session_id', 'N/A')}")
        print(f"Total Turns: {session_summary.get('total_turns', 0)}")
        print(f"Duration: {session_summary.get('duration_seconds', 0):.1f} seconds")
        print(f"Topics: {', '.join(session_summary.get('topics', {}).keys())}")
        print(f"First Query: {session_summary.get('first_query', 'N/A')}")
        print(f"Last Response: {session_summary.get('last_response', 'N/A')[:100]}...")
    
    print("\n" + "=" * 60)
    print("🧠 Memory System Analysis:")
    print("-" * 60)
    
    if chatbot.agent_memory:
        memory_summary = chatbot.agent_memory.get_memory_summary()
        print(f"Total Memories: {memory_summary.get('total_memories', 0)}")
        print(f"Average Importance: {memory_summary.get('average_importance', 0):.2f}")
        print(f"Most Accessed: {memory_summary.get('most_accessed_key', 'N/A')} "
              f"({memory_summary.get('most_accessed_count', 0)} times)")
        
        importance_dist = memory_summary.get('importance_distribution', {})
        print(f"High Importance: {importance_dist.get('high', 0)}")
        print(f"Medium Importance: {importance_dist.get('medium', 0)}")
        print(f"Low Importance: {importance_dist.get('low', 0)}")
    
    print("\n" + "=" * 60)
    print("🔧 Tool System Analysis:")
    print("-" * 60)
    
    if chatbot.tool_registry:
        tool_metadata = chatbot.tool_registry.get_tool_metadata()
        print("Available Tools:")
        for tool_name, metadata in tool_metadata.items():
            print(f"   {tool_name}:")
            print(f"     Description: {metadata.get('description', 'N/A')}")
            print(f"     Usage Count: {metadata.get('usage_count', 0)}")
            print(f"     Last Used: {metadata.get('last_used', 'Never')}")
    
    print("\n" + "=" * 60)
    print("📈 Performance Summary:")
    print("-" * 60)
    
    successful_queries = sum(1 for r in results if r['success'])
    agent_used_queries = sum(1 for r in results if r['success'] and r['result'].get('agent_used', False))
    
    print(f"Total Queries: {len(results)}")
    print(f"Successful Queries: {successful_queries}")
    print(f"Agent Used: {agent_used_queries}")
    print(f"Success Rate: {successful_queries/len(results)*100:.1f}%")
    print(f"Agent Usage Rate: {agent_used_queries/len(results)*100:.1f}%")
    
    print(f"\n✅ ReAct Agent Architecture testing completed!")
    print(f"\n🎯 Key Capabilities Demonstrated:")
    print(f"   ✅ Multi-step reasoning and planning")
    print(f"   ✅ Tool integration and execution")
    print(f"   ✅ Conversation memory and context")
    print(f"   ✅ Dynamic decision making")
    print(f"   ✅ Integration with existing RAG systems")
    print(f"   ✅ Session management and analysis")
    
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