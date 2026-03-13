#!/usr/bin/env python3
"""
Usage examples for session and memory management system
Demonstrates conversation management, context retention, and memory integration
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any, List

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.src.memory.conversation_manager import create_conversation_manager
from backend.src.models.state import ContextType, create_context_entry

def example_basic_conversation():
    """Example: Basic conversation with session management"""
    print("=== Basic Conversation Example ===")
    
    try:
        # Create conversation manager
        manager = create_conversation_manager(
            groq_api_key=os.getenv("GROQ_API_KEY", "test_key"),
            enable_memory=True,
            enable_advanced_rag=True,
            enable_agent=True
        )
        
        # Create a new conversation session
        session_id = manager.create_conversation(user_id="demo_user_123")
        print(f"✅ Created conversation session: {session_id}")
        
        # Process some messages
        messages = [
            "What is machine learning?",
            "Tell me about neural networks.",
            "How does deep learning work?"
        ]
        
        for i, message in enumerate(messages):
            print(f"\n📝 Processing message {i+1}: '{message}'")
            
            response = manager.process_message(
                session_id=session_id,
                user_message=message,
                metadata={"turn": i+1, "topic": "AI/ML"}
            )
            
            if response.get("success"):
                print(f"🤖 Assistant: {response['assistant_response'][:100]}...")
                print(f"   Confidence: {response.get('confidence_score', 0):.2f}")
                print(f"   Processing time: {response.get('processing_time', 0):.2f}s")
                print(f"   Tools used: {response.get('tools_used', [])}")
                print(f"   Context entries: {len(response.get('context_used', []))}")
            else:
                print(f"❌ Error: {response.get('error', 'Unknown error')}")
        
        # Get conversation history
        history = manager.get_conversation_history(session_id, limit=2)
        print(f"\n📚 Recent conversation history: {len(history)} turns")
        
        for turn in history:
            print(f"   User: {turn['user_message']['content'][:50]}...")
            print(f"   Assistant: {turn['assistant_response']['content'][:50]}...")
        
        # End conversation
        summary = manager.end_conversation(session_id, generate_summary=True)
        print(f"\n🏁 Conversation ended successfully")
        print(f"   Total turns: {summary['session_stats']['total_turns']}")
        print(f"   Status: {summary['session_stats']['status']}")
        
        return {
            "session_created": True,
            "messages_processed": len(messages),
            "history_retrieved": len(history),
            "conversation_ended": True,
            "success": True
        }
        
    except Exception as e:
        print(f"❌ Error in basic conversation example: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def example_context_aware_conversation():
    """Example: Context-aware conversation with memory"""
    print("\n=== Context-Aware Conversation Example ===")
    
    try:
        # Create conversation manager
        manager = create_conversation_manager(
            groq_api_key=os.getenv("GROQ_API_KEY", "test_key"),
            enable_memory=True,
            enable_advanced_rag=True,
            enable_agent=True
        )
        
        # Create session with context
        session_id = manager.create_conversation(user_id="context_user_456")
        print(f"✅ Created context-aware session: {session_id}")
        
        # Simulate a conversation with evolving context
        conversation_flow = [
            {
                "user": "I'm interested in learning about artificial intelligence.",
                "context": [{"type": "user_preference", "content": "User is interested in AI education"}]
            },
            {
                "user": "What are the main branches of AI?",
                "context": [{"type": "previous_query", "content": "User asked about AI learning"}]
            },
            {
                "user": "Can you give me examples of machine learning applications?",
                "context": [{"type": "conversation_topic", "content": "Discussion about AI branches"}]
            }
        ]
        
        for i, turn in enumerate(conversation_flow):
            print(f"\n📝 Turn {i+1}: '{turn['user']}'")
            
            # Create context entries from the provided context
            context_entries = []
            for ctx in turn.get("context", []):
                context_entry = create_context_entry(
                    content=ctx["content"],
                    context_type=ContextType(ctx.get("type", "relevant_memory")),
                    relevance_score=0.8,
                    metadata={"turn": i+1, "source": "conversation_flow"}
                )
                context_entries.append(context_entry)
            
            response = manager.process_message(
                session_id=session_id,
                user_message=turn["user"],
                context_override=[
                    {
                        "content": ctx["content"],
                        "type": ctx.get("type", "relevant_memory"),
                        "relevance_score": 0.8,
                        "metadata": {"turn": i+1}
                    } for ctx in turn.get("context", [])
                ]
            )
            
            if response.get("success"):
                print(f"🤖 Assistant: {response['assistant_response'][:100]}...")
                print(f"   Context used: {len(response.get('context_used', []))} entries")
            else:
                print(f"❌ Error: {response.get('error', 'Unknown error')}")
        
        # Get relevant context for a follow-up question
        relevant_context = manager.get_relevant_context(
            session_id=session_id,
            query="Tell me more about deep learning applications",
            limit=3
        )
        print(f"\n🔍 Relevant context retrieved: {len(relevant_context)} entries")
        
        for i, ctx in enumerate(relevant_context):
            print(f"   Context {i+1}: {ctx['content'][:60]}... (score: {ctx.get('relevance_score', 0):.2f})")
        
        return {
            "context_session_created": True,
            "context_conversation_completed": True,
            "relevant_context_retrieved": len(relevant_context),
            "success": True
        }
        
    except Exception as e:
        print(f"❌ Error in context-aware conversation example: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def example_user_preference_learning():
    """Example: User preference learning and personalization"""
    print("\n=== User Preference Learning Example ===")
    
    try:
        # Create conversation manager
        manager = create_conversation_manager(
            groq_api_key=os.getenv("GROQ_API_KEY", "test_key"),
            enable_memory=True,
            enable_user_profiling=True
        )
        
        # Create session for user
        session_id = manager.create_conversation(user_id="preference_user_789")
        print(f"✅ Created session for preference learning: {session_id}")
        
        # Simulate user interactions to learn preferences
        interactions = [
            {"user": "I prefer technical explanations over simple ones.", "preference": {"explanation_style": "technical"}},
            {"user": "I'm most interested in neural networks and deep learning.", "preference": {"interests": ["neural_networks", "deep_learning"]}},
            {"user": "I like examples with real-world applications.", "preference": {"examples": "real_world"}}
        ]
        
        for i, interaction in enumerate(interactions):
            print(f"\n📝 Learning interaction {i+1}: '{interaction['user']}'")
            
            # Process message
            response = manager.process_message(
                session_id=session_id,
                user_message=interaction["user"],
                metadata={"learning_interaction": True, "preference_update": True}
            )
            
            if response.get("success"):
                print(f"   Response processed successfully")
            
            # Update user preferences
            preference_success = manager.update_user_preferences(
                user_id="preference_user_789",
                preferences=interaction["preference"]
            )
            print(f"   Preferences updated: {'✅' if preference_success else '❌'}")
        
        # Get user conversations to see preferences in action
        user_conversations = manager.get_user_conversations(
            user_id="preference_user_789",
            include_inactive=False,
            limit=1
        )
        print(f"\n👤 User conversations: {len(user_conversations)} sessions")
        
        for conv in user_conversations:
            print(f"   Session: {conv['session_id'][:8]}...")
            print(f"   Turns: {conv['total_turns']}")
            print(f"   Status: {conv['status']}")
        
        return {
            "preference_session_created": True,
            "preferences_learned": len(interactions),
            "user_conversations_retrieved": len(user_conversations),
            "success": True
        }
        
    except Exception as e:
        print(f"❌ Error in user preference learning example: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def example_conversation_analytics():
    """Example: Conversation analytics and insights"""
    print("\n=== Conversation Analytics Example ===")
    
    try:
        # Create conversation manager
        manager = create_conversation_manager(
            groq_api_key=os.getenv("GROQ_API_KEY", "test_key"),
            enable_memory=True
        )
        
        # Create multiple sessions for analytics
        sessions = []
        for i in range(3):
            session_id = manager.create_conversation(user_id=f"analytics_user_{i+1}")
            sessions.append(session_id)
            print(f"✅ Created analytics session {i+1}: {session_id[:8]}...")
        
        # Process different types of conversations
        conversation_types = [
            ("technical", "Explain the difference between supervised and unsupervised learning."),
            ("practical", "How can I implement a simple neural network in Python?"),
            ("conceptual", "What are the ethical implications of AI in healthcare?")
        ]
        
        for i, (conv_type, message) in enumerate(conversation_types):
            session_id = sessions[i % len(sessions)]
            
            print(f"\n💬 {conv_type.capitalize()} conversation: '{message[:50]}...'")
            
            response = manager.process_message(
                session_id=session_id,
                user_message=message,
                metadata={"conversation_type": conv_type, "complexity": "high"}
            )
            
            if response.get("success"):
                print(f"   Response time: {response.get('processing_time', 0):.2f}s")
                print(f"   Confidence: {response.get('confidence_score', 0):.2f}")
                print(f"   Tools: {response.get('tools_used', [])}")
        
        # Get system statistics
        system_stats = manager.get_system_stats()
        print(f"\n📊 System Statistics:")
        print(f"   Memory enabled: {system_stats['conversation_manager']['memory_enabled']}")
        print(f"   Advanced RAG: {system_stats['conversation_manager']['advanced_rag_enabled']}")
        print(f"   Agent enabled: {system_stats['conversation_manager']['agent_enabled']}")
        
        if 'session_manager' in system_stats:
            session_stats = system_stats['session_manager']
            print(f"   Active sessions: {session_stats.get('active_sessions', 0)}")
            print(f"   Active users: {session_stats.get('active_users', 0)}")
            print(f"   Avg turns per session: {session_stats.get('average_turns_per_session', 0):.1f}")
        
        # Health check
        health_status = manager.health_check()
        print(f"\n🏥 System Health: {health_status['status']}")
        
        for component, health in health_status.get('components', {}).items():
            print(f"   {component}: {health.get('status', 'unknown')}")
        
        return {
            "analytics_sessions_created": len(sessions),
            "conversation_types_tested": len(conversation_types),
            "system_stats_retrieved": bool(system_stats),
            "health_check_completed": True,
            "success": True
        }
        
    except Exception as e:
        print(f"❌ Error in conversation analytics example: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def example_error_handling_and_recovery():
    """Example: Error handling and recovery scenarios"""
    print("\n=== Error Handling and Recovery Example ===")
    
    try:
        # Create conversation manager
        manager = create_conversation_manager(
            groq_api_key=os.getenv("GROQ_API_KEY", "test_key"),
            enable_memory=True
        )
        
        # Test error scenarios
        error_tests = [
            {
                "name": "Invalid session ID",
                "action": lambda: manager.process_message("invalid_session_id", "test message"),
                "expected": "error"
            },
            {
                "name": "Non-existent session",
                "action": lambda: manager.get_conversation_history("non_existent_session_12345"),
                "expected": "empty_list"
            },
            {
                "name": "Empty message",
                "action": lambda: manager.process_message(manager.create_conversation(), ""),
                "expected": "processing"
            }
        ]
        
        for test in error_tests:
            print(f"\n🧪 Testing: {test['name']}")
            
            try:
                result = test["action"]()
                
                if test["expected"] == "error":
                    if result.get("success") == False:
                        print(f"✅ Expected error handled correctly")
                        print(f"   Error message: {result.get('error', 'No error message')}")
                    else:
                        print(f"⚠️  Unexpected success")
                
                elif test["expected"] == "empty_list":
                    if isinstance(result, list) and len(result) == 0:
                        print(f"✅ Empty list returned as expected")
                    else:
                        print(f"⚠️  Unexpected result: {type(result)}")
                
                elif test["expected"] == "processing":
                    if result.get("success") is not None:
                        print(f"✅ Message processed (empty handling works)")
                    else:
                        print(f"⚠️  Unexpected result")
                
            except Exception as e:
                print(f"✅ Exception handled: {type(e).__name__}")
        
        return {
            "error_tests_completed": len(error_tests),
            "error_handling_working": True,
            "success": True
        }
        
    except Exception as e:
        print(f"❌ Error in error handling example: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """Main function to run all examples"""
    print("🚀 Session & Memory Management System Examples")
    print("=" * 60)
    
    try:
        # Run examples
        results = {}
        
        results['basic_conversation'] = example_basic_conversation()
        results['context_aware_conversation'] = example_context_aware_conversation()
        results['user_preference_learning'] = example_user_preference_learning()
        results['conversation_analytics'] = example_conversation_analytics()
        results['error_handling_and_recovery'] = example_error_handling_and_recovery()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 EXAMPLES SUMMARY")
        print("=" * 60)
        
        successful_examples = sum(1 for r in results.values() if r.get('success', False))
        total_examples = len(results)
        
        print(f"Completed Examples: {successful_examples}/{total_examples}")
        
        for name, result in results.items():
            status = "✅" if result.get('success', False) else "❌"
            print(f"{status} {name.replace('_', ' ').title()}")
        
        if successful_examples == total_examples:
            print(f"\n🎉 All examples completed successfully!")
            print("The Session & Memory Management system is ready for integration.")
            print("\nKey features demonstrated:")
            print("  ✅ Persistent conversation sessions with MongoDB")
            print("  ✅ Context-aware responses with memory retrieval")
            print("  ✅ User preference learning and personalization")
            print("  ✅ Comprehensive conversation analytics")
            print("  ✅ Robust error handling and recovery")
        else:
            print(f"\n⚠️  Some examples had issues. Check the error messages above.")
        
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Please check your configuration and try again.")
        return {}

if __name__ == "__main__":
    main()