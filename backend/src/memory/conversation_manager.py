import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

# Import existing components
from .session_manager import create_session_manager, SessionManager
from .enhanced_memory_manager import create_enhanced_memory_manager, EnhancedMemoryManager
from .chat_history import create_mongo_chat_history, MongoChatHistory
from .agent_memory import AgentMemory
from ..models.state import (
    ConversationSession, ConversationTurn, Message, ContextEntry,
    SessionStatus, MessageType, ContextType, create_message, create_context_entry
)

logger = logging.getLogger(__name__)

class ConversationManager:
    """
    High-level conversation manager that integrates session management, 
    memory retention, and the existing Agentic AI Chatbot
    """
    
    def __init__(self, 
                 groq_api_key: str,
                 connection_string: Optional[str] = None,
                 vector_store_path: Optional[str] = None,
                 enable_advanced_rag: bool = True,
                 enable_agent: bool = True,
                 enable_memory: bool = True):
        """
        Initialize conversation manager
        
        Args:
            groq_api_key: Groq API key for chat functionality
            connection_string: MongoDB connection string
            vector_store_path: Path to vector store
            enable_advanced_rag: Enable advanced RAG pipeline
            enable_agent: Enable ReAct agent
            enable_memory: Enable conversation memory
        """
        self.groq_api_key = groq_api_key
        self.connection_string = connection_string or os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017/")
        self.vector_store_path = vector_store_path
        self.enable_advanced_rag = enable_advanced_rag
        self.enable_agent = enable_agent
        self.enable_memory = enable_memory
        
        # Initialize components
        self.chat_history = create_mongo_chat_history(self.connection_string)
        self.session_manager = create_session_manager(self.connection_string)
        
        if enable_memory:
            self.memory_manager = create_enhanced_memory_manager(self.connection_string)
        else:
            self.memory_manager = None
        
        # Initialize Agentic AI Chatbot (lazy loading)
        self._chatbot = None
        
        logger.info("Conversation manager initialized")
    
    @property
    def chatbot(self):
        """Lazy loading of Agentic AI Chatbot"""
        if self._chatbot is None:
            try:
                from ..agentic_ai_chatbot import AgenticAIChatbot
                
                self._chatbot = AgenticAIChatbot(
                    groq_api_key=self.groq_api_key,
                    vector_store_path=self.vector_store_path,
                    use_advanced_rag=self.enable_advanced_rag,
                    enable_agent=self.enable_agent
                )
                logger.info("Agentic AI Chatbot initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Agentic AI Chatbot: {e}")
                raise
        
        return self._chatbot
    
    def create_conversation(self, user_id: Optional[str] = None, 
                          custom_duration_hours: Optional[int] = None) -> str:
        """
        Create a new conversation session
        
        Args:
            user_id: Optional user ID for session association
            custom_duration_hours: Custom session duration
        
        Returns:
            Session ID
        """
        try:
            # Create session through session manager
            session_id = self.session_manager.create_session(
                user_id=user_id,
                custom_duration_hours=custom_duration_hours
            )
            
            logger.info(f"Created conversation session: {session_id} for user: {user_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating conversation session: {e}")
            raise
    
    def process_message(self, session_id: str, user_message: str, 
                       context_override: Optional[List[Dict[str, Any]]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user message within a conversation session
        
        Args:
            session_id: Conversation session ID
            user_message: User's message
            context_override: Optional context to override default retrieval
            metadata: Additional metadata for the message
        
        Returns:
            Response with conversation details
        """
        start_time = datetime.now()
        
        try:
            # Validate session
            session = self.session_manager.get_session(session_id)
            if not session:
                return {
                    "error": "Session not found or expired",
                    "session_id": session_id,
                    "success": False
                }
            
            # Get relevant context if memory is enabled
            relevant_context = []
            if self.enable_memory and self.memory_manager:
                if context_override is None:
                    # Get context from memory
                    relevant_context = self.memory_manager.get_relevant_context(
                        session_id=session_id,
                        current_query=user_message,
                        include_session_memory=True,
                        include_agent_memory=True,
                        include_user_context=True,
                        limit=10
                    )
                else:
                    # Use provided context
                    relevant_context = [
                        create_context_entry(
                            content=ctx.get("content", ""),
                            context_type=ContextType(ctx.get("type", "retrieved_document")),
                            relevance_score=ctx.get("relevance_score", 0.7),
                            metadata=ctx.get("metadata", {})
                        ) for ctx in context_override
                    ]
            
            # Process through Agentic AI Chatbot
            chatbot_response = self.chatbot.ask(
                query=user_message,
                session_context={
                    "session_id": session_id,
                    "user_id": session.user_id,
                    "conversation_context": [ctx.content for ctx in relevant_context[:5]]  # Limit context
                } if relevant_context else None,
                use_advanced_rag=self.enable_advanced_rag,
                enable_agent=self.enable_agent
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Extract response details
            assistant_response = chatbot_response.get("answer", "Sorry, I couldn't process your message.")
            confidence_score = chatbot_response.get("confidence_score", 0.0)
            sources = chatbot_response.get("sources", [])
            tools_used = chatbot_response.get("tools_used", [])
            
            # Add conversation turn to session
            success = self.session_manager.add_conversation_turn(
                session_id=session_id,
                user_query=user_message,
                assistant_response=assistant_response,
                context_used=relevant_context,
                tools_used=tools_used,
                processing_time=processing_time,
                metadata={
                    "confidence_score": confidence_score,
                    "sources": sources,
                    **(metadata or {})
                }
            )
            
            if not success:
                logger.warning(f"Failed to save conversation turn for session: {session_id}")
            
            # Add to enhanced memory if enabled
            if self.enable_memory and self.memory_manager:
                self.memory_manager.add_conversation_turn(
                    session_id=session_id,
                    user_query=user_message,
                    assistant_response=assistant_response,
                    context_used=relevant_context,
                    tools_used=tools_used,
                    processing_time=processing_time,
                    metadata={
                        "confidence_score": confidence_score,
                        "sources": sources,
                        "model": "agentic_ai",
                        **(metadata or {})
                    }
                )
            
            # Build response
            response = {
                "session_id": session_id,
                "user_message": user_message,
                "assistant_response": assistant_response,
                "confidence_score": confidence_score,
                "processing_time": processing_time,
                "sources": sources,
                "tools_used": tools_used,
                "context_used": [
                    {
                        "content": ctx.content,
                        "type": ctx.type.value,
                        "relevance_score": ctx.relevance_score,
                        "metadata": ctx.metadata
                    } for ctx in relevant_context[:3]  # Include top 3 context entries
                ],
                "success": True
            }
            
            logger.info(f"Processed message for session {session_id} in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing message for session {session_id}: {e}")
            return {
                "error": str(e),
                "session_id": session_id,
                "success": False
            }
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session
        
        Args:
            session_id: Session ID
            limit: Maximum number of turns to return
        
        Returns:
            List of conversation turns
        """
        try:
            session = self.session_manager.get_session(session_id)
            if not session:
                return []
            
            recent_turns = session.get_recent_turns(limit=limit)
            
            history = []
            for turn in recent_turns:
                history.append({
                    "turn_id": turn.id,
                    "user_message": {
                        "content": turn.user_message.content if turn.user_message else "",
                        "timestamp": turn.user_message.timestamp.isoformat() if turn.user_message else None,
                        "metadata": turn.user_message.metadata if turn.user_message else {}
                    },
                    "assistant_response": {
                        "content": turn.assistant_message.content if turn.assistant_message else "",
                        "timestamp": turn.assistant_message.timestamp.isoformat() if turn.assistant_message else None,
                        "metadata": turn.assistant_message.metadata if turn.assistant_message else {}
                    },
                    "context_used": [
                        {
                            "content": ctx.content,
                            "type": ctx.type.value,
                            "relevance_score": ctx.relevance_score,
                            "metadata": ctx.metadata
                        } for ctx in turn.context_used
                    ],
                    "tools_used": turn.tools_used,
                    "processing_time": turn.processing_time,
                    "timestamp": turn.timestamp.isoformat()
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting conversation history for session {session_id}: {e}")
            return []
    
    def get_relevant_context(self, session_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant context for a query within a session
        
        Args:
            session_id: Session ID
            query: User query
            limit: Maximum number of context entries
        
        Returns:
            List of relevant context entries
        """
        if not self.enable_memory or not self.memory_manager:
            return []
        
        try:
            relevant_context = self.memory_manager.get_relevant_context(
                session_id=session_id,
                current_query=query,
                limit=limit
            )
            
            return [
                {
                    "content": ctx.content,
                    "type": ctx.type.value,
                    "relevance_score": ctx.relevance_score,
                    "metadata": ctx.metadata
                } for ctx in relevant_context
            ]
            
        except Exception as e:
            logger.error(f"Error getting relevant context for session {session_id}: {e}")
            return []
    
    def end_conversation(self, session_id: str, generate_summary: bool = True) -> Dict[str, Any]:
        """
        End a conversation session
        
        Args:
            session_id: Session ID
            generate_summary: Whether to generate conversation summary
        
        Returns:
            Session summary and statistics
        """
        try:
            # Get session statistics before ending
            session_stats = self.session_manager.get_session_statistics(session_id)
            
            # Get conversation summary if memory is enabled
            conversation_summary = None
            if self.enable_memory and self.memory_manager:
                conversation_summary = self.memory_manager.get_conversation_summary(session_id)
            
            # End the session
            success = self.session_manager.end_session(session_id, generate_summary=generate_summary)
            
            if success:
                logger.info(f"Ended conversation session: {session_id}")
                
                return {
                    "session_id": session_id,
                    "session_stats": session_stats,
                    "conversation_summary": conversation_summary,
                    "status": "ended",
                    "success": True
                }
            else:
                return {
                    "session_id": session_id,
                    "error": "Failed to end session",
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"Error ending conversation session {session_id}: {e}")
            return {
                "error": str(e),
                "session_id": session_id,
                "success": False
            }
    
    def get_user_conversations(self, user_id: str, include_inactive: bool = False, 
                              limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get all conversations for a user
        
        Args:
            user_id: User ID
            include_inactive: Whether to include inactive sessions
            limit: Maximum number of sessions
        
        Returns:
            List of user conversations
        """
        try:
            sessions = self.session_manager.get_user_sessions(user_id, include_inactive=include_inactive)
            
            conversations = []
            for session in sessions[:limit]:
                conversations.append({
                    "session_id": session.session_id,
                    "status": session.status.value,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "total_turns": session.total_turns,
                    "context_summary": session.get_context_summary(),
                    "metadata": session.metadata
                })
            
            return conversations
            
        except Exception as e:
            logger.error(f"Error getting user conversations for {user_id}: {e}")
            return []
    
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """
        Update user preferences
        
        Args:
            user_id: User ID
            preferences: User preferences
        
        Returns:
            Success status
        """
        try:
            if self.memory_manager:
                return self.memory_manager.update_user_preferences(user_id, preferences)
            else:
                # Fallback to session manager user profile
                return self.session_manager.update_user_preferences(user_id, preferences)
                
        except Exception as e:
            logger.error(f"Error updating user preferences for {user_id}: {e}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics
        
        Returns:
            System statistics
        """
        try:
            stats = {
                "conversation_manager": {
                    "memory_enabled": self.enable_memory,
                    "advanced_rag_enabled": self.enable_advanced_rag,
                    "agent_enabled": self.enable_agent
                },
                "session_manager": self.session_manager.get_system_stats() if hasattr(self.session_manager, 'get_system_stats') else {},
                "timestamp": datetime.now().isoformat()
            }
            
            if self.memory_manager:
                stats["memory_manager"] = self.memory_manager.get_memory_statistics()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check system health
        
        Returns:
            Health status
        """
        try:
            # Check chat history connection
            chat_history_health = self.chat_history.health_check()
            
            # Check session manager
            session_manager_health = self.session_manager.health_check() if hasattr(self.session_manager, 'health_check') else {"status": "unknown"}
            
            # Check memory manager if enabled
            memory_manager_health = {"status": "disabled"}
            if self.memory_manager:
                memory_manager_health = self.memory_manager.chat_history.health_check() if hasattr(self.memory_manager, 'chat_history') else {"status": "unknown"}
            
            # Check chatbot availability
            chatbot_health = {"status": "healthy"}
            try:
                # Test chatbot with simple query
                test_response = self.chatbot.ask("test", use_advanced_rag=False, enable_agent=False)
                if not test_response.get("answer"):
                    chatbot_health = {"status": "unhealthy", "error": "No response from chatbot"}
            except Exception as e:
                chatbot_health = {"status": "unhealthy", "error": str(e)}
            
            overall_health = all([
                chat_history_health.get("status") == "healthy",
                session_manager_health.get("status") == "healthy",
                memory_manager_health.get("status") in ["healthy", "disabled"],
                chatbot_health.get("status") == "healthy"
            ])
            
            return {
                "status": "healthy" if overall_health else "unhealthy",
                "components": {
                    "chat_history": chat_history_health,
                    "session_manager": session_manager_health,
                    "memory_manager": memory_manager_health,
                    "chatbot": chatbot_health
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Factory function for creating conversation manager
def create_conversation_manager(
    groq_api_key: Optional[str] = None,
    connection_string: Optional[str] = None,
    vector_store_path: Optional[str] = None,
    enable_advanced_rag: bool = True,
    enable_agent: bool = True,
    enable_memory: bool = True,
    **kwargs
) -> ConversationManager:
    """
    Factory function to create conversation manager
    
    Args:
        groq_api_key: Groq API key
        connection_string: MongoDB connection string
        vector_store_path: Path to vector store
        enable_advanced_rag: Enable advanced RAG
        enable_agent: Enable ReAct agent
        enable_memory: Enable conversation memory
        **kwargs: Additional arguments for ConversationManager
    
    Returns:
        ConversationManager instance
    """
    groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Groq API key is required")
    
    connection_string = connection_string or os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017/")
    
    return ConversationManager(
        groq_api_key=groq_api_key,
        connection_string=connection_string,
        vector_store_path=vector_store_path,
        enable_advanced_rag=enable_advanced_rag,
        enable_agent=enable_agent,
        enable_memory=enable_memory,
        **kwargs
    )