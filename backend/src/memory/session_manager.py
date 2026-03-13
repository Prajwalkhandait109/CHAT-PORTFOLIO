import os
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import uuid

from .chat_history import MongoChatHistory, create_mongo_chat_history
from ..models.state import (
    ConversationSession, ConversationTurn, Message, ContextEntry,
    UserProfile, ConversationSummary, SessionStatus, MessageType, ContextType,
    create_new_session, create_message, create_context_entry
)

logger = logging.getLogger(__name__)

@dataclass
class SessionConfig:
    """Configuration for session management"""
    default_session_duration_hours: int = 24
    max_concurrent_sessions_per_user: int = 5
    max_session_turns: int = 100
    auto_archive_after_days: int = 30
    context_retention_limit: int = 10
    enable_conversation_summaries: bool = True
    summary_generation_threshold: int = 10  # Generate summary after N turns

class SessionManager:
    """Manages conversation session lifecycle with memory and context retention"""
    
    def __init__(self, 
                 chat_history: Optional[MongoChatHistory] = None,
                 config: Optional[SessionConfig] = None,
                 connection_string: Optional[str] = None):
        """
        Initialize session manager
        
        Args:
            chat_history: MongoDB chat history instance
            config: Session configuration
            connection_string: MongoDB connection string
        """
        self.config = config or SessionConfig()
        
        # Initialize chat history
        if chat_history:
            self.chat_history = chat_history
        else:
            connection_string = connection_string or os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017/")
            self.chat_history = create_mongo_chat_history(connection_string)
        
        # Connect to database
        if not self.chat_history.connect():
            raise RuntimeError("Failed to connect to MongoDB for session management")
        
        # Active sessions cache (session_id -> ConversationSession)
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.user_sessions: Dict[str, List[str]] = {}  # user_id -> list of session_ids
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Start cleanup thread
        self.cleanup_thread = None
        self.stop_cleanup = False
        self._start_cleanup_thread()
        
        logger.info("Session manager initialized")
    
    def create_session(self, user_id: Optional[str] = None, custom_duration_hours: Optional[int] = None) -> str:
        """
        Create a new conversation session
        
        Args:
            user_id: Optional user ID for session association
            custom_duration_hours: Custom session duration (overrides default)
        
        Returns:
            Session ID
        """
        with self._lock:
            try:
                # Check user session limit
                if user_id and self._get_user_session_count(user_id) >= self.config.max_concurrent_sessions_per_user:
                    logger.warning(f"User {user_id} has reached maximum session limit")
                    # Archive oldest session
                    self._archive_oldest_user_session(user_id)
                
                # Create new session
                duration = custom_duration_hours or self.config.default_session_duration_hours
                session = create_new_session(user_id=user_id, expires_in_hours=duration)
                
                # Add to active sessions
                self.active_sessions[session.session_id] = session
                
                if user_id:
                    if user_id not in self.user_sessions:
                        self.user_sessions[user_id] = []
                    self.user_sessions[user_id].append(session.session_id)
                
                # Save to database
                if not self.chat_history.save_conversation_session(session):
                    logger.error(f"Failed to save session to database: {session.session_id}")
                    # Remove from cache if save failed
                    del self.active_sessions[session.session_id]
                    if user_id and session.session_id in self.user_sessions.get(user_id, []):
                        self.user_sessions[user_id].remove(session.session_id)
                    raise RuntimeError("Failed to create session")
                
                logger.info(f"Created new session: {session.session_id} for user: {user_id}")
                return session.session_id
                
            except Exception as e:
                logger.error(f"Error creating session: {e}")
                raise
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Get conversation session by ID
        
        Args:
            session_id: Session ID
        
        Returns:
            ConversationSession or None
        """
        with self._lock:
            try:
                # Check active sessions cache first
                if session_id in self.active_sessions:
                    session = self.active_sessions[session_id]
                    # Check if expired
                    if session.is_expired():
                        self._expire_session(session_id)
                        return None
                    return session
                
                # Load from database
                session = self.chat_history.get_conversation_session(session_id)
                if session:
                    # Check if expired
                    if session.is_expired():
                        self._expire_session(session_id)
                        return None
                    
                    # Add to active sessions cache
                    self.active_sessions[session_id] = session
                    if session.user_id:
                        if session.user_id not in self.user_sessions:
                            self.user_sessions[session.user_id] = []
                        if session_id not in self.user_sessions[session.user_id]:
                            self.user_sessions[session.user_id].append(session_id)
                    
                    return session
                
                return None
                
            except Exception as e:
                logger.error(f"Error getting session {session_id}: {e}")
                return None
    
    def add_conversation_turn(self, session_id: str, user_query: str, assistant_response: str,
                             context_used: Optional[List[ContextEntry]] = None,
                             tools_used: Optional[List[str]] = None,
                             processing_time: Optional[float] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a conversation turn to the session
        
        Args:
            session_id: Session ID
            user_query: User's query
            assistant_response: Assistant's response
            context_used: Context information used
            tools_used: Tools used during processing
            processing_time: Processing time in seconds
            metadata: Additional metadata
        
        Returns:
            Success status
        """
        with self._lock:
            try:
                session = self.get_session(session_id)
                if not session:
                    logger.error(f"Session not found or expired: {session_id}")
                    return False
                
                # Check session turn limit
                if session.total_turns >= self.config.max_session_turns:
                    logger.warning(f"Session {session_id} has reached maximum turns")
                    self._expire_session(session_id)
                    return False
                
                # Create conversation turn
                user_message = create_message(user_query, MessageType.USER)
                assistant_message = create_message(assistant_response, MessageType.ASSISTANT)
                
                turn = ConversationTurn(
                    id=str(uuid.uuid4()),
                    user_message=user_message,
                    assistant_message=assistant_message,
                    context_used=context_used or [],
                    tools_used=tools_used or [],
                    processing_time=processing_time,
                    metadata=metadata or {}
                )
                
                # Add turn to session
                session.add_turn(turn)
                
                # Update session in database
                if not self.chat_history.save_conversation_session(session):
                    logger.error(f"Failed to save session with new turn: {session_id}")
                    return False
                
                # Generate summary if threshold reached
                if (self.config.enable_conversation_summaries and 
                    session.total_turns >= self.config.summary_generation_threshold and
                    session.total_turns % self.config.summary_generation_threshold == 0):
                    self._generate_session_summary(session_id)
                
                logger.info(f"Added conversation turn to session: {session_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error adding conversation turn: {e}")
                return False
    
    def get_session_context(self, session_id: str, limit: int = None) -> List[ContextEntry]:
        """
        Get relevant context from the session
        
        Args:
            session_id: Session ID
            limit: Maximum number of context entries to return
        
        Returns:
            List of context entries
        """
        limit = limit or self.config.context_retention_limit
        
        with self._lock:
            try:
                session = self.get_session(session_id)
                if not session:
                    return []
                
                # Collect context from recent turns
                recent_turns = session.get_recent_turns(limit=5)
                all_context = []
                
                for turn in recent_turns:
                    all_context.extend(turn.context_used)
                
                # Sort by relevance score and timestamp
                all_context.sort(key=lambda x: (x.relevance_score, x.timestamp), reverse=True)
                
                # Return top context entries
                return all_context[:limit]
                
            except Exception as e:
                logger.error(f"Error getting session context: {e}")
                return []
    
    def get_recent_queries(self, session_id: str, limit: int = 5) -> List[str]:
        """
        Get recent user queries from the session
        
        Args:
            session_id: Session ID
            limit: Maximum number of queries to return
        
        Returns:
            List of recent queries
        """
        with self._lock:
            try:
                session = self.get_session(session_id)
                if not session:
                    return []
                
                recent_turns = session.get_recent_turns(limit=limit)
                queries = []
                
                for turn in recent_turns:
                    if turn.user_message and turn.user_message.content:
                        queries.append(turn.user_message.content)
                
                return queries
                
            except Exception as e:
                logger.error(f"Error getting recent queries: {e}")
                return []
    
    def end_session(self, session_id: str, generate_summary: bool = True) -> bool:
        """
        End a conversation session
        
        Args:
            session_id: Session ID
            generate_summary: Whether to generate session summary
        
        Returns:
            Success status
        """
        with self._lock:
            try:
                session = self.get_session(session_id)
                if not session:
                    logger.error(f"Session not found: {session_id}")
                    return False
                
                # Generate final summary if requested
                if generate_summary and self.config.enable_conversation_summaries:
                    self._generate_session_summary(session_id)
                
                # Update session status
                session.status = SessionStatus.INACTIVE
                session.updated_at = datetime.now()
                
                # Save to database
                if not self.chat_history.save_conversation_session(session):
                    logger.error(f"Failed to save ended session: {session_id}")
                    return False
                
                # Remove from active sessions
                self._remove_from_active_sessions(session_id)
                
                logger.info(f"Ended session: {session_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error ending session: {e}")
                return False
    
    def get_user_sessions(self, user_id: str, include_inactive: bool = False) -> List[ConversationSession]:
        """
        Get all sessions for a user
        
        Args:
            user_id: User ID
            include_inactive: Whether to include inactive sessions
        
        Returns:
            List of user sessions
        """
        with self._lock:
            try:
                # Get from database
                sessions = self.chat_history.get_user_sessions(user_id)
                
                # Filter by status if needed
                if not include_inactive:
                    sessions = [s for s in sessions if s.status == SessionStatus.ACTIVE]
                
                # Update active sessions cache
                for session in sessions:
                    if session.status == SessionStatus.ACTIVE and not session.is_expired():
                        self.active_sessions[session.session_id] = session
                
                return sessions
                
            except Exception as e:
                logger.error(f"Error getting user sessions: {e}")
                return []
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get user profile
        
        Args:
            user_id: User ID
        
        Returns:
            UserProfile or None
        """
        try:
            return self.chat_history.get_user_profile(user_id)
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None
    
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """
        Update user preferences
        
        Args:
            user_id: User ID
            preferences: Preferences to update
        
        Returns:
            Success status
        """
        try:
            # Get existing profile or create new one
            profile = self.get_user_profile(user_id)
            if not profile:
                profile = UserProfile(user_id=user_id)
            
            profile.update_preferences(preferences)
            
            return self.chat_history.save_user_profile(profile)
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            return False
    
    def get_conversation_summary(self, session_id: str) -> Optional[ConversationSummary]:
        """
        Get conversation summary for a session
        
        Args:
            session_id: Session ID
        
        Returns:
            ConversationSummary or None
        """
        try:
            return self.chat_history.get_conversation_summary(session_id)
        except Exception as e:
            logger.error(f"Error getting conversation summary: {e}")
            return None
    
    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session
        
        Args:
            session_id: Session ID
        
        Returns:
            Session statistics
        """
        with self._lock:
            try:
                session = self.get_session(session_id)
                if not session:
                    return {"error": "Session not found"}
                
                context_summary = session.get_context_summary()
                
                return {
                    "session_id": session_id,
                    "total_turns": session.total_turns,
                    "status": session.status.value,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "expires_at": session.expires_at.isoformat() if session.expires_at else None,
                    "context_summary": context_summary,
                    "user_id": session.user_id
                }
                
            except Exception as e:
                logger.error(f"Error getting session statistics: {e}")
                return {"error": str(e)}
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions
        
        Returns:
            Number of sessions cleaned up
        """
        with self._lock:
            try:
                expired_count = 0
                
                # Check active sessions
                expired_session_ids = []
                for session_id, session in self.active_sessions.items():
                    if session.is_expired():
                        expired_session_ids.append(session_id)
                
                # Expire sessions
                for session_id in expired_session_ids:
                    self._expire_session(session_id)
                    expired_count += 1
                
                # Use database cleanup
                db_expired_count = self.chat_history.cleanup_expired_sessions()
                expired_count += db_expired_count
                
                logger.info(f"Cleaned up {expired_count} expired sessions")
                return expired_count
                
            except Exception as e:
                logger.error(f"Error during session cleanup: {e}")
                return 0
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system-wide statistics
        
        Returns:
            System statistics
        """
        with self._lock:
            try:
                # Get database stats
                db_stats = self.chat_history.get_stats()
                
                # Calculate active session stats
                active_count = len(self.active_sessions)
                user_count = len(self.user_sessions)
                
                # Calculate average session metrics
                total_turns = sum(session.total_turns for session in self.active_sessions.values())
                avg_turns = total_turns / active_count if active_count > 0 else 0
                
                return {
                    "active_sessions": active_count,
                    "active_users": user_count,
                    "average_turns_per_session": round(avg_turns, 2),
                    "max_concurrent_sessions_per_user": self.config.max_concurrent_sessions_per_user,
                    "max_session_turns": self.config.max_session_turns,
                    "database_stats": db_stats,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting system stats: {e}")
                return {"error": str(e)}
    
    def _get_user_session_count(self, user_id: str) -> int:
        """Get number of active sessions for user"""
        return len(self.user_sessions.get(user_id, []))
    
    def _archive_oldest_user_session(self, user_id: str) -> bool:
        """Archive the oldest user session"""
        try:
            user_session_ids = self.user_sessions.get(user_id, [])
            if not user_session_ids:
                return False
            
            # Get oldest session
            oldest_session_id = min(user_session_ids, key=lambda sid: self.active_sessions[sid].created_at)
            
            # Archive it
            return self.end_session(oldest_session_id, generate_summary=True)
            
        except Exception as e:
            logger.error(f"Error archiving oldest user session: {e}")
            return False
    
    def _expire_session(self, session_id: str) -> bool:
        """Expire a session"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return False
            
            session.status = SessionStatus.EXPIRED
            session.updated_at = datetime.now()
            
            # Save to database
            self.chat_history.save_conversation_session(session)
            
            # Remove from active sessions
            self._remove_from_active_sessions(session_id)
            
            logger.info(f"Expired session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error expiring session {session_id}: {e}")
            return False
    
    def _remove_from_active_sessions(self, session_id: str) -> None:
        """Remove session from active sessions cache"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            user_id = session.user_id
            
            del self.active_sessions[session_id]
            
            if user_id and user_id in self.user_sessions:
                if session_id in self.user_sessions[user_id]:
                    self.user_sessions[user_id].remove(session_id)
                
                if not self.user_sessions[user_id]:
                    del self.user_sessions[user_id]
    
    def _generate_session_summary(self, session_id: str) -> bool:
        """Generate summary for session"""
        try:
            session = self.get_session(session_id)
            if not session or session.total_turns < 2:
                return False
            
            # This is a simplified summary generation
            # In a real implementation, you might use an LLM for better summaries
            
            recent_turns = session.get_recent_turns(limit=5)
            summary_text = f"Conversation with {session.total_turns} turns. Recent topics include: "
            
            topics = []
            for turn in recent_turns:
                if turn.user_message and turn.user_message.content:
                    # Simple topic extraction (first 50 chars)
                    topic = turn.user_message.content[:50]
                    if len(turn.user_message.content) > 50:
                        topic += "..."
                    topics.append(topic)
            
            summary_text += ", ".join(topics)
            
            # Extract key entities (simplified)
            key_topics = []
            important_entities = []
            
            for turn in recent_turns:
                for context in turn.context_used:
                    if context.relevance_score > 0.7:  # High relevance
                        key_topics.append(context.content[:30])
            
            summary = ConversationSummary(
                session_id=session_id,
                summary_text=summary_text,
                key_topics=list(set(key_topics))[:5],  # Unique topics
                important_entities=list(set(important_entities))[:5],
                sentiment_score=0.5,  # Default neutral
                relevance_score=0.8,  # Default high relevance
                metadata={"generated_by": "session_manager", "turn_count": session.total_turns}
            )
            
            return self.chat_history.save_conversation_summary(summary)
            
        except Exception as e:
            logger.error(f"Error generating session summary: {e}")
            return False
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        import threading
        import time
        
        def cleanup_worker():
            while not self.stop_cleanup:
                try:
                    self.cleanup_expired_sessions()
                    # Clean up every 5 minutes
                    time.sleep(300)
                except Exception as e:
                    logger.error(f"Error in cleanup worker: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
        
        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        logger.info("Session cleanup thread started")
    
    def stop_cleanup_thread(self):
        """Stop the cleanup thread"""
        self.stop_cleanup = True
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=10)
        logger.info("Session cleanup thread stopped")
    
    def shutdown(self):
        """Shutdown session manager"""
        try:
            logger.info("Shutting down session manager")
            
            # Stop cleanup thread
            self.stop_cleanup_thread()
            
            # Save all active sessions
            with self._lock:
                for session_id, session in self.active_sessions.items():
                    if session.status == SessionStatus.ACTIVE:
                        session.status = SessionStatus.INACTIVE
                        self.chat_history.save_conversation_session(session)
            
            # Disconnect from database
            self.chat_history.disconnect()
            
            logger.info("Session manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during session manager shutdown: {e}")

class ConversationContextManager:
    """Manages conversation context and memory retrieval"""
    
    def __init__(self, session_manager: SessionManager):
        """
        Initialize conversation context manager
        
        Args:
            session_manager: Session manager instance
        """
        self.session_manager = session_manager
    
    def get_relevant_context(self, session_id: str, current_query: str, 
                           context_types: Optional[List[ContextType]] = None,
                           limit: int = 10) -> List[ContextEntry]:
        """
        Get relevant context for current query
        
        Args:
            session_id: Current session ID
            current_query: Current user query
            context_types: Types of context to include
            limit: Maximum number of context entries
        
        Returns:
            List of relevant context entries
        """
        try:
            # Get session context
            session_context = self.session_manager.get_session_context(session_id, limit=limit)
            
            # Filter by context types if specified
            if context_types:
                session_context = [
                    ctx for ctx in session_context 
                    if ctx.type in context_types
                ]
            
            # Simple relevance scoring based on query similarity
            # In a real implementation, you might use embeddings for better relevance
            relevant_context = []
            current_query_lower = current_query.lower()
            
            for context in session_context:
                # Simple keyword matching for relevance
                context_content_lower = context.content.lower()
                
                # Count matching keywords
                query_words = set(current_query_lower.split())
                context_words = set(context_content_lower.split())
                
                matches = len(query_words.intersection(context_words))
                total_words = len(query_words)
                
                if total_words > 0:
                    relevance_boost = matches / total_words
                    adjusted_relevance = min(1.0, context.relevance_score + relevance_boost * 0.3)
                    context.relevance_score = adjusted_relevance
                
                relevant_context.append(context)
            
            # Sort by relevance score
            relevant_context.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return relevant_context[:limit]
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            return []
    
    def get_conversation_memory(self, session_id: str, memory_type: str = "recent", limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get conversation memory
        
        Args:
            session_id: Session ID
            memory_type: Type of memory ("recent", "important", "summary")
            limit: Maximum number of memory entries
        
        Returns:
            List of memory entries
        """
        try:
            if memory_type == "recent":
                return self._get_recent_memory(session_id, limit)
            elif memory_type == "important":
                return self._get_important_memory(session_id, limit)
            elif memory_type == "summary":
                return self._get_summary_memory(session_id)
            else:
                logger.warning(f"Unknown memory type: {memory_type}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting conversation memory: {e}")
            return []
    
    def _get_recent_memory(self, session_id: str, limit: int) -> List[Dict[str, Any]]:
        """Get recent conversation memory"""
        try:
            session = self.session_manager.get_session(session_id)
            if not session:
                return []
            
            recent_turns = session.get_recent_turns(limit=limit)
            memory = []
            
            for turn in recent_turns:
                memory_entry = {
                    "type": "conversation_turn",
                    "user_query": turn.user_message.content if turn.user_message else "",
                    "assistant_response": turn.assistant_message.content if turn.assistant_message else "",
                    "timestamp": turn.timestamp.isoformat(),
                    "tools_used": turn.tools_used,
                    "processing_time": turn.processing_time
                }
                memory.append(memory_entry)
            
            return memory
            
        except Exception as e:
            logger.error(f"Error getting recent memory: {e}")
            return []
    
    def _get_important_memory(self, session_id: str, limit: int) -> List[Dict[str, Any]]:
        """Get important conversation memory based on relevance scores"""
        try:
            session = self.session_manager.get_session(session_id)
            if not session:
                return []
            
            # Collect all context with high relevance
            important_context = []
            
            for turn in session.turns:
                for context in turn.context_used:
                    if context.relevance_score > 0.8:  # High relevance threshold
                        important_context.append({
                            "type": "important_context",
                            "content": context.content,
                            "relevance_score": context.relevance_score,
                            "context_type": context.type.value,
                            "timestamp": context.timestamp.isoformat()
                        })
            
            # Sort by relevance score and return top entries
            important_context.sort(key=lambda x: x["relevance_score"], reverse=True)
            return important_context[:limit]
            
        except Exception as e:
            logger.error(f"Error getting important memory: {e}")
            return []
    
    def _get_summary_memory(self, session_id: str) -> List[Dict[str, Any]]:
        """Get summary-based conversation memory"""
        try:
            summary = self.session_manager.get_conversation_summary(session_id)
            if not summary:
                return []
            
            return [{
                "type": "conversation_summary",
                "summary_text": summary.summary_text,
                "key_topics": summary.key_topics,
                "important_entities": summary.important_entities,
                "sentiment_score": summary.sentiment_score,
                "relevance_score": summary.relevance_score,
                "timestamp": summary.timestamp.isoformat()
            }]
            
        except Exception as e:
            logger.error(f"Error getting summary memory: {e}")
            return []
    
    def create_context_from_memory(self, memory_entries: List[Dict[str, Any]], relevance_score: float = 0.7) -> List[ContextEntry]:
        """
        Create context entries from memory
        
        Args:
            memory_entries: Memory entries
            relevance_score: Default relevance score
        
        Returns:
            List of context entries
        """
        try:
            context_entries = []
            
            for memory in memory_entries:
                content = ""
                context_type = ContextType.RELEVANT_MEMORY
                
                if memory.get("type") == "conversation_turn":
                    content = f"User: {memory.get('user_query', '')}\nAssistant: {memory.get('assistant_response', '')}"
                    context_type = ContextType.PREVIOUS_QUERY
                elif memory.get("type") == "important_context":
                    content = memory.get("content", "")
                    context_type = ContextType.RELEVANT_MEMORY
                elif memory.get("type") == "conversation_summary":
                    content = memory.get("summary_text", "")
                    context_type = ContextType.RELEVANT_MEMORY
                
                if content:
                    context_entry = create_context_entry(
                        content=content,
                        context_type=context_type,
                        relevance_score=memory.get("relevance_score", relevance_score),
                        metadata={
                            "memory_type": memory.get("type"),
                            "original_timestamp": memory.get("timestamp"),
                            "processing_time": memory.get("processing_time"),
                            "tools_used": memory.get("tools_used", [])
                        }
                    )
                    context_entries.append(context_entry)
            
            return context_entries
            
        except Exception as e:
            logger.error(f"Error creating context from memory: {e}")
            return []

# Factory function for creating session manager instances
def create_session_manager(
    connection_string: Optional[str] = None,
    config: Optional[SessionConfig] = None,
    **kwargs
) -> SessionManager:
    """
    Factory function to create session manager
    
    Args:
        connection_string: MongoDB connection string
        config: Session configuration
        **kwargs: Additional arguments for SessionManager
    
    Returns:
        SessionManager instance
    """
    connection_string = connection_string or os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017/")
    
    return SessionManager(
        connection_string=connection_string,
        config=config,
        **kwargs
    )