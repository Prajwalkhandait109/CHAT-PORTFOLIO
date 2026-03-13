import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import logging
import threading
from dataclasses import dataclass

# Import existing memory components
from .agent_memory import AgentMemory
from .session_manager import SessionManager, create_session_manager, ConversationContextManager
from .chat_history import MongoChatHistory, create_mongo_chat_history
from ..models.state import (
    ConversationSession, ConversationTurn, Message, ContextEntry,
    UserProfile, ConversationSummary, SessionStatus, MessageType, ContextType
)

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Configuration for enhanced memory management"""
    max_conversation_memory: int = 1000  # Maximum conversations to keep in memory
    context_retention_days: int = 30     # Days to retain conversation context
    relevance_threshold: float = 0.7     # Minimum relevance score for context
    enable_cross_session_memory: bool = True  # Enable memory across sessions
    enable_user_profiling: bool = True   # Enable user preference learning
    memory_compression_threshold: int = 50  # Compress memory after N sessions
    cleanup_interval_hours: int = 24     # Cleanup interval in hours

class EnhancedMemoryManager:
    """
    Enhanced memory manager that combines agent memory, session management, 
    and conversation history for comprehensive context retention
    """
    
    def __init__(self, 
                 connection_string: Optional[str] = None,
                 memory_config: Optional[MemoryConfig] = None,
                 agent_memory: Optional[AgentMemory] = None):
        """
        Initialize enhanced memory manager
        
        Args:
            connection_string: MongoDB connection string
            memory_config: Memory configuration
            agent_memory: Existing agent memory instance
        """
        self.config = memory_config or MemoryConfig()
        
        # Initialize components
        connection_string = connection_string or os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017/")
        
        self.chat_history = create_mongo_chat_history(connection_string)
        self.session_manager = create_session_manager(connection_string)
        self.context_manager = ConversationContextManager(self.session_manager)
        
        # Use provided agent memory or create new one
        self.agent_memory = agent_memory or AgentMemory()
        
        # In-memory cache for frequently accessed data
        self.conversation_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.user_context_cache: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background cleanup
        self.cleanup_thread = None
        self.stop_cleanup = False
        
        logger.info("Enhanced memory manager initialized")
    
    def create_conversation_session(self, user_id: Optional[str] = None, 
                                  custom_duration_hours: Optional[int] = None) -> str:
        """
        Create a new conversation session with enhanced memory support
        
        Args:
            user_id: Optional user ID
            custom_duration_hours: Custom session duration
        
        Returns:
            Session ID
        """
        try:
            # Create session through session manager
            session_id = self.session_manager.create_session(user_id, custom_duration_hours)
            
            # Load user context if available
            if user_id and self.config.enable_user_profiling:
                user_context = self._load_user_context(user_id)
                if user_context:
                    self.user_context_cache[user_id] = user_context
            
            logger.info(f"Created enhanced conversation session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating conversation session: {e}")
            raise
    
    def add_conversation_turn(self, session_id: str, user_query: str, assistant_response: str,
                           context_used: Optional[List[ContextEntry]] = None,
                           tools_used: Optional[List[str]] = None,
                           processing_time: Optional[float] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a conversation turn with enhanced memory processing
        
        Args:
            session_id: Session ID
            user_query: User's query
            assistant_response: Assistant's response
            context_used: Context information used
            tools_used: Tools used during processing
            processing_time: Processing time
            metadata: Additional metadata
        
        Returns:
            Success status
        """
        try:
            # Add to session manager
            success = self.session_manager.add_conversation_turn(
                session_id=session_id,
                user_query=user_query,
                assistant_response=assistant_response,
                context_used=context_used,
                tools_used=tools_used,
                processing_time=processing_time,
                metadata=metadata
            )
            
            if not success:
                return False
            
            # Add to agent memory for cross-session retention
            if self.config.enable_cross_session_memory:
                self._add_to_agent_memory(session_id, user_query, assistant_response, context_used)
            
            # Update conversation cache
            self._update_conversation_cache(session_id, user_query, assistant_response, context_used)
            
            logger.info(f"Added conversation turn to enhanced session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding conversation turn: {e}")
            return False
    
    def get_relevant_context(self, session_id: str, current_query: str,
                           include_session_memory: bool = True,
                           include_agent_memory: bool = True,
                           include_user_context: bool = True,
                           limit: int = 10) -> List[ContextEntry]:
        """
        Get comprehensive relevant context for current query
        
        Args:
            session_id: Current session ID
            current_query: Current user query
            include_session_memory: Include session-specific memory
            include_agent_memory: Include agent's general memory
            include_user_context: Include user-specific context
            limit: Maximum number of context entries
        
        Returns:
            List of relevant context entries
        """
        try:
            all_context = []
            
            # Get session-specific context
            if include_session_memory:
                session_context = self.session_manager.get_relevant_context(
                    session_id, current_query, limit=limit//2
                )
                all_context.extend(session_context)
            
            # Get agent memory context
            if include_agent_memory and self.config.enable_cross_session_memory:
                agent_memories = self.agent_memory.get_relevant_memories(current_query, limit=limit//3)
                agent_context = self._convert_agent_memories_to_context(agent_memories)
                all_context.extend(agent_context)
            
            # Get user-specific context
            if include_user_context:
                user_context = self._get_user_context_for_query(session_id, current_query)
                all_context.extend(user_context)
            
            # Deduplicate and rank by relevance
            unique_context = self._deduplicate_context(all_context)
            unique_context.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return unique_context[:limit]
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            return []
    
    def get_conversation_summary(self, session_id: str, 
                               include_context: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive conversation summary
        
        Args:
            session_id: Session ID
            include_context: Include context information
        
        Returns:
            Conversation summary
        """
        try:
            # Get session statistics
            session_stats = self.session_manager.get_session_statistics(session_id)
            
            # Get conversation summary from session manager
            summary = self.session_manager.get_conversation_summary(session_id)
            
            # Get agent memory summary
            agent_memory_summary = self._get_agent_memory_summary(session_id)
            
            # Get user context summary
            user_context_summary = self._get_user_context_summary(session_id)
            
            comprehensive_summary = {
                "session_id": session_id,
                "session_stats": session_stats,
                "conversation_summary": summary.to_dict() if summary else None,
                "agent_memory_summary": agent_memory_summary,
                "user_context_summary": user_context_summary,
                "generated_at": datetime.now().isoformat()
            }
            
            return comprehensive_summary
            
        except Exception as e:
            logger.error(f"Error getting conversation summary: {e}")
            return {"error": str(e)}
    
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """
        Update user preferences and learn from interactions
        
        Args:
            user_id: User ID
            preferences: User preferences to update
        
        Returns:
            Success status
        """
        try:
            if not self.config.enable_user_profiling:
                logger.info("User profiling disabled")
                return True
            
            # Update through session manager
            success = self.session_manager.update_user_preferences(user_id, preferences)
            
            if success:
                # Update user context cache
                if user_id in self.user_context_cache:
                    self.user_context_cache[user_id].update(preferences)
                else:
                    self.user_context_cache[user_id] = preferences
                
                # Add to agent memory for learning
                preference_context = create_context_entry(
                    content=f"User {user_id} preferences: {json.dumps(preferences)}",
                    context_type=ContextType.USER_PREFERENCE,
                    relevance_score=0.9,
                    metadata={"user_id": user_id, "preference_type": "explicit"}
                )
                
                self.agent_memory.add_memory(
                    content=preference_context.content,
                    metadata=preference_context.metadata
                )
            
            logger.info(f"Updated user preferences for {user_id}")
            return success
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            return False
    
    def compress_old_memories(self, user_id: Optional[str] = None, days_old: int = 30) -> int:
        """
        Compress old memories to save space
        
        Args:
            user_id: Optional user ID to compress specific user memories
            days_old: Compress memories older than this many days
        
        Returns:
            Number of memories compressed
        """
        try:
            # Get old sessions
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            if user_id:
                old_sessions = self.chat_history.search_conversations(
                    user_id=user_id,
                    date_to=cutoff_date
                )
            else:
                old_sessions = self.chat_history.search_conversations(
                    date_to=cutoff_date
                )
            
            compressed_count = 0
            
            for session in old_sessions:
                # Generate summary for old session
                summary = self._generate_session_summary(session.session_id)
                
                if summary:
                    # Store compressed summary
                    compressed_memory = create_context_entry(
                        content=summary.summary_text,
                        context_type=ContextType.RELEVANT_MEMORY,
                        relevance_score=summary.relevance_score,
                        metadata={
                            "original_session_id": session.session_id,
                            "compression_date": datetime.now().isoformat(),
                            "original_turn_count": session.total_turns,
                            "compressed": True
                        }
                    )
                    
                    self.agent_memory.add_memory(
                        content=compressed_memory.content,
                        metadata=compressed_memory.metadata
                    )
                    
                    # Optionally remove detailed session data
                    # self.chat_history.delete_conversation_session(session.session_id)
                    
                    compressed_count += 1
            
            logger.info(f"Compressed {compressed_count} old memories")
            return compressed_count
            
        except Exception as e:
            logger.error(f"Error compressing old memories: {e}")
            return 0
    
    def cleanup_expired_data(self) -> Dict[str, int]:
        """
        Clean up expired sessions and old data
        
        Returns:
            Cleanup statistics
        """
        try:
            stats = {
                "expired_sessions_cleaned": 0,
                "old_memories_compressed": 0,
                "cache_entries_cleared": 0
            }
            
            # Clean up expired sessions
            stats["expired_sessions_cleaned"] = self.session_manager.cleanup_expired_sessions()
            
            # Compress old memories
            if self.config.context_retention_days > 0:
                stats["old_memories_compressed"] = self.compress_old_memories(
                    days_old=self.config.context_retention_days
                )
            
            # Clear old cache entries
            stats["cache_entries_cleared"] = self._cleanup_cache()
            
            logger.info(f"Cleanup completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {"error": str(e)}
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics
        
        Returns:
            Memory statistics
        """
        try:
            # Session manager statistics
            session_stats = self.session_manager.get_system_stats()
            
            # Agent memory statistics
            agent_memory_stats = self.agent_memory.get_stats() if hasattr(self.agent_memory, 'get_stats') else {}
            
            # Chat history statistics
            chat_history_stats = self.chat_history.get_stats()
            
            # Cache statistics
            cache_stats = {
                "conversation_cache_entries": len(self.conversation_cache),
                "user_context_cache_entries": len(self.user_context_cache)
            }
            
            comprehensive_stats = {
                "session_manager": session_stats,
                "agent_memory": agent_memory_stats,
                "chat_history": chat_history_stats,
                "cache": cache_stats,
                "configuration": {
                    "max_conversation_memory": self.config.max_conversation_memory,
                    "context_retention_days": self.config.context_retention_days,
                    "relevance_threshold": self.config.relevance_threshold,
                    "enable_cross_session_memory": self.config.enable_cross_session_memory,
                    "enable_user_profiling": self.config.enable_user_profiling
                },
                "generated_at": datetime.now().isoformat()
            }
            
            return comprehensive_stats
            
        except Exception as e:
            logger.error(f"Error getting memory statistics: {e}")
            return {"error": str(e)}
    
    def _add_to_agent_memory(self, session_id: str, user_query: str, assistant_response: str,
                             context_used: Optional[List[ContextEntry]] = None) -> None:
        """Add conversation to agent memory"""
        try:
            # Create memory content
            memory_content = f"User: {user_query}\nAssistant: {assistant_response}"
            
            # Add context information
            if context_used:
                context_summary = " ".join([ctx.content for ctx in context_used[:3]])
                memory_content += f"\nContext: {context_summary}"
            
            # Add to agent memory
            self.agent_memory.add_memory(
                content=memory_content,
                metadata={
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "has_context": bool(context_used),
                    "context_count": len(context_used) if context_used else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Error adding to agent memory: {e}")
    
    def _update_conversation_cache(self, session_id: str, user_query: str, 
                                 assistant_response: str, context_used: Optional[List[ContextEntry]]) -> None:
        """Update conversation cache"""
        try:
            with self._lock:
                if session_id not in self.conversation_cache:
                    self.conversation_cache[session_id] = []
                
                # Add conversation entry
                entry = {
                    "user_query": user_query,
                    "assistant_response": assistant_response,
                    "context_used": [ctx.to_dict() for ctx in context_used] if context_used else [],
                    "timestamp": datetime.now().isoformat()
                }
                
                self.conversation_cache[session_id].append(entry)
                
                # Limit cache size
                if len(self.conversation_cache[session_id]) > self.config.max_conversation_memory:
                    self.conversation_cache[session_id] = self.conversation_cache[session_id][-self.config.max_conversation_memory:]
            
        except Exception as e:
            logger.error(f"Error updating conversation cache: {e}")
    
    def _convert_agent_memories_to_context(self, memories: List[Dict[str, Any]]) -> List[ContextEntry]:
        """Convert agent memories to context entries"""
        try:
            context_entries = []
            
            for memory in memories:
                relevance_score = memory.get("relevance_score", 0.5)
                
                # Boost relevance for recent memories
                memory_timestamp = memory.get("metadata", {}).get("timestamp")
                if memory_timestamp:
                    try:
                        memory_date = datetime.fromisoformat(memory_timestamp)
                        days_old = (datetime.now() - memory_date).days
                        
                        # Boost relevance for memories within the last 7 days
                        if days_old <= 7:
                            relevance_score = min(1.0, relevance_score + 0.2)
                    except (ValueError, TypeError):
                        pass
                
                context_entry = create_context_entry(
                    content=memory.get("content", ""),
                    context_type=ContextType.RELEVANT_MEMORY,
                    relevance_score=relevance_score,
                    metadata={
                        "memory_id": memory.get("id", ""),
                        "memory_type": "agent_memory",
                        "original_timestamp": memory_timestamp,
                        **memory.get("metadata", {})
                    }
                )
                
                context_entries.append(context_entry)
            
            return context_entries
            
        except Exception as e:
            logger.error(f"Error converting agent memories to context: {e}")
            return []
    
    def _load_user_context(self, user_id: str) -> Dict[str, Any]:
        """Load user context from database and cache"""
        try:
            user_profile = self.session_manager.get_user_profile(user_id)
            if not user_profile:
                return {}
            
            # Get recent sessions for context
            recent_sessions = self.chat_history.get_user_sessions(user_id, limit=5)
            
            # Extract preferences and patterns
            user_context = {
                "preferences": user_profile.preferences,
                "total_sessions": user_profile.total_sessions,
                "last_active": user_profile.last_active.isoformat(),
                "recent_session_ids": [s.session_id for s in recent_sessions],
                "common_topics": self._extract_common_topics(recent_sessions),
                "interaction_patterns": self._extract_interaction_patterns(recent_sessions)
            }
            
            return user_context
            
        except Exception as e:
            logger.error(f"Error loading user context for {user_id}: {e}")
            return {}
    
    def _get_user_context_for_query(self, session_id: str, current_query: str) -> List[ContextEntry]:
        """Get user-specific context for current query"""
        try:
            # Get session to find user ID
            session = self.session_manager.get_session(session_id)
            if not session or not session.user_id:
                return []
            
            user_id = session.user_id
            
            # Get user context from cache or load from database
            if user_id not in self.user_context_cache:
                self.user_context_cache[user_id] = self._load_user_context(user_id)
            
            user_context = self.user_context_cache[user_id]
            
            # Create context entries from user preferences
            context_entries = []
            
            # Add preferences as context
            if user_context.get("preferences"):
                preferences_context = create_context_entry(
                    content=f"User preferences: {json.dumps(user_context['preferences'])}",
                    context_type=ContextType.USER_PREFERENCE,
                    relevance_score=0.8,
                    metadata={"user_id": user_id, "context_type": "preferences"}
                )
                context_entries.append(preferences_context)
            
            # Add common topics as context
            common_topics = user_context.get("common_topics", [])
            if common_topics:
                topics_context = create_context_entry(
                    content=f"Common topics: {', '.join(common_topics[:5])}",
                    context_type=ContextType.USER_PREFERENCE,
                    relevance_score=0.7,
                    metadata={"user_id": user_id, "context_type": "common_topics"}
                )
                context_entries.append(topics_context)
            
            return context_entries
            
        except Exception as e:
            logger.error(f"Error getting user context for query: {e}")
            return []
    
    def _extract_common_topics(self, sessions: List[ConversationSession]) -> List[str]:
        """Extract common topics from user sessions"""
        try:
            topic_counter = {}
            
            for session in sessions:
                for turn in session.turns:
                    if turn.user_message and turn.user_message.content:
                        # Simple keyword extraction (in practice, use NLP)
                        words = turn.user_message.content.lower().split()
                        for word in words:
                            if len(word) > 4:  # Filter short words
                                topic_counter[word] = topic_counter.get(word, 0) + 1
            
            # Get top topics
            sorted_topics = sorted(topic_counter.items(), key=lambda x: x[1], reverse=True)
            return [topic for topic, count in sorted_topics[:10]]
            
        except Exception as e:
            logger.error(f"Error extracting common topics: {e}")
            return []
    
    def _extract_interaction_patterns(self, sessions: List[ConversationSession]) -> Dict[str, Any]:
        """Extract interaction patterns from user sessions"""
        try:
            patterns = {
                "avg_session_length": 0,
                "avg_response_time": 0,
                "preferred_tools": {},
                "context_usage_rate": 0
            }
            
            total_turns = 0
            total_response_time = 0
            tool_usage = {}
            context_usage_count = 0
            
            for session in sessions:
                total_turns += session.total_turns
                
                for turn in session.turns:
                    if turn.processing_time:
                        total_response_time += turn.processing_time
                    
                    for tool in turn.tools_used:
                        tool_usage[tool] = tool_usage.get(tool, 0) + 1
                    
                    if turn.context_used:
                        context_usage_count += 1
            
            if sessions:
                patterns["avg_session_length"] = total_turns / len(sessions)
                patterns["avg_response_time"] = total_response_time / total_turns if total_turns > 0 else 0
                patterns["preferred_tools"] = dict(sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:5])
                patterns["context_usage_rate"] = context_usage_count / total_turns if total_turns > 0 else 0
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error extracting interaction patterns: {e}")
            return {}
    
    def _generate_session_summary(self, session_id: str) -> Optional[ConversationSummary]:
        """Generate summary for a session"""
        try:
            session = self.session_manager.get_session(session_id)
            if not session or session.total_turns < 2:
                return None
            
            # Simple summary generation (in practice, use an LLM)
            recent_turns = session.get_recent_turns(limit=5)
            
            summary_text = f"Conversation with {session.total_turns} turns. "
            
            # Extract key topics
            topics = []
            for turn in recent_turns:
                if turn.user_message and turn.user_message.content:
                    # Simple topic extraction
                    words = turn.user_message.content.lower().split()
                    topics.extend([w for w in words if len(w) > 5][:3])
            
            # Get important entities from context
            entities = []
            for turn in recent_turns:
                for context in turn.context_used:
                    if context.relevance_score > 0.8:
                        entities.append(context.content[:50])
            
            summary = ConversationSummary(
                session_id=session_id,
                summary_text=summary_text + f"Topics: {', '.join(list(set(topics))[:5])}",
                key_topics=list(set(topics))[:5],
                important_entities=list(set(entities))[:5],
                sentiment_score=0.5,  # Default neutral
                relevance_score=0.8,
                metadata={
                    "generated_by": "enhanced_memory_manager",
                    "turn_count": session.total_turns,
                    "user_id": session.user_id
                }
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating session summary: {e}")
            return None
    
    def _get_agent_memory_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of agent memory for session"""
        try:
            # Get recent memories related to this session
            session_memories = []
            
            # This is a simplified implementation
            # In practice, you might query memories by session_id or content similarity
            
            return {
                "total_memories": len(session_memories),
                "recent_memories": session_memories[:5],
                "memory_types": {}
            }
            
        except Exception as e:
            logger.error(f"Error getting agent memory summary: {e}")
            return {}
    
    def _get_user_context_summary(self, session_id: str) -> Dict[str, Any]:
        """Get user context summary for session"""
        try:
            session = self.session_manager.get_session(session_id)
            if not session or not session.user_id:
                return {}
            
            user_id = session.user_id
            
            if user_id not in self.user_context_cache:
                self.user_context_cache[user_id] = self._load_user_context(user_id)
            
            return self.user_context_cache[user_id]
            
        except Exception as e:
            logger.error(f"Error getting user context summary: {e}")
            return {}
    
    def _deduplicate_context(self, context_entries: List[ContextEntry]) -> List[ContextEntry]:
        """Remove duplicate context entries"""
        try:
            seen_content = set()
            unique_context = []
            
            for entry in context_entries:
                # Create a hash of the content (first 100 chars for efficiency)
                content_hash = hash(entry.content[:100].lower().strip())
                
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_context.append(entry)
            
            return unique_context
            
        except Exception as e:
            logger.error(f"Error deduplicating context: {e}")
            return context_entries
    
    def _cleanup_cache(self) -> int:
        """Clean up old cache entries"""
        try:
            with self._lock:
                cleared_count = 0
                
                # Clean conversation cache
                current_time = datetime.now()
                
                for session_id in list(self.conversation_cache.keys()):
                    # Keep only recent conversations (last 7 days)
                    recent_entries = []
                    for entry in self.conversation_cache[session_id]:
                        try:
                            entry_time = datetime.fromisoformat(entry.get("timestamp", ""))
                            if (current_time - entry_time).days <= 7:
                                recent_entries.append(entry)
                        except (ValueError, TypeError):
                            continue
                    
                    if len(recent_entries) < len(self.conversation_cache[session_id]):
                        cleared_count += len(self.conversation_cache[session_id]) - len(recent_entries)
                        self.conversation_cache[session_id] = recent_entries
                
                # Clean user context cache (keep active users)
                active_users = set()
                for session_id, session in self.session_manager.active_sessions.items():
                    if session.user_id:
                        active_users.add(session.user_id)
                
                for user_id in list(self.user_context_cache.keys()):
                    if user_id not in active_users:
                        del self.user_context_cache[user_id]
                        cleared_count += 1
                
                return cleared_count
                
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
            return 0

# Factory function for creating enhanced memory manager
def create_enhanced_memory_manager(
    connection_string: Optional[str] = None,
    memory_config: Optional[MemoryConfig] = None,
    **kwargs
) -> EnhancedMemoryManager:
    """
    Factory function to create enhanced memory manager
    
    Args:
        connection_string: MongoDB connection string
        memory_config: Memory configuration
        **kwargs: Additional arguments for EnhancedMemoryManager
    
    Returns:
        EnhancedMemoryManager instance
    """
    connection_string = connection_string or os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017/")
    
    return EnhancedMemoryManager(
        connection_string=connection_string,
        memory_config=memory_config,
        **kwargs
    )