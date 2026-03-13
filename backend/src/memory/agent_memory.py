from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    timestamp: datetime
    user_query: str
    agent_response: str
    metadata: Dict[str, Any]
    session_id: str
    turn_number: int


@dataclass
class MemoryEntry:
    """Represents a memory entry with context and importance."""
    key: str
    value: Any
    context: str
    importance: float  # 0.0 to 1.0
    timestamp: datetime
    access_count: int
    last_accessed: datetime


class AgentMemory:
    """Conversation memory system for agents."""
    
    def __init__(self, max_conversation_history: int = 10, max_memory_entries: int = 50):
        self.max_conversation_history = max_conversation_history
        self.max_memory_entries = max_memory_entries
        
        # Conversation history (per session)
        self.conversation_history: Dict[str, deque] = {}
        
        # General memory (key-value pairs with metadata)
        self.memory_store: Dict[str, MemoryEntry] = {}
        
        # Current session
        self.current_session_id: Optional[str] = None
    
    def start_session(self, session_id: str) -> None:
        """Start a new conversation session."""
        self.current_session_id = session_id
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = deque(maxlen=self.max_conversation_history)
        logger.info(f"Started new session: {session_id}")
    
    def end_session(self, session_id: str) -> None:
        """End a conversation session."""
        if session_id in self.conversation_history:
            logger.info(f"Ended session: {session_id} with {len(self.conversation_history[session_id])} turns")
        else:
            logger.warning(f"Attempted to end non-existent session: {session_id}")
    
    def add_conversation_turn(self, user_query: str, agent_response: str, 
                            metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a conversation turn to the current session."""
        if not self.current_session_id:
            logger.warning("No active session. Starting default session.")
            self.start_session("default")
        
        session_history = self.conversation_history[self.current_session_id]
        turn_number = len(session_history) + 1
        
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_query=user_query,
            agent_response=agent_response,
            metadata=metadata or {},
            session_id=self.current_session_id,
            turn_number=turn_number
        )
        
        session_history.append(turn)
        logger.info(f"Added turn {turn_number} to session {self.current_session_id}")
    
    def get_conversation_context(self, session_id: Optional[str] = None, 
                               last_n: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation context."""
        session_id = session_id or self.current_session_id
        
        if not session_id or session_id not in self.conversation_history:
            return []
        
        history = self.conversation_history[session_id]
        recent_turns = list(history)[-last_n:]  # Get last n turns
        
        return [
            {
                "user": turn.user_query,
                "agent": turn.agent_response,
                "turn_number": turn.turn_number,
                "timestamp": turn.timestamp.isoformat(),
                "metadata": turn.metadata
            }
            for turn in recent_turns
        ]
    
    def add_memory(self, key: str, value: Any, context: str = "", 
                   importance: float = 0.5) -> None:
        """Add a memory entry."""
        now = datetime.now()
        
        if key in self.memory_store:
            # Update existing entry
            entry = self.memory_store[key]
            entry.value = value
            entry.context = context
            entry.importance = importance
            entry.timestamp = now
            entry.last_accessed = now
        else:
            # Create new entry
            entry = MemoryEntry(
                key=key,
                value=value,
                context=context,
                importance=importance,
                timestamp=now,
                access_count=0,
                last_accessed=now
            )
            
            # Check memory limit
            if len(self.memory_store) >= self.max_memory_entries:
                self._evict_least_important_memory()
            
            self.memory_store[key] = entry
        
        logger.info(f"Added/updated memory: {key} (importance: {importance})")
    
    def get_memory(self, key: str) -> Optional[Any]:
        """Get a memory value by key."""
        if key in self.memory_store:
            entry = self.memory_store[key]
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            logger.info(f"Accessed memory: {key} (count: {entry.access_count})")
            return entry.value
        return None
    
    def get_relevant_memories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get memories relevant to a query."""
        relevant_memories = []
        query_lower = query.lower()
        
        for key, entry in self.memory_store.items():
            # Simple relevance scoring based on keyword matching
            relevance_score = 0.0
            
            # Check if query terms appear in key, value, or context
            if query_lower in key.lower():
                relevance_score += 0.4
            
            if isinstance(entry.value, str) and query_lower in entry.value.lower():
                relevance_score += 0.3
            
            if query_lower in entry.context.lower():
                relevance_score += 0.2
            
            # Factor in importance and recency
            time_factor = self._calculate_time_factor(entry.timestamp)
            final_score = (relevance_score * 0.7) + (entry.importance * 0.2) + (time_factor * 0.1)
            
            if final_score > 0.1:  # Threshold for relevance
                relevant_memories.append({
                    "key": key,
                    "value": entry.value,
                    "context": entry.context,
                    "importance": entry.importance,
                    "relevance_score": final_score,
                    "access_count": entry.access_count,
                    "timestamp": entry.timestamp.isoformat()
                })
        
        # Sort by relevance score and return top results
        relevant_memories.sort(key=lambda x: x["relevance_score"], reverse=True)
        return relevant_memories[:limit]
    
    def _calculate_time_factor(self, timestamp: datetime) -> float:
        """Calculate time-based relevance factor."""
        now = datetime.now()
        age = (now - timestamp).total_seconds()
        
        # Exponential decay: newer memories are more relevant
        # Half-life of 7 days (604800 seconds)
        half_life = 604800
        return max(0.0, 1.0 * (0.5 ** (age / half_life)))
    
    def _evict_least_important_memory(self) -> None:
        """Evict the least important memory to make space."""
        if not self.memory_store:
            return
        
        # Find memory with lowest importance score
        least_important_key = min(
            self.memory_store.keys(),
            key=lambda k: self.memory_store[k].importance
        )
        
        del self.memory_store[least_important_key]
        logger.info(f"Evicted least important memory: {least_important_key}")
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of a conversation session."""
        if session_id not in self.conversation_history:
            return {"error": "Session not found"}
        
        history = self.conversation_history[session_id]
        if not history:
            return {"error": "Session is empty"}
        
        # Analyze conversation
        total_turns = len(history)
        duration = history[-1].timestamp - history[0].timestamp
        
        # Extract key topics (simple approach)
        all_queries = " ".join([turn.user_query for turn in history])
        words = all_queries.lower().split()
        common_words = ["the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "a", "an", "is", "are", "was", "were"]
        topics = [word for word in words if word not in common_words and len(word) > 3]
        
        from collections import Counter
        topic_counts = Counter(topics)
        
        return {
            "session_id": session_id,
            "total_turns": total_turns,
            "duration_seconds": duration.total_seconds(),
            "start_time": history[0].timestamp.isoformat(),
            "end_time": history[-1].timestamp.isoformat(),
            "topics": dict(topic_counts.most_common(5)),
            "first_query": history[0].user_query,
            "last_response": history[-1].agent_response
        }
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of memory usage."""
        total_memories = len(self.memory_store)
        total_sessions = len(self.conversation_history)
        
        if total_memories == 0:
            return {"total_memories": 0, "total_sessions": total_sessions}
        
        # Calculate average importance
        avg_importance = sum(entry.importance for entry in self.memory_store.values()) / total_memories
        
        # Find most accessed memory
        most_accessed = max(self.memory_store.values(), key=lambda x: x.access_count)
        
        # Memory by importance ranges
        high_importance = sum(1 for entry in self.memory_store.values() if entry.importance > 0.7)
        medium_importance = sum(1 for entry in self.memory_store.values() if 0.3 <= entry.importance <= 0.7)
        low_importance = sum(1 for entry in self.memory_store.values() if entry.importance < 0.3)
        
        return {
            "total_memories": total_memories,
            "total_sessions": total_sessions,
            "average_importance": avg_importance,
            "most_accessed_key": most_accessed.key,
            "most_accessed_count": most_accessed.access_count,
            "importance_distribution": {
                "high": high_importance,
                "medium": medium_importance,
                "low": low_importance
            }
        }
    
    def clear_memory(self, key: Optional[str] = None) -> bool:
        """Clear specific memory or all memories."""
        if key:
            if key in self.memory_store:
                del self.memory_store[key]
                logger.info(f"Cleared memory: {key}")
                return True
            return False
        else:
            # Clear all memories
            self.memory_store.clear()
            logger.info("Cleared all memories")
            return True
    
    def clear_session_history(self, session_id: str) -> bool:
        """Clear conversation history for a specific session."""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            logger.info(f"Cleared session history: {session_id}")
            return True
        return False