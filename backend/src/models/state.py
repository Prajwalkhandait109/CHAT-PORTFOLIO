from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import uuid

class MessageType(Enum):
    """Types of messages in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    ERROR = "error"

class SessionStatus(Enum):
    """Status of a conversation session"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    ARCHIVED = "archived"

class ContextType(Enum):
    """Types of context information"""
    RELEVANT_MEMORY = "relevant_memory"
    RETRIEVED_DOCUMENT = "retrieved_document"
    TOOL_RESULT = "tool_result"
    PREVIOUS_QUERY = "previous_query"
    USER_PREFERENCE = "user_preference"

@dataclass
class Message:
    """Individual message in a conversation"""
    id: str
    type: MessageType
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = MessageType(self.type)
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)

@dataclass
class ContextEntry:
    """Contextual information for conversation"""
    id: str
    type: ContextType
    content: str
    relevance_score: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = ContextType(self.type)
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)

@dataclass
class ConversationTurn:
    """A complete conversation turn (user query + assistant response)"""
    id: str
    user_message: Message
    assistant_message: Optional[Message] = None
    context_used: List[ContextEntry] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    processing_time: Optional[float] = None  # seconds
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationSession:
    """Complete conversation session with all turns and metadata"""
    session_id: str
    user_id: Optional[str] = None
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    turns: List[ConversationTurn] = field(default_factory=list)
    total_turns: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = SessionStatus(self.status)
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at)
        if isinstance(self.expires_at, str):
            self.expires_at = datetime.fromisoformat(self.expires_at)
    
    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a conversation turn to the session"""
        self.turns.append(turn)
        self.total_turns += 1
        self.updated_at = datetime.now()
    
    def get_recent_turns(self, limit: int = 5) -> List[ConversationTurn]:
        """Get the most recent conversation turns"""
        return self.turns[-limit:] if self.turns else []
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of context used across all turns"""
        context_summary = {
            "total_context_entries": 0,
            "context_types": {},
            "avg_relevance_score": 0.0,
            "recent_queries": []
        }
        
        total_relevance = 0.0
        total_entries = 0
        
        for turn in self.turns:
            for context in turn.context_used:
                total_entries += 1
                total_relevance += context.relevance_score
                
                context_type = context.type.value
                context_summary["context_types"][context_type] = context_summary["context_types"].get(context_type, 0) + 1
            
            # Collect recent queries
            if turn.user_message and turn.user_message.content:
                context_summary["recent_queries"].append({
                    "query": turn.user_message.content,
                    "timestamp": turn.timestamp.isoformat()
                })
        
        context_summary["total_context_entries"] = total_entries
        context_summary["avg_relevance_score"] = total_relevance / total_entries if total_entries > 0 else 0.0
        context_summary["recent_queries"] = context_summary["recent_queries"][-5:]  # Last 5 queries
        
        return context_summary
    
    def is_expired(self) -> bool:
        """Check if the session has expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for storage"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "total_turns": self.total_turns,
            "metadata": self.metadata,
            "turns": [
                {
                    "id": turn.id,
                    "user_message": {
                        "id": turn.user_message.id,
                        "type": turn.user_message.type.value,
                        "content": turn.user_message.content,
                        "timestamp": turn.user_message.timestamp.isoformat(),
                        "metadata": turn.user_message.metadata
                    },
                    "assistant_message": {
                        "id": turn.assistant_message.id,
                        "type": turn.assistant_message.type.value,
                        "content": turn.assistant_message.content,
                        "timestamp": turn.assistant_message.timestamp.isoformat(),
                        "metadata": turn.assistant_message.metadata
                    } if turn.assistant_message else None,
                    "context_used": [
                        {
                            "id": ctx.id,
                            "type": ctx.type.value,
                            "content": ctx.content,
                            "relevance_score": ctx.relevance_score,
                            "timestamp": ctx.timestamp.isoformat(),
                            "metadata": ctx.metadata
                        } for ctx in turn.context_used
                    ],
                    "tools_used": turn.tools_used,
                    "processing_time": turn.processing_time,
                    "timestamp": turn.timestamp.isoformat(),
                    "metadata": turn.metadata
                } for turn in self.turns
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationSession':
        """Create session from dictionary"""
        turns = []
        for turn_data in data.get("turns", []):
            user_msg_data = turn_data["user_message"]
            user_message = Message(
                id=user_msg_data["id"],
                type=MessageType(user_msg_data["type"]),
                content=user_msg_data["content"],
                timestamp=datetime.fromisoformat(user_msg_data["timestamp"]),
                metadata=user_msg_data.get("metadata", {})
            )
            
            assistant_message = None
            if turn_data.get("assistant_message"):
                assistant_msg_data = turn_data["assistant_message"]
                assistant_message = Message(
                    id=assistant_msg_data["id"],
                    type=MessageType(assistant_msg_data["type"]),
                    content=assistant_msg_data["content"],
                    timestamp=datetime.fromisoformat(assistant_msg_data["timestamp"]),
                    metadata=assistant_msg_data.get("metadata", {})
                )
            
            context_used = []
            for ctx_data in turn_data.get("context_used", []):
                context_entry = ContextEntry(
                    id=ctx_data["id"],
                    type=ContextType(ctx_data["type"]),
                    content=ctx_data["content"],
                    relevance_score=ctx_data["relevance_score"],
                    timestamp=datetime.fromisoformat(ctx_data["timestamp"]),
                    metadata=ctx_data.get("metadata", {})
                )
                context_used.append(context_entry)
            
            turn = ConversationTurn(
                id=turn_data["id"],
                user_message=user_message,
                assistant_message=assistant_message,
                context_used=context_used,
                tools_used=turn_data.get("tools_used", []),
                processing_time=turn_data.get("processing_time"),
                timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                metadata=turn_data.get("metadata", {})
            )
            turns.append(turn)
        
        return cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            status=SessionStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            turns=turns,
            total_turns=data.get("total_turns", len(turns)),
            metadata=data.get("metadata", {})
        )

@dataclass
class UserProfile:
    """User profile for personalized conversations"""
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[str] = field(default_factory=list)
    last_active: datetime = field(default_factory=datetime.now)
    total_sessions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.last_active, str):
            self.last_active = datetime.fromisoformat(self.last_active)
    
    def update_preferences(self, preferences: Dict[str, Any]) -> None:
        """Update user preferences"""
        self.preferences.update(preferences)
        self.last_active = datetime.now()
    
    def add_conversation_session(self, session_id: str) -> None:
        """Add conversation session to history"""
        self.conversation_history.append(session_id)
        self.total_sessions += 1
        self.last_active = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
            "conversation_history": self.conversation_history,
            "last_active": self.last_active.isoformat(),
            "total_sessions": self.total_sessions,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Create from dictionary"""
        return cls(
            user_id=data["user_id"],
            preferences=data.get("preferences", {}),
            conversation_history=data.get("conversation_history", []),
            last_active=datetime.fromisoformat(data["last_active"]) if data.get("last_active") else datetime.now(),
            total_sessions=data.get("total_sessions", 0),
            metadata=data.get("metadata", {})
        )

@dataclass
class ConversationSummary:
    """Summary of conversation for quick context retrieval"""
    session_id: str
    summary_text: str
    key_topics: List[str]
    important_entities: List[str]
    sentiment_score: float  # -1 to 1
    relevance_score: float  # 0 to 1
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "session_id": self.session_id,
            "summary_text": self.summary_text,
            "key_topics": self.key_topics,
            "important_entities": self.important_entities,
            "sentiment_score": self.sentiment_score,
            "relevance_score": self.relevance_score,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationSummary':
        """Create from dictionary"""
        return cls(
            session_id=data["session_id"],
            summary_text=data["summary_text"],
            key_topics=data.get("key_topics", []),
            important_entities=data.get("important_entities", []),
            sentiment_score=data.get("sentiment_score", 0.0),
            relevance_score=data.get("relevance_score", 0.0),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            metadata=data.get("metadata", {})
        )

# Utility functions for conversation management
def create_new_session(user_id: Optional[str] = None, expires_in_hours: int = 24) -> ConversationSession:
    """Create a new conversation session"""
    session_id = str(uuid.uuid4())
    expires_at = datetime.now() + timedelta(hours=expires_in_hours) if expires_in_hours > 0 else None
    
    return ConversationSession(
        session_id=session_id,
        user_id=user_id,
        expires_at=expires_at,
        metadata={"created_by": "system", "version": "1.0"}
    )

def create_message(content: str, message_type: MessageType, metadata: Optional[Dict[str, Any]] = None) -> Message:
    """Create a new message"""
    return Message(
        id=str(uuid.uuid4()),
        type=message_type,
        content=content,
        timestamp=datetime.now(),
        metadata=metadata or {}
    )

def create_context_entry(content: str, context_type: ContextType, relevance_score: float, metadata: Optional[Dict[str, Any]] = None) -> ContextEntry:
    """Create a new context entry"""
    return ContextEntry(
        id=str(uuid.uuid4()),
        type=context_type,
        content=content,
        relevance_score=relevance_score,
        timestamp=datetime.now(),
        metadata=metadata or {}
    )

# Export all models
__all__ = [
    'MessageType',
    'SessionStatus',
    'ContextType',
    'Message',
    'ContextEntry',
    'ConversationTurn',
    'ConversationSession',
    'UserProfile',
    'ConversationSummary',
    'create_new_session',
    'create_message',
    'create_context_entry'
]