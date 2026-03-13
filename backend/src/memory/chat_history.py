import os
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
import threading
from dataclasses import asdict

try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
    from pymongo.errors import ConnectionFailure, DuplicateKeyError, PyMongoError
    from pymongo.collection import Collection
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logging.warning("MongoDB not available. Install with: pip install pymongo motor")

from ..models.state import (
    ConversationSession, ConversationTurn, Message, ContextEntry,
    UserProfile, ConversationSummary, SessionStatus, MessageType, ContextType
)

logger = logging.getLogger(__name__)

class MongoChatHistory:
    """MongoDB-backed conversation history storage"""
    
    def __init__(self, 
                 connection_string: str = "mongodb://localhost:27017/",
                 database_name: str = "chatbot_history",
                 conversations_collection: str = "conversations",
                 users_collection: str = "users",
                 summaries_collection: str = "summaries"):
        """
        Initialize MongoDB chat history storage
        
        Args:
            connection_string: MongoDB connection string
            database_name: Database name
            conversations_collection: Collection name for conversations
            users_collection: Collection name for user profiles
            summaries_collection: Collection name for conversation summaries
        """
        if not MONGODB_AVAILABLE:
            raise ImportError("MongoDB dependencies not available. Install with: pip install pymongo motor")
        
        self.connection_string = connection_string
        self.database_name = database_name
        self.conversations_collection_name = conversations_collection
        self.users_collection_name = users_collection
        self.summaries_collection_name = summaries_collection
        
        self.client = None
        self.database = None
        self.conversations_collection = None
        self.users_collection = None
        self.summaries_collection = None
        
        self.is_connected = False
        self._lock = threading.RLock()
        
        # TTL indexes for automatic cleanup
        self.ttl_indexes_created = False
    
    def connect(self) -> bool:
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            
            # Test connection
            self.client.admin.command('ping')
            
            self.database = self.client[self.database_name]
            self.conversations_collection = self.database[self.conversations_collection_name]
            self.users_collection = self.database[self.users_collection_name]
            self.summaries_collection = self.database[self.summaries_collection_name]
            
            self.is_connected = True
            
            # Create indexes
            self._create_indexes()
            
            logger.info(f"Connected to MongoDB database: {self.database_name}")
            return True
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from MongoDB"""
        try:
            if self.client:
                self.client.close()
            
            self.is_connected = False
            logger.info("Disconnected from MongoDB")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from MongoDB: {e}")
            return False
    
    def _create_indexes(self):
        """Create necessary indexes for optimal performance"""
        try:
            with self._lock:
                # Conversations collection indexes
                self.conversations_collection.create_index([("session_id", ASCENDING)], unique=True)
                self.conversations_collection.create_index([("user_id", ASCENDING)])
                self.conversations_collection.create_index([("created_at", DESCENDING)])
                self.conversations_collection.create_index([("updated_at", DESCENDING)])
                self.conversations_collection.create_index([("status", ASCENDING)])
                
                # TTL index for automatic expiration
                if not self.ttl_indexes_created:
                    self.conversations_collection.create_index(
                        [("expires_at", ASCENDING)], 
                        expireAfterSeconds=0
                    )
                
                # Users collection indexes
                self.users_collection.create_index([("user_id", ASCENDING)], unique=True)
                self.users_collection.create_index([("last_active", DESCENDING)])
                
                # Summaries collection indexes
                self.summaries_collection.create_index([("session_id", ASCENDING)], unique=True)
                self.summaries_collection.create_index([("timestamp", DESCENDING)])
                
                self.ttl_indexes_created = True
                logger.info("MongoDB indexes created successfully")
                
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    def save_conversation_session(self, session: ConversationSession) -> bool:
        """Save a conversation session"""
        if not self.is_connected:
            logger.error("Not connected to MongoDB")
            return False
        
        try:
            with self._lock:
                session_dict = session.to_dict()
                
                # Use upsert to handle both insert and update
                result = self.conversations_collection.update_one(
                    {"session_id": session.session_id},
                    {"$set": session_dict},
                    upsert=True
                )
                
                logger.info(f"Saved conversation session: {session.session_id}")
                return True
                
        except DuplicateKeyError:
            logger.warning(f"Conversation session already exists: {session.session_id}")
            return self.update_conversation_session(session)
        except Exception as e:
            logger.error(f"Error saving conversation session: {e}")
            return False
    
    def get_conversation_session(self, session_id: str) -> Optional[ConversationSession]:
        """Retrieve a conversation session by ID"""
        if not self.is_connected:
            logger.error("Not connected to MongoDB")
            return None
        
        try:
            with self._lock:
                doc = self.conversations_collection.find_one({"session_id": session_id})
                
                if doc:
                    return ConversationSession.from_dict(doc)
                else:
                    logger.warning(f"Conversation session not found: {session_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving conversation session: {e}")
            return None
    
    def get_user_sessions(self, user_id: str, limit: int = 10, skip: int = 0) -> List[ConversationSession]:
        """Get all conversation sessions for a user"""
        if not self.is_connected:
            logger.error("Not connected to MongoDB")
            return []
        
        try:
            with self._lock:
                cursor = self.conversations_collection.find(
                    {"user_id": user_id}
                ).sort("created_at", DESCENDING).skip(skip).limit(limit)
                
                sessions = []
                for doc in cursor:
                    session = ConversationSession.from_dict(doc)
                    sessions.append(session)
                
                logger.info(f"Retrieved {len(sessions)} sessions for user: {user_id}")
                return sessions
                
        except Exception as e:
            logger.error(f"Error retrieving user sessions: {e}")
            return []
    
    def get_active_sessions(self, user_id: Optional[str] = None, limit: int = 50) -> List[ConversationSession]:
        """Get active conversation sessions"""
        if not self.is_connected:
            logger.error("Not connected to MongoDB")
            return []
        
        try:
            with self._lock:
                query = {"status": SessionStatus.ACTIVE.value}
                if user_id:
                    query["user_id"] = user_id
                
                cursor = self.conversations_collection.find(query).sort("updated_at", DESCENDING).limit(limit)
                
                sessions = []
                for doc in cursor:
                    session = ConversationSession.from_dict(doc)
                    if not session.is_expired():
                        sessions.append(session)
                
                logger.info(f"Retrieved {len(sessions)} active sessions")
                return sessions
                
        except Exception as e:
            logger.error(f"Error retrieving active sessions: {e}")
            return []
    
    def update_session_status(self, session_id: str, status: SessionStatus) -> bool:
        """Update the status of a conversation session"""
        if not self.is_connected:
            logger.error("Not connected to MongoDB")
            return False
        
        try:
            with self._lock:
                result = self.conversations_collection.update_one(
                    {"session_id": session_id},
                    {
                        "$set": {
                            "status": status.value,
                            "updated_at": datetime.now().isoformat()
                        }
                    }
                )
                
                if result.modified_count > 0:
                    logger.info(f"Updated session status: {session_id} -> {status.value}")
                    return True
                else:
                    logger.warning(f"Session not found or status unchanged: {session_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error updating session status: {e}")
            return False
    
    def add_conversation_turn(self, session_id: str, turn: ConversationTurn) -> bool:
        """Add a conversation turn to an existing session"""
        if not self.is_connected:
            logger.error("Not connected to MongoDB")
            return False
        
        try:
            with self._lock:
                turn_dict = {
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
                }
                
                result = self.conversations_collection.update_one(
                    {"session_id": session_id},
                    {
                        "$push": {"turns": turn_dict},
                        "$inc": {"total_turns": 1},
                        "$set": {"updated_at": datetime.now().isoformat()}
                    }
                )
                
                if result.modified_count > 0:
                    logger.info(f"Added conversation turn to session: {session_id}")
                    return True
                else:
                    logger.warning(f"Session not found: {session_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error adding conversation turn: {e}")
            return False
    
    def save_user_profile(self, user_profile: UserProfile) -> bool:
        """Save or update user profile"""
        if not self.is_connected:
            logger.error("Not connected to MongoDB")
            return False
        
        try:
            with self._lock:
                user_dict = user_profile.to_dict()
                
                result = self.users_collection.update_one(
                    {"user_id": user_profile.user_id},
                    {"$set": user_dict},
                    upsert=True
                )
                
                logger.info(f"Saved user profile: {user_profile.user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving user profile: {e}")
            return False
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve user profile"""
        if not self.is_connected:
            logger.error("Not connected to MongoDB")
            return None
        
        try:
            with self._lock:
                doc = self.users_collection.find_one({"user_id": user_id})
                
                if doc:
                    return UserProfile.from_dict(doc)
                else:
                    logger.warning(f"User profile not found: {user_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving user profile: {e}")
            return None
    
    def save_conversation_summary(self, summary: ConversationSummary) -> bool:
        """Save conversation summary"""
        if not self.is_connected:
            logger.error("Not connected to MongoDB")
            return False
        
        try:
            with self._lock:
                summary_dict = summary.to_dict()
                
                result = self.summaries_collection.update_one(
                    {"session_id": summary.session_id},
                    {"$set": summary_dict},
                    upsert=True
                )
                
                logger.info(f"Saved conversation summary: {summary.session_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving conversation summary: {e}")
            return False
    
    def get_conversation_summary(self, session_id: str) -> Optional[ConversationSummary]:
        """Retrieve conversation summary"""
        if not self.is_connected:
            logger.error("Not connected to MongoDB")
            return None
        
        try:
            with self._lock:
                doc = self.summaries_collection.find_one({"session_id": session_id})
                
                if doc:
                    return ConversationSummary.from_dict(doc)
                else:
                    logger.warning(f"Conversation summary not found: {session_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving conversation summary: {e}")
            return None
    
    def search_conversations(self, 
                           user_id: Optional[str] = None,
                           query_text: Optional[str] = None,
                           date_from: Optional[datetime] = None,
                           date_to: Optional[datetime] = None,
                           limit: int = 10) -> List[ConversationSession]:
        """Search conversations with various filters"""
        if not self.is_connected:
            logger.error("Not connected to MongoDB")
            return []
        
        try:
            with self._lock:
                mongo_query = {}
                
                if user_id:
                    mongo_query["user_id"] = user_id
                
                if date_from or date_to:
                    mongo_query["created_at"] = {}
                    if date_from:
                        mongo_query["created_at"]["$gte"] = date_from.isoformat()
                    if date_to:
                        mongo_query["created_at"]["$lte"] = date_to.isoformat()
                
                if query_text:
                    # Text search in conversation content
                    mongo_query["$or"] = [
                        {"turns.user_message.content": {"$regex": query_text, "$options": "i"}},
                        {"turns.assistant_message.content": {"$regex": query_text, "$options": "i"}}
                    ]
                
                cursor = self.conversations_collection.find(mongo_query).sort("created_at", DESCENDING).limit(limit)
                
                sessions = []
                for doc in cursor:
                    session = ConversationSession.from_dict(doc)
                    sessions.append(session)
                
                logger.info(f"Found {len(sessions)} conversations matching search criteria")
                return sessions
                
        except Exception as e:
            logger.error(f"Error searching conversations: {e}")
            return []
    
    def get_recent_conversations(self, limit: int = 10) -> List[ConversationSession]:
        """Get recent conversations across all users"""
        if not self.is_connected:
            logger.error("Not connected to MongoDB")
            return []
        
        try:
            with self._lock:
                cursor = self.conversations_collection.find().sort("updated_at", DESCENDING).limit(limit)
                
                sessions = []
                for doc in cursor:
                    session = ConversationSession.from_dict(doc)
                    sessions.append(session)
                
                logger.info(f"Retrieved {len(sessions)} recent conversations")
                return sessions
                
        except Exception as e:
            logger.error(f"Error retrieving recent conversations: {e}")
            return []
    
    def delete_conversation_session(self, session_id: str) -> bool:
        """Delete a conversation session"""
        if not self.is_connected:
            logger.error("Not connected to MongoDB")
            return False
        
        try:
            with self._lock:
                # Delete from conversations
                result1 = self.conversations_collection.delete_one({"session_id": session_id})
                
                # Delete from summaries
                result2 = self.summaries_collection.delete_one({"session_id": session_id})
                
                deleted_count = result1.deleted_count + result2.deleted_count
                
                if deleted_count > 0:
                    logger.info(f"Deleted conversation session: {session_id}")
                    return True
                else:
                    logger.warning(f"Conversation session not found: {session_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error deleting conversation session: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.is_connected:
            return {"status": "disconnected"}
        
        try:
            with self._lock:
                # Collection statistics
                conversations_count = self.conversations_collection.count_documents({})
                users_count = self.users_collection.count_documents({})
                summaries_count = self.summaries_collection.count_documents({})
                
                # Status distribution
                status_pipeline = [
                    {"$group": {"_id": "$status", "count": {"$sum": 1}}}
                ]
                status_counts = list(self.conversations_collection.aggregate(status_pipeline))
                
                # Recent activity (last 24 hours)
                recent_date = datetime.now() - timedelta(days=1)
                recent_conversations = self.conversations_collection.count_documents({
                    "created_at": {"$gte": recent_date.isoformat()}
                })
                
                # Average session length
                avg_turns_pipeline = [
                    {"$group": {"_id": None, "avg_turns": {"$avg": "$total_turns"}}}
                ]
                avg_turns_result = list(self.conversations_collection.aggregate(avg_turns_pipeline))
                avg_turns = avg_turns_result[0]["avg_turns"] if avg_turns_result else 0
                
                return {
                    "status": "connected",
                    "database_name": self.database_name,
                    "collections": {
                        "conversations": conversations_count,
                        "users": users_count,
                        "summaries": summaries_count
                    },
                    "status_distribution": {item["_id"]: item["count"] for item in status_counts},
                    "recent_activity_24h": recent_conversations,
                    "average_session_turns": round(avg_turns, 2),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        if not self.is_connected:
            logger.error("Not connected to MongoDB")
            return 0
        
        try:
            with self._lock:
                current_time = datetime.now().isoformat()
                
                # Find expired sessions
                expired_sessions = self.conversations_collection.find({
                    "expires_at": {"$lt": current_time},
                    "status": {"$ne": SessionStatus.EXPIRED.value}
                })
                
                expired_count = 0
                for session_doc in expired_sessions:
                    session_id = session_doc["session_id"]
                    
                    # Update status to expired
                    result = self.conversations_collection.update_one(
                        {"session_id": session_id},
                        {"$set": {"status": SessionStatus.EXPIRED.value}}
                    )
                    
                    if result.modified_count > 0:
                        expired_count += 1
                
                logger.info(f"Cleaned up {expired_count} expired sessions")
                return expired_count
                
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
            return 0
    
    def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        if not self.is_connected:
            return {
                "status": "disconnected",
                "message": "Not connected to MongoDB",
                "connection_string": self.connection_string
            }
        
        try:
            # Test basic operations
            self.client.admin.command('ping')
            
            # Test collection access
            self.conversations_collection.find_one({})
            self.users_collection.find_one({})
            self.summaries_collection.find_one({})
            
            stats = self.get_stats()
            
            return {
                "status": "healthy",
                "database_name": self.database_name,
                "collections": list(stats.get("collections", {}).keys()),
                "total_documents": sum(stats.get("collections", {}).values()),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "connection_string": self.connection_string,
                "timestamp": datetime.now().isoformat()
            }

class AsyncMongoChatHistory(MongoChatHistory):
    """Async version of MongoDB chat history storage"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.async_client = None
        self.async_database = None
        self.async_conversations_collection = None
        self.async_users_collection = None
        self.async_summaries_collection = None
    
    async def connect_async(self) -> bool:
        """Connect to MongoDB asynchronously"""
        try:
            self.async_client = AsyncIOMotorClient(self.connection_string)
            
            # Test connection
            await self.async_client.admin.command('ping')
            
            self.async_database = self.async_client[self.database_name]
            self.async_conversations_collection = self.async_database[self.conversations_collection_name]
            self.async_users_collection = self.async_database[self.users_collection_name]
            self.async_summaries_collection = self.async_database[self.summaries_collection_name]
            
            logger.info(f"Async connected to MongoDB database: {self.database_name}")
            return True
            
        except Exception as e:
            logger.error(f"Async connection to MongoDB failed: {e}")
            return False
    
    async def save_conversation_session_async(self, session: ConversationSession) -> bool:
        """Save conversation session asynchronously"""
        if not self.async_client:
            logger.error("Async MongoDB client not initialized")
            return False
        
        try:
            session_dict = session.to_dict()
            
            result = await self.async_conversations_collection.update_one(
                {"session_id": session.session_id},
                {"$set": session_dict},
                upsert=True
            )
            
            logger.info(f"Async saved conversation session: {session.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Async error saving conversation session: {e}")
            return False
    
    async def get_conversation_session_async(self, session_id: str) -> Optional[ConversationSession]:
        """Get conversation session asynchronously"""
        if not self.async_client:
            logger.error("Async MongoDB client not initialized")
            return None
        
        try:
            doc = await self.async_conversations_collection.find_one({"session_id": session_id})
            
            if doc:
                return ConversationSession.from_dict(doc)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Async error retrieving conversation session: {e}")
            return None

# Factory function for creating MongoDB chat history instances
def create_mongo_chat_history(
    connection_string: Optional[str] = None,
    database_name: str = "chatbot_history",
    **kwargs
) -> MongoChatHistory:
    """
    Factory function to create MongoDB chat history instance
    
    Args:
        connection_string: MongoDB connection string (defaults to env var or localhost)
        database_name: Database name
        **kwargs: Additional arguments for MongoChatHistory
    
    Returns:
        MongoChatHistory instance
    """
    # Get connection string from environment or use default
    if connection_string is None:
        connection_string = os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017/")
    
    return MongoChatHistory(
        connection_string=connection_string,
        database_name=database_name,
        **kwargs
    )