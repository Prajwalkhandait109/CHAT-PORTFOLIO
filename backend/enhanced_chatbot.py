from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Dict, Any, Optional
import logging

from src.rag.router import QueryRouter
from src.models.route_identifier import ClassificationResult

logger = logging.getLogger(__name__)


class EnhancedChatbot:
    """Enhanced chatbot with intelligent query routing."""
    
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.router = QueryRouter(self.client)
        
        # Initialize vector store for portfolio queries
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        try:
            self.db = FAISS.load_local("db", self.embedding, allow_dangerous_deserialization=True)
            logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            self.db = None
        
        # Register handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register all query handlers."""
        self.router.register_handler("portfolio_handler", self._handle_portfolio_query)
        self.router.register_handler("technical_handler", self._handle_technical_query)
        self.router.register_handler("general_handler", self._handle_general_query)
        self.router.register_handler("greeting_handler", self._handle_greeting_query)
        self.router.register_handler("goodbye_handler", self._handle_goodbye_query)
        self.router.register_handler("clarification_handler", self._handle_clarification_query)
        self.router.register_handler("scope_handler", self._handle_scope_query)
    
    def _handle_portfolio_query(self, query: str, classification: ClassificationResult, 
                               should_use_rag: bool, context_required: bool, **kwargs) -> str:
        """Handle portfolio-related queries using RAG."""
        if not self.db:
            return "I apologize, but I don't have access to portfolio information right now."
        
        try:
            # Search for relevant documents
            docs = self.db.similarity_search(query, k=3)
            
            if not docs:
                return "I couldn't find specific information about that in Prajwal's portfolio. Could you ask about something else, like skills, projects, or experience?"
            
            # Build context
            context = "\n".join([doc.page_content for doc in docs])
            
            # Create prompt
            system_prompt = """
You are Prajwal's AI portfolio assistant. Use the provided context to answer questions about Prajwal's professional background.

Guidelines:
- Be specific and detailed when possible
- Reference actual projects, skills, or experience from the context
- If asked about something not in the context, suggest related topics you can discuss
- Be professional and helpful
"""
            
            prompt = f"""
Context from Prajwal's portfolio:
{context}

User Question: {query}

Please provide a detailed answer based on the context above."""
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in portfolio handler: {e}")
            return "I encountered an error while searching Prajwal's portfolio. Please try again."
    
    def _handle_technical_query(self, query: str, classification: ClassificationResult, 
                               should_use_rag: bool, context_required: bool, **kwargs) -> str:
        """Handle technical questions using general knowledge."""
        system_prompt = """
You are a knowledgeable technical assistant. Answer technical questions clearly and accurately.

Guidelines:
- Provide accurate technical information
- Use examples when helpful
- Be concise but comprehensive
- If unsure about something, say so
"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=400
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in technical handler: {e}")
            return "I encountered an error while processing your technical question. Please try again."
    
    def _handle_general_query(self, query: str, classification: ClassificationResult, 
                             should_use_rag: bool, context_required: bool, **kwargs) -> str:
        """Handle general knowledge questions."""
        system_prompt = """
You are a helpful AI assistant. Answer general questions clearly and concisely.

Guidelines:
- Provide accurate information
- Be helpful and friendly
- Keep responses concise but informative
- If the question is too broad, ask for clarification
"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in general handler: {e}")
            return "I encountered an error while processing your question. Please try again."
    
    def _handle_greeting_query(self, query: str, classification: ClassificationResult, 
                              should_use_rag: bool, context_required: bool, **kwargs) -> str:
        """Handle greeting queries."""
        return "Hello! I'm Prajwal's AI portfolio assistant. I can help you learn about Prajwal's skills, projects, and professional experience. What would you like to know?"
    
    def _handle_goodbye_query(self, query: str, classification: ClassificationResult, 
                             should_use_rag: bool, context_required: bool, **kwargs) -> str:
        """Handle goodbye queries."""
        return "Thank you for your interest in Prajwal's portfolio! Feel free to ask more questions anytime. Goodbye!"
    
    def _handle_clarification_query(self, query: str, classification: ClassificationResult, 
                                   should_use_rag: bool, context_required: bool, **kwargs) -> str:
        """Handle unclear queries that need clarification."""
        return "I'm not quite sure what you're asking. Could you please rephrase your question? I can help you learn about Prajwal's skills, projects, and experience."
    
    def _handle_scope_query(self, query: str, classification: ClassificationResult, 
                           should_use_rag: bool, context_required: bool, **kwargs) -> str:
        """Handle out-of-scope queries."""
        return "I can only answer questions about Prajwal's portfolio, including skills, projects, and experience. Please ask something related to Prajwal's professional background."
    
    def ask(self, query: str, session_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query through the intelligent routing system.
        
        Args:
            query: The user's question
            session_context: Optional session context
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            return self.router.process_query(query, session_context)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again.",
                "category": "error",
                "confidence": 0.0,
                "handler_used": "error_handler",
                "error": str(e)
            }