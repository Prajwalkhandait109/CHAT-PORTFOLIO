from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Dict, Any, Optional
import logging

from src.rag.router import QueryRouter
from src.rag.graph_builder import AdvancedRAGPipeline, SimpleRAGPipeline
from src.models.route_identifier import ClassificationResult

logger = logging.getLogger(__name__)


class AdvancedChatbot:
    """Advanced chatbot with intelligent routing and multi-stage RAG pipeline."""
    
    def __init__(self, groq_api_key: str, use_advanced_rag: bool = True):
        self.client = Groq(api_key=groq_api_key)
        self.use_advanced_rag = use_advanced_rag
        self.router = QueryRouter(self.client)
        
        # Initialize vector store for portfolio queries
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        try:
            self.db = FAISS.load_local("db", self.embedding, allow_dangerous_deserialization=True)
            logger.info("Vector store loaded successfully")
            
            # Initialize RAG pipelines
            if use_advanced_rag:
                self.advanced_rag = AdvancedRAGPipeline(self.client, self.db)
                self.simple_rag = SimpleRAGPipeline(self.client, self.db)  # Also create simple for fallback
                logger.info("Advanced RAG pipeline initialized")
            else:
                self.advanced_rag = None
                self.simple_rag = SimpleRAGPipeline(self.client, self.db)
                logger.info("Simple RAG pipeline initialized")
                
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            self.db = None
            self.advanced_rag = None
            self.simple_rag = None
        
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
        """Handle portfolio-related queries using advanced RAG."""
        if not self.db:
            return "I apologize, but I don't have access to portfolio information right now."
        
        try:
            if self.use_advanced_rag and self.advanced_rag:
                # Use advanced RAG pipeline
                logger.info("Using advanced RAG pipeline for portfolio query")
                result = self.advanced_rag.invoke(query, optimization_strategy="hybrid")
                
                # Log detailed metrics
                metadata = result.get("metadata", {})
                logger.info(f"Advanced RAG completed: {result.get('relevant_documents', 0)} relevant docs, "
                          f"{metadata.get('retrieval_stats', {}).get('total_queries', 0)} queries, "
                          f"{metadata.get('grading_stats', {}).get('relevant_documents', 0)} graded relevant")
                
                return result["answer"]
            else:
                # Use simple RAG pipeline
                logger.info("Using simple RAG pipeline for portfolio query")
                result = self.simple_rag.simple_retrieve_and_generate(query, k=5)
                
                logger.info(f"Simple RAG completed: {result.get('relevant_documents', 0)} relevant docs")
                return result["answer"]
                
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
    
    def ask(self, query: str, session_context: Optional[Dict[str, Any]] = None, 
            use_advanced_rag: Optional[bool] = None) -> Dict[str, Any]:
        """
        Process a query through the intelligent routing system.
        
        Args:
            query: The user's question
            session_context: Optional session context
            use_advanced_rag: Override default RAG mode
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Use provided RAG mode or default
            rag_mode = use_advanced_rag if use_advanced_rag is not None else self.use_advanced_rag
            
            # Route the query
            route_decision = self.router.route(query, session_context)
            
            # Execute with appropriate RAG mode
            if route_decision.should_use_rag:
                # For portfolio queries, use the specified RAG mode
                original_mode = self.use_advanced_rag
                self.use_advanced_rag = rag_mode
                
                result = self.router.execute_route(route_decision)
                
                # Restore original mode
                self.use_advanced_rag = original_mode
            else:
                # For non-portfolio queries, use regular routing
                result = self.router.execute_route(route_decision)
            
            # Add RAG mode information
            if route_decision.should_use_rag:
                result["rag_mode"] = "advanced" if rag_mode else "simple"
                result["rag_metadata"] = {
                    "optimization_strategy": "hybrid" if rag_mode else "basic",
                    "pipeline_type": "multi-stage" if rag_mode else "single-stage"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again.",
                "category": "error",
                "confidence": 0.0,
                "handler_used": "error_handler",
                "error": str(e)
            }
    
    def get_rag_stats(self) -> Dict[str, Any]:
        """Get statistics about RAG pipeline performance."""
        stats = {
            "advanced_rag_enabled": self.use_advanced_rag,
            "vector_store_available": self.db is not None,
            "advanced_pipeline_available": self.advanced_rag is not None,
            "simple_pipeline_available": self.simple_rag is not None
        }
        
        if self.advanced_rag:
            stats["advanced_features"] = [
                "query_optimization",
                "document_grading", 
                "hallucination_checking",
                "multi_stage_processing"
            ]
        
        return stats