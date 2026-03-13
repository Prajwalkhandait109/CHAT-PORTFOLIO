from typing import Dict, Any, Optional, Callable
import logging
from src.models.route_identifier import (
    QueryCategory, RouteDecision, ClassificationResult, HANDLER_CONFIG
)
from .query_classifier import QueryClassifier

logger = logging.getLogger(__name__)


class QueryRouter:
    """Intelligent router that classifies and routes queries to appropriate handlers."""
    
    def __init__(self, groq_client):
        self.classifier = QueryClassifier(groq_client)
        self.handlers: Dict[str, Callable] = {}
        self.default_handler = "portfolio_handler"  # Fallback to portfolio for safety
    
    def register_handler(self, name: str, handler_func: Callable):
        """Register a query handler function."""
        self.handlers[name] = handler_func
        logger.info(f"Registered handler: {name}")
    
    def route(self, query: str, session_context: Optional[Dict[str, Any]] = None) -> RouteDecision:
        """
        Classify and route a query to the appropriate handler.
        
        Args:
            query: The user's input query
            session_context: Optional session context for enhanced routing
            
        Returns:
            RouteDecision with routing information
        """
        logger.info(f"Routing query: '{query[:50]}...'")
        
        # Classify the query
        classification = self.classifier.classify(query)
        logger.info(f"Classified as {classification.category.value} with confidence {classification.confidence}")
        
        # Get handler configuration
        handler_config = HANDLER_CONFIG.get(
            classification.category,
            HANDLER_CONFIG[QueryCategory.PORTFOLIO]  # Default to portfolio
        )
        
        # Create routing decision
        route_decision = RouteDecision(
            query=query,
            classification=classification,
            handler_name=handler_config["name"],
            should_use_rag=handler_config["should_use_rag"],
            context_required=handler_config["context_required"],
            response_template=handler_config.get("response_template"),
            metadata={
                "session_context": session_context or {},
                "handler_config": handler_config,
                "routing_timestamp": logger.handlers[0].baseFilename if logger.handlers else "unknown"
            }
        )
        
        logger.info(f"Routing to handler: {route_decision.handler_name}")
        return route_decision
    
    def execute_route(self, route_decision: RouteDecision) -> Dict[str, Any]:
        """
        Execute the routing decision by calling the appropriate handler.
        
        Args:
            route_decision: The routing decision to execute
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            handler_name = route_decision.handler_name
            
            # Check if we have a predefined response template
            if route_decision.response_template:
                logger.info(f"Using predefined response for {handler_name}")
                return {
                    "response": route_decision.response_template,
                    "category": route_decision.classification.category.value,
                    "confidence": route_decision.classification.confidence,
                    "handler_used": handler_name,
                    "used_template": True,
                    "used_rag": False
                }
            
            # Check if handler is registered
            if handler_name not in self.handlers:
                logger.warning(f"Handler {handler_name} not registered, using default")
                handler_name = self.default_handler
            
            # Execute the handler
            handler_func = self.handlers[handler_name]
            
            # Prepare handler arguments
            handler_kwargs = {
                "query": route_decision.query,
                "classification": route_decision.classification,
                "should_use_rag": route_decision.should_use_rag,
                "context_required": route_decision.context_required,
                "metadata": route_decision.metadata
            }
            
            logger.info(f"Executing handler {handler_name} with RAG={route_decision.should_use_rag}")
            response = handler_func(**handler_kwargs)
            
            return {
                "response": response,
                "category": route_decision.classification.category.value,
                "confidence": route_decision.classification.confidence,
                "handler_used": handler_name,
                "used_template": False,
                "used_rag": route_decision.should_use_rag
            }
            
        except Exception as e:
            logger.error(f"Error executing route {handler_name}: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again.",
                "category": "error",
                "confidence": 0.0,
                "handler_used": "error_handler",
                "error": str(e),
                "used_template": True,
                "used_rag": False
            }
    
    def process_query(self, query: str, session_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete query processing pipeline: classify, route, and execute.
        
        Args:
            query: The user's input query
            session_context: Optional session context
            
        Returns:
            Complete response with metadata
        """
        # Route the query
        route_decision = self.route(query, session_context)
        
        # Execute the routing decision
        result = self.execute_route(route_decision)
        
        # Add routing metadata
        result["routing_metadata"] = {
            "classification_reasoning": route_decision.classification.reasoning,
            "keywords": route_decision.classification.keywords,
            "handler_config": route_decision.metadata.get("handler_config", {})
        }
        
        return result