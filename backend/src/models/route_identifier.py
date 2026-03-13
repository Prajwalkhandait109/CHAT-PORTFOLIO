from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass


class QueryCategory(Enum):
    """Categories for query classification."""
    PORTFOLIO = "portfolio"  # About Prajwal's skills, projects, experience
    TECHNICAL = "technical"  # Technical questions about technologies
    GENERAL = "general"      # General knowledge questions
    GREETING = "greeting"    # Hello, hi, etc.
    GOODBYE = "goodbye"     # Bye, thanks, etc.
    UNCLEAR = "unclear"    # Ambiguous or unclear queries
    OUT_OF_SCOPE = "out_of_scope"  # Completely unrelated to portfolio


@dataclass
class ClassificationResult:
    """Result of query classification."""
    category: QueryCategory
    confidence: float  # 0.0 to 1.0
    reasoning: str     # Why this classification was chosen
    keywords: list[str] = None  # Key terms that influenced classification
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


@dataclass
class RouteDecision:
    """Final routing decision for a query."""
    query: str
    classification: ClassificationResult
    handler_name: str
    should_use_rag: bool  # Whether to use vector search
    context_required: bool  # Whether additional context is needed
    response_template: Optional[str] = None  # Predefined response if applicable
    metadata: Dict[str, Any] = None  # Additional routing metadata
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# Handler configuration
HANDLER_CONFIG = {
    QueryCategory.PORTFOLIO: {
        "name": "portfolio_handler",
        "should_use_rag": True,
        "context_required": True,
        "system_prompt": "You are Prajwal's AI portfolio assistant. Answer questions about Prajwal's skills, projects, and experience using the provided context."
    },
    QueryCategory.TECHNICAL: {
        "name": "technical_handler", 
        "should_use_rag": False,  # Use general knowledge
        "context_required": False,
        "system_prompt": "You are a helpful technical assistant. Answer technical questions clearly and accurately."
    },
    QueryCategory.GENERAL: {
        "name": "general_handler",
        "should_use_rag": False,
        "context_required": False,
        "system_prompt": "You are a helpful AI assistant. Answer general questions clearly and concisely."
    },
    QueryCategory.GREETING: {
        "name": "greeting_handler",
        "should_use_rag": False,
        "context_required": False,
        "response_template": "Hello! I'm Prajwal's AI portfolio assistant. I can help you learn about Prajwal's skills, projects, and experience. What would you like to know?"
    },
    QueryCategory.GOODBYE: {
        "name": "goodbye_handler",
        "should_use_rag": False,
        "context_required": False,
        "response_template": "Thank you for your interest in Prajwal's portfolio! Feel free to ask more questions anytime. Goodbye!"
    },
    QueryCategory.UNCLEAR: {
        "name": "clarification_handler",
        "should_use_rag": False,
        "context_required": False,
        "response_template": "I'm not quite sure what you're asking. Could you please rephrase your question? I can help you learn about Prajwal's skills, projects, and experience."
    },
    QueryCategory.OUT_OF_SCOPE: {
        "name": "scope_handler",
        "should_use_rag": False,
        "context_required": False,
        "response_template": "I can only answer questions about Prajwal's portfolio, including skills, projects, and experience. Please ask something related to Prajwal's professional background."
    }
}