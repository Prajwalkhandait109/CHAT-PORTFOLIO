from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from config import GROQ_API_KEY
import groq
import logging
from typing import Dict, Any, Optional

# Import the advanced chatbot with multi-stage RAG
from advanced_chatbot import AdvancedChatbot

logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize the advanced chatbot
advanced_bot = None
if GROQ_API_KEY:
    try:
        advanced_bot = AdvancedChatbot(GROQ_API_KEY, use_advanced_rag=True)
        logger.info("Advanced chatbot with multi-stage RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize advanced chatbot: {e}")
        # Fallback to enhanced chatbot if advanced version fails
        try:
            from enhanced_chatbot import EnhancedChatbot
            advanced_bot = EnhancedChatbot(GROQ_API_KEY)
            logger.info("Falling back to enhanced chatbot")
        except Exception as e2:
            logger.error(f"Enhanced chatbot also failed: {e2}")

# Allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Prajwal AI Portfolio API Running with Advanced RAG Pipeline"}

@app.get("/health")
def health():
    rag_stats = advanced_bot.get_rag_stats() if advanced_bot else {}
    
    return {
        "status": "ok",
        "api_key_loaded": bool(GROQ_API_KEY),
        "advanced_rag_enabled": advanced_bot is not None,
        "rag_stats": rag_stats,
        "features": {
            "intelligent_routing": advanced_bot is not None,
            "query_classification": advanced_bot is not None,
            "multi_handler_support": advanced_bot is not None,
            "advanced_rag": rag_stats.get("advanced_rag_enabled", False),
            "document_grading": rag_stats.get("advanced_pipeline_available", False),
            "hallucination_checking": rag_stats.get("advanced_pipeline_available", False)
        }
    }

@app.post("/chat")
def chat(question: str, use_advanced_rag: Optional[bool] = None):
    """
    Process a chat query with optional advanced RAG pipeline.
    
    Args:
        question: The user's question
        use_advanced_rag: Override default RAG mode (true=advanced, false=simple, null=default)
    """
    if not GROQ_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"error": "GROQ_API_KEY is not loaded. Check your .env file encoding (must be UTF-8) and content."}
        )
    
    if not question or not question.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Question cannot be empty"}
        )
    
    try:
        if advanced_bot:
            # Use the advanced chatbot with multi-stage RAG
            result = advanced_bot.ask(question, use_advanced_rag=use_advanced_rag)
            return result
        else:
            # Fallback to original chatbot
            from chatbot import ask_bot
            response = ask_bot(question)
            return {"response": response}
            
    except groq.AuthenticationError as e:
        logger.error(f"Invalid API key error: {e}")
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid Groq API key. Check your .env file and ensure GROQ_API_KEY is set correctly."}
        )
    except groq.APIError as e:
        logger.error(f"Groq API error: {e}")
        return JSONResponse(
            status_code=502,
            content={"error": f"Groq API error: {str(e)}"}
        )
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred: {str(e)}"}
        )

@app.post("/chat/classify")
def classify_query(question: str):
    """Endpoint to test query classification without full response."""
    if not GROQ_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"error": "GROQ_API_KEY is not loaded."}
        )
    
    if not question or not question.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Question cannot be empty"}
        )
    
    try:
        if advanced_bot and advanced_bot.router:
            # Get classification only
            classification = advanced_bot.router.classifier.classify(question)
            route_decision = advanced_bot.router.route(question)
            
            return {
                "query": question,
                "classification": {
                    "category": classification.category.value,
                    "confidence": classification.confidence,
                    "reasoning": classification.reasoning,
                    "keywords": classification.keywords
                },
                "route_decision": {
                    "handler_name": route_decision.handler_name,
                    "should_use_rag": route_decision.should_use_rag,
                    "context_required": route_decision.context_required,
                    "response_template": route_decision.response_template
                }
            }
        else:
            return JSONResponse(
                status_code=503,
                content={"error": "Advanced routing system not available"}
            )
            
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Classification error: {str(e)}"}
        )

@app.post("/chat/rag")
def test_rag_pipeline(question: str, strategy: str = "hybrid", use_advanced: bool = True):
    """Endpoint to test RAG pipeline directly."""
    if not GROQ_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"error": "GROQ_API_KEY is not loaded."}
        )
    
    if not question or not question.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Question cannot be empty"}
        )
    
    try:
        if advanced_bot and advanced_bot.advanced_rag and use_advanced:
            # Test advanced RAG pipeline
            result = advanced_bot.advanced_rag.invoke(question, optimization_strategy=strategy)
            return result
        elif advanced_bot and advanced_bot.simple_rag and not use_advanced:
            # Test simple RAG pipeline
            result = advanced_bot.simple_rag.simple_retrieve_and_generate(question)
            return result
        else:
            return JSONResponse(
                status_code=503,
                content={"error": "RAG pipeline not available"}
            )
            
    except Exception as e:
        logger.error(f"RAG pipeline error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"RAG pipeline error: {str(e)}"}
        )

@app.get("/routing/handlers")
def get_handlers():
    """Get information about available handlers."""
    if not advanced_bot or not advanced_bot.router:
        return JSONResponse(
            status_code=503,
            content={"error": "Advanced routing system not available"}
        )
    
    from src.models.route_identifier import HANDLER_CONFIG
    
    return {
        "handlers": [
            {
                "name": config["name"],
                "category": category.value,
                "should_use_rag": config["should_use_rag"],
                "context_required": config["context_required"],
                "has_template": "response_template" in config
            }
            for category, config in HANDLER_CONFIG.items()
        ],
        "rag_modes": [
            {"mode": "advanced", "description": "Multi-stage RAG with query optimization and document grading"},
            {"mode": "simple", "description": "Basic RAG with relevance filtering"}
        ]
    }