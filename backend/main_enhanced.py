from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from config import GROQ_API_KEY
import groq
import logging
from typing import Dict, Any, Optional

# Import the enhanced chatbot with routing
from enhanced_chatbot import EnhancedChatbot

logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize the enhanced chatbot
enhanced_bot = None
if GROQ_API_KEY:
    try:
        enhanced_bot = EnhancedChatbot(GROQ_API_KEY)
        logger.info("Enhanced chatbot with routing system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize enhanced chatbot: {e}")
        # Fallback to original chatbot if enhanced version fails
        try:
            from chatbot import ask_bot
            logger.info("Falling back to original chatbot")
        except ImportError:
            logger.error("Neither enhanced nor original chatbot available")

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
    return {"message": "Prajwal AI Portfolio API Running with Intelligent Routing"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "api_key_loaded": bool(GROQ_API_KEY),
        "routing_enabled": enhanced_bot is not None,
        "features": {
            "intelligent_routing": enhanced_bot is not None,
            "query_classification": enhanced_bot is not None,
            "multi_handler_support": enhanced_bot is not None
        }
    }

@app.post("/chat")
def chat(question: str):
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
        if enhanced_bot:
            # Use the enhanced chatbot with routing
            result = enhanced_bot.ask(question)
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
        if enhanced_bot and enhanced_bot.router:
            # Get classification only
            classification = enhanced_bot.router.classifier.classify(question)
            route_decision = enhanced_bot.router.route(question)
            
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
                content={"error": "Enhanced routing system not available"}
            )
            
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Classification error: {str(e)}"}
        )

@app.get("/routing/handlers")
def get_handlers():
    """Get information about available handlers."""
    if not enhanced_bot or not enhanced_bot.router:
        return JSONResponse(
            status_code=503,
            content={"error": "Enhanced routing system not available"}
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
        ]
    }