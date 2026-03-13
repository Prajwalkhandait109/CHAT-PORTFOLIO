from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from chatbot import ask_bot
from config import GROQ_API_KEY
import groq
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

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
    return {"message": "Prajwal AI Portfolio API Running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "api_key_loaded": bool(GROQ_API_KEY),
    }

@app.post("/chat")
def chat(question: str):
    if not GROQ_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"error": "GROQ_API_KEY is not loaded. Check your .env file encoding (must be UTF-8) and content."}
        )
    try:
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