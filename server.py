"""
Gemma 4 E2B Instruct - OpenAI-Compatible API Server
Deployed on Railway with llama-cpp-python
"""

import os
import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://huggingface.co/bartowski/google_gemma-4-2b-it-GGUF/resolve/main/google_gemma-4-2b-it-Q4_K_M.gguf"
)
MODEL_FILE = "/app/models/model.gguf"
MODEL_DIR = "/app/models"
PORT = int(os.environ.get("PORT", "8000"))
CTX_SIZE = int(os.environ.get("CTX_SIZE", "4096"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))

# Global model instance
llm = None


def download_model():
    """Download model from HuggingFace if not present."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_FILE):
        logger.info(f"Model already exists at {MODEL_FILE}")
        return

    logger.info(f"Downloading model from {MODEL_URL}...")
    import urllib.request
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        logger.info("Model downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        # Try alternative URL
        alt_url = "https://huggingface.co/bartowski/google_gemma-4-2b-it-GGUF/resolve/main/google_gemma-4-2b-it-Q4_K_M.gguf"
        logger.info(f"Trying alternative URL: {alt_url}")
        urllib.request.urlretrieve(alt_url, MODEL_FILE)


def load_model():
    """Load the GGUF model with llama-cpp-python."""
    global llm
    logger.info(f"Loading model from {MODEL_FILE}...")
    logger.info(f"Context size: {CTX_SIZE}, Max tokens: {MAX_TOKENS}")

    from llama_cpp import Llama
    llm = Llama(
        model_path=MODEL_FILE,
        n_ctx=CTX_SIZE,
        n_gpu_layers=0,
        verbose=False,
        chat_format="chatml",
    )
    logger.info("Model loaded successfully!")
    return llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("Starting Gemma 4 Server...")
    download_model()
    load_model()
    logger.info(f"Gemma 4 Server ready on port {PORT}")
    yield
    logger.info("Shutting down...")


# FastAPI app
app = FastAPI(
    title="Gemma 4 API",
    description="OpenAI-compatible API for Gemma 4 E2B Instruct",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ───────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gemma-4-2b-it"
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = MAX_TOKENS
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-gemma4"
    object: str = "chat.completion"
    model: str = "gemma-4-2b-it"
    choices: list[ChatCompletionChoice]


# ─── Endpoints ─────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "Gemma 4 E2B Instruct",
        "status": "running" if llm else "loading",
        "endpoints": {
            "chat": "POST /v1/chat/completions",
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if llm else "loading",
        "model_loaded": llm is not None,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model is still loading...")

    try:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        completion = llm.create_chat_completion(
            messages=messages,
            max_tokens=request.max_tokens or MAX_TOKENS,
            temperature=request.temperature or 0.7,
            stream=False,
        )

        return ChatCompletionResponse(
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(
                        role=completion["choices"][0]["message"]["role"],
                        content=completion["choices"][0]["message"]["content"],
                    ),
                    finish_reason=completion["choices"][0].get("finish_reason", "stop"),
                )
            ]
        )
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions/stream")
async def chat_completions_stream(request: ChatCompletionRequest):
    """Streaming chat completions."""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model is still loading...")

    import json
    from fastapi.responses import StreamingResponse

    async def generate():
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        try:
            completion = llm.create_chat_completion(
                messages=messages,
                max_tokens=request.max_tokens or MAX_TOKENS,
                temperature=request.temperature or 0.7,
                stream=True,
            )

            for chunk in completion:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    data = json.dumps({
                        "id": "chatcmpl-gemma4",
                        "object": "chat.completion.chunk",
                        "model": "gemma-4-2b-it",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None,
                        }]
                    })
                    yield f"data: {data}\n\n"

                finish = chunk["choices"][0].get("finish_reason")
                if finish:
                    yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
