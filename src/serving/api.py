import time
import asyncio
import logging
import os
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import yaml
import json

from .schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    ReadyResponse,
    ErrorResponse,
)
from .client import VLLMClient
from .metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    IN_PROGRESS,
    ERROR_COUNT,
    record_app_info,
    get_metrics,
)

from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
def load_config(path: str = "config.yaml") -> dict:
    """Load config from yaml file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


config = load_config()

# From finetuning section (single source of truth)
BASE_MODEL_NAME = config["finetuning"]["base_model"]
LORA_ADAPTER_PATH = config["finetuning"]["final_model_dir"]
LORA_ADAPTER_NAME = Path(LORA_ADAPTER_PATH).name

# From serving section
VLLM_HOST = config["serving"]["vllm_host"]
VLLM_PORT = config["serving"]["vllm_port"]
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}"
VLLM_TIMEOUT = config["serving"]["timeout"]
WARMUP_ON_START = config["serving"]["warmup_on_start"]

# System prompt
SYSTEM_PROMPT = """You are a helpful assistant specializing in answering questions.

[ANSWER STYLE]
- Be concise and directly address the question
- Do NOT add explanations unless explicitly required
- Avoid examples, lists, or extra background unless asked
- Keep answers brief and factual"""


# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Main API logger (all requests)
api_logger = logging.getLogger("api")
api_logger.setLevel(logging.INFO)
api_handler = logging.FileHandler(LOG_DIR / "api.log")
api_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
api_logger.addHandler(api_handler)
api_logger.addHandler(logging.StreamHandler())  # Also print to terminal

# Request logger (question/answer content)
request_logger = logging.getLogger("requests")
request_logger.setLevel(logging.INFO)
request_handler = logging.FileHandler(LOG_DIR / "requests.log")
request_handler.setFormatter(logging.Formatter("%(message)s"))  # JSON lines
request_logger.addHandler(request_handler)


# -----------------------------------------------------------------------------
# Authentication
# -----------------------------------------------------------------------------
security = HTTPBearer(auto_error=False)
API_TOKEN = os.getenv("API_TOKEN")

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token for protected endpoints."""
    if not API_TOKEN:
        return None  # Auth disabled if no token configured
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials

# -----------------------------------------------------------------------------
# Application State
# -----------------------------------------------------------------------------
vllm_client: VLLMClient = None
available_models: list[str] = []


# -----------------------------------------------------------------------------
# Lifespan Management
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    global vllm_client, available_models
    
    api_logger.info("Starting API server...")
    api_logger.info(f"vLLM URL: {VLLM_BASE_URL}")
    api_logger.info(f"Base model: {BASE_MODEL_NAME}")
    api_logger.info(f"LoRA adapter: {LORA_ADAPTER_NAME}")
    
    # Record app info for metrics
    record_app_info(
        model=BASE_MODEL_NAME,
        adapter=LORA_ADAPTER_NAME,
        domain=config["domain"],
        quantization=config["finetuning"].get("quantization", "none"),
        lora_r=config["finetuning"].get("lora_r", 0),
        vllm_port=VLLM_PORT,
    )
    
    vllm_client = VLLMClient(
        base_url=VLLM_BASE_URL,
        timeout=VLLM_TIMEOUT,
    )
    
    # Wait for vLLM to be ready
    api_logger.info("Waiting for vLLM server...")
    max_retries = 60
    for i in range(max_retries):
        if await vllm_client.health_check():
            api_logger.info("vLLM server is ready")
            break
        if i < max_retries - 1:
            await asyncio.sleep(2)
    else:
        api_logger.warning("vLLM server not ready after timeout, continuing anyway...")
    
    # Get available models
    available_models = await vllm_client.list_models()
    api_logger.info(f"Available models: {available_models}")
    
    # Warmup
    if WARMUP_ON_START and available_models:
        warmup_model = LORA_ADAPTER_NAME if LORA_ADAPTER_NAME in available_models else available_models[0]
        await vllm_client.warmup(warmup_model)
    
    api_logger.info("API server ready")
    
    yield
    
    api_logger.info("Shutting down API server...")
    await vllm_client.close()
    api_logger.info("Shutdown complete")


# -----------------------------------------------------------------------------
# FastAPI Application
# -----------------------------------------------------------------------------
app = FastAPI(
    title=f"{config['domain'].title()} QA API",
    description=f"Domain-specific QA API for {config['domain']}",
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


# -----------------------------------------------------------------------------
# Metrics Middleware
# -----------------------------------------------------------------------------
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track metrics and log all requests."""
    if request.url.path == "/metrics":
        return await call_next(request)
    
    endpoint = request.url.path
    method = request.method
    
    IN_PROGRESS.labels(endpoint=endpoint).inc()
    start_time = time.time()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=response.status_code).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(duration)
        
        # Log request
        api_logger.info(f"{method} {endpoint} - status={response.status_code} - latency={duration:.3f}s")
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        
        ERROR_COUNT.labels(endpoint=endpoint, method=method, error_type=type(e).__name__).inc()
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=500).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(duration)
        
        api_logger.error(f"{method} {endpoint} - status=500 - latency={duration:.3f}s - error={type(e).__name__}: {e}")
        
        raise
        
    finally:
        IN_PROGRESS.labels(endpoint=endpoint).dec()


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=get_metrics(), media_type="text/plain; charset=utf-8")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint."""
    vllm_connected = await vllm_client.health_check()
    return HealthResponse(
        status="healthy" if vllm_connected else "degraded",
        vllm_connected=vllm_connected,
    )


@app.get("/ready", response_model=ReadyResponse, tags=["Health"])
async def ready():
    """Readiness check endpoint."""
    global available_models
    available_models = await vllm_client.list_models()
    is_ready = len(available_models) > 0
    return ReadyResponse(ready=is_ready, models_loaded=available_models)


@app.get("/models", tags=["Models"])
async def list_models():
    """List available models."""
    global available_models
    available_models = await vllm_client.list_models()
    return {"models": available_models}


@app.post(
    "/ask",
    response_model=AskResponse,
    responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Inference"],
)
async def ask(request: AskRequest, _: HTTPAuthorizationCredentials = Depends(verify_token)):

    """Ask a question and get an answer."""
    
    model = LORA_ADAPTER_NAME
    
    if model not in available_models:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model}' not available. Available: {available_models}",
        )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request.question},
    ]
    
    start_time = time.time()
    
    try:
        answer = await vllm_client.chat_generate(
            messages=messages,
            model=model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except Exception as e:
        api_logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    latency_ms = (time.time() - start_time) * 1000
    
    # Log question and answer to requests.log
    request_logger.info(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "question": request.question,
        "answer": answer,
        "model": model,
        "latency_ms": round(latency_ms, 2),
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }))
    
    return AskResponse(
        answer=answer,
        model=model,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/warmup", tags=["Admin"])
async def warmup():
    """Manually trigger model warmup."""
    results = {}
    for model in available_models:
        success = await vllm_client.warmup(model)
        results[model] = "success" if success else "failed"
    return {"warmup_results": results}