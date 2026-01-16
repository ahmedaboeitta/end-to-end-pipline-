from pydantic import BaseModel, Field
from typing import Optional


class AskRequest(BaseModel):
    """Request model for /ask endpoint."""
    question: str = Field(..., min_length=1, max_length=2000, description="Question to ask")
    max_tokens: int = Field(default=256, ge=1, le=1024, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")


class AskResponse(BaseModel):
    """Response model for /ask endpoint."""
    answer: str = Field(..., description="Generated answer")
    model: str = Field(..., description="Model used for generation")
    latency_ms: float = Field(..., description="Generation latency in milliseconds")


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str = Field(..., description="Service status")
    vllm_connected: bool = Field(..., description="Whether vLLM server is reachable")


class ReadyResponse(BaseModel):
    """Response model for /ready endpoint."""
    ready: bool = Field(..., description="Whether service is ready to serve requests")
    models_loaded: list[str] = Field(default=[], description="List of loaded models")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Error details")