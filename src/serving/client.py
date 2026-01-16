import httpx
import asyncio
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class VLLMClient:
    """Async client for vLLM OpenAI-compatible API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def health_check(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"vLLM health check failed: {e}")
            return False
    
    async def list_models(self) -> list[str]:
        """List available models."""
        try:
            client = await self._get_client()
            response = await client.get("/v1/models")
            response.raise_for_status()
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def generate(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Generate completion from vLLM."""
        client = await self._get_client()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop or [],
        }
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = await client.post("/v1/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Extract generated text
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("text", "").strip()
                return ""
                
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
                    
            except httpx.HTTPStatusError as e:
                last_error = e
                logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
                raise
                
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
        
        raise RuntimeError(f"Failed after {self.max_retries} attempts: {last_error}")
    
    async def chat_generate(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Generate chat completion from vLLM."""
        client = await self._get_client()
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = await client.post("/v1/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Extract generated text
                choices = data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    return message.get("content", "").strip()
                return ""
                
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
                    
            except httpx.HTTPStatusError as e:
                last_error = e
                logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
                raise
                
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
        
        raise RuntimeError(f"Failed after {self.max_retries} attempts: {last_error}")
    
    async def warmup(self, model: str) -> bool:
        """Send warmup request to load model into GPU cache."""
        logger.info(f"Warming up model: {model}")
        try:
            await self.generate(
                prompt="Hello",
                model=model,
                max_tokens=5,
                temperature=0,
            )
            logger.info("Warmup complete")
            return True
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            return False