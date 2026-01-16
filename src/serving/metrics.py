import time
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, REGISTRY

# -----------------------------------------------------------------------------
# Metrics Definitions
# -----------------------------------------------------------------------------

# Request counter - tracks total requests by endpoint, method, status
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["endpoint", "method", "status"]
)

# Request latency histogram - tracks response time distribution
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint", "method"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# In-progress requests gauge - tracks concurrent requests
IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests currently being processed",
    ["endpoint"]
)

# Error counter - tracks errors separately for quick alerting
ERROR_COUNT = Counter(
    "http_request_errors_total",
    "Total HTTP request errors",
    ["endpoint", "method", "error_type"]
)

# Application info - static metadata
APP_INFO = Info(
    "app",
    "Application information"
)

# Startup timestamp
STARTUP_TIME = Gauge(
    "app_startup_timestamp_seconds",
    "Unix timestamp when the application started"
)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def record_app_info(
    model: str,
    adapter: str,
    domain: str = "",
    quantization: str = "none",
    lora_r: int = 0,
    vllm_port: int = 0,
    version: str = "1.0.0"
):
    """Record application metadata on startup."""
    APP_INFO.info({
        "model": model,
        "adapter": adapter,
        "domain": domain,
        "quantization": quantization,
        "lora_r": str(lora_r),
        "vllm_port": str(vllm_port),
        "version": version,
    })
    STARTUP_TIME.set(time.time())


def get_metrics() -> bytes:
    """Generate Prometheus metrics output."""
    return generate_latest(REGISTRY)