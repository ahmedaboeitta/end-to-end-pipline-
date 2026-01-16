#!/bin/bash
# =============================================================================
# LLM Fine-tuning Pipeline - Bash Wrapper
# =============================================================================
# 
# Usage:
#   ./scripts/run_pipeline.sh                    # Run full pipeline (stages 1-5)
#   ./scripts/run_pipeline.sh --stages 1,2,3    # Run specific stages
#   ./scripts/run_pipeline.sh --build           # Build Docker images
#   ./scripts/run_pipeline.sh --serve           # Start servers + run tests
#   ./scripts/run_pipeline.sh --test            # Run tests only (servers must be running)
#   ./scripts/run_pipeline.sh --stop            # Stop servers
#   ./scripts/run_pipeline.sh --all             # Full pipeline + build + serve + test
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_ROOT}/config.yaml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Load Environment Variables
# =============================================================================

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
    log_info "Loaded .env file"
else
    log_warn ".env file not found at $PROJECT_ROOT/.env"
fi

# =============================================================================
# Config Reader
# =============================================================================

read_config() {
    python3 << EOF
import yaml
with open("$CONFIG_FILE", "r") as f:
    config = yaml.safe_load(f)
print(f'VLLM_PORT={config["serving"]["vllm_port"]}')
print(f'API_PORT={config["serving"]["api_port"]}')
print(f'VLLM_HOST={config["serving"]["vllm_host"]}')
EOF
}

eval $(read_config)

# =============================================================================
# Failure Handler
# =============================================================================

print_debug_info() {
    echo ""
    log_error "=========================================="
    log_error "DEPLOYMENT FAILED"
    log_error "=========================================="
    echo ""
    log_warn "Running containers left for debugging:"
    
    # Check which containers are running
    VLLM_RUNNING=$(docker ps --filter "name=vllm-server" --format "{{.Names}}" 2>/dev/null)
    API_RUNNING=$(docker ps --filter "name=qa-api" --format "{{.Names}}" 2>/dev/null)
    
    if [ -n "$VLLM_RUNNING" ]; then
        echo "  - vllm-server (vLLM inference)"
    fi
    if [ -n "$API_RUNNING" ]; then
        echo "  - qa-api (FastAPI gateway)"
    fi
    
    if [ -z "$VLLM_RUNNING" ] && [ -z "$API_RUNNING" ]; then
        echo "  (no containers running)"
    fi
    
    echo ""
    log_warn "To view logs:"
    echo "  docker logs vllm-server"
    echo "  docker logs qa-api"
    echo ""
    log_warn "To cleanup:"
    echo "  docker compose -f src/docker/docker-compose.yaml down"
    echo "  # or: ./scripts/run_pipeline.sh --stop"
    echo ""
}

# =============================================================================
# Health Check Functions
# =============================================================================

wait_for_vllm() {
    log_info "Waiting for vLLM server (port $VLLM_PORT)..."
    local max_attempts=300
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
            log_info "✅ vLLM server is ready"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    log_error "❌ vLLM server failed to start (timeout after $((max_attempts * 2)) seconds)"
    return 1
}

wait_for_api() {
    log_info "Waiting for API server (port $API_PORT)..."
    local max_attempts=60
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
            log_info "✅ API server is ready"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    log_error "❌ API server failed to start (timeout after $((max_attempts * 2)) seconds)"
    return 1
}

# =============================================================================
# Test Function
# =============================================================================

run_tests() {
    log_info "Running integration tests..."
    cd "$PROJECT_ROOT"
    
    if [ ! -f "$SCRIPT_DIR/test_serving.sh" ]; then
        log_error "Test script not found: $SCRIPT_DIR/test_serving.sh"
        return 1
    fi
    
    if bash "$SCRIPT_DIR/test_serving.sh"; then
        log_info "✅ All integration tests passed"
        return 0
    else
        log_error "❌ Integration tests failed"
        return 1
    fi
}

# =============================================================================
# Docker Functions
# =============================================================================

docker_build() {
    log_info "Building Docker images..."
    cd "$PROJECT_ROOT"
    
    docker compose -f src/docker/docker-compose.yaml build
    
    log_info "✅ Docker images built"
}

docker_serve() {
    log_info "Starting servers with Docker..."
    cd "$PROJECT_ROOT"
    
    # Auto-build if API image doesn't exist
    if ! docker images | grep -q "qa-api"; then
        log_info "Docker images not found, building first..."
        docker_build
    fi
    
    # Start vLLM first
    log_info "Starting vLLM container..."
    docker compose -f src/docker/docker-compose.yaml up -d vllm
    
    if ! wait_for_vllm; then
        print_debug_info
        exit 1
    fi
    
    # Then start API
    log_info "Starting API container..."
    docker compose -f src/docker/docker-compose.yaml up -d api
    
    if ! wait_for_api; then
        print_debug_info
        exit 1
    fi
    
    log_info "✅ All servers running"
    log_info "   vLLM:  http://localhost:$VLLM_PORT"
    log_info "   API:   http://localhost:$API_PORT"
    log_info "   Docs:  http://localhost:$API_PORT/docs"
    
    # Run integration tests
    echo ""
    if ! run_tests; then
        print_debug_info
        exit 1
    fi
    
    echo ""
    log_info "=========================================="
    log_info "🎉 DEPLOYMENT SUCCESSFUL"
    log_info "=========================================="
    log_info "   vLLM:  http://localhost:$VLLM_PORT"
    log_info "   API:   http://localhost:$API_PORT"
    log_info "   Docs:  http://localhost:$API_PORT/docs"
}

docker_stop() {
    log_info "Stopping servers..."
    cd "$PROJECT_ROOT"
    
    docker compose -f src/docker/docker-compose.yaml down
    
    log_info "✅ Servers stopped"
}

# =============================================================================
# Local Serve (without Docker)
# =============================================================================

local_serve() {
    log_info "Starting servers locally..."
    cd "$PROJECT_ROOT"
    
    # Start vLLM in background
    log_info "Starting vLLM..."
    bash "$SCRIPT_DIR/serve_vllm.sh" &
    VLLM_PID=$!
    
    if ! wait_for_vllm; then
        log_error "vLLM failed to start"
        kill $VLLM_PID 2>/dev/null
        echo ""
        log_warn "vLLM process (PID: $VLLM_PID) may still be running"
        log_warn "To cleanup: kill $VLLM_PID"
        exit 1
    fi
    
    # Start API
    log_info "Starting API..."
    python -m src.serving.run &
    API_PID=$!
    
    if ! wait_for_api; then
        log_error "API failed to start"
        echo ""
        log_warn "Processes left for debugging:"
        log_warn "  vLLM PID: $VLLM_PID"
        log_warn "  API PID:  $API_PID"
        log_warn "To cleanup: kill $VLLM_PID $API_PID"
        exit 1
    fi
    
    # Run integration tests
    echo ""
    if ! run_tests; then
        echo ""
        log_warn "Processes left for debugging:"
        log_warn "  vLLM PID: $VLLM_PID"
        log_warn "  API PID:  $API_PID"
        log_warn "To cleanup: kill $VLLM_PID $API_PID"
        exit 1
    fi
    
    echo ""
    log_info "=========================================="
    log_info "🎉 LOCAL DEPLOYMENT SUCCESSFUL"
    log_info "=========================================="
    log_info "   vLLM PID: $VLLM_PID"
    log_info "   API PID:  $API_PID"
    log_info ""
    log_info "Press Ctrl+C to stop servers"
    
    # Wait for interrupt
    trap "kill $VLLM_PID $API_PID 2>/dev/null; exit 0" SIGINT SIGTERM
    wait
}

# =============================================================================
# Pipeline Functions
# =============================================================================

run_pipeline() {
    local stages="$1"
    cd "$PROJECT_ROOT"
    
    if [ -n "$stages" ]; then
        python pipeline.py --stage "$stages"
    else
        python pipeline.py
    fi
}

run_pipeline_range() {
    local start="$1"
    local end="$2"
    cd "$PROJECT_ROOT"
    
    python pipeline.py --start "$start" --end "$end"
}

# =============================================================================
# Usage
# =============================================================================

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Pipeline Options:"
    echo "  (no args)           Run full pipeline (stages 1-5)"
    echo "  --stage STAGE       Run specific stage (1, 2, 3, 4, 4a, 4b, 4c, 4d, 5)"
    echo "  --start N --end M   Run stages from N to M"
    echo "  --list              List available stages"
    echo ""
    echo "Serving Options:"
    echo "  --build             Build Docker images"
    echo "  --serve             Start vLLM + API servers + run tests (Docker)"
    echo "  --serve-local       Start servers locally + run tests (no Docker)"
    echo "  --test              Run integration tests only (servers must be running)"
    echo "  --stop              Stop Docker servers"
    echo ""
    echo "Combined:"
    echo "  --all               Run pipeline + build + serve + test"
    echo ""
    echo "Other:"
    echo "  --config FILE       Use custom config file"
    echo "  --help              Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                          # Run stages 1-5"
    echo "  $0 --stage 3                # Run stage 3 only"
    echo "  $0 --stage 4b               # Run fine-tuning only"
    echo "  $0 --start 2 --end 4        # Run stages 2, 3, 4"
    echo "  $0 --serve                  # Start servers + test"
    echo "  $0 --test                   # Test running servers"
    echo "  $0 --all                    # Everything"
}

# =============================================================================
# Main
# =============================================================================

# Parse arguments
STAGE=""
START=""
END=""
ACTION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --start)
            START="$2"
            shift 2
            ;;
        --end)
            END="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --build)
            ACTION="build"
            shift
            ;;
        --serve)
            ACTION="serve"
            shift
            ;;
        --serve-local)
            ACTION="serve-local"
            shift
            ;;
        --test)
            ACTION="test"
            shift
            ;;
        --stop)
            ACTION="stop"
            shift
            ;;
        --all)
            ACTION="all"
            shift
            ;;
        --list)
            cd "$PROJECT_ROOT"
            python pipeline.py --list
            exit 0
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Execute based on action
case $ACTION in
    build)
        docker_build
        ;;
    serve)
        docker_serve
        ;;
    serve-local)
        local_serve
        ;;
    test)
        run_tests
        ;;
    stop)
        docker_stop
        ;;
    all)
        run_pipeline
        docker_build
        docker_serve
        ;;
    *)
        # Default: run pipeline
        if [ -n "$STAGE" ]; then
            run_pipeline "$STAGE"
        elif [ -n "$START" ] || [ -n "$END" ]; then
            run_pipeline_range "${START:-1}" "${END:-5}"
        else
            run_pipeline
        fi
        ;;
esac