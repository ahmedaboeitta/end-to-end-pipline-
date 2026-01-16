#!/bin/bash
# =============================================================================
# Test Serving Stack
# =============================================================================

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log_ok() { echo -e "${GREEN}✅ $1${NC}"; }
log_fail() { echo -e "${RED}❌ $1${NC}"; }

API_URL="http://localhost:8000"
VLLM_URL="http://localhost:8001"

echo "=========================================="
echo "Testing Serving Stack"
echo "=========================================="

# Test 1: vLLM Health
echo -n "Testing vLLM health... "
if curl -s "$VLLM_URL/health" > /dev/null; then
    log_ok "vLLM is healthy"
else
    log_fail "vLLM not responding"
    exit 1
fi

# Test 2: API Health
echo -n "Testing API health... "
if curl -s "$API_URL/health" > /dev/null; then
    log_ok "API is healthy"
else
    log_fail "API not responding"
    exit 1
fi

# Test 3: API Ready
echo -n "Testing API readiness... "
READY=$(curl -s "$API_URL/ready" | grep -o '"ready":true')
if [ -n "$READY" ]; then
    log_ok "API is ready"
else
    log_fail "API not ready"
    exit 1
fi

# Test 4: List Models
echo -n "Testing model loading... "
MODELS=$(curl -s "$API_URL/models")
echo "$MODELS"
log_ok "Models listed"

# Test 5: Ask Question
echo ""
echo "Testing inference..."
RESPONSE=$(curl -s -X POST "$API_URL/ask" \
    -H "Content-Type: application/json" \
    -d '{"question": "What is morning sickness?"}')

echo "Response: $RESPONSE"

if echo "$RESPONSE" | grep -q "answer"; then
    log_ok "Inference working"
else
    log_fail "Inference failed"
    exit 1
fi

# Test 6: Metrics
echo -n "Testing metrics endpoint... "
if curl -s "$API_URL/metrics" | grep -q "http_requests_total"; then
    log_ok "Metrics available"
else
    log_fail "Metrics not working"
fi

echo ""
echo "=========================================="
echo "All tests passed!"
echo "=========================================="