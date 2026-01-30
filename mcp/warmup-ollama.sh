#!/bin/bash
# Ollama Model Warmup Script
# This script pre-loads specified models into memory to avoid cold-start delays

set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
MODELS_TO_WARMUP="${OLLAMA_WARMUP_MODELS:-gpt-oss:20b}"

echo "🔥 Ollama Warmup Script Starting..."
echo "📍 Ollama Host: $OLLAMA_HOST"
echo "🎯 Models to warmup: $MODELS_TO_WARMUP"

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to be ready..."
max_attempts=30
attempt=0
while ! curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ $attempt -ge $max_attempts ]; then
        echo "❌ Ollama did not become ready in time"
        exit 1
    fi
    echo "   Attempt $attempt/$max_attempts..."
    sleep 2
done

echo "✅ Ollama is ready!"

# Warmup each model
# Split comma-separated models (POSIX-compliant)
echo "$MODELS_TO_WARMUP" | tr ',' '\n' | while read -r model; do
    model=$(echo "$model" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//') # trim whitespace
    echo ""
    echo "🚀 Warming up model: $model"
    echo "   This may take 30-60 seconds on first load..."

    # Send a minimal request to trigger model loading
    start_time=$(date +%s)

    response=$(curl -s -X POST "$OLLAMA_HOST/api/generate" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$model\",
            \"prompt\": \"Hi\",
            \"stream\": false,
            \"options\": {
                \"num_predict\": 1
            }
        }")

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    if echo "$response" | grep -q "error"; then
        echo "❌ Failed to warmup $model"
        echo "   Error: $response"
    else
        echo "✅ Model $model loaded successfully in ${duration}s"
        echo "   Model is now ready for inference!"
    fi
done

echo ""
echo "🎉 Warmup complete! All models are loaded and ready."
echo "💡 Models will stay in memory based on OLLAMA_KEEP_ALIVE setting"

# Optional: Keep checking if models are loaded
if [ "${OLLAMA_KEEP_MONITORING:-false}" = "true" ]; then
    echo ""
    echo "📊 Monitoring loaded models (CTRL+C to stop)..."
    while true; do
        echo "---"
        curl -s "$OLLAMA_HOST/api/ps" | grep -o '"name":"[^"]*"' || echo "No models loaded"
        sleep 30
    done
fi
