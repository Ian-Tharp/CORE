import requests
import json

print("Making sync request...", flush=True)
response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "gpt-oss:20b",
        "messages": [
            {"role": "user", "content": "Hi, respond briefly."}
        ],
        "stream": False
    },
    timeout=120
)
print(f"Status: {response.status_code}", flush=True)
if response.status_code == 200:
    result = response.json()
    print(f"Response: {result.get('message', {}).get('content', 'no content')}", flush=True)
else:
    print(f"Error: {response.text}", flush=True)
