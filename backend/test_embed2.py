import asyncio
import httpx

async def test():
    url = "http://ollama:11434/api/embeddings"
    payloads = [
        {"model": "nomic-embed-text", "input": "test"},
        {"model": "nomic-embed-text", "prompt": "test"},
    ]
    async with httpx.AsyncClient(timeout=None) as client:
        for pl in payloads:
            try:
                r = await client.post(url, json=pl)
                print(f"Payload key: {list(pl.keys())[-1]}, status: {r.status_code}")
                data = r.json()
                print(f"  Keys: {list(data.keys())}")
                if "embedding" in data:
                    emb = data["embedding"]
                    print(f"  embedding type={type(emb).__name__}, len={len(emb) if isinstance(emb, list) else 'N/A'}")
                    if isinstance(emb, list) and emb:
                        print(f"  first 3: {emb[:3]}")
                if "embeddings" in data:
                    print(f"  embeddings: {type(data['embeddings'])}")
                if "data" in data:
                    print(f"  data: {data['data'][:1] if data['data'] else 'empty'}")
            except Exception as e:
                print(f"Payload {pl}: error {e}")

asyncio.run(test())
