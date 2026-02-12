import asyncio
from app.services.ollama_embeddings import embed_texts_via_ollama

async def test():
    vecs, dims = await embed_texts_via_ollama(model="nomic-embed-text", texts=["Big Brother is watching you"])
    print(f"Returned {len(vecs)} vectors, original dims={dims}")
    if vecs:
        v = vecs[0]
        print(f"Vector length: {len(v)}")
        print(f"First 5 values: {v[:5]}")
        print(f"Values at 768-772: {v[768:772]}")
        nonzero = sum(1 for x in v if x != 0.0)
        print(f"Non-zero values: {nonzero} / {len(v)}")

asyncio.run(test())
