import asyncio
from app.services.knowledgebase_service import retrieve_context

async def test():
    ctx = await retrieve_context(
        query="Big Brother is watching you",
        mode="all",
        max_docs=5,
        max_chunks=5,
        provider="local",
        local_model="nomic-embed-text",
    )
    chunks = ctx.get("chunks", [])
    doc_ids = ctx.get("doc_ids", [])
    print(f"Chunks returned: {len(chunks)}")
    print(f"Doc IDs: {doc_ids}")
    for c in chunks[:5]:
        dist = c.get("distance", "?")
        text = c.get("text", "")[:120].replace("\n", " ")
        print(f"  dist={dist:.4f} | doc={c.get('document_id')} | {text}")

asyncio.run(test())
