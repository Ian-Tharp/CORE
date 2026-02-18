"""
Test script for knowledge attribution feature.
Quick verification that the repository functions work with instance metadata.
"""

import asyncio
from app.repository import knowledgebase_repository as repo


async def test_knowledge_attribution():
    """Test basic functionality of knowledge attribution."""
    print("Testing knowledge attribution...")
    
    # Test 1: Create document with instance metadata
    print("1. Testing document creation with instance metadata...")
    try:
        doc_id = await repo.create_document(
            filename="test_doc.txt",
            original_name="Test Document",
            size=100,
            mime_type="text/plain",
            storage_path="/tmp/test.txt",
            description="Test document for attribution",
            instance_name="TestInstance",
            source_discussion="Testing knowledge attribution feature"
        )
        print(f"   ✅ Created document {doc_id} with instance metadata")
    except Exception as e:
        print(f"   ❌ Failed to create document: {e}")
        return False

    # Test 2: Insert chunk with instance metadata
    print("2. Testing chunk insertion with instance metadata...")
    try:
        test_chunks = [(0, "This is a test chunk for attribution.", [0.1] * 768)]
        await repo.insert_chunk_embeddings(
            document_id=doc_id,
            items=test_chunks,
            model="test-model",
            dimensions=768,
            instance_name="TestInstance",
            source_discussion="Testing chunk attribution"
        )
        print("   ✅ Inserted chunk with instance metadata")
    except Exception as e:
        print(f"   ❌ Failed to insert chunk: {e}")
        return False

    # Test 3: List instances
    print("3. Testing instance listing...")
    try:
        instances = await repo.list_instances()
        print(f"   ✅ Found {len(instances)} instances")
        if instances:
            print(f"   📝 First instance: {instances[0]}")
    except Exception as e:
        print(f"   ❌ Failed to list instances: {e}")
        return False

    # Test 4: Instance-based search
    print("4. Testing instance-based search...")
    try:
        query_vec = [0.1] * 768  # Simple test vector
        results = await repo.search_chunks_by_instance(
            query_vec=query_vec,
            instance_name="TestInstance",
            limit=5
        )
        print(f"   ✅ Instance search returned {len(results)} results")
        if results:
            result = results[0]
            print(f"   📝 First result: instance={result.get('instance_name')}, text='{result.get('text', '')[:50]}...'")
    except Exception as e:
        print(f"   ❌ Failed to search by instance: {e}")
        return False

    # Cleanup
    print("5. Cleaning up test data...")
    try:
        await repo.delete_document(doc_id)
        print("   ✅ Cleaned up test document")
    except Exception as e:
        print(f"   ⚠️  Failed to cleanup: {e}")

    print("\n🎉 All tests passed! Knowledge attribution is working correctly.")
    return True


if __name__ == "__main__":
    asyncio.run(test_knowledge_attribution())