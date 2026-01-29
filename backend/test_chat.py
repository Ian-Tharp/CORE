import asyncio
import httpx
import sys

async def test():
    print('Making request...', flush=True)
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                'http://localhost:11434/api/chat',
                json={
                    'model': 'gpt-oss:20b',
                    'messages': [
                        {'role': 'user', 'content': 'Say hello in 5 words or less'}
                    ],
                    'stream': False
                },
                timeout=120.0
            )
            print(f'Status: {response.status_code}', flush=True)
            if response.status_code == 200:
                result = response.json()
                print(f'Response: {result.get("message", {}).get("content", "no content")}', flush=True)
            else:
                print(f'Error body: {response.text}', flush=True)
        except Exception as e:
            print(f'Exception: {e}', flush=True)

asyncio.run(test())
