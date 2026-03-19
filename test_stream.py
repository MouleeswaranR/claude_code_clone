import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
import os

async def main():
    client = AsyncOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    try:
        stream = await client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct",  # change if needed
            messages=[{"role": "user", "content": "Say hello world"}],
            stream=True,
            timeout=60,
        )
        print("[Stream started]")
        async for chunk in stream:
            print("[Chunk]", chunk)
    except Exception as e:
        print("[Test failed]", type(e).__name__, str(e))

asyncio.run(main())