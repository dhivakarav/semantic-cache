import asyncio
import hashlib
import json
import time
from typing import AsyncIterator, List

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from src.cache.store import SemanticCacheStore
from src.embeddings.encoder import QueryEncoder

app = FastAPI(title="Semantic Cache Proxy")

cache_store = SemanticCacheStore()
encoder = QueryEncoder()

UPSTREAM_URL = "https://api.openai.com/v1/chat/completions"


async def _upstream_post(body: dict, auth_header: str) -> dict:
    headers = {"Authorization": auth_header, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(UPSTREAM_URL, json=body, headers=headers)
        return resp.json()


async def _upstream_stream(body: dict, auth_header: str) -> AsyncIterator[str]:
    headers = {"Authorization": auth_header, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", UPSTREAM_URL, json=body, headers=headers) as resp:
            async for line in resp.aiter_lines():
                yield line


def fingerprint_system(messages: List[dict]) -> str:
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    return hashlib.sha256("|".join(system_parts).encode()).hexdigest()[:16]


def extract_user_turn(messages: List[dict]) -> str:
    user_parts = [m["content"] for m in messages if m["role"] == "user"]
    return " ".join(user_parts)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    tenant_id = request.headers.get("X-Tenant-ID", "default")
    api_key = request.headers.get("Authorization", "")

    messages = body.get("messages", [])
    stream = body.get("stream", False)

    sys_fp = fingerprint_system(messages)
    user_text = extract_user_turn(messages)

    embedding = await encoder.encode(user_text)
    cache_key_prefix = f"{tenant_id}:{sys_fp}"

    hit = await cache_store.lookup(embedding, cache_key_prefix)
    if hit:
        if stream:
            return StreamingResponse(
                _replay_stream(hit), media_type="text/event-stream"
            )
        return Response(content=json.dumps(hit["response"]), media_type="application/json")

    if stream:
        return StreamingResponse(
            _forward_and_cache_stream(body, api_key, embedding, cache_key_prefix),
            media_type="text/event-stream",
        )
    response_data = await _upstream_post(body, api_key)
    await cache_store.store(embedding, cache_key_prefix, response_data)
    return Response(content=json.dumps(response_data), media_type="application/json")


async def _replay_stream(cached: dict) -> AsyncIterator[str]:
    content = cached["response"]["choices"][0]["message"]["content"]
    chunk_size = 20
    for i in range(0, len(content), chunk_size):
        chunk = content[i : i + chunk_size]
        delta = {"choices": [{"delta": {"content": chunk}, "index": 0, "finish_reason": None}]}
        yield f"data: {json.dumps(delta)}\n\n"
        await asyncio.sleep(0.01)
    yield "data: [DONE]\n\n"


async def _forward_and_cache_stream(
    body: dict,
    api_key: str,
    embedding: List[float],
    cache_key_prefix: str,
) -> AsyncIterator[str]:
    full_content = []
    async for line in _upstream_stream(body, api_key):
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0].get("delta", {}).get("content", "")
                if delta:
                    full_content.append(delta)
            except Exception:
                pass
        yield line + "\n\n"

    if full_content:
        synthetic_response = {
            "choices": [{"message": {"role": "assistant", "content": "".join(full_content)}}]
        }
        await cache_store.store(embedding, cache_key_prefix, synthetic_response)
