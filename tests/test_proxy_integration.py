"""
Proxy integration tests.
- FastAPI app served via httpx.AsyncClient (no real network)
- Upstream OpenAI calls mocked via unittest.mock
- Embeddings replaced with deterministic synthetic vectors
- Qdrant runs in-memory
"""
import asyncio
import json
import sys
import os
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
from qdrant_client import AsyncQdrantClient

from src.cache.store import SemanticCacheStore, VECTOR_DIM
import src.proxy.server as server_module


# ── Helpers ───────────────────────────────────────────────────────────────────

def fixed_embedding(seed: int = 0, dim: int = VECTOR_DIM) -> list:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return (v / np.linalg.norm(v)).tolist()


def make_openai_response(content: str, model: str = "gpt-4o") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def make_upstream_post_mock(content: str) -> AsyncMock:
    """Mocks src.proxy.server._upstream_post to return a fixed response dict."""
    return AsyncMock(return_value=make_openai_response(content))


async def fake_upstream_stream(sse_lines):
    """Async generator that yields pre-baked SSE lines (for _upstream_stream mock)."""
    for line in sse_lines:
        yield line


def make_in_memory_store() -> SemanticCacheStore:
    return SemanticCacheStore(client=AsyncQdrantClient(":memory:"))


def make_encoder_mock(embedding: list) -> MagicMock:
    enc = MagicMock()
    enc.encode = AsyncMock(return_value=embedding)
    return enc


def build_request_body(user_msg: str, system_msg: str = "You are helpful.", stream: bool = False) -> dict:
    return {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "stream": stream,
    }


async def post_completion(app, body: dict, tenant: str = "tenantA", api_key: str = "Bearer sk-test") -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            "/v1/chat/completions",
            json=body,
            headers={"X-Tenant-ID": tenant, "Authorization": api_key},
        )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCacheMissForwardsToUpstream:
    def test_miss_calls_upstream_and_returns_response(self):
        async def run():
            store = make_in_memory_store()
            emb = fixed_embedding(seed=1)
            enc = make_encoder_mock(emb)
            upstream_mock = make_upstream_post_mock("The capital of France is Paris.")

            with patch.object(server_module, "cache_store", store), \
                 patch.object(server_module, "encoder", enc), \
                 patch.object(server_module, "_upstream_post", upstream_mock):

                body = build_request_body("What is the capital of France?")
                resp = await post_completion(server_module.app, body)

            assert resp.status_code == 200
            data = resp.json()
            assert data["choices"][0]["message"]["content"] == "The capital of France is Paris."
            upstream_mock.assert_called_once()

        asyncio.run(run())
        print("  PASS: cache miss forwards to upstream and returns response")


class TestCacheHitSkipsUpstream:
    def test_second_identical_request_hits_cache(self):
        async def run():
            store = make_in_memory_store()
            emb = fixed_embedding(seed=2)
            enc = make_encoder_mock(emb)
            upstream_mock = make_upstream_post_mock("Python is a high-level language.")

            with patch.object(server_module, "cache_store", store), \
                 patch.object(server_module, "encoder", enc), \
                 patch.object(server_module, "_upstream_post", upstream_mock):

                body = build_request_body("What is Python?")

                # first request: miss → upstream
                resp1 = await post_completion(server_module.app, body)
                assert resp1.status_code == 200

                # lower threshold so same-vector lookup hits
                store.thresholds.default = 0.5

                # second request: hit → no upstream
                resp2 = await post_completion(server_module.app, body)
                assert resp2.status_code == 200
                assert resp2.json()["choices"][0]["message"]["content"] == "Python is a high-level language."

            assert upstream_mock.call_count == 1, \
                f"Expected 1 upstream call, got {upstream_mock.call_count}"

        asyncio.run(run())
        print("  PASS: second identical request served from cache, upstream not called again")


class TestTenantIsolation:
    def test_different_tenants_get_separate_caches(self):
        async def run():
            store = make_in_memory_store()
            emb = fixed_embedding(seed=3)
            enc = make_encoder_mock(emb)

            responses = ["Answer for tenant A.", "Answer for tenant B."]
            call_count = 0

            async def mock_upstream(body, auth):
                nonlocal call_count
                result = make_openai_response(responses[call_count])
                call_count += 1
                return result

            body = build_request_body("Same question.")

            with patch.object(server_module, "cache_store", store), \
                 patch.object(server_module, "encoder", enc), \
                 patch.object(server_module, "_upstream_post", mock_upstream):

                resp_a = await post_completion(server_module.app, body, tenant="tenantA")
                resp_b = await post_completion(server_module.app, body, tenant="tenantB")

            assert resp_a.json()["choices"][0]["message"]["content"] == "Answer for tenant A."
            assert resp_b.json()["choices"][0]["message"]["content"] == "Answer for tenant B."
            assert call_count == 2, "Both tenants should have hit upstream independently"

        asyncio.run(run())
        print("  PASS: different tenants get separate upstream calls and cache namespaces")


class TestSystemPromptIsolation:
    def test_different_system_prompts_produce_different_cache_keys(self):
        async def run():
            store = make_in_memory_store()
            emb = fixed_embedding(seed=4)
            enc = make_encoder_mock(emb)
            upstream_mock = make_upstream_post_mock("Answer.")

            body_a = build_request_body("Tell me a joke.", system_msg="You are a comedian.")
            body_b = build_request_body("Tell me a joke.", system_msg="You are a lawyer.")

            with patch.object(server_module, "cache_store", store), \
                 patch.object(server_module, "encoder", enc), \
                 patch.object(server_module, "_upstream_post", upstream_mock):

                await post_completion(server_module.app, body_a)
                await post_completion(server_module.app, body_b)

            assert upstream_mock.call_count == 2, \
                "Different system prompts should produce different cache keys and both hit upstream"

        asyncio.run(run())
        print("  PASS: different system prompts produce different cache keys")


class TestStreamingCacheReplay:
    def test_cached_response_replayed_as_sse_stream(self):
        async def run():
            store = make_in_memory_store()
            emb = fixed_embedding(seed=5)

            # pre-populate cache
            await store._ensure_collection()
            cached_content = "Streaming cached answer here."
            await store.store(
                emb, "tenantA:cachedkey",
                {"choices": [{"message": {"role": "assistant", "content": cached_content}}]},
            )
            store.thresholds.default = 0.5

            enc = make_encoder_mock(emb)

            # patch fingerprint_system to return "cachedkey" so namespace matches
            with patch.object(server_module, "cache_store", store), \
                 patch.object(server_module, "encoder", enc), \
                 patch.object(server_module, "fingerprint_system", return_value="cachedkey"):

                body = build_request_body("Any question.", stream=True)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server_module.app), base_url="http://test") as client:
                    resp = await client.post(
                        "/v1/chat/completions",
                        json=body,
                        headers={"X-Tenant-ID": "tenantA", "Authorization": "Bearer sk-test"},
                    )

            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            raw = resp.text
            assert "data: " in raw
            assert "[DONE]" in raw

            # reconstruct content from SSE chunks
            assembled = ""
            for line in raw.splitlines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    assembled += chunk["choices"][0]["delta"].get("content", "")

            assert assembled == cached_content

        asyncio.run(run())
        print("  PASS: cached response replayed correctly as SSE stream")


class TestStreamingForwardAndCache:
    def test_streaming_upstream_response_is_cached_for_next_request(self):
        async def run():
            store = make_in_memory_store()
            emb = fixed_embedding(seed=6)
            enc = make_encoder_mock(emb)

            streamed_content = "Streamed answer from upstream."
            sse_lines = [
                f'data: {json.dumps({"choices": [{"delta": {"content": chunk}, "index": 0}]})}'
                for chunk in [streamed_content[i:i+10] for i in range(0, len(streamed_content), 10)]
            ] + ["data: [DONE]"]

            async def mock_upstream_stream(body, auth):
                for line in sse_lines:
                    yield line

            body = build_request_body("Stream this.", stream=True)

            with patch.object(server_module, "cache_store", store), \
                 patch.object(server_module, "encoder", enc), \
                 patch.object(server_module, "_upstream_stream", mock_upstream_stream):

                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server_module.app), base_url="http://test"
                ) as client:
                    resp = await client.post(
                        "/v1/chat/completions",
                        json=body,
                        headers={"X-Tenant-ID": "tenantA", "Authorization": "Bearer sk-test"},
                    )

            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            # confirm streamed content was written to cache
            store.thresholds.default = 0.5
            hit = await store.lookup(emb, "tenantA:" + server_module.fingerprint_system(body["messages"]))
            assert hit is not None, "Streamed response should have been written to cache"
            assert hit["response"]["choices"][0]["message"]["content"] == streamed_content

        asyncio.run(run())
        print("  PASS: streaming upstream response is forwarded and stored in cache")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    suites = [
        TestCacheMissForwardsToUpstream(),
        TestCacheHitSkipsUpstream(),
        TestTenantIsolation(),
        TestSystemPromptIsolation(),
        TestStreamingCacheReplay(),
        TestStreamingForwardAndCache(),
    ]

    passed = 0
    failed = 0

    for suite in suites:
        methods = [m for m in dir(suite) if m.startswith("test_")]
        for method in methods:
            name = f"{type(suite).__name__}.{method}"
            try:
                print(f"\n[RUN] {name}")
                getattr(suite, method)()
                passed += 1
            except Exception as e:
                import traceback
                print(f"  FAIL: {e}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed!")
    else:
        sys.exit(1)
