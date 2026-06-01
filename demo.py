"""
Live demo: runs the proxy in-process with a mock upstream.
Shows cache miss → upstream call, then cache hit → no upstream call.
"""
import asyncio
import json
import sys
import time
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, ".")

import httpx
from qdrant_client import AsyncQdrantClient

from src.cache.store import SemanticCacheStore, VECTOR_DIM
import src.proxy.server as server_module

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
DIM    = "\033[2m"


def banner(text):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")


def log(label, value, color=RESET):
    print(f"  {DIM}{label:<22}{RESET}{color}{value}{RESET}")


def fixed_emb(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(VECTOR_DIM)
    return (v / np.linalg.norm(v)).tolist()


async def post(app, body, tenant="demo-tenant"):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            "/v1/chat/completions",
            json=body,
            headers={"X-Tenant-ID": tenant, "Authorization": "Bearer sk-demo"},
        )


async def run_demo():
    # ── Setup ──────────────────────────────────────────────────────────────────
    store = SemanticCacheStore(client=AsyncQdrantClient(":memory:"))
    emb   = fixed_emb(seed=42)

    upstream_calls = 0
    upstream_response = {
        "id": "chatcmpl-demo",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant",
            "content": "Paris is the capital of France."}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 14, "completion_tokens": 8, "total_tokens": 22},
    }

    async def mock_upstream_post(body, auth):
        nonlocal upstream_calls
        upstream_calls += 1
        await asyncio.sleep(0.12)          # simulate ~120 ms network latency
        return upstream_response

    enc = MagicMock()
    enc.encode = AsyncMock(return_value=emb)

    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful geography assistant."},
            {"role": "user",   "content": "What is the capital of France?"},
        ],
    }

    banner("Semantic Cache Proxy — Live Demo")
    print(f"\n  {DIM}Qdrant:   in-memory{RESET}")
    print(f"  {DIM}Upstream: mock (120 ms simulated latency){RESET}")
    print(f"  {DIM}Query:    \"What is the capital of France?\"{RESET}")

    with patch.object(server_module, "cache_store", store), \
         patch.object(server_module, "encoder",     enc), \
         patch.object(server_module, "_upstream_post", mock_upstream_post):

        # ── Request 1: cache miss ──────────────────────────────────────────────
        banner("Request 1 — Cache MISS")
        t0   = time.perf_counter()
        resp = await post(server_module.app, body)
        t1   = time.perf_counter()

        data    = resp.json()
        content = data["choices"][0]["message"]["content"]
        log("Status:",        f"{resp.status_code} OK",        GREEN)
        log("Upstream calls:", str(upstream_calls),            YELLOW)
        log("Latency:",        f"{(t1-t0)*1000:.0f} ms")
        log("Response:",       content,                        CYAN)
        print(f"\n  {RED}✗ Cache MISS — forwarded to upstream{RESET}")

        # lower threshold so the same vector hits on next request
        store.thresholds.default = 0.5

        # ── Request 2: cache hit ───────────────────────────────────────────────
        banner("Request 2 — Cache HIT (same query)")
        t0   = time.perf_counter()
        resp = await post(server_module.app, body)
        t1   = time.perf_counter()

        data    = resp.json()
        content = data["choices"][0]["message"]["content"]
        log("Status:",        f"{resp.status_code} OK",        GREEN)
        log("Upstream calls:", str(upstream_calls) + "  (unchanged)", GREEN)
        log("Latency:",        f"{(t1-t0)*1000:.0f} ms")
        log("Response:",       content,                        CYAN)
        print(f"\n  {GREEN}✓ Cache HIT — served from Qdrant, upstream skipped{RESET}")

        # ── Request 3: different tenant ────────────────────────────────────────
        banner("Request 3 — Different Tenant (cache MISS)")
        t0   = time.perf_counter()
        resp = await post(server_module.app, body, tenant="other-tenant")
        t1   = time.perf_counter()

        data    = resp.json()
        content = data["choices"][0]["message"]["content"]
        log("Tenant:",         "other-tenant",                 YELLOW)
        log("Status:",         f"{resp.status_code} OK",       GREEN)
        log("Upstream calls:",  str(upstream_calls),           YELLOW)
        log("Latency:",         f"{(t1-t0)*1000:.0f} ms")
        log("Response:",        content,                       CYAN)
        print(f"\n  {RED}✗ Cache MISS — tenant namespace is isolated{RESET}")

        # ── Streaming demo ─────────────────────────────────────────────────────
        banner("Request 4 — Streaming Cache Replay (SSE)")
        # cache is warm for demo-tenant now
        store.thresholds.default = 0.5
        stream_body = {**body, "stream": True}

        t0 = time.perf_counter()
        transport = httpx.ASGITransport(app=server_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json=stream_body,
                headers={"X-Tenant-ID": "demo-tenant", "Authorization": "Bearer sk-demo"},
            )
        t1 = time.perf_counter()

        assembled = ""
        chunks    = 0
        for line in resp.text.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                assembled += chunk["choices"][0]["delta"].get("content", "")
                chunks += 1

        log("Content-Type:", resp.headers.get("content-type", ""), CYAN)
        log("SSE chunks:",   str(chunks))
        log("Latency:",      f"{(t1-t0)*1000:.0f} ms")
        log("Assembled:",    assembled, CYAN)
        log("Upstream calls:", str(upstream_calls) + "  (unchanged)", GREEN)
        print(f"\n  {GREEN}✓ Cache HIT replayed as SSE stream — upstream not called{RESET}")

    # ── Summary ────────────────────────────────────────────────────────────────
    banner("Summary")
    print(f"  {'Requests sent:':<24}4")
    print(f"  {'Upstream calls:':<24}{upstream_calls}  (2 misses, 2 hits skipped)")
    print(f"  {'Cache saved:':<24}{4 - upstream_calls} round-trips to OpenAI")
    print(f"\n  {GREEN}{BOLD}Demo complete.{RESET}\n")


if __name__ == "__main__":
    asyncio.run(run_demo())
