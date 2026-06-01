"""
Self-contained tests for the semantic cache.
Uses in-memory Qdrant + random/synthetic embeddings — no API keys or Docker needed.
"""
import asyncio
import random
import time
import numpy as np
from qdrant_client import AsyncQdrantClient

# Patch sys.path so imports work without install
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cache.store import SemanticCacheStore, VECTOR_DIM
from src.embeddings.encoder import QueryEncoder


def rand_embedding(dim: int = VECTOR_DIM) -> list:
    v = np.random.randn(dim).astype(float)
    return (v / np.linalg.norm(v)).tolist()


def similar_embedding(base: list, noise: float = 0.02) -> list:
    v = np.array(base) + np.random.randn(len(base)) * noise
    return (v / np.linalg.norm(v)).tolist()


def dissimilar_embedding(dim: int = VECTOR_DIM) -> list:
    return rand_embedding(dim)


def make_store() -> SemanticCacheStore:
    client = AsyncQdrantClient(":memory:")
    return SemanticCacheStore(client=client)


def make_response(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCacheStoreLookupMiss:
    def test_empty_cache_returns_none(self):
        async def run():
            store = make_store()
            emb = rand_embedding()
            result = await store.lookup(emb, "tenant1:sys001")
            assert result is None, "Empty cache should return None"
        asyncio.run(run())
        print("  PASS: empty cache returns None")


class TestCacheStoreStoreAndHit:
    def test_exact_embedding_hits(self):
        async def run():
            store = make_store()
            emb = rand_embedding()
            response = make_response("Paris is the capital of France.")
            await store.store(emb, "tenant1:sys001", response)

            # Lower threshold to 0.5 for exact same vector (score will be 1.0)
            store.thresholds.default = 0.5
            hit = await store.lookup(emb, "tenant1:sys001")
            assert hit is not None, "Exact vector should hit cache"
            assert hit["response"]["choices"][0]["message"]["content"] == "Paris is the capital of France."
            assert hit["score"] >= 0.99
        asyncio.run(run())
        print("  PASS: exact embedding hits cache with correct response")


class TestCacheStoreSimilarHit:
    def test_similar_embedding_hits(self):
        async def run():
            store = make_store()
            base_emb = rand_embedding()
            similar_emb = similar_embedding(base_emb, noise=0.01)  # very similar
            response = make_response("The sky is blue.")
            await store.store(base_emb, "tenant1:sys001", response)

            store.thresholds.default = 0.90
            hit = await store.lookup(similar_emb, "tenant1:sys001")
            assert hit is not None, "Very similar vector should hit cache"
        asyncio.run(run())
        print("  PASS: similar embedding hits cache")


class TestCacheStoreDissimilarMiss:
    def test_dissimilar_embedding_misses(self):
        async def run():
            store = make_store()
            base_emb = rand_embedding()
            different_emb = dissimilar_embedding()
            response = make_response("Cached answer.")
            await store.store(base_emb, "tenant1:sys001", response)

            store.thresholds.default = 0.93
            hit = await store.lookup(different_emb, "tenant1:sys001")
            assert hit is None, "Dissimilar vector should not hit cache"
        asyncio.run(run())
        print("  PASS: dissimilar embedding correctly misses cache")


class TestCacheStoreTTL:
    def test_expired_entry_returns_none(self):
        async def run():
            store = make_store()
            emb = rand_embedding()
            response = make_response("This answer expires fast.")
            await store.store(emb, "tenant1:sys001", response, ttl=1)  # 1 second TTL

            # Manually age the entry by patching lookup's time check
            store.thresholds.default = 0.5
            time.sleep(2)  # wait for TTL to expire
            hit = await store.lookup(emb, "tenant1:sys001")
            assert hit is None, "Expired entry should return None"
        asyncio.run(run())
        print("  PASS: expired TTL returns None")


class TestCacheStoreMultiTenancy:
    def test_namespaces_are_isolated(self):
        async def run():
            store = make_store()
            emb = rand_embedding()
            response_a = make_response("Tenant A answer.")
            response_b = make_response("Tenant B answer.")

            await store.store(emb, "tenantA:sys001", response_a)
            await store.store(emb, "tenantB:sys001", response_b)

            store.thresholds.default = 0.5
            hit_a = await store.lookup(emb, "tenantA:sys001")
            hit_b = await store.lookup(emb, "tenantB:sys001")

            assert hit_a["response"]["choices"][0]["message"]["content"] == "Tenant A answer."
            assert hit_b["response"]["choices"][0]["message"]["content"] == "Tenant B answer."
        asyncio.run(run())
        print("  PASS: tenant namespaces are isolated")


class TestCacheStoreInvalidation:
    def test_namespace_invalidation(self):
        async def run():
            store = make_store()
            emb = rand_embedding()
            response = make_response("Will be deleted.")
            await store.store(emb, "tenant1:sys001", response)

            await store.invalidate_namespace("tenant1:sys001")

            store.thresholds.default = 0.5
            hit = await store.lookup(emb, "tenant1:sys001")
            assert hit is None, "Invalidated namespace should return None"
        asyncio.run(run())
        print("  PASS: namespace invalidation works")


class TestCacheStoreThresholds:
    def test_per_query_type_thresholds(self):
        async def run():
            store = make_store()
            base_emb = rand_embedding()
            # noisy embedding — moderate similarity
            noisy_emb = similar_embedding(base_emb, noise=0.15)
            response = make_response("Cached.")
            await store.store(base_emb, "tenant1:factual", response, query_type="factual")

            # high threshold: should miss
            hit_strict = await store.lookup(noisy_emb, "tenant1:factual", query_type="factual")

            # lower threshold: should hit
            store.thresholds.creative = 0.1
            await store.store(base_emb, "tenant1:creative", response, query_type="creative")
            hit_loose = await store.lookup(noisy_emb, "tenant1:creative", query_type="creative")

            assert hit_strict is None, "Factual threshold (0.96) should reject noisy vector"
            assert hit_loose is not None, "Creative threshold (0.1) should accept noisy vector"
        asyncio.run(run())
        print("  PASS: per-query-type thresholds work correctly")


class TestCosineSimilarity:
    def test_cosine_similarity_identical(self):
        v = rand_embedding()
        score = QueryEncoder.cosine_similarity(v, v)
        assert abs(score - 1.0) < 1e-6
        print("  PASS: cosine similarity of identical vectors = 1.0")

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        score = QueryEncoder.cosine_similarity(a, b)
        assert abs(score) < 1e-6
        print("  PASS: cosine similarity of orthogonal vectors = 0.0")

    def test_cosine_similarity_range(self):
        a = rand_embedding(128)
        b = rand_embedding(128)
        score = QueryEncoder.cosine_similarity(a, b)
        assert -1.0 <= score <= 1.0
        print(f"  PASS: cosine similarity in [-1, 1] range (got {score:.4f})")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    suites = [
        TestCacheStoreLookupMiss(),
        TestCacheStoreStoreAndHit(),
        TestCacheStoreSimilarHit(),
        TestCacheStoreDissimilarMiss(),
        TestCacheStoreTTL(),
        TestCacheStoreMultiTenancy(),
        TestCacheStoreInvalidation(),
        TestCacheStoreThresholds(),
        TestCosineSimilarity(),
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
                print(f"  FAIL: {e}")
                failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed!")
    else:
        sys.exit(1)
