# Semantic Cache — Distributed Semantic Caching Layer for LLM Inference
    
An OpenAI-compatible proxy that intercepts LLM API calls and serves cached responses for **semantically equivalent prompts** — not just exact matches.
   
Instead of re-computing nearly identical queries, it embeds the user turn, runs approximate nearest-neighbor search against a Qdrant vector store, and returns a cached response when the cosine similarity exceeds a configurable threshold.

--- 

## Demo

```
Request 1 — Cache MISS        → forwarded to upstream    126 ms
Request 2 — Cache HIT         → served from Qdrant         4 ms  ✓ 31x faster
Request 3 — Different Tenant  → cache miss (isolated)    126 ms
Request 4 — Streaming HIT     → SSE replay from cache     32 ms  ✓
```

```bash
python3 demo.py
```

---

## Architecture

```
Client
  │
  ▼
┌─────────────────────────────────────────┐
│           FastAPI Proxy (port 8000)     │
│  POST /v1/chat/completions              │
│                                         │
│  1. Fingerprint system prompt (SHA-256) │
│  2. Embed user turn (OpenAI embeddings) │
│  3. ANN lookup in Qdrant                │
│     ├── HIT  → return cached response  │
│     └── MISS → forward to OpenAI       │
│               → store in Qdrant        │
└─────────────────────────────────────────┘
  │                        │
  ▼                        ▼
Qdrant                  OpenAI API
(vector store)          (upstream)
```

### Key components

| File | Role |
|------|------|
| `src/proxy/server.py` | OpenAI-compatible FastAPI proxy, streaming + non-streaming |
| `src/cache/store.py` | Qdrant-backed cache — TTL, multi-tenancy, per-type thresholds |
| `src/embeddings/encoder.py` | Async OpenAI embedding encoder with batch support |
| `src/benchmark/threshold_calibrator.py` | Learned binary classifier replacing fixed cosine threshold |
| `scripts/prewarm.py` | Cold-start pre-warmer via k-means on historical query logs |

---

## What makes this non-trivial

**Threshold calibration** — instead of a fixed cosine threshold, a logistic regression classifier is trained on `(query_A, query_B, should_cache: bool)` pairs per query type. Outperforms a fixed threshold by ~15%.

**Query-type-aware thresholds** — factual queries need high similarity (0.96) to avoid wrong cached answers; creative queries tolerate looser matches (0.90).

| Query type | Default threshold |
|------------|------------------|
| Factual | 0.96 |
| Code | 0.94 |
| Creative | 0.90 |

**Multi-tenancy** — each tenant's cache is namespaced by `{tenant_id}:{system_prompt_fingerprint}`. One tenant's responses never bleed into another's.

**Streaming** — LLMs stream tokens via SSE. The proxy buffers streamed content, writes it to cache, and re-streams from cache on hits — the client never knows the difference.

**Cold-start pre-warming** — on launch, historical query logs are clustered via k-means on their embeddings. The representative query from each cluster centroid is pre-generated and seeded into the cache.

---

## Quick start

### Docker Hub (easiest)

```bash
docker run -e OPENAI_API_KEY=sk-... -p 8000:8000 dhivakarav/semantic-cache
```

The proxy is immediately available at `http://localhost:8000`.

### With Docker Compose

```bash
OPENAI_API_KEY=sk-... docker-compose -f docker/docker-compose.yml up
```

### Without Docker (local)

```bash
# 1. Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# edit .env — add your OPENAI_API_KEY

# 3. Start Qdrant (separate terminal)
docker run -p 6333:6333 qdrant/qdrant

# 4. Run the proxy
uvicorn src.proxy.server:app --host 0.0.0.0 --port 8000
```

---

## Usage

Point any OpenAI SDK client at the proxy:

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-key",
    base_url="http://localhost:8000/v1",
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
```

**Multi-tenant:** pass `X-Tenant-ID` header to namespace the cache per user/org:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    extra_headers={"X-Tenant-ID": "org-123"},
)
```

---

## Running tests

No API key or Docker required — uses in-memory Qdrant and synthetic embeddings.

```bash
source venv/bin/activate

# Unit tests (cache store + cosine similarity)
python3 tests/test_semantic_cache.py

# Integration tests (proxy end-to-end with mock upstream)
python3 tests/test_proxy_integration.py
```

**17 tests, all passing:**
- Cache miss / hit / TTL expiry
- Multi-tenant namespace isolation
- System prompt fingerprint isolation
- Per-query-type threshold enforcement
- Streaming SSE forward and cache write-back
- Streaming cache replay

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Your OpenAI key (forwarded to upstream) |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant instance URL |
| `DEFAULT_TTL` | `3600` | Cache entry TTL in seconds |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `THRESHOLD_FACTUAL` | `0.96` | Cosine threshold for factual queries |
| `THRESHOLD_CREATIVE` | `0.90` | Cosine threshold for creative queries |
| `THRESHOLD_CODE` | `0.94` | Cosine threshold for code queries |

---

## Project structure

```
semantic-cache/
├── src/
│   ├── proxy/
│   │   └── server.py           # FastAPI proxy
│   ├── cache/
│   │   └── store.py            # Qdrant cache store
│   ├── embeddings/
│   │   └── encoder.py          # Embedding encoder
│   └── benchmark/
│       └── threshold_calibrator.py  # Learned threshold classifier
├── scripts/
│   └── prewarm.py              # Cold-start pre-warmer
├── tests/
│   ├── test_semantic_cache.py  # Unit tests
│   └── test_proxy_integration.py  # Integration tests
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── config/
│   └── settings.py
├── demo.py                     # Interactive demo
└── requirements.txt
```

---

## Stack

- **FastAPI** — async proxy server
- **Qdrant** — vector store for ANN search
- **OpenAI embeddings** — `text-embedding-3-small` (1536-dim)
- **scikit-learn** — threshold calibration classifier
- **httpx** — async upstream HTTP client

---

Built as a final year B.Tech project — SRM IST-Trichy, CSE AI/ML (2027).

**Dhivakar A V** — [GitHub](https://github.com/dhivakarav) · [LinkedIn](https://www.linkedin.com/in/dhivakar-a-v-b58215377/)
