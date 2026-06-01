import asyncio
from typing import List

import httpx
import numpy as np

OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"
DEFAULT_MODEL = "text-embedding-3-small"


class QueryEncoder:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str = ""):
        self.model = model
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=30)

    async def encode(self, text: str) -> List[float]:
        resp = await self._client.post(
            OPENAI_EMBED_URL,
            json={"input": text, "model": self.model},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    async def encode_batch(self, texts: List[str]) -> List[List[float]]:
        resp = await self._client.post(
            OPENAI_EMBED_URL,
            json={"input": texts, "model": self.model},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda x: x["index"])
        return [d["embedding"] for d in data]

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        va, vb = np.array(a), np.array(b)
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
