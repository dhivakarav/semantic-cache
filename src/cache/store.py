import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

COLLECTION = "semantic_cache"
VECTOR_DIM = 1536  # text-embedding-3-small
DEFAULT_TTL = 3600  # 1 hour


@dataclass
class ThresholdConfig:
    factual: float = 0.96
    creative: float = 0.90
    code: float = 0.94
    default: float = 0.93


class SemanticCacheStore:
    def __init__(self, url: str = "http://localhost:6333", client: Optional[AsyncQdrantClient] = None):
        self.client = client if client is not None else AsyncQdrantClient(url=url)
        self.thresholds = ThresholdConfig()
        self._initialized = False

    async def _ensure_collection(self):
        if self._initialized:
            return
        collections = await self.client.get_collections()
        names = [c.name for c in collections.collections]
        if COLLECTION not in names:
            await self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
        self._initialized = True

    def _get_threshold(self, query_type: str = "default") -> float:
        return getattr(self.thresholds, query_type, self.thresholds.default)

    async def lookup(
        self, embedding: List[float], namespace: str, query_type: str = "default"
    ) -> Optional[dict]:
        await self._ensure_collection()
        threshold = self._get_threshold(query_type)
        now = time.time()

        result = await self.client.query_points(
            collection_name=COLLECTION,
            query=embedding,
            query_filter=Filter(
                must=[FieldCondition(key="namespace", match=MatchValue(value=namespace))]
            ),
            limit=1,
            score_threshold=threshold,
            with_payload=True,
        )

        if not result.points:
            return None

        hit = result.points[0]
        payload = hit.payload

        # TTL check
        if now - payload.get("created_at", 0) > payload.get("ttl", DEFAULT_TTL):
            await self.client.delete(
                collection_name=COLLECTION,
                points_selector=[hit.id],
            )
            return None

        return {"response": payload["response"], "score": hit.score}

    async def store(
        self,
        embedding: List[float],
        namespace: str,
        response: dict,
        ttl: int = DEFAULT_TTL,
        query_type: str = "default",
    ):
        await self._ensure_collection()
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "namespace": namespace,
                "response": response,
                "created_at": time.time(),
                "ttl": ttl,
                "query_type": query_type,
            },
        )
        await self.client.upsert(collection_name=COLLECTION, points=[point])

    async def invalidate_namespace(self, namespace: str):
        await self.client.delete(
            collection_name=COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="namespace", match=MatchValue(value=namespace))]
            ),
        )

    async def delete_by_id(self, point_id: str):
        await self.client.delete(
            collection_name=COLLECTION,
            points_selector=[point_id],
        )
