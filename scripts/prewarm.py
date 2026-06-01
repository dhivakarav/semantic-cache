"""
Cold-start pre-warming: cluster historical query logs via k-means on embeddings,
generate responses for cluster centroids, and seed the cache.
"""
import asyncio
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from src.cache.store import SemanticCacheStore
from src.embeddings.encoder import QueryEncoder


async def prewarm(
    query_log_path: str,
    openai_api_key: str,
    n_clusters: int = 50,
    namespace: str = "default:prewarm",
):
    logs = json.loads(Path(query_log_path).read_text())
    texts = [entry["user_text"] for entry in logs]

    encoder = QueryEncoder(api_key=openai_api_key)
    print(f"Embedding {len(texts)} historical queries...")
    embeddings = await encoder.encode_batch(texts)

    X = np.array(embeddings)
    print(f"Clustering into {n_clusters} centroids...")
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    km.fit(X)

    centroids = km.cluster_centers_.tolist()
    store = SemanticCacheStore()

    print("Seeding cache with centroid queries...")
    for i, (centroid, label) in enumerate(zip(centroids, range(n_clusters))):
        # find closest real query to this centroid
        dists = np.linalg.norm(X - np.array(centroid), axis=1)
        closest_idx = int(np.argmin(dists))
        representative_text = texts[closest_idx]
        print(f"  [{i+1}/{n_clusters}] Centroid representative: {representative_text[:60]}...")

        # placeholder: in real usage, call LLM here and store real response
        synthetic_response = {
            "choices": [{"message": {"role": "assistant", "content": f"[prewarm-{i}]"}}]
        }
        await store.store(centroid, namespace, synthetic_response, ttl=86400)

    print("Pre-warming complete.")


if __name__ == "__main__":
    import sys
    asyncio.run(prewarm(sys.argv[1], sys.argv[2]))
