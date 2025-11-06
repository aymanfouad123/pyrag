from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid

class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=3072):
        """Initialize Qdrant storage with a collection."""
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        
        # create collection if it doesn't exist
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
    
    def add_vectors(self, ids, vectors, payloads=None):
        """
        Add vectors to the collection.
        """
        
        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i]
            )
            for i in range(len(ids))
        ]
        
        self.client.upsert(
            collection_name=self.collection,
            points=points
        )
        
    def search(self, query, k=5):
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query,
            with_payload=True,
            limit=k
        )
        context = []
        sources = set()
        for r in results:
            payload = getattr(r, "payload", {})
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                context.append(text)
                sources.add(source)
        
        return {
            "context": context,
            "sources": sources
        }
    