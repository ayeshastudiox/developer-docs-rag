from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


def query_pinecone(embedding, top_k=3):
    results = index.query(
        vector=embedding, top_k=top_k, include_metadata=True
    )

    contexts = []
    for match in results.get("matches", []):
        if "metadata" in match and "text" in match["metadata"]:
            contexts.append(match["metadata"]["text"])

    return "\n".join(contexts)