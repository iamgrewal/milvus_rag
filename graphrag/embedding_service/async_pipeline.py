import asyncio
from graphrag.embedding_service.service import embed_and_store_batch

async def process_documents_async(documents, batch_size=100):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        await embed_and_store_batch(batch)
