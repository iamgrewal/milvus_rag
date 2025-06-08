#!/bin/bash

# Install virtualenv if not present
if ! command -v virtualenv &> /dev/null; then
    echo "Installing virtualenv..."
    pip install virtualenv==20.31.2
fi

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    virtualenv venv
fi

# Activate virtual environment
source venv/bin/activate

## Secure Configuration Utility

#```python
cat <<EOF > graphrag/config/secure_config.py
# graphrag/config/secure_config.py
import os
from cryptography.fernet import Fernet

class SecureConfig:
    def __init__(self):
        encryption_key = os.getenv("ENCRYPTION_KEY")
        if not encryption_key:
            raise ValueError("ENCRYPTION_KEY must be set in the environment.")
        self.cipher = Fernet(encryption_key)

    def get_api_key(self, service: str) -> str:
        encrypted = os.getenv(f"{service.upper()}_API_KEY_ENCRYPTED")
        if not encrypted:
            raise ValueError(f"Encrypted API key for {service} not found in environment.")
        return self.cipher.decrypt(encrypted.encode()).decode()
EOF
#```

#---

## Asynchronous Document Processing

#```python
# graphrag/embedding_service/async_pipeline.py
cat <<EOF > graphrag/embedding_service/async_pipeline.py
import asyncio
from graphrag.embedding_service.service import embed_and_store_batch

async def process_documents_async(documents, batch_size=100):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        await embed_and_store_batch(batch)
EOF
#```

#---

## Structured Logging with Retry

#```python
cat <<EOF > graphrag/rag_system/service.py
# graphrag/rag_system/service.py
import logging
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger()

class RAGService:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def query_with_retry(self, query: str):
        try:
            result = await self._perform_rag_query(query)
            logger.info("query_successful", query_length=len(query))
            return result
        except Exception as e:
            logger.error("query_failed", error=str(e), query_hash=hash(query))
            raise

    async def _perform_rag_query(self, query: str):
        # Placeholder for actual RAG logic
        pass
EOF
##```

#---

## FastAPI Integration

#```python

cat <<EOF > graphrag/rag_system/main.py
# graphrag/rag_system/main.py
from fastapi import FastAPI, HTTPException
from graphrag.rag_system.service import RAGService

app = FastAPI()
rag_service = RAGService()

@app.get("/query")
async def query_rag(q: str):
    try:
        return await rag_service.query_with_retry(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
EOF
#```

#---

## Environment `.env` Example

#```env
#ENCRYPTION_KEY=your_fernet_key
#OPENAI_API_KEY_ENCRYPTED=gAAAAABexampleencryptedtoken
#```

#---

#These updates enable:
#- ✅ Secure credential storage using environment encryption
#- ✅ Asynchronous batch processing
#- ✅ Retry handling with logging
#- ✅ FastAPI-based REST API layer for RAG queries

# Optional: Generate ctags if available
if command -v ctags &> /dev/null; then
    echo "Generating ctags..."
    ctags -R --languages=Python --python-kinds=-iv -f .tags .
else
    echo "Warning: ctags not found. Skipping tag generation."
fi

