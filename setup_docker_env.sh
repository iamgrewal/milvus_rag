#!/bin/bash

cd /mnt/d/projects/wslprojects/milvus_env || exit

echo "Creating .env file..."
cat <<EOF > .env
MILVUS_HOST=milvus-standalone
MILVUS_PORT=19530
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
COLLECTION_NAME=graph_rag
LOG_LEVEL=INFO
EOF

echo "Creating Dockerfile..."
cat <<EOF > Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY ./graphrag /app/graphrag

RUN apt-get update && apt-get install -y git curl && \\
    pip install --no-cache-dir -r /app/graphrag/requirements.txt && \\
    python -m nltk.downloader stopwords && \\
    python -m spacy download en_core_web_sm

ENV PYTHONPATH="/app"

CMD ["python", "graphrag/rag_system/main.py"]
EOF

echo "Creating requirements.txt..."
cat <<EOF > graphrag/requirements.txt
pymilvus==2.3.4
neo4j==5.13.0
sentence-transformers==2.2.2
spacy==3.6.1
nltk==3.8.1
transformers==4.40.1
dateparser==1.2.0
python-dotenv==1.0.1
EOF

echo "Creating docker-compose.yml..."
cat <<EOF > docker-compose.yml
version: '3.8'

services:
  milvus-standalone:
    image: milvusdb/milvus:v2.3.4
    container_name: milvus-standalone
    ports:
      - \"19530:19530\"
    healthcheck:
      test: [\"CMD\", \"curl\", \"-f\", \"http://localhost:9091\"] || exit 1
      interval: 30s
      timeout: 10s
      retries: 3

  neo4j:
    image: neo4j:5.13
    container_name: neo4j
    ports:
      - \"7474:7474\"
      - \"7687:7687\"
    environment:
      - NEO4J_AUTH=neo4j/password
    volumes:
      - neo4j-data:/data

  graphrag-app:
    build: .
    depends_on:
      - milvus-standalone
      - neo4j
    env_file:
      - .env
    volumes:
      - ./graphrag:/app/graphrag

volumes:
  neo4j-data:
EOF

echo "Docker environment setup files created."
