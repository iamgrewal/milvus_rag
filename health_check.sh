#!/bin/bash

echo "Running service health check..."

# Check Neo4j
NEO4J_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:7474)
if [ "$NEO4J_HEALTH" != "200" ]; then
  echo "Neo4j service not healthy (HTTP $NEO4J_HEALTH)"
  exit 1
else
  echo "Neo4j is healthy."
fi

# Check Milvus Port
MILVUS_PORT=$(nc -zv localhost 19530 2>&1)
echo "$MILVUS_PORT" | grep -q succeeded
if [ $? -ne 0 ]; then
  echo "Milvus port 19530 not open."
  exit 1
else
  echo "Milvus is reachable."
fi

# Check App
APP_LOG=$(docker logs graphrag-app 2>&1 | tail -n 10)
echo "$APP_LOG" | grep -iq "Embedding model loaded"
if [ $? -ne 0 ]; then
  echo "GraphRAG app not initialized correctly."
  exit 1
else
  echo "GraphRAG app appears healthy."
fi

exit 0
