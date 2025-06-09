---
description: Health check script for verifying the status of services in the application.
globs: ['health_check.sh']
alwaysApply: false
---

# Health Check Script Documentation

## Overview
The `health_check.sh` script is designed to perform health checks on critical services used by the application. It checks the status of the Neo4j database, the Milvus service port, and the GraphRAG application to ensure they are operational before proceeding with further tasks.

## Key Components
- **Neo4j Health Check**: The script sends a request to the Neo4j service and checks if it returns an HTTP status code of 200. If not, it indicates that the Neo4j service is not healthy.
- **Milvus Port Check**: It uses the `nc` (netcat) command to verify if the Milvus service is reachable on port 19530. If the port is not open, it reports the service as unreachable.
- **GraphRAG Application Check**: The script retrieves the last 10 lines of the GraphRAG application logs from Docker and checks for the presence of the phrase "Embedding model loaded" to confirm that the application has initialized correctly.

## Dependencies
This script does not import any other files in the repository, nor is it imported by any other files. It operates independently to check the health of the services.

## Usage Example
To run the health check, execute the script in a terminal:
```bash
bash health_check.sh
```

## Best Practices
- Ensure that the services being checked (Neo4j, Milvus, GraphRAG) are running before executing this script.
- Regularly monitor the output of this script to catch any service issues early.
- Consider integrating this health check into a CI/CD pipeline to automate service verification during deployments.