---
description: "Security and deployment guidelines for hybrid RAG."
globs: ["**/deploy/**", "**/config/**"]
alwaysApply: true
---
# Security & Deployment

- Use .env or Vault for secrets.
- TLS encryption for all services.
- Containerize with Docker; orchestrate via Docker Compose or Kubernetes.

@deploy/secure_startup.sh
