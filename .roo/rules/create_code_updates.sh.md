---
description: Documentation for the create_code_updates.sh script that sets up a Python environment and configures various components of the application.
globs: ['create_code_updates.sh']
alwaysApply: false
---

# Overview
The `create_code_updates.sh` script is a Bash script designed to set up a Python virtual environment, install necessary dependencies, and create several key configuration files for a Python application. It facilitates secure credential storage, asynchronous document processing, structured logging with retry mechanisms, and a FastAPI-based REST API for querying.

# Key Components

## Virtual Environment Setup
- **Installation of `virtualenv`:** The script checks if `virtualenv` is installed and installs it if not present. This is crucial for creating isolated Python environments.
- **Creation and Activation of Virtual Environment:** If a directory named `venv` does not exist, it creates one and activates it to ensure that subsequent Python commands run in this isolated environment.

## Secure Configuration Utility
- **File:** `graphrag/config/secure_config.py`
- **Class:** `SecureConfig`
  - **Purpose:** Manages secure access to API keys by decrypting them using a provided encryption key from the environment. It raises errors if the required environment variables are not set.
  - **Method:** `get_api_key(service: str)`: Retrieves and decrypts the API key for a specified service.

## Asynchronous Document Processing
- **File:** `graphrag/embedding_service/async_pipeline.py`
- **Function:** `process_documents_async(documents, batch_size=100)`
  - **Purpose:** Processes documents in batches asynchronously, calling the `embed_and_store_batch` function for each batch.

## Structured Logging with Retry
- **File:** `graphrag/rag_system/service.py`
- **Class:** `RAGService`
  - **Method:** `query_with_retry(query: str)`
    - **Purpose:** Executes a query with retry logic, logging success and failure events. It uses the `tenacity` library to handle retries with exponential backoff.

## FastAPI Integration
- **File:** `graphrag/rag_system/main.py`
- **Function:** `query_rag(q: str)`
  - **Purpose:** Exposes an HTTP GET endpoint for querying the RAG service, handling exceptions and returning appropriate HTTP responses.

## Environment Configuration
- **Example `.env` File:**
  - Contains environment variables for encryption keys and encrypted API keys, which are essential for the secure configuration utility.

# Usage Examples
1. **Run the Script:** Execute `bash create_code_updates.sh` to set up the environment and create necessary files.
2. **Environment Variables:** Ensure to set the `ENCRYPTION_KEY` and any service-specific API keys in your environment before running the application.

# Best Practices
- **Environment Management:** Always use a virtual environment for Python projects to avoid dependency conflicts.
- **Secure Handling of Secrets:** Store sensitive information like API keys in environment variables and never hard-code them in your source code.
- **Error Handling:** Implement robust error handling in your asynchronous functions to ensure that failures are logged and managed appropriately.
- **Documentation:** Keep your code well-documented, especially when dealing with encryption and sensitive data handling.