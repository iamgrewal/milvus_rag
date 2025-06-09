---
description: Documentation for the graphrag/config directory, detailing its purpose and key files.
globs: ['graphrag/config/*']
alwaysApply: false
---

# Overview of the `graphrag/config` Directory

The `graphrag/config` directory is responsible for managing configuration settings and secure configurations for the Graphrag application. It contains essential Python files that define how the application interacts with various services and manages its operational parameters.

## Key Files

### `settings.py`
This file contains general configuration settings for the Graphrag application. It typically includes parameters such as database connection strings, application modes (development, testing, production), and other environment-specific settings.

### `secure_config.py`
This file is dedicated to managing sensitive information, such as API keys, passwords, and other credentials. It is crucial to ensure that this file is kept secure and not exposed in version control systems.

## Relationship to Other Parts of the Codebase

Files in the `graphrag/config` directory are imported by various managers within the Graphrag application, specifically:
- `graphrag/neo4j/manager.py`
- `graphrag/milvus/manager.py`
- `src/graphrag/neo4j/manager.py`
- `src/graphrag/milvus/manager.py`

These managers utilize the configurations defined in `settings.py` and `secure_config.py` to establish connections and manage interactions with external services like Neo4j and Milvus.

## Common Patterns and Conventions

- **Environment Variables**: It is a common practice to use environment variables for sensitive information, which can be loaded into `secure_config.py` to avoid hardcoding credentials.
- **Modular Configuration**: The settings are often modularized to allow for easy overrides based on the environment (e.g., development vs. production).

## Best Practices

- **Keep Sensitive Information Secure**: Always ensure that `secure_config.py` is excluded from version control (e.g., by adding it to `.gitignore`). Use environment variables or secret management tools to handle sensitive data.
- **Document Configuration Options**: Clearly comment on the purpose of each configuration option in `settings.py` to aid other developers in understanding the configuration.
- **Consistent Naming Conventions**: Follow consistent naming conventions for configuration variables to enhance readability and maintainability.

By adhering to these practices, developers can ensure that the configuration management within the Graphrag application remains organized, secure, and easy to understand.