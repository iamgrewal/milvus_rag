---
description: 
globs: 
alwaysApply: false
---
---
description: This rule outlines best practices for using the dotenv library in Python projects, focusing on security, maintainability, and configuration management. It provides guidelines for managing environment variables effectively across different environments.
globs: **/.env
---
# dotenv Library Best Practices

This guide outlines best practices for effectively using the `dotenv` library in Python projects. It covers code organization, common patterns, performance considerations, security best practices, testing approaches, common pitfalls, and tooling.

## Library Information:
- Name: dotenv
- Tags: python, configuration, environment-variables, security

## 1. Code Organization and Structure

*   **Project Root Placement:** Place the `.env` file in the root directory of your project. This ensures it's easily discoverable and accessible during development.
*   **Configuration Files:** Use separate `.env` files for different environments (e.g., `.env.development`, `.env.test`, `.env.production`). This allows you to manage environment-specific configurations effectively.
*   **Centralized Configuration:** Create a dedicated module (e.g., `config.py`) to load and manage environment variables. This provides a single point of access for all configuration settings, improving code maintainability.

    python
    # config.py
    import os
    from dotenv import load_dotenv

    load_dotenv()

    class Config:
        SECRET_KEY = os.getenv("SECRET_KEY")
        DATABASE_URL = os.getenv("DATABASE_URL")
        DEBUG = os.getenv("DEBUG", "False").lower() == "true" # Default to False

    config = Config()
    

*   **Modularization:** Divide your configuration settings into logical groups within the `config.py` module. This improves code readability and maintainability, especially for large projects.

## 2. Common Patterns and Anti-patterns

*   **Pattern: Configuration Objects:** Use configuration objects (e.g., classes) to encapsulate environment variables. This allows for type hinting, validation, and easier access to configuration settings.

    python
    # config.py (using dataclasses)
    import os
    from dataclasses import dataclass
    from dotenv import load_dotenv

    load_dotenv()

    @dataclass
    class Config:
        secret_key: str = os.getenv("SECRET_KEY")
        database_url: str = os.getenv("DATABASE_URL")
        debug: bool = os.getenv("DEBUG", "False").lower() == "true" # Default to False

    config = Config()

    print(f"Secret Key: {config.secret_key}")
    print(f"Database URL: {config.database_url}")
    print(f"Debug Mode: {config.debug}")
    

*   **Pattern: Environment-Specific Loading:** Dynamically load the appropriate `.env` file based on the current environment.

    python
    # config.py
    import os
    from dotenv import load_dotenv

    environment = os.getenv("ENVIRONMENT", "development")
    dotenv_path = f".env.{environment}"
    load_dotenv(dotenv_path=dotenv_path, verbose=True) # Ensure load succeeds or fails loudly

    SECRET_KEY = os.getenv("SECRET_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true" # Default to False

    print(f"Secret Key: {SECRET_KEY}")
    print(f"Database URL: {DATABASE_URL}")
    print(f"Debug Mode: {DEBUG}")
    

*   **Anti-pattern: Hardcoding Secrets:** Avoid hardcoding sensitive information (e.g., API keys, passwords) directly in your code. Always store them in environment variables.
*   **Anti-pattern: Over-reliance on `.env` in Production:** While `.env` files are convenient for development, avoid relying on them directly in production. Use environment variables set at the system level or secrets management solutions.
*   **Anti-pattern: Ignoring Validation:** Failing to validate environment variables can lead to unexpected runtime errors. Always validate that required variables are set and conform to expected formats.

## 3. Performance Considerations

*   **Loading on Startup:** Load environment variables only once at application startup. Avoid repeatedly loading the `.env` file during runtime.
*   **Caching:** If necessary, cache the configuration object after loading it. This can improve performance, especially for frequently accessed settings. However, make sure to invalidate the cache when the environment changes.
*   **Lazy Loading:** For less frequently used configuration options, consider lazy loading them. This can reduce startup time and memory footprint.

## 4. Security Best Practices

*   **.gitignore:** Always add the `.env` file to your `.gitignore` file to prevent it from being committed to version control.
*   **Environment Variables in Production:** In production environments, set environment variables directly on the server or container platform (e.g., Docker, Kubernetes) instead of relying on `.env` files.
*   **Secrets Management:** Use dedicated secrets management tools (e.g., AWS Secrets Manager, HashiCorp Vault) to store and manage sensitive information in production. These tools provide encryption, access control, and auditing capabilities.
*   **Least Privilege:** Grant only the necessary permissions to access environment variables. Avoid granting broad access to sensitive configuration settings.
*   **Regular Rotation:** Regularly rotate sensitive credentials (e.g., API keys, passwords) to minimize the impact of potential security breaches.
*   **Validation:** Validate environment variables to ensure they conform to expected formats and values. This can help prevent injection attacks and other security vulnerabilities.

    python
    import os
    from dotenv import load_dotenv
    import validators

    load_dotenv()

    DATABASE_URL = os.getenv("DATABASE_URL")
    if not validators.url(DATABASE_URL):
        raise ValueError("Invalid DATABASE_URL format")
    

## 5. Testing Approaches

*   **Unit Tests:** Mock the `os.getenv` function to simulate different environment variable values during unit testing.

    python
    import os
    import unittest
    from unittest.mock import patch

    class TestConfig(unittest.TestCase):
        @patch.dict(os.environ, {"DATABASE_URL": "test_db_url", "DEBUG": "True"})
        def test_config_values(self):
            from your_module import config # Replace your_module
            self.assertEqual(config.database_url, "test_db_url")
            self.assertEqual(config.debug, True)
    

*   **Integration Tests:** Use separate `.env` files for testing environments to ensure that tests are isolated and do not affect other environments.
*   **End-to-End Tests:** Verify that environment variables are correctly loaded and used in end-to-end tests.

## 6. Common Pitfalls and Gotchas

*   **Variable Shadowing:** Be aware of variable shadowing, where environment variables override local variables with the same name.
*   **Type Conversion:** Remember that environment variables are always strings. You may need to convert them to the appropriate data type (e.g., integer, boolean) before using them.

    python
    import os

    port = int(os.getenv("PORT", "8000")) # Convert to integer
    debug = os.getenv("DEBUG", "False").lower() == "true" # Convert to boolean
    

*   **Missing Variables:** Handle cases where environment variables are not set. Provide default values or raise informative errors.
*   **Whitespace:** Be mindful of leading or trailing whitespace in environment variable values. Use the `strip()` method to remove whitespace if necessary.
*   **Case Sensitivity:** Environment variable names are case-sensitive on some operating systems. Use consistent naming conventions to avoid issues.
*   **Encoding:** Ensure that your `.env` file is saved with UTF-8 encoding to avoid issues with special characters.

## 7. Tooling and Environment

*   **python-dotenv:** The primary library for loading environment variables from `.env` files.
*   **direnv:** A tool that automatically loads environment variables when you enter a directory. This can be useful for development environments.
*   **Docker Compose:** A tool for defining and running multi-container Docker applications. Docker Compose can be used to set environment variables for containers.
*   **Kubernetes:** A container orchestration platform. Kubernetes provides mechanisms for managing environment variables and secrets for pods.
*   **Pydantic:** A data validation and settings management library. Pydantic can be used to define configuration objects and validate environment variables.
*   **Envalid:** A library for validating environment variables.
*   **dotenv-vault:** A secure way to manage your .env files and sync them across your team.

## 8. Additional Best Practices

*   **Use Descriptive Names:** Choose descriptive and consistent names for your environment variables (e.g., `DATABASE_URL`, `API_KEY`).
*   **Document Environment Variables:** Maintain clear documentation outlining each environment variable, its purpose, and expected format.
*   **Avoid Default Secrets:** Do not use default values for sensitive secrets in your `.env` files. Force developers to explicitly set these values.
*   **Regularly Review Configuration:** Periodically review your configuration settings to ensure they are still valid and secure.
*   **Automated Configuration:** Use configuration management tools (e.g., Ansible, Chef) to automate the deployment and configuration of environment variables in production.

By following these best practices, you can effectively manage environment variables in your Python projects, improving security, maintainability, and configuration management.