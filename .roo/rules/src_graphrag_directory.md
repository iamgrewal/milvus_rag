---
description: Documentation for the src/graphrag directory, which contains logging functionality.
globs: ['src/graphrag/*']
alwaysApply: false
---

# Overview of src/graphrag

The `src/graphrag` directory is dedicated to logging functionality within the codebase. It serves as a centralized location for managing log outputs, which can be crucial for debugging and monitoring the application's behavior.

## Key Files

- **logger.py**: This is the primary file in the `src/graphrag` directory. It contains the implementation of the logging system, including configuration settings, log formatting, and methods for logging messages at various severity levels (e.g., info, warning, error). This file is essential for ensuring that all parts of the application can log messages consistently.

## Relationship to Other Parts of the Codebase

The `src/graphrag` directory does not import from or get imported by any other directories, indicating that it is self-contained. However, the logging functionality provided by `logger.py` is likely utilized throughout the application wherever logging is necessary. Developers should ensure that they import and use the logger correctly to maintain a consistent logging strategy across the codebase.

## Common Patterns and Conventions

- **Logging Levels**: The logger should support multiple levels of logging (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL) to allow for flexible logging based on the application's needs.
- **Configuration**: It is common to configure the logger at the start of the application, setting parameters such as log file location, log level, and format. This configuration should be done in `logger.py` to ensure that all logging follows the same rules.

## Best Practices

- **Consistent Usage**: Always use the logging methods provided in `logger.py` instead of using print statements for debugging. This ensures that all logs are captured and formatted consistently.
- **Log Meaningful Messages**: When logging, provide clear and meaningful messages that can help in diagnosing issues later. Avoid vague messages that do not provide context.
- **Avoid Logging Sensitive Information**: Be cautious not to log sensitive information such as passwords or personal data to comply with privacy regulations and best practices.
- **Review Log Levels**: Regularly review the log levels used in the application to ensure that they are appropriate for the production environment. For example, DEBUG logs may be useful during development but should be minimized in production.