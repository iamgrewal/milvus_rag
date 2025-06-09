---
description: 
globs: 
alwaysApply: false
---
---
description: This rule file provides comprehensive guidelines for using the structlog library in Python, covering best practices for code organization, performance, security, testing, and common pitfalls. It aims to ensure consistent, efficient, and secure logging practices across projects using structlog.
globs: **/*.py
---
# structlog Best Practices: A Comprehensive Guide

This document outlines best practices for using the `structlog` library in Python. Following these guidelines will lead to more maintainable, efficient, and secure logging practices.

## Library Information:
- Name: structlog
- Tags: python, logging, structured-logging, observability

## 1. Code Organization and Structure

### 1.1. Centralized Configuration

- **Best Practice:** Configure `structlog` in a single, well-defined location in your project. This avoids inconsistent configurations across different modules.
- **Example:**

  python
  # config.py
  import structlog
  import logging
  import sys

  def configure_logging():
      shared_processors = [
          structlog.stdlib.add_log_level,
          structlog.stdlib.add_logger_name,
          structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S.%f"),
          structlog.processors.StackInfoRenderer(),
          structlog.processors.format_exc_info,
      ]

      if sys.stderr.isatty():
          # Development: Pretty console output
          processors = shared_processors + [
              structlog.dev.ConsoleRenderer(),
          ]
      else:
          # Production: JSON output for log aggregators
          processors = shared_processors + [
              structlog.processors.JSONRenderer(),
          ]

      structlog.configure(
          processors=processors,
          logger_factory=structlog.stdlib.LoggerFactory(),
          wrapper_class=structlog.stdlib.BoundLogger,
          cache_logger_on_first_use=True,
      )

  # In your main application file:
  # main.py
  from config import configure_logging
  
  configure_logging()
  import structlog
  logger = structlog.get_logger(__name__)
  logger.info("Application started")
  

### 1.2. Modular Logging

- **Best Practice:** Use a separate logger instance for each module or class. This provides clearer context for log entries and simplifies debugging.
- **Example:**

  python
  # my_module.py
  import structlog
  logger = structlog.get_logger(__name__)

  def my_function(arg):
      logger.debug("Function called", argument=arg)
      # ... function logic ...
  

### 1.3. Consistent Naming

- **Best Practice:** Adopt a consistent naming convention for loggers (e.g., using `__name__`).  This aids in identifying the origin of log messages.

## 2. Common Patterns and Anti-patterns

### 2.1. Canonical Log Lines

- **Best Practice:** Aim for a single, comprehensive log entry per significant event or request.  Use bound loggers to incrementally add context.
- **Example:**

  python
  logger = structlog.get_logger().bind(request_id=request.id, user_id=user.id)
  logger.info("Request processed", status_code=response.status_code, processing_time=1.23)
  

### 2.2. Contextual Logging

- **Best Practice:** Enrich log entries with relevant context data. This makes logs more informative and easier to analyze.
- **Example:**

  python
  logger.info("User logged in", username=user.name, ip_address=request.remote_addr)
  

### 2.3. Avoiding Prose in Logs

- **Anti-pattern:**  Avoid writing verbose, prose-like log messages.  Instead, focus on structured data.
- **Instead:** Use key-value pairs to represent log data.
- **Example (Bad):**

  python
  logger.info("The user " + user.name + " logged in successfully from IP " + request.remote_addr)
  

- **Example (Good):**

  python
  logger.info("User logged in", username=user.name, ip_address=request.remote_addr)
  

### 2.4. Using Bound Loggers

- **Best Practice:** Use bound loggers to add context that persists across multiple log entries within a specific scope (e.g., a request, a session).  This avoids redundant data in each log message.
- **Example:**

  python
  bound_logger = logger.bind(request_id=request.id)
  bound_logger.info("Processing request")
  # ... more code ...
  bound_logger.info("Request completed", status_code=200)
  

### 2.5. Dynamic Key Names

- **Anti-pattern:** Avoid generating keys dynamically, as this can make log analysis difficult. Strive for consistent, predefined key names.
- **Instead:** Plan your log structure and define keys in advance.

## 3. Performance Considerations

### 3.1. Processor Chains

- **Best Practice:** Optimize your processor chain to minimize overhead. Avoid unnecessary or computationally expensive processors.
- **Consider:** Reordering processors so that cheaper operations happen first, filtering out log entries early, and caching results where appropriate.

### 3.2. Asynchronous Logging

- **Best Practice:**  For high-throughput applications, use asynchronous logging to avoid blocking the main thread.  This can be achieved using libraries like `asyncio` and `threading`.
- **Example (using threading):**

  python
  import structlog
  import logging
  import threading
  import queue

  log_queue = queue.Queue(-1)  # Unlimited size

  def logging_thread(q):
      while True:
          record = q.get()
          if record is None:
              break
          logger = logging.getLogger(record.name)
          logger.handle(record)

  log_thread = threading.Thread(target=logging_thread, args=(log_queue,))
  log_thread.daemon = True
  log_thread.start()

  class QueueHandler(logging.Handler):
      def emit(self, record):
          log_queue.put(record)

  # Configure standard logging to use the QueueHandler
  root = logging.getLogger()
  root.setLevel(logging.DEBUG)
  queue_handler = QueueHandler()
  root.addHandler(queue_handler)

  def configure_structlog():
      structlog.configure(
          processors=[
              structlog.stdlib.add_log_level,
              structlog.stdlib.PositionalArgumentsFormatter(),
              structlog.processors.StackInfoRenderer(),
              structlog.processors.format_exc_info,
              structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
          ],
          context_class=dict,
          logger_factory=structlog.stdlib.LoggerFactory(),
          wrapper_class=structlog.stdlib.BoundLogger,
          cache_logger_on_first_use=True,
      )

  configure_structlog()

  # Example usage
  logger = structlog.get_logger()
  logger.info("Message from structlog")

  # Shutdown logging thread
  log_queue.put(None)
  log_thread.join()
  

### 3.3. Log Levels

- **Best Practice:** Use appropriate log levels to control the volume of log output. Avoid excessive logging at high verbosity levels in production.
- **Levels:** DEBUG, INFO, WARNING, ERROR, CRITICAL

## 4. Security Best Practices

### 4.1. Sensitive Data

- **Best Practice:** Never log sensitive data such as passwords, API keys, or personal information directly. Hash or redact sensitive data before logging.
- **Example (redaction):**

  python
  def redact_sensitive_data(logger, method_name, event_dict):
      if 'password' in event_dict:
          event_dict['password'] = '********'  # Redact password
      return event_dict

  structlog.configure(processors=[redact_sensitive_data, structlog.processors.JSONRenderer()])
  

### 4.2. Input Validation

- **Best Practice:** Validate and sanitize any user-provided input before including it in log messages to prevent log injection attacks.

### 4.3. Rate Limiting

- **Best Practice:** Implement rate limiting for logging to prevent denial-of-service attacks that flood the logs with malicious data.

## 5. Testing Approaches

### 5.1. Unit Testing

- **Best Practice:** Write unit tests to verify that log messages are generated correctly under various conditions.
- **Example (using `structlog.testing`):

  python
  import structlog
  from structlog.testing import LogCapture

  def test_logging_output():
      with LogCapture() as log_capture:
          logger = structlog.get_logger()
          logger.info("Test message", key1="value1")

      assert len(log_capture.entries) == 1
      assert log_capture.entries[0].event == "Test message"
      assert log_capture.entries[0].key1 == "value1"
  

### 5.2. Integration Testing

- **Best Practice:**  Include integration tests to ensure that logging is correctly integrated with other systems (e.g., log aggregators).

## 6. Common Pitfalls and Gotchas

### 6.1. Incorrect Configuration

- **Pitfall:**  Failing to configure `structlog` correctly can lead to unexpected log output or errors. Double-check your processor chain and settings.

### 6.2. Circular Dependencies

- **Pitfall:** Circular dependencies between modules can cause issues during `structlog` configuration. Ensure that your modules are properly organized to avoid these dependencies.

### 6.3. Exception Handling

- **Pitfall:** Not handling exceptions within processors can lead to log entries being dropped. Wrap processor logic in `try...except` blocks.

### 6.4. Mixing with Standard Logging

- **Pitfall:** Mixing `structlog` with the standard library `logging` module without proper integration can lead to inconsistent log formats. Use `structlog.stdlib` to bridge the gap.

## 7. Tooling and Environment

### 7.1. Log Aggregators

- **Best Practice:** Use log aggregators like Elasticsearch, Graylog, or Splunk to centralize and analyze logs.

### 7.2. Monitoring Tools

- **Best Practice:** Integrate logging with monitoring tools like Prometheus or Grafana to visualize log data and set up alerts.

### 7.3. Development Environment

- **Best Practice:** Use pretty console output for development and structured JSON output for production.

### 7.4. Environment Variables

- **Best Practice:** Use environment variables to configure logging levels and other settings. This allows you to easily change logging behavior without modifying code.

  python
  import os
  import structlog
  import logging

  log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
  numeric_level = getattr(logging, log_level, None)
  if not isinstance(numeric_level, int):
      raise ValueError('Invalid log level: {}'.format(log_level))

  structlog.configure(
      processors=[
          structlog.stdlib.add_log_level,
          structlog.stdlib.PositionalArgumentsFormatter(),
          structlog.processors.StackInfoRenderer(),
          structlog.processors.format_exc_info,
          structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
      ],
      context_class=dict,
      logger_factory=structlog.stdlib.LoggerFactory(),
      wrapper_class=structlog.stdlib.BoundLogger,
      cache_logger_on_first_use=True,
  )

  logger = structlog.get_logger()
  logger.info("Application started", log_level=log_level)
  

By adhering to these best practices, you can leverage the full power of `structlog` to create robust, informative, and manageable logging systems in your Python projects.