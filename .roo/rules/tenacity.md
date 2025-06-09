---
description: 
globs: 
alwaysApply: false
---
---
description: This rule outlines best practices for using the Tenacity library in Python to implement robust retry mechanisms. It covers configuration, error handling, performance, and testing to ensure reliable application behavior.
globs: **/*.py
---
# Tenacity: Best Practices for Reliable Retry Logic in Python

This document provides a comprehensive guide to using the Tenacity library effectively in Python. It covers various aspects of retry implementation, including configuration, error handling, performance optimization, and testing strategies.

## 1. Code Organization and Structure

### 1.1. Decoupling Retry Logic

*   **Best Practice:** Keep retry logic separate from the core business logic. Use decorators or context managers provided by Tenacity to wrap functions that require retry behavior.
*   **Rationale:** This improves code readability, maintainability, and testability. It also allows you to easily modify retry policies without altering the underlying business logic.

    python
    from tenacity import retry, stop_after_attempt

    @retry(stop=stop_after_attempt(3))
    def unreliable_function():
        # Business logic that might fail
        pass
    

### 1.2. Configuration Management

*   **Best Practice:** Centralize retry configuration in a dedicated module or configuration file.
*   **Rationale:** This makes it easier to manage and update retry policies across the application. Use environment variables or configuration files to define retry parameters.

    python
    # config.py
    RETRY_ATTEMPTS = 3
    RETRY_WAIT_SECONDS = 1

    from tenacity import retry, stop_after_attempt, wait_fixed
    import config

    @retry(stop=stop_after_attempt(config.RETRY_ATTEMPTS), wait=wait_fixed(config.RETRY_WAIT_SECONDS))
    def unreliable_function():
        # Business logic
        pass
    

### 1.3. Modular Retry Policies

*   **Best Practice:** Define reusable retry policies as functions or classes.
*   **Rationale:** This promotes code reuse and reduces duplication. Create specific policies for different types of operations or services.

    python
    from tenacity import retry, stop_after_attempt, wait_exponential

    def create_exponential_retry_policy(attempts=3, multiplier=1, min_delay=1, max_delay=10):
        return retry(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=multiplier, min=min_delay, max=max_delay)
        )

    @create_exponential_retry_policy()
    def unreliable_function():
        # Business logic
        pass
    

## 2. Common Patterns and Anti-patterns

### 2.1. Exponential Backoff

*   **Pattern:** Use exponential backoff with jitter for most network-related operations.
*   **Rationale:** Exponential backoff gradually increases the delay between retries, preventing overwhelming the server. Jitter adds randomness to the delay, avoiding thundering herd problems.

    python
    from tenacity import retry, wait_exponential, wait_random

    @retry(wait=wait_exponential(multiplier=1, min=1, max=60) + wait_random(0, 2))
    def unreliable_network_call():
        # Network call
        pass
    

### 2.2. Retry on Specific Exceptions

*   **Pattern:** Retry only on specific exceptions that are likely to be transient.
*   **Rationale:** Retrying on all exceptions can mask underlying issues or lead to infinite loops.  Focus on exceptions like `ConnectionError`, `TimeoutError`, `ServiceUnavailable`, and `RequestException` (from `requests` library).

    python
    from tenacity import retry, retry_if_exception_type
    import requests

    @retry(retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout)))
    def unreliable_api_call():
        # API Call
        pass
    

### 2.3. Circuit Breaker Integration (Anti-pattern mitigation)

*   **Pattern:** Integrate Tenacity with a circuit breaker pattern to prevent cascading failures.
*   **Rationale:** A circuit breaker stops retries after a certain number of failures, giving the downstream service time to recover.  Libraries like `pybreaker` can be used in conjunction with Tenacity.

    python
    from tenacity import retry, stop_after_attempt
    from pybreaker import CircuitBreaker

    breaker = CircuitBreaker(fail_max=3, reset_timeout=10)

    @retry(stop=stop_after_attempt(3))
    def unreliable_function():
        with breaker:
            # Business logic
            pass
    

### 2.4. Avoid Infinite Retries (Anti-pattern)

*   **Anti-pattern:** Retrying indefinitely without a stop condition.
*   **Rationale:** This can lead to resource exhaustion and cascading failures. Always set a stop condition, such as `stop_after_attempt` or `stop_after_delay`.

    python
    from tenacity import retry, stop_after_attempt

    @retry(stop=stop_after_attempt(5))  # Correct: Stop after 5 attempts
    def unreliable_function():
        # Business logic
        pass

    # Incorrect:  @retry  # No stop condition - Avoid!
    # def unreliable_function():
    #     pass
    

### 2.5. Ignoring Non-Retryable Errors (Anti-pattern)

*   **Anti-pattern:** Retrying on errors that indicate a permanent failure.
*   **Rationale:** This wastes resources and delays error detection. Avoid retrying on errors like `ValueError`, `TypeError`, or HTTP 400 errors (Bad Request).

### 2.6. Overly Aggressive Retries (Anti-pattern)

*   **Anti-pattern:** Retrying too frequently or with short delays.
*   **Rationale:** This can overwhelm the downstream service and exacerbate the problem. Use exponential backoff and appropriate delays.

## 3. Performance Considerations

### 3.1. Asynchronous Retries

*   **Best Practice:** Use asynchronous retries for I/O-bound operations to avoid blocking the event loop.
*   **Rationale:** Asynchronous retries allow other tasks to run while waiting for the retry delay.

    python
    import asyncio
    from tenacity import retry, stop_after_attempt

    @retry(stop=stop_after_attempt(3))
    async def unreliable_async_function():
        # Asynchronous business logic
        await asyncio.sleep(1) # Simulate async operation
        pass
    

### 3.2. Optimize Retry Delays

*   **Best Practice:** Tune retry delays based on the characteristics of the downstream service.
*   **Rationale:**  Too short delays can overwhelm the service, while too long delays can degrade performance. Consider service-level agreements (SLAs) and historical performance data.

### 3.3. Threading Considerations

*   **Best Practice:** When using threads, ensure that the retry logic is thread-safe.
*   **Rationale:**  Tenacity itself is generally thread-safe, but the code within the retried function might not be.  Use appropriate locking mechanisms if necessary.

## 4. Security Best Practices

### 4.1. Sensitive Data Handling

*   **Best Practice:** Avoid logging sensitive data during retry attempts.
*   **Rationale:** Sensitive data in logs can expose security vulnerabilities.  Sanitize or mask sensitive information before logging.

### 4.2. Rate Limiting Awareness

*   **Best Practice:** Be aware of rate limits imposed by downstream services and adjust retry policies accordingly.
*   **Rationale:**  Exceeding rate limits can lead to account suspension or denial of service. Implement strategies like exponential backoff with jitter to avoid overwhelming the service.

### 4.3. Authentication and Authorization

*   **Best Practice:** Ensure that authentication and authorization tokens are valid during retry attempts.
*   **Rationale:**  Tokens can expire or be revoked, leading to failed retries.  Implement token refresh mechanisms if necessary.

## 5. Testing Approaches

### 5.1. Unit Testing

*   **Best Practice:** Unit test retry logic in isolation.
*   **Rationale:**  Verify that the retry policy is configured correctly and that the function is retried the expected number of times.

    python
    import unittest
    from unittest.mock import patch, Mock
    from tenacity import RetryError

    class TestRetry(unittest.TestCase):
        @patch('your_module.unreliable_function')
        def test_unreliable_function_retries(self, mock_function):
            mock_function.side_effect = [Exception('Failed'), Exception('Failed'), 'Success']
            from your_module import unreliable_function # Import inside the test to apply the patch
            result = unreliable_function()
            self.assertEqual(mock_function.call_count, 3)
            self.assertEqual(result, 'Success')

        @patch('your_module.unreliable_function')
        def test_unreliable_function_fails_after_max_retries(self, mock_function):
            mock_function.side_effect = Exception('Failed')
            from your_module import unreliable_function # Import inside the test to apply the patch
            with self.assertRaises(RetryError):
                unreliable_function()
            self.assertEqual(mock_function.call_count, 3) # Assuming max attempts is 3
    

### 5.2. Integration Testing

*   **Best Practice:** Integration test retry logic with downstream services.
*   **Rationale:**  Simulate transient failures in the downstream service to verify that retries are handled correctly.

### 5.3. Mocking and Patching

*   **Best Practice:** Use mocking and patching to simulate different error scenarios during testing.
*   **Rationale:**  This allows you to test retry logic without relying on actual service failures.

## 6. Common Pitfalls and Gotchas

### 6.1. State Management

*   **Pitfall:** Incorrectly managing state within the retried function.
*   **Solution:** Ensure that the function is idempotent or that state is properly reset between retries.

### 6.2. Exception Handling

*   **Pitfall:** Catching and ignoring exceptions within the retried function.
*   **Solution:** Re-raise exceptions to allow Tenacity to handle them.

### 6.3. Context Manager Usage

*   **Pitfall:** Improper use of Tenacity's context manager.
*   **Solution:** Ensure that the context manager is used correctly to wrap the code block that needs to be retried.

### 6.4. Reraise Configuration
*   **Pitfall:** Not understanding the impact of `reraise=True`.
*   **Solution:** Understand that `reraise=True` raises the original exception at the end of all the retries instead of `RetryError`.

## 7. Tooling and Environment

### 7.1. Logging

*   **Best Practice:** Integrate Tenacity with a logging framework to track retry attempts and errors.
*   **Rationale:** Logging provides valuable insights into retry behavior and helps identify potential issues.

    python
    import logging
    from tenacity import retry, stop_after_attempt, before_log

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    @retry(stop=stop_after_attempt(3), before=before_log(logger, logging.INFO))
    def unreliable_function():
        # Business logic
        pass
    

### 7.2. Monitoring

*   **Best Practice:** Monitor retry metrics to detect and address recurring failures.
*   **Rationale:** Monitoring helps identify patterns and trends that might indicate underlying problems.

### 7.3. Environment Variables

*   **Best Practice:** Use environment variables to configure retry parameters.
*   **Rationale:** This allows you to easily adjust retry policies without modifying the code.

### 7.4. Stamina library
*   **Best Practice:** Consider using Stamina, an opinionated wrapper around Tenacity, for production-grade retries with sensible defaults and out-of-the-box instrumentation.
*   **Rationale:** Stamina simplifies configuration and promotes best practices, reducing the potential for misuse.

By following these best practices, you can effectively use the Tenacity library to build robust and reliable applications that gracefully handle transient failures.