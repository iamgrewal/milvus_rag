---
description: This rule provides comprehensive best practices for developing with the Milvus vector database, including code structure, performance optimization, security, and testing. It covers common patterns, anti-patterns, and tooling to ensure high-quality, maintainable Milvus-based applications.
globs: **/*.{go,cpp,h,hpp,py}
---
# Milvus Library Best Practices and Coding Standards

This document outlines the recommended best practices for developing applications and contributing to the Milvus vector database. Following these guidelines will ensure code quality, maintainability, performance, and security.

## Library Information:
- Name: Milvus
- Tags: vector-database, ai, similarity-search_files, data-management, unstructured-data

## 1. Code Organization and Structure

### 1.1. Project Structure:

-   **Go Components:** Follow the standard Go project layout.
    -   `cmd/`: Main applications for Milvus services (e.g., `milvus`, `milvus-standalone`).
    -   `internal/`: Private application and library code. This directory should not be imported by other projects.
    -   `pkg/`: Code that *can* be used by external applications/libraries. Minimize the use of this.
    -   `api/`: API definitions (protobuf, gRPC).
    -   `configs/`: Configuration files and schemas.
    -   `docs/`: Documentation.
    -   `tests/`: Unit and integration tests.
-   **C++ Components:**
    -   `include/`: Header files defining interfaces and data structures.
    -   `src/`: Implementation files.
    -   `test/`: Unit tests.
    -   `CMakeLists.txt`: Build configuration.
-   **Python SDK:**
    -   `pymilvus/`: Python package source code.
    -   `tests/`: Unit and integration tests.
    -   `setup.py`: Package installation.

### 1.2. Module Design:

-   **Loose Coupling:** Design modules with minimal dependencies on each other. Use interfaces to abstract dependencies and promote modularity.
-   **High Cohesion:** Modules should have a single, well-defined purpose.  All elements within a module should be related to that purpose.
-   **Clear Abstractions:** Define clear abstractions for different components of the system. This makes it easier to understand and maintain the code.
-   **Separation of Concerns:**  Separate concerns into distinct modules or layers.  For example, separate data access logic from business logic.

### 1.3. Naming Conventions:

-   **Go:** Follow Go's naming conventions (e.g., `camelCase` for variables, `PascalCase` for types, `snake_case` for protobuf fields where relevant).
-   **C++:** Follow Google C++ Style Guide. Camel case for file names, 4 space indentation, adopt `.cpp` file extension, 120-character line length.
-   **Python:** Follow PEP 8 guidelines (e.g., `snake_case` for variables and functions, `CamelCase` for classes).
-   **Consistent Naming:** Use consistent naming conventions throughout the codebase.

### 1.4. Code Comments and Documentation:

-   **Comprehensive Comments:** Write clear, concise comments to explain complex logic, non-obvious code, and design decisions.
-   **API Documentation:** Document all public APIs using appropriate documentation tools (e.g., Go's `godoc`, Doxygen for C++, Sphinx for Python).
-   **Example Usage:** Provide example code snippets to illustrate how to use the library.
-   **Keep Documentation Up-to-Date:** Ensure that the documentation is always consistent with the code.

## 2. Common Patterns and Anti-patterns

### 2.1. Common Patterns:

-   **Factory Pattern:** Use factory patterns to create instances of objects, especially when the object creation logic is complex.
-   **Strategy Pattern:** Use strategy patterns to implement different algorithms or behaviors that can be easily switched at runtime.
-   **Observer Pattern:** Use observer patterns to notify interested parties when a state change occurs.
-   **Dependency Injection:** Use dependency injection to provide dependencies to components, promoting testability and modularity.
-   **Resource Pooling:** Implement resource pooling for frequently used resources (e.g., database connections, memory buffers) to improve performance.
-   **Command Pattern:** Encapsulate a request as an object, thereby letting you parameterize clients with different requests, queue or log requests, and support undoable operations.

### 2.2. Anti-patterns:

-   **God Class:** Avoid creating classes that are too large and have too many responsibilities. Break down large classes into smaller, more manageable ones.
-   **Code Duplication:** Avoid duplicating code. Extract common logic into reusable functions or classes.
-   **Magic Numbers:** Avoid using magic numbers in the code. Define constants with meaningful names instead.
-   **Tight Coupling:** Avoid tight coupling between modules. Use interfaces and dependency injection to reduce coupling.
-   **Ignoring Errors:** Never ignore errors. Always handle errors gracefully and log them appropriately.
-   **Premature Optimization:** Don't optimize code prematurely. Focus on writing clear, correct code first, and then optimize only if necessary.

## 3. Performance Considerations

### 3.1. Indexing Strategies:

-   **Choose the Right Index:** Select the appropriate index type based on the dataset characteristics, query patterns, and performance requirements. Consider factors such as data size, dimensionality, query latency, and recall.
    -   `IVF_FLAT`: Good for general purpose, balanced performance.
    -   `IVF_SQ8`:  Quantization-based index, smaller size, slightly lower accuracy.
    -   `HNSW`: Graph-based index, high performance for high-dimensional data.
    -   `SCANN`: Optimized for large-scale datasets.
    -   `DiskANN`: Suitable for large datasets that don't fit in memory.
-   **Index Parameters:**  Tune index parameters (e.g., `nlist`, `nprobe` for IVF indexes) to optimize search_files performance and precision.  Experiment with different values to find the best trade-off for your specific dataset and workload. See Milvus documentation for recommended ranges.
    -   `nlist`: Number of buckets during clustering (IVF indexes).  `nlist = 4 * sqrt(n)` is a good starting point, where `n` is the total number of vectors.
    -   `nprobe`: Number of buckets to search_files during query (IVF indexes). Trade-off between precision and efficiency. Trial and error is the best approach.
-   **`index_file_size`:** When creating a table, the `index_file_size` parameter is used to specify the size, in MB, of a single file for data storage. The default is 1024. For the IVFLAT index type, the index file size approximately equals to the size of the corresponding raw data file. For the SQ8 index, the size of an index file is approximately 30 percent of the corresponding raw data file. During a similarity search_files, the Milvus vector database searches each index file one by one.  Consider increasing `index_file_size` to 2048 MB for potentially improved search_files performance (30-50% improvement in some cases), but be mindful of GPU/CPU memory limitations.

### 3.2. Data Ingestion:

-   **Batch Inserts:** Use batch inserts to improve data ingestion performance. Inserting data in batches reduces the overhead of individual insert operations.
-   **Parallel Ingestion:**  Ingest data in parallel using multiple threads or processes to maximize throughput.
-   **Optimize Vector Embeddings:** Generate appropriate vector embeddings for your data.  Using an image model to vectorize text (or vice versa) will yield poor results. Choose the embedding model that is trained on the same type of data as your input data.

### 3.3. Query Optimization:

-   **Limit Results:** Use the `limit` parameter to restrict the number of results returned by a query. This can significantly improve query latency.
-   **Metadata Filtering:**  Use metadata filtering to narrow down the search_files space and improve query performance.  Filter by scalar fields to reduce the number of vectors that need to be compared.
-   **Range Search:** If applicable, use range search_files to find vectors within a specific distance range.
-   **Hybrid Search:** Combine semantic search_files with full-text search_files or other search_files techniques to improve search_files relevance.

### 3.4. Hardware Acceleration:

-   **GPU Acceleration:** Leverage GPU acceleration (e.g., NVIDIA's CAGRA) to enhance vector search_files performance. Milvus supports GPU indexing.
-   **CPU Optimization:**  Optimize code for CPU performance by using appropriate data structures and algorithms.

### 3.5. Resource Management:

-   **Memory Management:**  Monitor memory usage and avoid memory leaks. Use appropriate memory management techniques to prevent excessive memory consumption.
-   **Connection Pooling:** Use connection pooling to reuse database connections and reduce connection overhead.
-   **Caching:** Implement caching mechanisms to store frequently accessed data in memory.

## 4. Security Best Practices

### 4.1. Authentication and Authorization:

-   **Mandatory Authentication:** Enable mandatory user authentication to ensure that only authorized users can access the database.
-   **Role-Based Access Control (RBAC):** Implement RBAC to control access to data and resources based on user roles. Assign specific permissions to users based on their roles.
-   **Secure Credentials:** Store credentials securely and avoid hardcoding them in the code.  Use environment variables or configuration files to manage credentials.

### 4.2. Data Encryption:

-   **TLS Encryption:** Use TLS encryption to secure all communications within the network. This protects sensitive data from unauthorized access.
-   **Data-at-Rest Encryption:** Consider encrypting data at rest to protect against data breaches.

### 4.3. Input Validation:

-   **Validate Inputs:** Validate all inputs to prevent injection attacks and other security vulnerabilities.
-   **Sanitize Inputs:** Sanitize inputs to remove potentially harmful characters or code.

### 4.4. Security Auditing:

-   **Audit Logs:** Enable audit logging to track user activity and identify potential security threats. Regularly review audit logs to detect suspicious behavior.
-   **Security Scans:** Perform regular security scans to identify vulnerabilities in the code and infrastructure.

### 4.5. Dependency Management:

-   **Dependency Scanning:** Scan dependencies for known vulnerabilities and update them regularly.
-   **Use Trusted Sources:** Obtain dependencies from trusted sources to avoid introducing malicious code into the project.

## 5. Testing Approaches

### 5.1. Unit Testing:

-   **Comprehensive Unit Tests:** Write comprehensive unit tests to verify the correctness of individual components.
-   **Mock Dependencies:** Use mocks to isolate components during unit testing.
-   **Code Coverage:** Aim for high code coverage to ensure that all parts of the code are tested.

### 5.2. Integration Testing:

-   **Integration Tests:** Write integration tests to verify the interaction between different components.
-   **Test Data:** Use realistic test data to simulate real-world scenarios.

### 5.3. Performance Testing:

-   **Load Testing:** Perform load testing to measure the performance of the system under high load.
-   **Stress Testing:** Perform stress testing to identify the breaking points of the system.
-   **Benchmark Tests:** Use benchmark tests to compare the performance of different configurations or implementations.

### 5.4. End-to-End Testing:

-   **End-to-End Tests:** Write end-to-end tests to verify the entire system from the user's perspective.

### 5.5. Test-Driven Development (TDD):

-   **Write Tests First:** Write tests before writing the actual code. This helps to ensure that the code meets the requirements.

## 6. Common Pitfalls and Gotchas

### 6.1. Data Type Mismatches:

-   **Ensure Data Type Consistency:** Ensure that the data types of the vectors and metadata fields are consistent. Inconsistent data types can lead to errors and unexpected behavior.

### 6.2. Indexing Issues:

-   **Incorrect Index Parameters:** Using incorrect index parameters can significantly degrade performance. Carefully tune index parameters based on the dataset characteristics and query patterns.
-   **Missing Indexes:**  Forgetting to create indexes on frequently queried fields can result in slow queries.

### 6.3. Concurrency Issues:

-   **Race Conditions:** Be aware of race conditions when multiple threads or processes are accessing shared resources. Use appropriate synchronization mechanisms to prevent race conditions.
-   **Deadlocks:** Avoid deadlocks by carefully designing the locking strategy.

### 6.4. Error Handling:

-   **Ignoring Errors:** Ignoring errors can lead to unexpected behavior and data corruption. Always handle errors gracefully and log them appropriately.
-   **Insufficient Logging:** Insufficient logging can make it difficult to diagnose problems. Log important events and errors to help with debugging.

### 6.5. Version Compatibility:

-   **Dependency Conflicts:** Dependency conflicts can lead to unexpected behavior and runtime errors. Manage dependencies carefully and use a dependency management tool to resolve conflicts.
-   **API Changes:** Be aware of API changes when upgrading Milvus or its dependencies. Ensure that the code is compatible with the new API.

## 7. Tooling and Environment

### 7.1. Development Environment:

-   **IDE:** Use a suitable Integrated Development Environment (IDE) for code development (e.g., VS Code, GoLand, CLion, PyCharm).
-   **Debugging Tools:**  Use debugging tools to help identify and fix bugs in the code (e.g., gdb, delve, pdb).

### 7.2. Build Tools:

-   **Go:** Use `go build` and `go test` for building and testing Go code.  Use `go mod` for dependency management.
-   **C++:** Use CMake for building C++ code.  Use a suitable build system (e.g., Make, Ninja).
-   **Python:** Use `pip` and `setuptools` for managing Python packages.  Use `venv` or `conda` for creating virtual environments.

### 7.3. Testing Tools:

-   **Go:** Use `go test` for running unit tests. Use `mockery` for generating mock implementations.
-   **C++:** Use a unit testing framework such as Google Test (gtest).
-   **Python:** Use `unittest` or `pytest` for running unit tests. Use `mock` for mocking dependencies.

### 7.4. Profiling Tools:

-   **Go:** Use `go tool pprof` for profiling Go code.
-   **C++:** Use profiling tools such as perf or gprof.
-   **Python:** Use `cProfile` or `line_profiler` for profiling Python code.

### 7.5. Monitoring Tools:

-   **Prometheus/Grafana:** Use Prometheus and Grafana for monitoring Milvus performance.
-   **Logging:** Use a logging framework such as `logrus` (Go), `spdlog` (C++), or `logging` (Python) for logging events and errors.

### 7.6. Deployment Tools:

-   **Docker:** Use Docker for containerizing Milvus applications.
-   **Kubernetes:** Use Kubernetes for deploying and managing Milvus in a cluster environment.
-   **Helm:** Use Helm for managing Kubernetes deployments.

## References

-   [Milvus Documentation](https://milvus.io/docs/)
-   [Milvus GitHub Repository](https://github.com/milvus-io/milvus)
-   [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
-   [Effective Go](https://go.dev/doc/effective_go)
-   [PEP 8 -- Style Guide for Python Code](https://peps.python.org/pep-0008/)
-   [How to Select Index Parameters for IVF Index - Milvus](https://medium.com/vector-database/best-practices-for-setting-parameters-in-milvus-clients-9b8a8984d3dd)
-   [How to Get the Right Vector Embeddings](https://medium.com/vector-database/how-to-get-the-right-vector-embeddings-83295ced7f35)
-   [CONTRIBUTING.md](https://github.com/milvus-io/milvus/blob/master/CONTRIBUTING.md)