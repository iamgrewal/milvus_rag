---
description: Documentation for the tests directory containing unit tests for NLP and RAG system components.
globs: ['tests/*']
alwaysApply: false
---

# Tests Directory Documentation

## Overview
The `tests` directory contains unit tests for the components of the Graphrag project, specifically focusing on the Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG) systems. These tests are essential for ensuring the reliability and correctness of the codebase as it evolves.

## Key Files
- **test_nlp_processor.py**: This file contains unit tests for the NLP processor module. It verifies the functionality of various NLP-related features and ensures that the processor behaves as expected under different scenarios.
- **test_rag_system.py**: This file includes tests for the RAG system, which integrates retrieval and generation capabilities. The tests here check the correctness of the RAG system's implementation and its interaction with other components.

## Relation to Other Parts of the Codebase
The tests in this directory import functionality from the following external modules:
- `graphrag/nlp/processor.py`: This module contains the core logic for processing natural language, which is tested in `test_nlp_processor.py`.
- `graphrag/rag_system/main.py`: This module implements the RAG system's main functionality, which is validated through the tests in `test_rag_system.py`.

No other parts of the codebase import from this directory, emphasizing its role as a testing suite rather than a functional component.

## Common Patterns and Conventions
- Each test file typically corresponds to a specific module in the codebase, following a naming convention that includes the prefix `test_`.
- Tests are organized into functions that assert expected outcomes, making it easy to identify what functionality is being tested.

## Best Practices
- Ensure that tests are comprehensive and cover edge cases to maintain high code quality.
- Keep tests isolated; each test should not depend on the state of another test to avoid flaky tests.
- Regularly run the test suite to catch regressions early in the development process.
- Use descriptive names for test functions to clarify what aspect of the code is being tested.