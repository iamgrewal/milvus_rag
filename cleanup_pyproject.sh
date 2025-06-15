#!/bin/bash
# cleanup_pyproject.sh - Fix the broken pyproject.toml file

set -euo pipefail

echo "🧹 Cleaning up broken pyproject.toml file..."

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Backup the broken file
if [ -f "pyproject.toml" ]; then
    print_warning "Backing up broken pyproject.toml..."
    cp pyproject.toml pyproject.toml.broken.backup
    print_status "Backup saved as pyproject.toml.broken.backup"
fi

# Create the clean version
cat >pyproject.toml <<'EOF'
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "milvus_rag"
version = "0.1.1"
description = "Hybrid RAG system using Milvus and Neo4j for enterprise AI applications"
authors = [
    {name = "Jay Grewal", email = "jay@Rhobytenet.com"}
]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.9"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

# Core dependencies - Phase 1 optimized versions
dependencies = [
    # Core RAG framework
    "langchain>=0.1.0,<0.4.0",
    "langchain-core>=0.3.0,<0.4.0",
    
    # LLM integration
    "openai>=1.0.0,<2.0.0",
    
    # Vector database
    "pymilvus>=2.5.0,<2.6.0",
    
    # Graph database
    "neo4j>=5.15.0,<6.0.0",
    "neo4j-graphrag[openai]>=0.1.0",
    
    # NLP and embeddings
    "sentence-transformers>=2.2.0,<3.0.0",
    "spacy>=3.7.0,<4.0.0",
    "transformers>=4.30.0,<5.0.0",
    
    # Core utilities
    "python-dotenv>=1.0.0",
    "tqdm>=4.65.0",
    "requests>=2.32.0,<3.0.0",
    "numpy>=1.24.0,<2.0.0",
    "pydantic>=2.0.0,<3.0.0",
    "fastapi>=0.104.0,<1.0.0",
    
    # Critical version constraints for Phase 1 compatibility
    "protobuf>=4.25.0,<5.0.0",  # Avoid 6.x breaking changes
    "grpcio>=1.67.0,<1.74.0",   # Compatible with protobuf 4.x
    "grpcio-tools>=1.67.0,<1.74.0",
    
    # Package management
    "packaging>=23.2,<25.0",
    "setuptools>=61.0",
]

[project.optional-dependencies]
# Development dependencies
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "pytest-benchmark>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "isort>=5.12.0",
    "mypy>=1.5.0",
    "pre-commit>=3.0.0",
    "safety>=3.0.0",
    "pip-audit>=2.6.0",
]

# Performance testing
benchmark = [
    "pytest-benchmark>=4.0.0",
    "memory-profiler>=0.61.0",
    "psutil>=5.9.0",
]

# Production deployment
prod = [
    "uvicorn[standard]>=0.24.0",
    "gunicorn>=21.0.0",
    "prometheus-client>=0.19.0",
    "sentry-sdk>=1.38.0",
]

[project.urls]
homepage = "https://github.com/iamgrewal/milvus_rag"
repository = "https://github.com/iamgrewal/milvus_rag"
documentation = "https://github.com/iamgrewal/milvus_rag#readme"
issues = "https://github.com/iamgrewal/milvus_rag/issues"

[tool.setuptools]
package-dir = {"" = "src"}
packages = ["graphrag"]

[tool.setuptools.package-data]
"*" = ["*.yaml", "*.yml", "*.json", "*.md"]

# Black configuration
[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
  | milvus_docs
)/
'''

# Ruff configuration (replaces flake8, isort, etc.)
[tool.ruff]
target-version = "py39"
line-length = 88
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "N",   # pep8-naming
    "S",   # flake8-bandit (security)
    "T20", # flake8-print
    "PT",  # flake8-pytest-style
]
ignore = [
    "E501",  # line too long, handled by black
    "B008",  # do not perform function calls in argument defaults
    "C901",  # too complex
    "S101",  # assert used (fine in tests)
    "T201",  # print found (fine for CLI tools)
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]
"tests/**/*" = ["S101", "T201", "PT"]
"scripts/**/*" = ["T201"]

[tool.ruff.isort]
known-first-party = ["graphrag"]
known-third-party = [
    "pymilvus",
    "neo4j",
    "langchain",
    "openai",
    "sentence_transformers",
    "spacy",
    "transformers",
    "fastapi",
    "pydantic",
]

# MyPy configuration
[tool.mypy]
python_version = "3.9"
check_untyped_defs = true
disallow_any_generics = true
disallow_incomplete_defs = true
disallow_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_return_any = true
strict_equality = true
show_error_codes = true

[[tool.mypy.overrides]]
module = [
    "pymilvus.*",
    "neo4j.*",
    "spacy.*",
    "sentence_transformers.*",
    "transformers.*",
]
ignore_missing_imports = true

# Pytest configuration
[tool.pytest.ini_options]
minversion = "7.0"
addopts = [
    "-ra",
    "-q", 
    "--strict-markers",
    "--strict-config",
    "--cov=src/graphrag",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "benchmark: marks tests as performance benchmarks",
    "unit: marks tests as unit tests",
    "phase1: marks tests for Phase 1 development",
    "phase2: marks tests for Phase 2 development",
    "milvus: marks tests requiring Milvus",
    "neo4j: marks tests requiring Neo4j",
]
asyncio_mode = "auto"
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore::PendingDeprecationWarning",
]

# Coverage configuration
[tool.coverage.run]
source = ["src/graphrag"]
omit = [
    "*/tests/*",
    "*/__pycache__/*",
    "*/venv/*",
    "*/env/*",
    "*/build/*",
    "*/dist/*",
    "*/milvus_docs/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "def __str__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "@abstractmethod",
]
show_missing = true
skip_covered = true

[tool.coverage.html]
directory = "htmlcov"

[tool.coverage.xml]
output = "coverage.xml"
EOF

print_status "Created clean pyproject.toml"

# Validate the new file
echo ""
echo "🔍 Validating new pyproject.toml..."

# Check TOML syntax
python3 -c "
import tomllib
import sys
try:
    with open('pyproject.toml', 'rb') as f:
        config = tomllib.load(f)
    print('✅ TOML syntax is valid')
    print(f'📦 Project: {config[\"project\"][\"name\"]} v{config[\"project\"][\"version\"]}')
    print(f'👤 Author: {config[\"project\"][\"authors\"][0][\"name\"]}')
    print(f'📊 Dependencies: {len(config[\"project\"][\"dependencies\"])} core packages')
except Exception as e:
    print(f'❌ TOML validation failed: {e}')
    sys.exit(1)
" || print_error "Failed to validate TOML syntax"

# Install development tools
echo ""
echo "🔧 Installing development tools..."
./run_in_env.sh pip install safety pip-audit

echo ""
print_status "✨ pyproject.toml cleanup complete!"

echo ""
echo "📋 What was fixed:"
echo "   🔧 Removed duplicate [project] sections"
echo "   🔧 Fixed broken TOML syntax and missing quotes"
echo "   🔧 Resolved conflicting project names and author info"
echo "   🔧 Added Phase 1 optimized dependency constraints"
echo "   🔧 Fixed protobuf version constraint (4.x instead of 6.x)"
echo "   🔧 Added comprehensive development tools configuration"
echo "   🔧 Added security and benchmarking dependencies"

echo ""
echo "🎯 Next steps:"
echo "   1. ./run_in_env.sh pip install -e .[dev] # Install with dev dependencies"
echo "   2. ./run_in_env.sh safety check          # Run security scan"
echo "   3. ./run_in_env.sh pytest               # Run tests"
echo "   4. git add pyproject.toml && git commit -m '🔧 Fix broken pyproject.toml configuration'"

echo ""
print_status "Ready for Phase 1 development!"
EOF

#chmod +x cleanup_pyproject.sh
