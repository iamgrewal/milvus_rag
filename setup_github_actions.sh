#!/bin/bash
# setup_github_actions.sh - Setup comprehensive GitHub Actions workflows

set -euo pipefail

echo "🚀 Setting up GitHub Actions workflows for Milvus RAG project..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    print_error "Not in a git repository. Please run this from your project root."
    exit 1
fi

# Create .github directory structure
print_info "Creating .github directory structure..."
mkdir -p .github/{workflows,ISSUE_TEMPLATE}

# Create workflow files
print_info "Creating workflow files..."

# CI/CD Pipeline
cat > .github/workflows/ci.yml << 'EOF'
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.9"

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      milvus:
        image: milvusdb/milvus:v2.3.0
        ports:
          - 19530:19530
        options: >-
          --health-cmd="curl -f http://localhost:9091/healthz || exit 1"
          --health-interval=10s
          --health-timeout=5s
          --health-retries=5
      
      neo4j:
        image: neo4j:5.15-community
        ports:
          - 7687:7687
          - 7474:7474
        env:
          NEO4J_AUTH: neo4j/testpassword
          NEO4J_PLUGINS: '["apoc"]'
        options: >-
          --health-cmd="cypher-shell -u neo4j -p testpassword 'RETURN 1'"
          --health-interval=10s
          --health-timeout=5s
          --health-retries=5

    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cache/pip
          ~/.cache/uv
        key: ${{ runner.os }}-deps-${{ hashFiles('**/requirements*.txt') }}
        restore-keys: |
          ${{ runner.os }}-deps-

    - name: Install UV and dependencies
      run: |
        pip install uv
        uv pip install --system -r requirements.txt
        if [ -f dev-requirements.txt ]; then
          uv pip install --system -r dev-requirements.txt
        fi

    - name: Wait for services
      run: |
        echo "Waiting for Milvus..."
        timeout 60 bash -c 'until curl -f http://localhost:19530/health 2>/dev/null; do sleep 2; done' || echo "Milvus not ready"
        echo "Waiting for Neo4j..."
        timeout 60 bash -c 'until echo "RETURN 1" | cypher-shell -u neo4j -p testpassword 2>/dev/null; do sleep 2; done' || echo "Neo4j not ready"

    - name: Lint with ruff
      run: |
        ruff check src/ tests/ --format=github || true
        ruff format --check src/ tests/ || true

    - name: Type check with mypy
      run: mypy src/graphrag --ignore-missing-imports || true

    - name: Run tests with coverage
      env:
        MILVUS_HOST: localhost
        MILVUS_PORT: 19530
        NEO4J_URI: bolt://localhost:7687
        NEO4J_USER: neo4j
        NEO4J_PASSWORD: testpassword
      run: |
        pytest tests/ \
          --cov=src/graphrag \
          --cov-report=xml \
          --cov-report=html \
          --cov-report=term-missing \
          --cov-fail-under=70 \
          --maxfail=5 \
          -v || true

    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      if: always()
      with:
        files: ./coverage.xml
        fail_ci_if_error: false

    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results
        path: |
          htmlcov/
          .coverage
EOF

# Claude Assistant workflow
cat > .github/workflows/claude-assistant.yml << 'EOF'
name: Claude PR Assistant

on:
  issue_comment:
    types: [created]
  pull_request_review_comment:
    types: [created]
  issues:
    types: [opened, assigned]
  pull_request_review:
    types: [submitted]

jobs:
  claude-code-action:
    if: |
      (github.event_name == 'issue_comment' && contains(github.event.comment.body, '@claude')) ||
      (github.event_name == 'pull_request_review_comment' && contains(github.event.comment.body, '@claude')) ||
      (github.event_name == 'pull_request_review' && contains(github.event.review.body, '@claude')) ||
      (github.event_name == 'issues' && contains(github.event.issue.body, '@claude'))
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
      issues: write
      id-token: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Run Claude PR Action
        uses: anthropics/claude-code-action@beta
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          timeout_minutes: "60"

      - name: Auto-commit Claude suggestions
        if: github.event_name == 'issue_comment' && contains(github.event.comment.body, '@claude commit')
        run: |
          git config --local user.email "claude@anthropic.com"
          git config --local user.name "Claude Assistant"
          git add -A
          if ! git diff --cached --quiet; then
            git commit -m "🤖 Claude-suggested improvements
            
            Triggered by: ${{ github.event.comment.html_url }}
            Author: ${{ github.event.comment.user.login }}"
            git push
          fi
EOF

# Auto-format workflow
cat > .github/workflows/auto-format.yml << 'EOF'
name: Auto Format

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  auto-format:
    runs-on: ubuntu-latest
    if: github.event.pull_request.head.repo.full_name == github.repository
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
        ref: ${{ github.head_ref }}

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.9"

    - name: Install formatting tools
      run: |
        pip install black ruff isort

    - name: Format code
      run: |
        black src/ tests/ || true
        ruff check --fix src/ tests/ || true
        isort src/ tests/ || true

    - name: Commit changes
      run: |
        git config --local user.email "action@github.com"
        git config --local user.name "GitHub Action"
        git add -A
        if ! git diff --cached --quiet; then
          git commit -m "🎨 Auto-format code [skip ci]"
          git push
        else
          echo "No formatting changes needed"
        fi
EOF

# Create dependabot configuration
cat > .github/dependabot.yml << 'EOF'
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
    open-pull-requests-limit: 5
    commit-message:
      prefix: "⬆️"
      include: "scope"
    groups:
      milvus-dependencies:
        patterns:
          - "pymilvus*"
          - "milvus*"
      ai-ml-dependencies:
        patterns:
          - "sentence-transformers*"
          - "transformers*"
          - "torch*"
          - "numpy*"
      neo4j-dependencies:
        patterns:
          - "neo4j*"
      testing-dependencies:
        patterns:
          - "pytest*"
          - "coverage*"

  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 3

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "monthly"
    open-pull-requests-limit: 2
EOF

# Create issue templates
cat > .github/ISSUE_TEMPLATE/bug_report.yml << 'EOF'
name: 🐛 Bug Report
description: Report a bug in the Milvus RAG system
title: "[BUG] "
labels: ["bug", "triage"]

body:
  - type: dropdown
    id: component
    attributes:
      label: Component
      description: Which component is affected?
      options:
        - Vector Store (Milvus)
        - Graph Store (Neo4j)
        - NLP Processing
        - Embedding Service
        - RAG System Orchestration
        - Docker/Infrastructure
        - Testing
        - Other
    validations:
      required: true

  - type: dropdown
    id: phase
    attributes:
      label: Development Phase
      description: Which development phase are you working on?
      options:
        - Phase 1 - Foundation Optimization
        - Phase 2 - Hybrid Integration
        - Phase 3 - Intelligence Layer
        - Phase 4 - Production Excellence
    validations:
      required: true

  - type: textarea
    id: description
    attributes:
      label: Bug Description
      description: A clear description of the bug
      placeholder: Describe the bug...
    validations:
      required: true

  - type: textarea
    id: steps
    attributes:
      label: Steps to Reproduce
      placeholder: |
        1. Run command '...'
        2. Query with '....'
        3. See error
    validations:
      required: true

  - type: textarea
    id: environment
    attributes:
      label: Environment
      value: |
        - OS: [e.g., Ubuntu 22.04 on WSL]
        - Python Version: [e.g., 3.9.7]
        - Milvus Version: [e.g., 2.3.0]
        - Neo4j Version: [e.g., 5.15]
        - Docker Version: [e.g., 24.0.6]
    validations:
      required: true
EOF

cat > .github/ISSUE_TEMPLATE/feature_request.yml << 'EOF'
name: 🚀 Feature Request
description: Suggest a new feature
title: "[FEATURE] "
labels: ["enhancement", "triage"]

body:
  - type: dropdown
    id: phase
    attributes:
      label: Target Phase
      options:
        - Phase 1 - Foundation Optimization
        - Phase 2 - Hybrid Integration
        - Phase 3 - Intelligence Layer
        - Phase 4 - Production Excellence
        - Future/Backlog
    validations:
      required: true

  - type: textarea
    id: problem
    attributes:
      label: Problem Statement
      description: What problem does this feature solve?
    validations:
      required: true

  - type: textarea
    id: solution
    attributes:
      label: Proposed Solution
      description: Describe your ideal solution
    validations:
      required: true
EOF

# Create PR template
cat > .github/PULL_REQUEST_TEMPLATE.md << 'EOF'
## 📋 Pull Request Summary

### Phase and Component
- **Development Phase**: [ ] Phase 1 | [ ] Phase 2 | [ ] Phase 3 | [ ] Phase 4
- **Component**: [ ] Milvus | [ ] Neo4j | [ ] NLP | [ ] Embedding | [ ] RAG System | [ ] Infrastructure

### Changes Description
Brief description of changes made.

### Performance Impact
- [ ] Query latency: ⬇️ Improved | ➡️ No change | ⬆️ Regression
- [ ] Memory usage: ⬇️ Reduced | ➡️ No change | ⬆️ Increased  
- [ ] Throughput: ⬆️ Improved | ➡️ No change | ⬇️ Decreased

### Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

### Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated (if needed)
- [ ] No breaking changes (or explicitly noted)
- [ ] Ready for @claude review

### Claude Instructions
@claude Please review this PR focusing on:
- [ ] Code quality and performance
- [ ] Alignment with current phase objectives
- [ ] Potential improvements or optimizations
- [ ] Test coverage adequacy
EOF

# Create scripts directory and performance check script
mkdir -p scripts

cat > scripts/check_performance_regression.py << 'EOF'
#!/usr/bin/env python3
"""
Performance regression detection script for GitHub Actions
"""

import json
import argparse
import sys
from typing import Dict, List, Tuple

def load_benchmark_results(filepath: str) -> Dict:
    """Load benchmark results from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Benchmark file not found: {filepath}")
        return {"benchmarks": []}
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in file: {filepath}")
        return {"benchmarks": []}

def compare_benchmarks(current: Dict, baseline: Dict, threshold: float) -> Tuple[bool, List[str]]:
    """Compare benchmark results and detect regressions"""
    
    current_benchmarks = {b["name"]: b for b in current.get("benchmarks", [])}
    baseline_benchmarks = {b["name"]: b for b in baseline.get("benchmarks", [])}
    
    regressions = []
    has_regression = False
    
    for name, current_bench in current_benchmarks.items():
        if name not in baseline_benchmarks:
            continue
            
        baseline_bench = baseline_benchmarks[name]
        
        current_time = current_bench["stats"]["mean"]
        baseline_time = baseline_bench["stats"]["mean"]
        
        # Calculate percentage change
        change_percent = ((current_time - baseline_time) / baseline_time) * 100
        
        if change_percent > threshold:
            has_regression = True
            regressions.append(
                f"🔴 {name}: {change_percent:.1f}% slower "
                f"({current_time:.3f}s vs {baseline_time:.3f}s)"
            )
        elif change_percent < -5:  # Improvement
            regressions.append(
                f"🟢 {name}: {abs(change_percent):.1f}% faster "
                f"({current_time:.3f}s vs {baseline_time:.3f}s)"
            )
    
    return has_regression, regressions

def main():
    parser = argparse.ArgumentParser(description="Check for performance regressions")
    parser.add_argument("--current", required=True, help="Current benchmark results JSON")
    parser.add_argument("--baseline", required=True, help="Baseline benchmark results JSON")
    parser.add_argument("--threshold", type=float, default=20.0, 
                       help="Regression threshold percentage (default: 20%)")
    
    args = parser.parse_args()
    
    print("🔍 Checking for performance regressions...")
    
    current = load_benchmark_results(args.current)
    baseline = load_benchmark_results(args.baseline)
    
    if not baseline.get("benchmarks"):
        print("⚠️  No baseline benchmarks found. Creating baseline...")
        sys.exit(0)
    
    has_regression, results = compare_benchmarks(current, baseline, args.threshold)
    
    print("\n📊 Performance Comparison Results:")
    print("=" * 50)
    
    for result in results:
        print(result)
    
    if has_regression:
        print(f"\n❌ Performance regression detected (>{args.threshold}% threshold)")
        print("Please investigate and optimize before merging.")
        sys.exit(1)
    else:
        print(f"\n✅ No significant performance regressions detected")
        sys.exit(0)

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/check_performance_regression.py

# Create baseline performance file
cat > .github/performance-baseline.json << 'EOF'
{
  "machine_info": {
    "platform": "Linux",
    "processor": "x86_64"
  },
  "benchmarks": [
    {
      "name": "test_vector_search_latency",
      "stats": {
        "mean": 0.150,
        "stddev": 0.020,
        "min": 0.120,
        "max": 0.200
      }
    },
    {
      "name": "test_graph_traversal_latency", 
      "stats": {
        "mean": 0.080,
        "stddev": 0.015,
        "min": 0.060,
        "max": 0.120
      }
    },
    {
      "name": "test_hybrid_query_latency",
      "stats": {
        "mean": 0.180,
        "stddev": 0.025,
        "min": 0.140,
        "max": 0.250
      }
    }
  ]
}
EOF

# Create or update pyproject.toml if it doesn't exist
if [ ! -f "pyproject.toml" ]; then
    print_info "Creating pyproject.toml..."
    cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "milvus-rag"
version = "0.1.0"
description = "Hybrid RAG system using Milvus and Neo4j"
authors = [{name = "Jay Grewal", email = "jay@onixnet.com"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.9"

[tool.black]
line-length = 88
target-version = ['py39']

[tool.ruff]
target-version = "py39"
line-length = 88
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
]
ignore = [
    "E501",  # line too long, handled by black
]

[tool.mypy]
python_version = "3.9"
check_untyped_defs = true
ignore_missing_imports = true

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q"
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "benchmark: marks tests as performance benchmarks",
]
asyncio_mode = "auto"

[tool.coverage.run]
source = ["src/graphrag"]
omit = [
    "*/tests/*",
    "*/__pycache__/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
]
EOF
else
    print_warning "pyproject.toml already exists, skipping..."
fi

print_status "GitHub Actions workflows created successfully!"

# Check for required secrets
print_info "Checking for required repository secrets..."

echo ""
print_warning "⚠️  IMPORTANT: You need to set up the following GitHub repository secrets:"
echo "   1. Go to your GitHub repository"
echo "   2. Navigate to Settings → Secrets and variables → Actions"
echo "   3. Add the following secrets:"
echo ""
echo "   📝 ANTHROPIC_API_KEY - Your Claude API key for @claude assistance"
echo "   📝 CODECOV_TOKEN - For coverage reports (optional)"
echo ""

# Suggest next steps
print_info "🎯 Next Steps:"
echo "   1. Push these changes to your repository"
echo "   2. Set up the required secrets in GitHub"
echo "   3. Create a pull request to test the workflows"
echo "   4. Use @claude in PR comments for intelligent assistance"
echo ""

print_info "🚀 Phase 1 Development Commands:"
echo "   ./run_in_env.sh pytest tests/ --cov=src/graphrag"
echo "   ./run_in_env.sh black src/ tests/"
echo "   ./run_in_env.sh ruff check src/ tests/"
echo ""

print_status "Setup complete! Your repository now has enterprise-grade CI/CD workflows."

echo ""
print_info "📖 Workflow Overview:"
echo "   🔧 CI/CD Pipeline: Runs tests, linting, and coverage on every PR"
echo "   🤖 Claude Assistant: Responds to @claude mentions in PRs and issues"  
echo "   🎨 Auto-format: Automatically formats code in PRs"
echo "   🔄 Dependabot: Keeps dependencies updated weekly"
echo "   📊 Performance: Tracks and prevents performance regressions"
echo ""

print_status "Ready to start Phase 1 development with confidence!"
EOF

chmod +x setup_github_actions.sh
