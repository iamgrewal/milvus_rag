# Milvus RAG Project

A robust Retrieval-Augmented Generation (RAG) system built with Milvus vector database.

## Prerequisites

- Python 3.8+
- Conda (Miniconda or Anaconda)
- Git
- Universal-ctags (for code navigation)

## Project Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd milvus_rag
```

### 2. Environment Setup

The project uses Conda for environment management. Two scripts are provided to handle environment setup and execution:

#### Option 1: Automated Setup (Recommended)

Run the setup script to create and configure the Conda environment:

```bash
./create_conda.sh
```

This script will:
- Create a new Conda environment named `milvus_env`
- Install all required dependencies
- Set up development tools
- Generate ctags for code navigation

#### Option 2: Manual Setup

If you prefer manual setup:

```bash
# Create and activate the environment
conda create -n milvus_env python=3.8
conda activate milvus_env

# Install dependencies
pip install -U pip
pip install -r requirements.txt
pip install -r dev-requirements.txt
pip install -e .
```

### 3. Running Commands

Always ensure you're using the correct environment. Two methods are available:

#### Method 1: Using the Wrapper Script (Recommended)

```bash
# Run any command in the correct environment
./run_in_env.sh python your_script.py
./run_in_env.sh pytest
./run_in_env.sh python -m graphrag.rag_system.main
```

#### Method 2: Manual Activation

```bash
# Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate milvus_env

# Run your commands
python your_script.py
```

## Development

### Code Navigation

The project uses ctags for code navigation. Tags are automatically generated during setup. To regenerate tags:

```bash
ctags -R --languages=Python --python-kinds=-iv -f .tags .
```

### IDE Setup

#### VSCode

Add the following to `.vscode/settings.json`:

```json
{
    "python.pythonPath": "/home/<your-user>/anaconda3/envs/milvus_env/bin/python",
    "python.analysis.extraPaths": ["src"]
}
```

## Project Structure

```
milvus_rag/
├── create_conda.sh      # Environment setup script
├── run_in_env.sh        # Environment wrapper script
├── requirements.txt     # Production dependencies
├── dev-requirements.txt # Development dependencies
├── pyproject.toml       # Project configuration
└── src/
    └── graphrag/        # Main package
        ├── config/      # Configuration
        ├── embedding_service/  # Embedding service
        ├── milvus/      # Milvus integration
        ├── neo4j/       # Neo4j integration
        ├── nlp/         # NLP processing
        └── rag_system/  # RAG system implementation
```

## Contributing

1. Ensure you're using the correct Conda environment
2. Follow the project's coding standards
3. Run tests before submitting changes
4. Update documentation as needed

## License

[Your License Here]

## Contact

[Your Contact Information]
