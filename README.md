
Here’s a tailored README for your **"Build RAG with Milvus"** project, reflecting your Debian WSL environment, conda setup, and Docker Milvus configuration. You can adapt or expand sections as needed.

---

# Build RAG with Milvus
### PO1 – Projects. Open. One.
#### A unified hub to Automate. Innovate. Elevate. Encourages automation and innovation for growth.

## Introduction

This project demonstrates how to build a Retrieval-Augmented Generation (RAG) system using [Milvus](https://milvus.io), a high-performance vector database. It follows the official guide from [Milvus Documentation](https://milvus.io/docs/build-rag-with-milvus.md), adapted for a local environment on Debian running via WSL (Windows Subsystem for Linux).

Milvus is deployed via Docker, and the project environment is managed using Conda within WSL.

---

## Table of Contents

* [Introduction](#introduction)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Features](#features)
* [Configuration](#configuration)
* [Dependencies](#dependencies)
* [Examples](#examples)
* [Troubleshooting](#troubleshooting)
* [Contributors](#contributors)
* [License](#license)

---

## Project Structure

env file
```
# Path to Docker volume directory inside WSL
DOCKER_VOLUME_DIRECTORY=/mnt/d/projects/wslprojects/milvus_env

# MinIO credentials
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
```

Project Directory

```bash
BASE_DIR=$DOCKER_VOLUME_DIRECTORY
BASE_DIR/
├── docker-compose.yml         # Milvus, etcd, minio containers
├── .env                       # Environment variable file
├── app/                       # RAG-related Python code
├── data/                      # Sample or ingested documents
└── notebooks/                 # Optional Jupyter notebooks
```

---

## Installation

### 1. Prerequisites

Ensure the following are installed in your WSL Debian environment:

* [Docker](https://docs.docker.com/engine/install/)
* [Conda (Miniconda/Anaconda)](https://docs.conda.io/en/latest/)
* Git, Curl, Python 3.9+

### 2. Create and Activate Conda Environment

```bash
conda create -n milvus_env python=3.9
conda activate milvus_env
```

### 3. Clone Project Repository

```bash
BASE_DIR=$DOCKER_VOLUME_DIRECTORY
git clone https://github.com/iamgrewal/milvus_rag BASE_DIR
cd BASE_DIR
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure `.env` File

```ini
DOCKER_VOLUME_DIRECTORY=BASE_DIR
```

---
### 6. Requirements 

```bash
langchain>=0.1.0
pymilvus==2.2.9
openai>=1.0.0
sentence-transformers>=2.2.2
python-dotenv>=1.0.0
tqdm
requests
numpy
```

## Usage

### 1. Start Milvus via Docker Compose

```bash
docker-compose up -d
```

This will spin up:

* **Milvus Standalone** (port 19530)
* **etcd** for metadata
* **MinIO** for object storage (access key: `minioadmin`, secret key: `minioadmin`)

### 2. Run the RAG Pipeline

Follow the Milvus documentation or provided scripts in `app/` to:

* Ingest documents
* Embed and store vectors in Milvus
* Query documents using natural language and LLM

---

## Features

* Full local deployment of Milvus via Docker
* Persistent volumes for Milvus, MinIO, and etcd
* Integration-ready for RAG pipelines using LangChain or custom logic
* Optimized for WSL on Windows

---

## Configuration

Edit the `.env` file for custom paths or volume mappings. Docker containers are set to use:

```env
DOCKER_VOLUME_DIRECTORY=BASE_DIR
```

You may also configure:

* Milvus port: `19530`
* MinIO port: `9000`
* etcd port: `2379`

---

## Dependencies

* [Milvus v2.2.9](https://milvus.io)
* [MinIO](https://min.io)
* [etcd](https://etcd.io)
* Python libraries:

  * langchain
  * pymilvus
  * sentence-transformers
  * openai
  * dotenv

---

## Examples

```bash
python app/load_data.py       # Embeds and inserts data into Milvus
python app/query_rag.py       # Queries Milvus and uses LLM to respond
```

You can also explore via Jupyter notebooks in the `notebooks/` directory (optional).

---

## Troubleshooting

* **Milvus container won't start?**

  * Ensure ports `19530`, `2379`, and `9000` are not blocked or already in use.
  * Check volumes have proper write permissions from WSL.

* **WSL volume mount issues?**

  * Ensure Docker Desktop has access to your mounted D: drive.

* **Python dependency errors?**

  * Make sure your conda environment is activated (`milvus_env`).

---

## Contributors

* Jatinder GRewal ([@yourhandle](https://github.com/iamgrewal))

---

## License

This project is licensed under the [MIT License](LICENSE), unless otherwise specified by upstream dependencies.
