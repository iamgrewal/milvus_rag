#! /bin/bash

conda create  -c conda-forge  --name milvus_env python=3.13.4  -y
conda activate milvus_env

conda install -c conda-forge uv -y
conda install uv pip -y

conda update conda -y
conda config --set channel_priority strict -y

python -c "import platform;print(platform.machine())"  

uv pip install -U pip -y
uv pip install -r requirements.txt -y
uv pip install -e . -y

uv pip install -U setuptools -y
uv pip install -U wheel -y
uv pip install -U twine -y
uv pip install -U mkdocs-material -y

uv pip install -U langchain -y
uv pip install -U pymilvus -y
uv pip install -U openai -y
uv pip install -U sentence-transformers -y
uv pip install -U python-dotenv -y
uv pip install -U tqdm -y
uv pip install -U requests -y
uv pip install -U numpy -y
uv pip install -U protobuf -y
uv pip install -U grpcio-tools -y
uv pip install -U neo4j -y
uv pip install -U neo4j-graphrag[openai] -y
uv pip install -U python-multipart -y
uv pip install -U pydantic -y
uv pip install -U pydantic-settings -y
uv pip install -U pydantic-core -y
uv pip install -U pydantic-core-fastapi -y
uv pip install -U pydantic-core-fastapi -y
