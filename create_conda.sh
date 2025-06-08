#! /bin/bash

conda create  -c conda-forge  --name milvus_env python=3.13.4
conda activate milvus_env
conda update conda
conda config --set channel_priority strict
python -c "import platform;print(platform.machine())"  
conda install uv pip -y
uv pip install -r requirements.txt

uv pip install -e .
