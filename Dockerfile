FROM python:3.10-slim

WORKDIR /app

COPY ./graphrag /app/graphrag

RUN apt-get update && apt-get install -y git curl && \
    pip install --no-cache-dir -r /app/graphrag/requirements.txt && \
    python -m nltk.downloader stopwords && \
    python -m spacy download en_core_web_sm

ENV PYTHONPATH="/app"

CMD ["python", "graphrag/rag_system/main.py"]
