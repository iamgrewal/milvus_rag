---
description: Documentation for the Dockerfile used to set up the environment for the Graphrag application.
globs: ['Dockerfile']
alwaysApply: false
---

# Dockerfile Documentation

## Overview
This Dockerfile is used to create a Docker image for the Graphrag application, which is a Python-based project. The image is built on top of a slim version of Python 3.10, ensuring a lightweight environment suitable for running the application.

## Key Components
- **FROM python:3.10-slim**: This line specifies the base image for the Docker container, which is a minimal version of Python 3.10.
- **WORKDIR /app**: Sets the working directory inside the container to `/app`, where the application code will reside.
- **COPY ./graphrag /app/graphrag**: Copies the local `graphrag` directory into the container's `/app/graphrag` directory, making the application code available in the container.
- **RUN apt-get update && apt-get install -y git curl**: Updates the package list and installs necessary system packages, including Git and cURL, which may be required for the application.
- **pip install --no-cache-dir -r /app/graphrag/requirements.txt**: Installs the Python dependencies listed in the `requirements.txt` file without caching, which helps keep the image size smaller.
- **python -m nltk.downloader stopwords**: Downloads the NLTK stopwords dataset, which is likely used in the application for natural language processing tasks.
- **python -m spacy download en_core_web_sm**: Downloads the small English model for SpaCy, another NLP library that the application may utilize.
- **ENV PYTHONPATH="/app"**: Sets the `PYTHONPATH` environment variable to include the `/app` directory, allowing Python to locate the application modules.
- **CMD ["python", "graphrag/rag_system/main.py"]**: Specifies the command to run when the container starts, which is to execute the main script of the Graphrag application.

## Dependencies
This Dockerfile does not import any other files in the repository, nor is it imported by any other files. It is a standalone file that defines the environment for the Graphrag application.

## Usage Example
To build the Docker image, navigate to the directory containing the Dockerfile and run:
```bash
docker build -t graphrag .
```
To run the container, use:
```bash
docker run -it graphrag
```

## Best Practices
- Ensure that the `requirements.txt` file is kept up to date with all necessary dependencies for the application.
- Regularly review and update the base image to include security patches and improvements.
- Minimize the number of layers in the Dockerfile by combining commands where possible to reduce the final image size.