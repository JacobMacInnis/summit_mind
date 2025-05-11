# Use an official Python runtime as parent image
FROM python:3.10-slim

# Accept model name at build time
ARG MODEL_NAME
ENV MODEL_NAME=$MODEL_NAME

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Copy poetry files first to leverage Docker cache
COPY pyproject.toml poetry.lock* /app/

# Install Poetry
RUN pip install poetry

# Install dependencies
RUN poetry config virtualenvs.create false \
 && poetry install --only main --no-root --no-interaction --no-ansi

COPY models/${MODEL_NAME} /app/models/${MODEL_NAME}

# Copy just the minimal required application code
COPY start_server.py /app/
COPY server/ /app/server/

# Expose port
EXPOSE 8000

LABEL cache_bust="rebuild-$(date +%s)"
# Command to run the server
# CMD ["poetry", "run", "uvicorn", "server.index:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python", "start_server.py"]

