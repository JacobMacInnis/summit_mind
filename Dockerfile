# Use an official Python runtime as parent image
FROM python:3.10-slim

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

# Copy the rest of your application
COPY . /app

# Expose port
EXPOSE 8000

# Command to run the server
# CMD ["poetry", "run", "uvicorn", "server.index:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python", "start_server.py"]

