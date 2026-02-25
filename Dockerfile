FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies first
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy pyproject.toml and uv.lock first
COPY pyproject.toml uv.lock ./

# Copy the source code (required for building the local package)
COPY ./src /app/src
COPY ./public /app/public
COPY README.md ./

# Install dependencies and build the local package
RUN uv sync

# Expose the port
EXPOSE 7855

# Start the application
ENTRYPOINT ["uv", "run", "chainlit", "run", "src/frontend/app.py", "--host", "0.0.0.0", "--port", "7855", "-h"]
