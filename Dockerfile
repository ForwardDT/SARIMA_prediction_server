# Use an official Python runtime
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only requirements first (leveraging Docker layer caching)
COPY requirements.txt .

# Install OS dependencies (if any; e.g. build-essential, freetype for matplotlib—
# slim may need libpng, ttf-dejavu, etc.)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
      libfreetype6-dev \
      libpng-dev \
      libjpeg-dev \
 && pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get purge -y --auto-remove build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy your application code
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Launch via uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
