# Use official Python image
FROM python:3.10-slim

COPY . /app

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY embeddings/faiss_index /app/embeddings/faiss_index

# Copy backend code
COPY . .

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]