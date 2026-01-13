FROM python:3.11-slim

# System deps:
# - tesseract-ocr: free OCR engine
# - (optional) tesseract-ocr-eng: English language data (usually included)
RUN apt-get update && apt-get install -y \
  tesseract-ocr \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
