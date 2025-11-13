# ------------ Base Image -------------
FROM python:3.10-slim

# ------------ Work Directory -------------
WORKDIR /app

# ------------ Install Dependencies -------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------ Copy Project Files -------------
COPY . .

# ------------ Expose API Port -------------
EXPOSE 5000

# ------------ Run FastAPI -------------
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5000"]
