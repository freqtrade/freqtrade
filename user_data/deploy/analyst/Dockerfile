FROM python:3.12-slim
RUN pip install --no-cache-dir requests psycopg2-binary ccxt
WORKDIR /app
COPY analyst.py .
CMD ["python", "-u", "analyst.py"]
