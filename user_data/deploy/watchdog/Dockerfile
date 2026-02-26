FROM python:3.12-slim
RUN pip install --no-cache-dir requests psycopg2-binary ccxt
WORKDIR /app
COPY watchdog.py .
CMD ["python", "-u", "watchdog.py"]
