FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --no-cache-dir "mcp[cli]>=1.0.0" aiohttp langdetect

COPY main.py /app/main.py

EXPOSE 8886

CMD ["python", "/app/main.py"]
