FROM python:3.10-slim
WORKDIR /app
COPY web/requirements.txt web/requirements.txt
RUN pip install --no-cache-dir -r web/requirements.txt
COPY common.py ./common.py
COPY data ./data
COPY outputs ./outputs
COPY README.md ./README.md
COPY web ./web
EXPOSE 8000
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
