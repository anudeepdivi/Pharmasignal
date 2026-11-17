FROM python:3.12.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PHARMASIGNAL_MODEL_DIR=/app/model \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY docker/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes -r /app/requirements.txt \
    && useradd --uid 10001 --create-home appuser
COPY src/ /app/src/
COPY artifacts/models/m4/pipeline.joblib /app/model/pipeline.joblib
COPY artifacts/models/m4/report.json /app/model/report.json
COPY artifacts/models/m4/metadata.json /app/model/metadata.json
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"
CMD ["python", "-m", "pharmasignal.serving", "--host", "0.0.0.0", "--port", "8000"]
