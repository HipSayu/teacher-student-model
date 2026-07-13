# =============================================================================
# Dockerfile — API CAKD Trash Classifier (CPU-only).
# -----------------------------------------------------------------------------
# Build context = THU MUC GOC repo (de COPY duoc CAKD/models). Vi du:
#   docker build -f deploy/Dockerfile -t cakd-api .
# Model weights (model_60.pth) KHONG bake vao image (499MB) — mount luc chay
# (xem docker-compose.yml) qua MODEL_PATH=/models/model_60.pth.
# =============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/models/model_60.pth \
    DEVICE=cpu

WORKDIR /app

# Torch CPU tu index rieng cua PyTorch (nhe hon ban CUDA rat nhieu)
COPY deploy/requirements.txt .
RUN pip install --no-cache-dir \
        torch==2.3.1 torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
        einops==0.8.0 pillow==10.4.0 fastapi==0.111.0 \
        "uvicorn[standard]==0.30.1" python-multipart==0.0.9 pydantic==2.8.2

# Dinh nghia model (student ResNet-50 CAKD) + code API
COPY CAKD/models /app/CAKD/models
COPY deploy/inference.py deploy/app.py /app/
COPY deploy/static /app/static

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
