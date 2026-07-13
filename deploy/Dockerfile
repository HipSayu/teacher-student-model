# =============================================================================
# Dockerfile — API CAKD Trash Classifier (CPU-only), TU CHUA trong deploy/.
# -----------------------------------------------------------------------------
# Build context = chinh thu muc deploy/. Chay tu trong deploy/:
#   docker build -t cakd-api .
# Model weights (model_60.pth) nam ngay trong deploy/ va duoc COPY thang vao image
# (bake) -> khong can mount, khong lo sai duong dan.
# =============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/models/model_60.pth \
    DEVICE=cpu

WORKDIR /app

# Torch CPU tu index rieng cua PyTorch (nhe hon ban CUDA rat nhieu)
COPY requirements.txt .
RUN pip install --no-cache-dir \
        torch==2.3.1 torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
        einops==0.8.0 pillow==10.4.0 fastapi==0.111.0 \
        "uvicorn[standard]==0.30.1" python-multipart==0.0.9 pydantic==2.8.2

# Dinh nghia model (student ResNet-50 CAKD) + code API + trang demo
COPY CAKD /app/CAKD
COPY inference.py app.py /app/
COPY static /app/static

# Weights: bake thang vao image (deploy/model_60.pth -> /models/model_60.pth)
COPY model_60.pth /models/model_60.pth

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
