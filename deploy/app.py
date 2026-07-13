# =============================================================================
# app.py — API FastAPI phuc vu model_60.pth (student ResNet-50 CAKD, 3 lop).
# -----------------------------------------------------------------------------
# Endpoint:
#   GET  /            -> trang demo camera realtime (deploy/static/index.html)
#   GET  /health      -> trang thai + metadata model
#   GET  /classes     -> danh sach lop
#   POST /predict     -> nhan anh (multipart file HOAC JSON base64) -> nhan du doan
# Chay:  uvicorn app:app --host 0.0.0.0 --port 8000
# =============================================================================
import base64
import binascii
import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from inference import TrashClassifier

# --- Cau hinh qua bien moi truong ---------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model_60.pth")
DEVICE = os.environ.get("DEVICE", None)                 # None -> tu chon cuda/cpu
USE_EMA = os.environ.get("USE_EMA", "0") == "1"
_HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(_HERE, "static")

app = FastAPI(
    title="CAKD Trash Classifier API",
    description="Phan loai rac tai che (glass / paper / plastic) — student ResNet-50 chung cat tu ViT-B/16 (CAKD).",
    version="1.0.0",
)

# Mo CORS cho app mobile / web goi tu domain khac
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier: TrashClassifier = None  # nap khi startup


@app.on_event("startup")
def _load_model():
    global classifier
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Khong tim thay weights tai MODEL_PATH={MODEL_PATH}. "
            "Mount file model_60.pth vao container (xem docker-compose.yml)."
        )
    classifier = TrashClassifier(MODEL_PATH, device=DEVICE, use_ema=USE_EMA)


class Base64Request(BaseModel):
    image: str          # anh base64 (co the kem tien to 'data:image/...;base64,')
    topk: int = 3


def _decode_image(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Khong doc duoc anh (dinh dang khong hop le).")


@app.get("/health")
def health():
    return {
        "status": "ok" if classifier is not None else "loading",
        "model_path": MODEL_PATH,
        "device": str(classifier.device) if classifier else None,
        "epoch": classifier.epoch if classifier else None,
        "classes": classifier.classes if classifier else None,
        "use_ema": USE_EMA,
    }


@app.get("/classes")
def classes():
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model dang nap.")
    return {"classes": classifier.classes}


@app.post("/predict")
async def predict(file: UploadFile = File(...), topk: int = 3):
    """Suy luan tu file anh gui theo multipart/form-data (field 'file')."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model dang nap.")
    img = _decode_image(await file.read())
    return classifier.predict(img, topk=topk)


@app.post("/predict_base64")
def predict_base64(req: Base64Request):
    """Suy luan tu chuoi base64 (tien loi cho app mobile gui khung hinh camera)."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model dang nap.")
    data = req.image.split(",", 1)[-1]  # bo tien to data-URI neu co
    try:
        raw = base64.b64decode(data)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Chuoi base64 khong hop le.")
    return classifier.predict(_decode_image(raw), topk=req.topk)


# --- Trang demo camera realtime (phuc vu file tinh) ---------------------------
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
