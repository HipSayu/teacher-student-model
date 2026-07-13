# CAKD Trash Classifier — API + Docker

Đóng gói `model_60.pth` (student **ResNet-50 CAKD**, chưng cất từ teacher ViT-B/16 —
epoch 60, **test acc@1 = 96.73%**) thành REST API để triển khai lên server, kèm trang
demo camera realtime.

Phân loại **3 lớp**: `glass` · `paper` · `plastic` (thứ tự alphabet của ImageFolder).

---

## Cấu trúc

```
deploy/
  inference.py        # nạp checkpoint + tiền xử lý + suy luận (dùng CAKD/models/resnet_cakd.py)
  app.py              # FastAPI: /health /classes /predict /predict_base64 + trang demo
  static/index.html   # demo camera realtime (mở bằng trình duyệt điện thoại)
  requirements.txt
  Dockerfile          # CPU-only (torch CPU từ index riêng)
  docker-compose.yml  # mount model_60.pth vào container, expose cổng 8000
```

Model **không** bị bake vào image (499 MB) — được **mount** lúc chạy từ `../model_60.pth`.

---

## Chạy bằng Docker (khuyến nghị)

Từ **thư mục gốc repo** (nơi có `model_60.pth`):

```bash
docker compose -f deploy/docker-compose.yml up --build
```

Kiểm tra:

```bash
curl http://localhost:8000/health
```

Mở trang demo camera: **http://localhost:8000/** (trên điện thoại dùng IP LAN của máy chủ,
ví dụ `http://192.168.1.10:8000/`; camera trình duyệt cần HTTPS hoặc `localhost`).

---

## Chạy trực tiếp (không Docker) — để dev

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r deploy/requirements.txt
MODEL_PATH=./model_60.pth uvicorn app:app --host 0.0.0.0 --port 8000 --app-dir deploy
```

---

## Endpoints

| Method | Path              | Mô tả |
|--------|-------------------|-------|
| GET    | `/health`         | trạng thái + metadata (device, epoch, classes) |
| GET    | `/classes`        | danh sách lớp |
| POST   | `/predict`        | multipart `file=<ảnh>` → dự đoán |
| POST   | `/predict_base64` | JSON `{"image":"<base64|data-uri>","topk":3}` |
| GET    | `/`               | trang demo camera realtime |

Ví dụ:

```bash
curl -F "file=@anh_thu.jpg" http://localhost:8000/predict
```

Kết quả:

```json
{
  "label": "plastic",
  "confidence": 0.981,
  "topk": [{"label":"plastic","prob":0.981}, {"label":"glass","prob":0.014}, ...],
  "probs": {"glass":0.014,"paper":0.005,"plastic":0.981},
  "inference_ms": 142.3
}
```

---

## Biến môi trường

| Biến          | Mặc định                | Ý nghĩa |
|---------------|-------------------------|---------|
| `MODEL_PATH`  | `/models/model_60.pth`  | đường dẫn checkpoint |
| `DEVICE`      | tự chọn (`cpu`)         | `cpu` hoặc `cuda` |
| `CLASS_NAMES` | `glass,paper,plastic`   | tên lớp (đúng thứ tự train) |
| `USE_EMA`     | `0`                     | `1` để dùng bản EMA của model |

---

## Triển khai lên server

1. Copy repo (hoặc chỉ `deploy/`, `CAKD/models/`, `model_60.pth`) lên server.
2. Cài Docker + Docker Compose.
3. `docker compose -f deploy/docker-compose.yml up --build -d`.
4. Mở cổng `8000` (hoặc đặt Nginx/Caddy reverse-proxy + HTTPS phía trước — cần HTTPS
   để camera trình duyệt hoạt động ngoài `localhost`).

App Flutter kết nối tới `http://<IP-server>:8000` (xem `mobile_flutter/README.md`).
