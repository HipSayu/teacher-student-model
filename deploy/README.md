# CAKD Trash Classifier — API + Docker

Đóng gói `model_60.pth` (student **ResNet-50 CAKD**, chưng cất từ teacher ViT-B/16 —
epoch 60, **test acc@1 = 96.73%**) thành REST API để triển khai lên server, kèm trang
demo camera realtime.

Phân loại **3 lớp**: `glass` · `paper` · `plastic` (thứ tự alphabet của ImageFolder).

---

## Cấu trúc — `deploy/` TỰ CHỨA HOÀN TOÀN

```
deploy/
  inference.py        # nạp checkpoint + tiền xử lý + suy luận
  app.py              # FastAPI: /health /classes /predict /predict_base64 + trang demo
  static/index.html   # demo camera realtime (mở bằng trình duyệt điện thoại)
  CAKD/models/        # định nghĩa model (resnet_cakd.py) — vendored, không cần repo gốc
  model_60.pth        # weights (476MB) — bake thẳng vào image
  requirements.txt
  Dockerfile          # CPU-only, COPY model vào image (không cần mount)
  docker-compose.yml
```

Chỉ cần copy **nguyên thư mục `deploy/`** lên server là chạy được — không phụ thuộc phần
còn lại của repo.

> ⚠️ `model_60.pth` bị `.gitignore` (rule `*.pth`) → **push bằng git sẽ KHÔNG mang theo model**.
> Dùng `scp`/`rsync` để copy cả thư mục `deploy/` (gồm `model_60.pth`) lên server. Ví dụ:
> `rsync -avz deploy/ user@10.20.0.82:~/cakd/deploy/`

---

## Chạy bằng Docker (khuyến nghị)

Copy `deploy/` lên server, rồi **chạy ngay trong `deploy/`**:

```bash
cd deploy
docker compose up --build -d
```

Kiểm tra:

```bash
curl http://localhost:8000/health
```

Mở trang demo camera: **http://localhost:8000/** (trên điện thoại dùng IP LAN của máy chủ,
ví dụ `http://192.168.1.10:8000/`; camera trình duyệt cần HTTPS hoặc `localhost`).

---

## Chạy trực tiếp (không Docker) — để dev

Trong thư mục `deploy/`:

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
MODEL_PATH=./model_60.pth uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Endpoints

| Method | Path              | Mô tả |
|--------|-------------------|-------|
| GET    | `/health`         | trạng thái + metadata (device, epoch, classes) |
| GET    | `/classes`        | danh sách lớp |
| POST   | `/predict`        | multipart `file=<ảnh>` → nhãn + xác suất |
| POST   | `/predict_base64` | JSON `{"image":"<base64|data-uri>","topk":3}` |
| POST   | `/detect`         | multipart `file=<ảnh>` → nhãn + xác suất + **bounding box** (CAM) |
| POST   | `/detect_base64`  | JSON `{"image":"...","thresh":0.35}` → như trên |
| GET    | `/`               | trang demo camera realtime |

> **`/detect` — bounding box bằng CAM (weakly-supervised localization).** Model là bộ
> **phân loại**, không phải detector. `/detect` dùng feature map cuối + trọng số lớp `fc`
> để tính vùng model chú ý nhất cho lớp dự đoán, rồi suy ra **1** bounding box
> (`box: {x,y,w,h}` normalize `[0,1]` theo toàn khung). Phù hợp 1 vật thể chính trên nền
> sạch; **không** tách được nhiều vật. Muốn multi-object thật cần train detector (YOLO) riêng.

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

## Triển khai lên server (ví dụ 10.20.0.82)

1. Copy **nguyên thư mục `deploy/`** (gồm cả `model_60.pth`) lên server bằng `scp`/`rsync`:
   ```bash
   rsync -avz deploy/ user@10.20.0.82:~/cakd/deploy/
   ```
2. Trên server, cài Docker + Docker Compose, rồi:
   ```bash
   cd ~/cakd/deploy
   docker compose up --build -d
   curl http://localhost:8000/health          # kỳ vọng status: ok
   ```
3. Mở firewall cổng `8000` để điện thoại LAN gọi vào.
4. (Tùy chọn) đặt Nginx/Caddy reverse-proxy + HTTPS phía trước — cần HTTPS nếu muốn
   dùng **trang demo web** camera ngoài `localhost` (app Flutter thì không cần).

App Flutter kết nối tới `http://10.20.0.82:8000` (đã cắm sẵn — xem `mobile_flutter/README.md`).
