# CAKD Trash — App Flutter (camera realtime → build APK)

App Android chụp khung hình camera theo thời gian thực (~2 fps), gửi lên **API CAKD**
(`deploy/`) và hiển thị nhãn `glass / paper / plastic` cùng thanh xác suất.

Repo này đã **scaffold sẵn** (`android/`) và **build sẵn APK**:
`mobile_flutter/cakd_trash.apk` (~22MB). Cứ copy file này vào điện thoại và cài là chạy —
API URL mặc định đã trỏ tới `http://10.20.0.82:8000`.

Muốn build lại: `flutter pub get && flutter build apk --release`
→ `build/app/outputs/flutter-apk/app-release.apk`.

> Cấu hình đã dùng: AGP **8.3.0**, Kotlin 1.9.22, Gradle 8.7 (nâng từ mặc định để tránh
> lỗi JdkImage khi Flutter dùng JBR Java 21 của Android Studio).

---

## Yêu cầu

- [Flutter SDK](https://docs.flutter.dev/get-started/install) (kênh stable, ≥ 3.19).
- Android SDK (đi kèm Android Studio) để `flutter build apk`.
- API CAKD đang chạy (xem `deploy/README.md`) và điện thoại **cùng mạng LAN** với server.

Kiểm tra môi trường: `flutter doctor`.

---

## Bước 1 — Sinh khung project

Trong thư mục `mobile_flutter/` (đã có sẵn `pubspec.yaml` + `lib/main.dart`):

```bash
cd mobile_flutter
flutter create .          # sinh android/, ios/, ... KHÔNG đè pubspec.yaml/lib có sẵn
flutter pub get
```

> Nếu `flutter create .` hỏi/ghi đè `lib/main.dart`, giữ lại **bản trong repo** (dùng
> `git checkout lib/main.dart pubspec.yaml` để khôi phục nếu lỡ bị đè).

## Bước 2 — Cấp quyền camera + cho phép HTTP (cleartext)

Mở `android/app/src/main/AndroidManifest.xml`, trong thẻ `<manifest>` thêm quyền và
trong thẻ `<application>` thêm `usesCleartextTraffic` (vì API demo dùng `http://`, không phải https):

```xml
<manifest ...>
    <uses-permission android:name="android.permission.CAMERA" />
    <uses-permission android:name="android.permission.INTERNET" />

    <application
        android:usesCleartextTraffic="true"
        ... >
```

Trong `android/app/build.gradle` (hoặc `build.gradle.kts`) đảm bảo `minSdkVersion` ≥ **21**
(gói `camera` yêu cầu):

```gradle
defaultConfig {
    minSdkVersion 21
}
```

## Bước 3 — Build APK

```bash
flutter build apk --release
```

APK nằm ở `build/app/outputs/flutter-apk/app-release.apk`. Copy vào điện thoại và cài
(bật "Cài từ nguồn không xác định"). Chạy nhanh khi cắm máy: `flutter run --release`.

---

## Cấu hình địa chỉ server

Trong app, bấm biểu tượng 🔗 (góc trên phải) để nhập **API URL**:

- **Máy thật (điện thoại):** `http://<IP-LAN-server>:8000` — ví dụ `http://192.168.1.10:8000`.
  Lấy IP server bằng `ipconfig` (Windows) / `ifconfig` (Linux).
- **Android emulator:** `http://10.0.2.2:8000` (đã đặt sẵn làm mặc định).

Bấm **Bắt đầu** để chạy nhận diện realtime, **Dừng** để tạm ngưng, 🔄 để đổi camera trước/sau.

---

## Cách hoạt động

`lib/main.dart`:
- `camera` mở preview, `Timer.periodic` mỗi 600ms gọi `takePicture()` (có cờ `_busy`
  chống chồng request).
- Gửi ảnh JPEG dạng `multipart/form-data` (field `file`) tới `POST /detect`.
- Nhận `{label, confidence, probs, box, inference_ms}` → `DetectionBoxPainter` vẽ
  **bounding box** quanh vật thể + tên vật ở trên box, kèm overlay thanh xác suất bên dưới.

> **Về bounding box:** box do server tính bằng **CAM** (vùng model chú ý nhất cho lớp
> dự đoán) — 1 box cho vật chính, toạ độ normalize `[0,1]` theo khung. Alignment với
> preview là **gần đúng** (ảnh `takePicture` và preview có thể lệch tỉ lệ đôi chút). Đây
> là weakly-supervised localization, không phải detector đa vật.

Muốn nhẹ băng thông hơn, có thể chuyển sang `POST /predict_base64` hoặc hạ
`ResolutionPreset` / tăng `_intervalMs`.
