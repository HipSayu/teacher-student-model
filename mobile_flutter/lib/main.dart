// =============================================================================
// main.dart — App Flutter: camera realtime -> goi API CAKD -> hien nhan + xac suat.
// -----------------------------------------------------------------------------
// Luong: mo camera -> moi ~600ms chup 1 khung -> POST multipart len /predict ->
// nhan {label, confidence, probs, inference_ms} -> ve overlay len preview.
// Doi dia chi server o o "API URL" tren dau man hinh (luu bang shared_preferences).
// =============================================================================
import 'dart:async';
import 'dart:convert';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    cameras = await availableCameras();
  } catch (_) {
    cameras = [];
  }
  runApp(const CakdApp());
}

class CakdApp extends StatelessWidget {
  const CakdApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'CAKD Trash Classifier',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(useMaterial3: true),
      home: const CameraPage(),
    );
  }
}

class Prediction {
  final String label;
  final double confidence;
  final Map<String, double> probs;
  final double inferenceMs;
  final Rect? box; // bounding box normalize [0,1] theo khung (null neu API khong tra)
  Prediction(this.label, this.confidence, this.probs, this.inferenceMs, this.box);

  factory Prediction.fromJson(Map<String, dynamic> j) {
    final probs = <String, double>{};
    (j['probs'] as Map<String, dynamic>).forEach(
      (k, v) => probs[k] = (v as num).toDouble(),
    );
    Rect? box;
    if (j['box'] != null) {
      final b = j['box'] as Map<String, dynamic>;
      box = Rect.fromLTWH(
        (b['x'] as num).toDouble(),
        (b['y'] as num).toDouble(),
        (b['w'] as num).toDouble(),
        (b['h'] as num).toDouble(),
      );
    }
    return Prediction(
      j['label'] as String,
      (j['confidence'] as num).toDouble(),
      probs,
      (j['inference_ms'] as num).toDouble(),
      box,
    );
  }
}

class CameraPage extends StatefulWidget {
  const CameraPage({super.key});
  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> with WidgetsBindingObserver {
  CameraController? _controller;
  Timer? _timer;
  bool _busy = false;
  bool _streaming = false;
  int _camIndex = 0;

  // Server trien khai API CAKD. Doi qua nut 🔗 neu can (Android emulator dung http://10.0.2.2:8000).
  String _apiUrl = 'http://10.20.0.82:8000';
  Prediction? _pred;
  String _status = '';

  static const _intervalMs = 600;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _restoreUrl().then((_) => _initCamera(_camIndex));
  }

  Future<void> _restoreUrl() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() => _apiUrl = prefs.getString('api_url') ?? _apiUrl);
  }

  Future<void> _saveUrl(String url) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('api_url', url);
  }

  Future<void> _initCamera(int index) async {
    if (cameras.isEmpty) {
      setState(() => _status = 'Khong tim thay camera tren thiet bi.');
      return;
    }
    await _controller?.dispose();
    final controller = CameraController(
      cameras[index % cameras.length],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );
    _controller = controller;
    try {
      await controller.initialize();
      if (mounted) setState(() {});
    } catch (e) {
      setState(() => _status = 'Loi mo camera: $e');
    }
  }

  void _toggleStream() {
    if (_streaming) {
      _timer?.cancel();
      setState(() => _streaming = false);
    } else {
      setState(() => _streaming = true);
      _timer = Timer.periodic(
        const Duration(milliseconds: _intervalMs),
        (_) => _captureAndSend(),
      );
    }
  }

  Future<void> _captureAndSend() async {
    final controller = _controller;
    if (_busy || controller == null || !controller.value.isInitialized) return;
    _busy = true;
    try {
      final shot = await controller.takePicture();
      final bytes = await shot.readAsBytes();
      final sw = Stopwatch()..start();
      final req = http.MultipartRequest('POST', Uri.parse('$_apiUrl/detect'))
        ..files.add(
          http.MultipartFile.fromBytes('file', bytes, filename: 'frame.jpg'),
        );
      final resp = await http.Response.fromStream(
        await req.send().timeout(const Duration(seconds: 8)),
      );
      if (resp.statusCode == 200) {
        final pred = Prediction.fromJson(jsonDecode(resp.body));
        if (mounted) {
          setState(() {
            _pred = pred;
            _status = 'infer ${pred.inferenceMs}ms · rtt ${sw.elapsedMilliseconds}ms';
          });
        }
      } else {
        if (mounted) setState(() => _status = 'HTTP ${resp.statusCode}');
      }
    } catch (e) {
      if (mounted) setState(() => _status = 'Loi: $e');
    } finally {
      _busy = false;
    }
  }

  Future<void> _editUrl() async {
    final ctrl = TextEditingController(text: _apiUrl);
    final result = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('API URL'),
        content: TextField(
          controller: ctrl,
          autofocus: true,
          keyboardType: TextInputType.url,
          decoration: const InputDecoration(hintText: 'http://192.168.1.10:8000'),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Huy')),
          TextButton(
            onPressed: () => Navigator.pop(ctx, ctrl.text.trim()),
            child: const Text('Luu'),
          ),
        ],
      ),
    );
    if (result != null && result.isNotEmpty) {
      setState(() => _apiUrl = result);
      await _saveUrl(result);
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      _timer?.cancel();
      _streaming = false;
      controller.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initCamera(_camIndex);
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _timer?.cancel();
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    final ready = controller != null && controller.value.isInitialized;
    final pred = _pred;

    return Scaffold(
      appBar: AppBar(
        title: const Text('♻️ CAKD Trash'),
        actions: [
          IconButton(icon: const Icon(Icons.link), tooltip: _apiUrl, onPressed: _editUrl),
          if (cameras.length > 1)
            IconButton(
              icon: const Icon(Icons.flip_camera_ios),
              onPressed: () {
                _camIndex = (_camIndex + 1) % cameras.length;
                _initCamera(_camIndex);
              },
            ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: Stack(
              fit: StackFit.expand,
              children: [
                if (ready)
                  CameraPreview(controller)
                else
                  const Center(child: CircularProgressIndicator()),
                if (pred?.box != null)
                  Positioned.fill(
                    child: CustomPaint(
                      painter: DetectionBoxPainter(pred!.box!, pred.label, pred.confidence),
                    ),
                  ),
                if (pred != null)
                  Positioned(
                    left: 0,
                    right: 0,
                    bottom: 0,
                    child: _ResultOverlay(pred: pred),
                  ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(vertical: 6),
            color: Colors.black,
            child: Text(
              _status.isEmpty ? _apiUrl : _status,
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 11, color: Colors.white54),
            ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: ready ? _toggleStream : null,
        icon: Icon(_streaming ? Icons.stop : Icons.play_arrow),
        label: Text(_streaming ? 'Dung' : 'Bat dau'),
      ),
    );
  }
}

// Ve bounding box (toa do normalize [0,1] theo khung) + ten vat the phia tren box.
class DetectionBoxPainter extends CustomPainter {
  final Rect nbox; // normalize [0,1]
  final String label;
  final double conf;
  const DetectionBoxPainter(this.nbox, this.label, this.conf);

  static const _accent = Color(0xFF39C6FF);

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Rect.fromLTWH(
      nbox.left * size.width,
      nbox.top * size.height,
      nbox.width * size.width,
      nbox.height * size.height,
    );
    final stroke = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = _accent;
    canvas.drawRRect(
      RRect.fromRectAndRadius(rect, const Radius.circular(8)),
      stroke,
    );

    // Nhan ten + do tin cay, dat ngay tren canh tren cua box.
    final tp = TextPainter(
      text: TextSpan(
        text: '  $label ${(conf * 100).toStringAsFixed(0)}%  ',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 14,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    final labelH = tp.height + 4;
    var labelTop = rect.top - labelH;
    if (labelTop < 0) labelTop = rect.top; // neu sat mep tren thi dat trong box
    final bg = Paint()..color = _accent;
    canvas.drawRect(
      Rect.fromLTWH(rect.left, labelTop, tp.width, labelH),
      bg,
    );
    tp.paint(canvas, Offset(rect.left, labelTop + 2));
  }

  @override
  bool shouldRepaint(covariant DetectionBoxPainter oldDelegate) =>
      oldDelegate.nbox != nbox ||
      oldDelegate.label != label ||
      oldDelegate.conf != conf;
}


class _ResultOverlay extends StatelessWidget {
  final Prediction pred;
  const _ResultOverlay({required this.pred});

  @override
  Widget build(BuildContext context) {
    final sorted = pred.probs.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 24, 16, 16),
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [Colors.transparent, Colors.black87],
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.baseline,
            textBaseline: TextBaseline.alphabetic,
            children: [
              Text(
                pred.label,
                style: const TextStyle(fontSize: 30, fontWeight: FontWeight.bold),
              ),
              const SizedBox(width: 10),
              Text(
                '${(pred.confidence * 100).toStringAsFixed(1)}%',
                style: const TextStyle(fontSize: 18, color: Color(0xFF9FE6B0)),
              ),
            ],
          ),
          const SizedBox(height: 10),
          ...sorted.map(
            (e) => Padding(
              padding: const EdgeInsets.symmetric(vertical: 3),
              child: Row(
                children: [
                  SizedBox(width: 64, child: Text(e.key, style: const TextStyle(fontSize: 13))),
                  Expanded(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(6),
                      child: LinearProgressIndicator(
                        value: e.value,
                        minHeight: 8,
                        backgroundColor: Colors.white12,
                        valueColor: const AlwaysStoppedAnimation(Color(0xFF39C6FF)),
                      ),
                    ),
                  ),
                  SizedBox(
                    width: 44,
                    child: Text(
                      '${(e.value * 100).toStringAsFixed(0)}%',
                      textAlign: TextAlign.right,
                      style: const TextStyle(fontSize: 12),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
