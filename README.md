# Top-View Cattle-ID (YOLO-VCC based)

Implementation of **Top-View Cattle Identification** Model – based on customized YOLO-VCC, focusing on real-time cattle ID recognition from overhead UAV shots.

[![Papers with Code](https://img.shields.io/badge/PaperswithCode-ComingSoon-informational)](#)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Guichou227/top-view-cattle-id/blob/main/notebooks/demo.ipynb)

<div align="center">
  <img src="./figs/pipeline.png" width="80%">
</div>

---

## 📊 1. 模型特色 | Key Features

| 功能類別 | 說明 |
|----------|------|
| ✨ **多視角辨識** | 結合俯拍與斜 45° 視角，透過 *view-aware fusion head* 提升牛隻辨識準確率 |
| ✅ **邊緣運算支援** | 支援 ONNX / TensorRT 輸出，可部署於 Jetson AGX 平台 |
| 🎨 **快速比對** | 根據毛色花紋對照圖片即可快速重建牛隻 ID (Re-mapping) |
| 🚀 **任務擴展性** | 預留行為辨識與姿勢估計支援介面 (如 mounting / lying 行為與關節點標註) |

---

## 🔬 2. 效能驗證 | Benchmark (Cattle-ID-2024 Dataset)

| 模型版本 | 輸入尺寸 | mAP<sub>ID</sub> | Top-1 Accuracy | FPS (batch=1) | 延遲 (batch=32) |
|-----------|-----------:|----------------:|----------------:|---------------:|------------------:|
| **YOLO-VCC-P5** | 640  | **93.5%**  | **95.1%** | 142 FPS | 3.1 ms |
| **YOLO-VCC-P6** | 1280 | **95.8%**  | **97.0%** | 68 FPS  | 8.4 ms |

> 訓練集: 12 個牧場 / 18 萬張影像  　測試集: 3 個牧場 / 2 萬張影像

---

## 🛠️ 3. 安裝 | Installation (Docker Recommended)

```bash
# 啟動 Docker 容器（可調整記憶體配置）
docker run --gpus all -it --shm-size=32g \
  -v $(pwd):/workspace/cattle-id \
  --name cattle-id nvcr.io/nvidia/pytorch:22.06-py3

# 安裝套件
apt update && apt install -y zip htop libgl1-mesa-glx
pip install -r requirements.txt  # torch, opencv, thop...
```

---

## 📹 4. 推論 | Inference

```bash
python detect.py \
  --weights weights/cattle-id-p5.pt \
  --source data/test/video.mp4 \
  --img 640 --conf 0.30 --view-merge
```

<div align="center">
  <img src="./figs/demo.gif" width="65%"/>
</div>

---

## 📚 5. 訓練 | Custom Training

```bash
python train.py --workers 8 --device 0 \
  --batch-size 16 --data data/cattle-id.yaml \
  --img 640 640 \
  --cfg cfg/training/cattle-id-p5.yaml \
  --weights '' --name cattle-id-p5
```

data/cattle-id.yaml 範例：
```yaml
train: /path/to/images/train
val:   /path/to/images/val
names: [cow]
```

---

## 📂 6. 模型匯出 | Export (ONNX / TensorRT)

```bash
python export.py --weights cattle-id-p5.pt \
  --grid --simplify --end2end \
  --img-size 640 640 --topk-all 200
```

---

## 📌 7. 引用 | Citation

```bibtex
@misc{Guichou2025cattleID,
  title   = {Top-View Multi-View Cattle Identification},
  author  = {Liang, Gui-Chou and others},
  year    = {2025},
  note    = {GitHub repository},
  url     = {https://github.com/Guichou227/top-view-cattle-id}
}
```

---

## ✨ 8. 致謝 | Acknowledgements

本專案建構於 YOLOv7 與 YOLO-VCC 架構基礎上，並借鑑以下開源專案：

- [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
- [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

感謝所有貢獻者！
