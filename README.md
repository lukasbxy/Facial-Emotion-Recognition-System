# 🎭 Facial Emotion Recognition System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A robust deep learning system for **Facial Emotion Recognition (FER)** built with PyTorch. Features a custom training pipeline, real-time webcam inference with Grad-CAM visualization, and an interactive training dashboard.

<p align="center">
  <img src="https://img.shields.io/badge/GPU-Accelerated-brightgreen" alt="GPU Accelerated"/>
  <img src="https://img.shields.io/badge/Platform-Windows-blue" alt="Windows"/>
</p>

---

## 📌 Overview

This project provides a complete deep learning workflow for emotion classification from facial images:

- **Training Pipeline**: Full-featured training with mixed precision (AMP), learning rate scheduling, and checkpoint management.
- **Live Inference**: Modern GUI application for real-time emotion detection using your webcam.
- **Explainability**: Grad-CAM heatmaps to visualize what the model focuses on.
- **Monitoring**: Custom web dashboard for tracking training metrics in real-time.

---

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Support** | ResNet18, ConvNeXt V2 Nano, ImprovedCNN |
| **Live Inference GUI** | Tkinter-based real-time detection with face tracking |
| **Grad-CAM Visualization** | See *where* the model is looking |
| **Training Dashboard** | Browser-based live metrics visualization |
| **Data Augmentation** | Mixup, Cutmix, TrivialAugment for robust training |
| **Auto-Tuning** | Automatic batch size and worker optimization |
| **Checkpoint Resume** | Seamlessly resume interrupted training |

---

## 📚 Model Architectures

### 1. ImprovedCNN (Recommended for Speed)
A custom 4-layer Sequential ConvNet architecture optimized for real-time FER, based on:
- *"Four-layer ConvNet to facial emotion recognition with minimal epochs..."*
- *"An improved facial emotion recognition system using CNN for human-robot interaction"*

**Architecture:**
```
Conv → ReLU → BatchNorm → MaxPool → Dropout (×4 blocks)
     → Adaptive Pooling → FC → Softmax
```
- **Parameters**: ~2.3M
- **Inference**: <5ms per face (GPU)

### 2. ResNet18
Standard transfer learning baseline with ImageNet pretrained weights.
- **Parameters**: ~11M
- **Best for**: High accuracy when training time is not critical

### 3. ConvNeXt V2 Nano
Modern architecture with GRN and improved training stability.
- **Parameters**: ~15M
- **Best for**: State-of-the-art performance

---

## 🛠️ Quick Start

### Prerequisites
- Windows 10/11
- Python 3.10+
- NVIDIA GPU (recommended)
- Conda (Anaconda/Miniconda)

### Installation

```powershell
# 1. Create environment
conda create -n emotion-model python=3.10 -y
conda activate emotion-model

# 2. Install PyTorch with GPU support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. Install dependencies
pip install -r requirements.txt
```

> 📖 For detailed instructions, see [SETUP.md](SETUP.md)

### Usage

```powershell
# Train the model
python src/train.py

# Launch Live Inference GUI
python Live_Inference_GUI/gui_app.py

# Check model parameters
python count_params.py
```

---

## 📂 Project Structure

```
📦 Facial-Emotion-Recognition/
├── 📁 configs/              # YAML configuration files
│   └── config.yaml          # Main training configuration
├── 📁 dashboard/            # Web-based training dashboard
│   └── index.html           # Dashboard frontend
├── 📁 Live_Inference_GUI/   # Real-time inference application
│   ├── gui_app.py           # Main GUI application
│   ├── inference.py         # Model wrapper & Grad-CAM
│   └── evaluator.py         # Batch evaluation module
├── 📁 src/                  # Core source code
│   ├── dataset.py           # Data loading & augmentation
│   ├── model.py             # Model architectures
│   ├── train.py             # Main training script
│   ├── evaluate.py          # Standalone evaluation script
│   ├── json_logger.py       # Dashboard metrics logger
│   ├── tuner.py             # Hyperparameter auto-tuning
│   ├── utils.py             # Metrics & visualization utilities
│   └── split_data.py        # Test set splitting utility
├── 📁 Docs/                 # Reference documentation (PDFs)
├── 📁 checkpoints/          # Saved model weights (git-ignored)
├── 📁 runs/                 # Training logs (git-ignored)
├── 📁 archive/              # Dataset storage (git-ignored)
│   ├── data/                # Training data
│   └── test_holdout/        # Holdout test set
├── .gitignore               # Git ignore rules
├── count_params.py          # Model parameter counter
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── SETUP.md                 # Detailed setup instructions
```

---

## ⚙️ Configuration

All training parameters are configured in `configs/config.yaml`:

```yaml
data:
  data_dir: "archive/data"
  batch_size: 32
  image_size: 64
  validation_split: 0.2

model:
  name: "improved_cnn"  # resnet18, convnext_v2_nano, improved_cnn
  num_classes: 7
  pretrained: false

augmentation:
  trivial_augment: true
  mixup_alpha: 0.8
  cutmix_alpha: 1.0
  label_smoothing: 0.1

training:
  epochs: 50
  learning_rate: 4.0e-3
  weight_decay: 0.05
  warmup_epochs: 20
```

---

## 📊 Training Dashboard

The training script automatically launches a web-based dashboard at `http://localhost:8000/dashboard/index.html`.

**Metrics tracked:**
- Loss (Train/Validation)
- Accuracy
- Precision, Recall, F1-Score
- Learning Rate
- Throughput (samples/sec)

---

## 🖼️ Live Inference Features

The GUI application (`Live_Inference_GUI/gui_app.py`) provides:

- **Real-time face detection** using Haar Cascades
- **Emotion classification** with confidence scores
- **Grad-CAM heatmaps** showing model attention
- **Adjustable inference frequency** (every frame, 5 frames, 10 frames, 1 second)
- **Model switching** between saved checkpoints
- **Batch evaluation** on holdout test sets

---

## 🔬 Supported Emotions

The standard FER-2013 emotion classes:

| Class | Emoji |
|-------|-------|
| Angry | 😠 |
| Disgust | 🤢 |
| Fear | 😨 |
| Happy | 😊 |
| Neutral | 😐 |
| Sad | 😢 |
| Surprise | 😲 |

---

## 📈 Results

Expected performance on FER-2013 with `ImprovedCNN`:

| Metric | Value |
|--------|-------|
| Validation Accuracy | ~68-72% |
| Inference Time (GPU) | <5ms |
| Model Size | ~9 MB |

*Note: Results may vary based on training configuration and data quality.*

---

## 📝 Changelog

| Date | Changes |
|------|---------|
| **2025-12-10** | Project cleanup for GitHub release. Added SETUP.md, .gitignore. |
| **2025-12-10** | Implemented `ImprovedCNN` architecture based on research papers. |
| **2025-11-28** | Added Live Inference GUI with Grad-CAM visualization. |
| **2025-11-27** | Added auto-tuning for batch size and num workers. |
| **Initial** | Basic ResNet18 training pipeline with dashboard. |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📚 References

- Deepface: A Lightweight Face Recognition and Facial Attribute Analysis Framework
- FER-2013 Dataset
- Grad-CAM: Visual Explanations from Deep Networks
- ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders

---

## 👤 Author

**Lukas Boguth**
- GitHub: [@lukasbxy](https://github.com/lukasbxy)

---

<p align="center">
  Made with ❤️ for Computer Vision research
</p>

