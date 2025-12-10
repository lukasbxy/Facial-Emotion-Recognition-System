# 🚀 Setup-Anleitung (Windows mit GPU-Support)

Diese Anleitung führt dich durch die komplette Einrichtung des Projekts auf einem Windows-System mit NVIDIA GPU.

---

## 📋 Voraussetzungen

- **Betriebssystem**: Windows 10/11
- **Python**: 3.10 (wird via Conda installiert)
- **GPU**: NVIDIA GPU mit CUDA-Unterstützung (empfohlen)
- **Conda**: Anaconda oder Miniconda installiert

> **Hinweis**: Das Training funktioniert auch ohne GPU auf der CPU, ist jedoch deutlich langsamer.

---

## 🛠️ Installation

### 1. Umgebung erstellen & aktivieren

Öffne dein Terminal (Anaconda Prompt oder PowerShell) und navigiere zum Projektordner:

```powershell
# Erstelle eine neue Conda-Umgebung mit Python 3.10
conda create -n emotion-model python=3.10 -y

# Aktiviere die Umgebung
conda activate emotion-model
```

### 2. PyTorch mit GPU-Support installieren

Dies ist der wichtigste Schritt für die GPU-Nutzung:

```powershell
# PyTorch mit CUDA 11.8 Support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

> **CPU-Only Alternative**: Falls du keine NVIDIA GPU hast:
> ```powershell
> conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
> ```

### 3. Restliche Bibliotheken installieren

Installiere die Python-Dependencies für Dashboard, GUI und Datenverarbeitung:

```powershell
pip install -r requirements.txt
```

### 4. GPU-Installation verifizieren (optional)

Überprüfe, ob PyTorch deine GPU erkennt:

```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Erwartete Ausgabe (beispielhaft):
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3080
```

---

## 🎯 Programm starten

### Training starten

```powershell
python src/train.py
```

Das Skript bietet eine interaktive Auswahl zwischen CPU und GPU.

**Mit automatischem Batch-Size Tuning:**
```powershell
python src/train.py --auto-tune
```

### Live Inference GUI starten

Die GUI ermöglicht Echtzeit-Gesichtserkennung mit deiner Webcam:

```powershell
python Live_Inference_GUI/gui_app.py
```

### Modellgröße prüfen

Um die Anzahl der Parameter des aktuellen Modells zu sehen:

```powershell
python count_params.py
```

---

## 📁 Datensatz einrichten

1. Erstelle den Ordner `archive/data/` im Projektverzeichnis
2. Organisiere deine Bilder nach Klassen:
   ```
   archive/
   └── data/
       ├── Angry/
       ├── Disgust/
       ├── Fear/
       ├── Happy/
       ├── Neutral/
       ├── Sad/
       └── Surprise/
   ```
3. **Optional**: Erstelle einen Test-Holdout-Datensatz:
   ```powershell
   python src/split_data.py
   ```

---

## ⚙️ Konfiguration anpassen

Die Hauptkonfiguration befindet sich in `configs/config.yaml`:

```yaml
# Wichtige Parameter
data:
  batch_size: 32        # Erhöhen für mehr GPU-Speicher
  image_size: 64        # Bildgröße (64x64 oder 48x48 üblich)

model:
  name: "improved_cnn"  # Optionen: resnet18, convnext_v2_nano, improved_cnn
  
training:
  epochs: 50            # Anzahl Trainingsepochen
  learning_rate: 4.0e-3 # Lernrate
```

---

## 🔧 Fehlerbehebung

### `conda` wird nicht erkannt

1. Öffne Anaconda Prompt statt PowerShell
2. Oder führe folgendes aus:
   ```powershell
   conda init powershell
   ```
   Dann PowerShell neu starten.

### GPU wird nicht erkannt

1. Prüfe NVIDIA-Treiber: `nvidia-smi`
2. Stelle sicher, dass PyTorch mit CUDA installiert wurde
3. Neuinstallation:
   ```powershell
   conda uninstall pytorch torchvision torchaudio
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
   ```

### Out of Memory (OOM) beim Training

Reduziere die `batch_size` in `configs/config.yaml` (z.B. auf 16 oder 8).

---

## 📊 Dashboard

Beim Training wird automatisch ein Web-Dashboard geöffnet unter:
```
http://localhost:8000/dashboard/index.html
```

Dort kannst du in Echtzeit folgende Metriken verfolgen:
- Loss (Training & Validation)
- Accuracy
- Precision, Recall, F1-Score
- Learning Rate

---

## 💡 Quick Reference

| Aktion | Befehl |
|--------|--------|
| Umgebung aktivieren | `conda activate emotion-model` |
| Training starten | `python src/train.py` |
| Auto-Tuning Training | `python src/train.py --auto-tune` |
| Live GUI starten | `python Live_Inference_GUI/gui_app.py` |
| Parameter zählen | `python count_params.py` |
| Test-Split erstellen | `python src/split_data.py` |

---

*Bei Fragen oder Problemen: Erstelle ein [Issue im GitHub Repository](https://github.com/lukasbxy/Facial-Emotion-Recognition-System/issues).*

