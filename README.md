# Industrial Eye - ML & Hardware

Industrial sensor monitoring system with machine learning anomaly detection for sound, temperature, and vibration analysis.

---

## 🎯 Overview

Industrial Eye is an IoT-based predictive maintenance system that monitors industrial equipment using:
- **Sound Analysis** - Detects acoustic anomalies using MFCC and spectral features
- **Temperature Monitoring** - Tracks thermal patterns and rapid changes
- **Vibration Detection** - Analyzes 3-axis accelerometer data for mechanical issues

---

## 🏗️ System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Arduino Nano   │───▶│    Firebase      │───▶│   ML Models     │
│   RP2040        │     │  Realtime DB    │     │  (Autoencoder)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   Sensor Data            Feature Storage         Anomaly Detection
   - Microphone           - Sound Features        - TFLite Models
   - IMU (Accel+Temp)     - Temp Features         - Supabase Storage
                          - Vibration Features
```

---

## 🔧 Hardware Specifications

### Microcontroller
| Component | Specification |
|-----------|---------------|
| Board | Arduino Nano RP 2040 |
| IMU Sensor | LSM6DSOX (Accelerometer + Temperature) |
| Microphone | PDM (Pulse Density Modulation) |

### Network Configuration
| Parameter | Value |
|-----------|-------|
| WiFi Module | WiFiNINA |
| Protocol | HTTPS (SSL/TLS port 443) |
| NTP Server | time.google.com |
| Timezone | UTC+2 (7200 seconds offset) |

### Microphone Specifications
| Parameter | Value |
|-----------|-------|
| Sample Rate | 16000 Hz (16 kHz) |
| Buffer Size | 1024 samples |
| Collection window | 64 ms |
| Channels | 1 (Mono) |
| Data Type | 16-bit signed integer |
| FFT Size | 256 samples |
| Analysis window (FFT) | 16 m |

### Timing & Performance
| Parameter | Value |
|-----------|-------|
| Main Loop Interval | 1000 ms |
| Serial Baud Rate | 115200 |
| HTTP Timeout | 500 ms |
| WiFi Timeout | 10 seconds |

---

## 🤖 ML Models

### Common Architecture
All three models use **Autoencoder** architecture for unsupervised anomaly detection:
- Framework: TensorFlow
- Output: TensorFlow Lite (.tflite)
- Detection Method: Reconstruction Error
- Threshold: 95th percentile of training MSE

### Sound Model

| Parameter | Value |
|-----------|-------|
| Input Features | 8 |
| Hidden Layer | 6 units |
| Bottleneck | 3 units |
| Dropout Rate | 10% |


**Input Features:**
1. `audioMFCC1` - Mel-Frequency Cepstral Coefficient 1
2. `audioMFCC2` - Mel-Frequency Cepstral Coefficient 2
3. `audioMFCC3` - Mel-Frequency Cepstral Coefficient 3
4. `audioMFCC4` - Mel-Frequency Cepstral Coefficient 4
5. `audioPeak` - Maximum amplitude
6. `audioRMS` - Root Mean Square
7. `audioSpectralCentroid` - Center of mass of spectrum
8. `audioZCR` - Zero Crossing Rate

### Temperature Model

| Parameter | Value |
|-----------|-------|
| Input Features | 4 |
| Hidden Layer | 3 units |
| Bottleneck | 2 units |
| Dropout Rate | 15% |


**Input Features:**
1. `tempCurrent` - Current temperature (°C)
2. `tempDerivative` - Rate of change
3. `tempMovingAverage` - Average over history
4. `tempStdDev` - Standard deviation

### Vibration Model

| Parameter | Value |
|-----------|-------|
| Input Features | 5 |
| Hidden Layer | 4 units |
| Bottleneck | 2 units |
| Dropout Rate | 10% |


**Input Features:**
1. `vibMagnitude` - 3D Euclidean magnitude
2. `vibRMS_X` - X-axis acceleration
3. `vibRMS_Y` - Y-axis acceleration
4. `vibRMS_Z` - Z-axis acceleration
5. `vibStdDev` - Standard deviation

### Training Configuration (Default)

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Batch Size | 32 |
| Validation Split | 15% |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | MSE |
| Data Augmentation | Gaussian noise (1%) |
| Early Stopping | Configurable |

---

## 🔥 Firebase Configuration

### Project Details
| Parameter | Value |
|-----------|-------|
| Project ID | graduation2-75f83 |
| Database | Realtime Database |

### Data Endpoints
```
/soundSensorDataFeatures.json    - Audio features
/tempSensorDataFeatures.json     - Temperature features
/vibrationSensorDataFeatures.json - Vibration features
```

### Firestore Config Paths
```
sound/config      - Sound model configuration
temp/config       - Temperature model configuration
vibration/config  - Vibration model configuration
learn_phase/data  - Training data path configuration
sound/data        - Sound training trigger & time
temp/data         - Temp training trigger & time
vibration/data    - Vibration training trigger & time
```

### User Configurable Parameters (via Mobile App)

These parameters are configured by the user through the mobile application interface and stored in Firebase:

| Parameter | Options | Description |
|-----------|---------|-------------|
| Training Time | 5, 15, 30, 60, 120 minutes | Duration for data collection before training |
| Threshold | 90%, 93%, 95%, 97%, 99% | Anomaly detection sensitivity |
| Epochs | 50, 100, 150, 200 | Number of training iterations |
| Batch Size | 4, 8, 16, 32, 64 | Samples per training batch |
| use_data_augmentation | True, False | Enable Gaussian noise augmentation |
| use_early_stopping | True, False | Stop training when no improvement |

### Advanced Parameters (via Firebase Console)

These parameters can be modified directly in Firebase Firestore for fine-tuning:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bottleneck_dim` | number | 2-3 | Autoencoder compression layer size |
| `hidden_dim` | number | 3-6 | Hidden layer neurons |
| `dropout_rate` | number | 0.1-0.15 | Regularization dropout percentage |
| `learning_rate` | number | 0.001 | Adam optimizer learning rate |
| `validation_split` | number | 0.15 | Data split for validation |
| `use_data_augmentation` | Boolean | true | Enable Gaussian noise augmentation |
| `augmentation_noise` | number | 0.01 | Noise level for augmentation |
| `use_early_stopping` | Boolean | true | Stop training when no improvement |
| `early_stopping_patience` | number | 10 | Epochs to wait before stopping |

### Firebase Document Structure

**Training Trigger (`{sensor}/data`):**
```json
{
  "learn": true,
  "time": 15
}
```

**Model Config (`{sensor}/config`):**
```json
{
  "epochs": 100,
  "batch_size": 32,
  "threshold_percentile": 95,
  "bottleneck_dim": 3,
  "hidden_dim": 6,
  "dropout_rate": 0.1,
  "learning_rate": 0.001,
  "use_data_augmentation": true,
  "use_early_stopping": false,
  "early_stopping_patience": 10
}
```

**Data Path (`learn_phase/data`):**
```json
{
  "sound_name": "sound_data",
  "temp_name": "temp_data",
  "vibration_name": "vibration_data",
  "phase_name": "phase_1"
}
```

---

## 🚀 Deployment

### Hugging Face Spaces
Each model has a Gradio-based web interface:
- Monitors Firestore for `learn=true` trigger
- Countdown timer before training starts
- Real-time status updates
- Manual training with custom parameters

### Kaggle Notebooks
Direct execution scripts that:
1. Load config from Firebase
2. Fetch data path from `learn_phase/data`
3. Train model and generate visualizations
4. Upload results to Supabase

### Supabase Storage

| Bucket | Purpose |
|--------|---------|
| `sound_learn` | Sound model (Kaggle) |
| `temp_learn` | Temp model (Kaggle) |
| `vibration_learn` | Vibration model (Kaggle) |
| `sound` | Sound model (Hugging Face) |
| `temp` | Temp model (Hugging Face) |
| `vibration` | Vibration model (Hugging Face) |

### Output Files
Each model generates:
- `{model}_anomaly_detector.tflite` - Optimized model
- `{model}_params.json` - Scaler parameters & metrics
- `visualizations/` - Training plots (ROC, Confusion Matrix, etc.)

---

## 📊 Visualizations Generated

1. **Training History** - Loss curves over epochs
2. **Error Distribution** - Histogram & box plot of reconstruction errors
3. **Error Timeline** - Reconstruction error over samples
4. **ROC Curve** - Receiver Operating Characteristic
5. **Confusion Matrix** - Classification performance
6. **Reconstruction Comparison** - Original vs reconstructed features
7. **Feature Analysis** - Error heatmap & correlation
8. **Model Summary** - Architecture & metrics overview

---

## 📁 Repository Structure

```
industrial_eye_ML-HW/
├── HARDWARE/
│   ├── hardware_code_modified.txt
├── Sound_Model/
│   ├── sound_model.py
│   └── huggingface_sound/
│       ├── app.py
│       └── requirements.txt
├── Temp_Model/
│   ├── temp_model.py
│   └── huggingface_temp/
│       ├── app.py
│       └── requirements.txt
├── Vibration_Model/
│   ├── vibration_model.py
│   └── huggingface_vibration/
│       ├── app.py
│       └── requirements.txt
└── README.md
```

Industrial Eye - Graduation Project 2
