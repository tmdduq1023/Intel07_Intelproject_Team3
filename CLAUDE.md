# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a skin improvement dispenser project (피부 개선 디스펜서) that uses camera technology to measure skin condition (moisture, elasticity, pigmentation, pore analysis) and dispense appropriate cosmetic products in the right amounts.

## Development Commands

### Server Development
```bash
# Activate virtual environment
cd Server
source venv/bin/activate

# Run main image processing server (receives images from Qt)
python3 node.py

# Run Raspberry Pi communication server (sends data to hardware)
python3 rasp.py
```

### AI Model Training and Inference
```bash
cd AI

# Train ROI detection models
python3 train_roi_detector.py
python3 train_skin_roi_v4.py

# Run inference on skin features
python3 predict_all_features.py
python3 infer_roi.py

# Evaluate models
python3 evaluate_skin_roi.py
python3 evaluation.py

# Dataset utilities
python3 create_coco_dataset.py
python3 prepare_roi_dataset.py
```

### Qt Application Build
```bash
cd Qt/camera_Qt
qmake camera_Qt.pro
make

# Run the application
./camera_Qt

# Clean build
make clean
qmake camera_Qt.pro
make
```

### Docker Deployment
```bash
# Build Docker image for AI training (PyTorch 2.0.1 with CUDA 11.8)
docker build -t skin-analyzer .

# Run AI training in container with GPU support
docker run --gpus all skin-analyzer
```

## Architecture Overview

The system implements a multi-tier distributed architecture with four main components:

### 1. Qt GUI Client (`Qt/camera_Qt/`)
- **Framework**: C++17 with Qt5/Qt6 (core, gui, widgets, multimedia, network, sql)
- **Components**: Camera capture, user interface, SQLite database integration, HTTP client
- **Files**: `mainwindow.cpp`, `databasemanager.cpp`, `analysisresultdialog.cpp`, `nameinputdialog.cpp`
- **Communication**: HTTP POST requests to Flask servers

### 2. Image Processing Server (`Server/node.py`)
- **Framework**: Flask web server
- **Purpose**: Receives images from Qt client, processes with AI models, forwards results
- **Endpoints**: `/upload` (receives image files)
- **Output**: Sends JSON analysis results to Raspberry Pi server

### 3. Hardware Communication Server (`Server/rasp.py`)
- **Framework**: Flask web server with UART serial communication
- **Purpose**: Receives analysis results and controls dispensing hardware
- **Endpoints**: `/receive` (receives JSON analysis data)
- **Hardware**: UART communication via `/dev/serial0` at 9600 baud
- **Protocol**: Sends 14 metrics as "@"-delimited string to hardware

### 4. AI Models (`AI/`)
- **Framework**: PyTorch with ResNet-50 backbone
- **Core Model**: `PersonalizedSkinModel` with person embeddings for personalized analysis
- **Features**: Multi-region skin analysis (forehead, left/right cheek, chin, lip)
- **Metrics**: Moisture, elasticity, pigmentation, pore analysis (14 total values)
- **Training**: Multiple model variants (`v2`, `v3`, `v4`) with separate models for different features

## Data Flow Architecture

```
Qt Client → [HTTP POST /upload] → node.py → [AI Processing] → [HTTP POST /receive] → rasp.py → [UART] → Hardware
```

1. **Image Capture**: Qt GUI captures facial image via camera
2. **Upload**: Image uploaded to `node.py` Flask server at port 5000
3. **AI Analysis**: Server processes image using PyTorch models for skin feature detection
4. **Result Forwarding**: Analysis results (JSON) sent to `rasp.py` at 192.168.0.90:5000
5. **Hardware Control**: `rasp.py` converts JSON to UART protocol and controls dispensing hardware

## Network Configuration

- **Qt Client**: Connects to image processing server
- **Image Server**: `node.py` on port 5000, endpoint `/upload`
- **Hardware Server**: `rasp.py` on 192.168.0.90:5000, endpoint `/receive`
- **UART**: Serial communication at `/dev/serial0`, 9600 baud rate
- **Protocol**: 14 numerical values separated by "@" symbols

## AI Model Architecture

### Core Models
- **PersonalizedSkinModel**: ResNet-50 backbone with person embeddings (32-dim)
- **Dataset**: PersonDataset with JSON profiles and image paths
- **Features**: Multi-feature prediction across 5 facial regions
- **Training Scripts**: Multiple versions (v2, v3, v4) with different architectures

### Model Variants
- **ROI Detection**: `train_roi_detector.py`, `train_roi_detector_ssdlite.py`
- **Skin Analysis**: `train_skin_roi_v4.py` (latest), `train_skin_roi_v3.py`, `train_skin_roi_v2_1.py`
- **Personalized Models**: `train_personalized.py`, `evaluate_personalized.py`
- **Separate Models**: `train_separate_models.py` for individual feature prediction

## Development Environment

### Python Environment
- **Python Version**: 3.10.12
- **Virtual Environment**: `Server/venv/` (pre-configured with Flask, requests, pyserial)
- **AI Dependencies**: PyTorch 2.0.1, torchvision, Pillow, numpy, tqdm

### Qt Environment  
- **Qt Version**: Qt5/Qt6 compatible
- **C++ Standard**: C++17
- **Modules**: core, gui, widgets, multimedia, multimediawidgets, network, sql
- **Build System**: qmake + make

### Hardware Requirements
- **GPU**: CUDA 11.8 support for AI training
- **Serial Port**: `/dev/serial0` for UART communication (Raspberry Pi)
- **Camera**: Compatible with Qt multimedia framework

## Configuration Files

### Server Configuration
- **Qt Client Config**: `Qt/camera_Qt/config.ini` - Contains server URLs and endpoints
- **Default URLs**: 
  - Image processing server: `http://192.168.0.90:5000/upload`
  - Raspberry Pi server: `http://192.168.0.90:5000/receive`
- **UART Settings**: 9600 baud rate, `/dev/serial0` device path

### Qt Project Structure
- **Main Files**: `mainwindow.cpp`, `databasemanager.cpp`, `analysisresultdialog.cpp`, `nameinputdialog.cpp`
- **Project Config**: `camera_Qt.pro` - Defines Qt modules, C++17 standard, source/header files
- **Database**: SQLite integration for user profiles and analysis results
- **Networking**: HTTP client for Flask server communication

## Debugging and Development

### Server Debugging
```bash
# Check if servers are running
ps aux | grep python

# Monitor server logs
cd Server
source venv/bin/activate
python3 node.py  # Check terminal output for image processing
python3 rasp.py  # Check terminal output for hardware communication

# Test server endpoints manually
curl -X POST http://localhost:5000/upload -F "image=@test.jpg"
curl -X POST http://192.168.0.90:5000/receive -H "Content-Type: application/json" -d '{"test":"data"}'
```

### Qt Application Debugging
```bash
# Check Qt application logs
cd Qt/camera_Qt
./camera_Qt  # Check terminal output for debug messages

# Verify camera access
v4l2-ctl --list-devices  # Linux camera detection
```

### Network Troubleshooting
- **Port Conflicts**: Ensure ports 5000 are available on both servers
- **Network Connectivity**: Verify Raspberry Pi IP address (192.168.0.90) is accessible
- **UART Permissions**: Check `/dev/serial0` permissions on Raspberry Pi
- **Firewall**: Ensure Flask servers can accept connections