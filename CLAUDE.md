# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a skin improvement dispenser project (피부 개선 디스펜서) that uses camera technology to measure skin condition (moisture, elasticity, pigmentation, pore analysis) and dispense appropriate cosmetic products in the right amounts.

## Code Architecture

The project follows a multi-tier architecture with distinct components:

- **AI/**: Machine learning models for skin analysis using PyTorch and ResNet-50 backbone
- **Server/**: Flask-based server components for image processing and hardware communication
- **Qt/**: C++ Qt GUI application for user interface
- **HW/**: Hardware control code (placeholder for Raspberry Pi integration)

## Core Components

### AI Models (`AI/`)
- `model.py`: Core PyTorch model using ResNet-50 for skin feature prediction
- `train_roi_detector.py`: Training scripts for region-of-interest detection
- `predict_all_features.py`: Inference pipeline for skin feature analysis
- `create_coco_dataset.py`: COCO dataset creation utilities
- Uses personalized embeddings and multi-feature prediction (moisture, elasticity, pigmentation, pore)

### Server Architecture (`Server/`)
- `node.py`: Main Flask server for receiving images from Qt client
- `rasp.py`: Raspberry Pi communication server with UART serial interface
- Communication flow: Qt → node.py → rasp.py → Hardware
- Handles JSON data with skin analysis results: forehead, left/right cheek, chin, lip regions

### Qt GUI (`Qt/camera_Qt/`)
- C++ Qt application with camera integration
- Communicates with Flask servers via HTTP POST requests
- Handles image capture and display of analysis results

## Development Commands

### Server Development
```bash
# Run main image processing server
cd Server
python3 node.py

# Run Raspberry Pi communication server  
cd Server
python3 rasp.py
```

### AI Model Training
```bash
# Train ROI detection model
cd AI
python3 train_roi_detector.py

# Run feature prediction
cd AI  
python3 predict_all_features.py
```

### Qt Application Build
```bash
cd Qt/camera_Qt
qmake camera_Qt.pro
make
```

### Docker Deployment
```bash
# Build Docker image for AI training
docker build -t skin-analyzer .

# Run AI training in container
docker run --gpus all skin-analyzer
```

## Key Dependencies

### Python Dependencies
- **PyTorch**: Deep learning framework for AI models
- **Flask**: Web framework for server components
- **Pillow**: Image processing
- **pyserial**: UART communication with hardware
- **requests**: HTTP communication between servers

### System Dependencies  
- **Qt5/Qt6**: For GUI application
- **CUDA**: For GPU-accelerated AI training
- **Docker**: For containerized deployment

## Network Architecture

The system uses multiple network endpoints:
- Qt client uploads images to `node.py` at port 5000 (`/upload`)
- `node.py` forwards analysis results to `rasp.py` at port 5000 (`/receive`)
- `rasp.py` communicates with hardware via UART serial interface
- IP configuration: Server at 192.168.0.90, Raspberry Pi communication enabled

## Data Flow

1. Qt GUI captures image and uploads via HTTP POST
2. Server processes image and runs AI analysis
3. Analysis results (14 metrics across 5 facial regions) sent to Raspberry Pi
4. Raspberry Pi controls dispensing hardware via UART protocol