# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a skin improvement dispenser project (피부 개선 디스펜서) that uses camera technology to measure skin condition (moisture, elasticity) and dispense appropriate cosmetic products in the right amounts.

## Code Architecture

The project follows a three-tier architecture:

- **AI/**: AI and machine learning components for skin analysis (currently empty, prepared for future implementation)
- **HW/**: Hardware control code, specifically for Raspberry Pi
  - `HW/raspi/`: Raspberry Pi specific code for camera control and system operations
- **Server/**: Server-side components (currently empty, prepared for future implementation)

## Hardware Components

### Raspberry Pi Camera Control
- `HW/raspi/picam_picture.py`: Uses Picamera2 library to capture images
- `HW/raspi/subprocess.py`: Handles file transfer operations via SCP

## Development Commands

Since this project doesn't have package.json, requirements.txt, or other standard dependency files, development appears to be done directly with Python scripts on the Raspberry Pi.

### Running Camera Capture
```bash
cd HW/raspi
python3 picam_picture.py
```

### File Transfer Operations
```bash
cd HW/raspi  
python3 subprocess.py
```

## Key Dependencies

- **Picamera2**: Required for Raspberry Pi camera operations
- **subprocess**: Used for system operations and file transfers

## Development Notes

- The project is in early stages with placeholder directories for AI and Server components
- Hardware code is focused on Raspberry Pi camera control
- File transfers use SCP to remote servers (IP: 192.168.0.37)
- Images are captured as "test.jpg" files