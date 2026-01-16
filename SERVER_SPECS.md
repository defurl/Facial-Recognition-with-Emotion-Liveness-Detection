# Server Specifications

This document outlines the hardware and software specifications of the backend server used for the Facial Recognition with Emotion & Liveness Detection project.

## Hardware

### CPU
- **Model:** QEMU Virtual CPU version 2.5+
- **Architecture:** x86_64
- **Cores:** 40 vCPUs
- **Topology:** 4 Sockets, 10 Cores per socket, 1 Thread per core
- **Vendor:** GenuineIntel
- **Hypervisor:** KVM

### GPU
The server is equipped with dual high-end consumer GPUs suitable for deep learning inference and training.

- **GPU 0:** NVIDIA GeForce RTX 4090
  - **VRAM:** 24 GB GDDR6X
  - **Power Limit:** 450W
- **GPU 1:** NVIDIA GeForce RTX 4090
  - **VRAM:** 24 GB GDDR6X
  - **Power Limit:** 450W

**Total VRAM:** 48 GB

### Memory (RAM)
- **Total:** ~98 GiB
- **Available:** ~87 GiB (at time of check)

### Storage
- **Root Filesystem:** ~688 GB Total
- **Status:** **CRITICAL WARNING** - The root filesystem is reporting 100% usage (647GB used / 5.7GB avail). Immediate cleanup is recommended.

## Software Environment

### Operating System
- **Distro:** Ubuntu 22.04.5 LTS (Jammy Jellyfish)
- **Kernel:** Linux (Generic)

### Drivers & Libraries
- **NVIDIA Driver:** 555.42.06
- **CUDA Version:** 12.5

## Notes for Web Backend
- The 40 vCPUs providing ample parallelism for handling multiple concurrent API requests (Flask/FastAPI workers).
- The Dual RTX 4090s allow for high-throughput batch processing of facial recognition and emotion detection models.
- **Action Item:** Free up disk space on `/` immediately to prevent service interruptions.
