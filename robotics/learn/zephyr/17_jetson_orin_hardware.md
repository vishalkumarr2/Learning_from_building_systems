# Jetson Orin — The Robot's Brain

## What Is the Jetson Orin?

NVIDIA Jetson Orin is a **System-on-Module (SoM)** designed for edge AI and robotics.
It combines a powerful CPU, NVIDIA GPU, and specialized AI accelerators into a package the size of a credit card.

"Edge AI" means running AI inference locally on the robot — not sending data to the cloud.

Product family: Orin Nano, Orin NX, AGX Orin. Different power/performance levels.

---

## Hardware Architecture

```
┌─────────────────────────────────────────────────────────────┐
│               Jetson AGX Orin (top-end module)               │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  ARM CPU     │  │  GPU         │  │  DLA (AI engine) │  │
│  │ 12-core A78  │  │  2048 CUDA   │  │  2× NVDLA 3.0   │  │
│  │ Cortex (A78) │  │  cores       │  │  (matrix ops)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────────────┐   │
│  │ 32GB LPDDR5│ │ NVMe SSD │ │ Peripheral Interfaces   │   │
│  │  (shared  │ │  storage │ │ UART/I2C/SPI/CAN/MIPI   │   │
│  │  CPU+GPU) │ │          │ │ USB3/PCIe/Ethernet       │   │
│  └──────────┘  └──────────┘  └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### SKU Comparison

| Model | CPU | GPU CUDA | DLA | RAM | Power | Use case |
|---|---|---|---|---|---|---|
| Orin Nano 4GB | 6-core A78AE | 1024 | 1× DLA | 4GB | 7-15W | Small robot, drone |
| Orin Nano 8GB | 6-core A78AE | 1024 | 1× DLA | 8GB | 7-25W | Mid robot |
| Orin NX 8GB | 6-core A78AE | 1024 | 2× DLA | 8GB | 10-25W | Advanced robot |
| Orin NX 16GB | 8-core A78AE | 1024 | 2× DLA | 16GB | 10-25W | Full autonomy |
| AGX Orin 32GB | 12-core A78AE | 2048 | 2× DLA | 32GB | 15-60W | Highest perf |
| AGX Orin 64GB | 12-core A78AE | 2048 | 2× DLA | 64GB | 15-60W | Research/industrial |

For mobile warehouse AMR (like AMR): typically **Orin NX 16GB** or **AGX Orin 32GB**.

---

## Key Components Explained

### ARM CPU — 12-core Cortex-A78AE

The A78AE is an "Automotive Enhanced" version — designed for safety-critical applications.
- 2 clusters of cores, each can run independently
- Runs Linux (Ubuntu 20.04 via JetPack)
- This is where your ROS2 nodes run

### GPU — NVIDIA Ampere architecture

2048 CUDA cores (AGX Orin) for parallel computation:
- Camera perception (object detection, semantic segmentation)
- Point cloud processing (LiDAR)
- Neural network inference (with TensorRT)
- Training is NOT done on robot — inference only

```python
# Example: running YOLOv8 at 30fps on Orin GPU
import torch

model = torch.hub.load('ultralytics/yolov8', 'yolov8n')
model = model.cuda()   # move to GPU

frame = capture_camera()
frame_gpu = torch.from_numpy(frame).cuda()
detections = model(frame_gpu)   # runs on GPU, ~30ms on Orin NX
```

### DLA — Deep Learning Accelerator

The DLA is a **fixed-function neural network accelerator** — even more efficient than the GPU for common neural network ops (conv layers, pooling):

- 2× DLA on AGX Orin/NX
- Can run two networks simultaneously (one on GPU, one on DLA)
- 5-10× better power efficiency than running on GPU
- Limited to supported operations (can't run arbitrary CUDA code)

Using TensorRT, you compile your model to run on DLA:

```python
import tensorrt as trt

# TRT builder assigns layers to DLA where possible
config.default_device_type = trt.DeviceType.DLA
config.DLA_core = 0
# Convolution, pooling → DLA; unsupported ops fall back to GPU
```

### Shared Memory — CPU+GPU in same pool

Unlike desktop PCs where CPU RAM and GPU VRAM are separate, Jetson uses **Unified Memory Architecture (UMA)**:
- CPU and GPU share the same 32GB LPDDR5
- No explicit `cudaMemcpy` needed for small tensors
- Huge win for robotics: camera frame on CPU, pass to GPU with zero copy

```python
# Desktop: must copy GPU → CPU → GPU multiple times
# Jetson: unified memory, pointer just works on both
tensor = torch.cuda.FloatTensor(1, 3, 640, 640)  # allocated once
result = model(tensor)  # GPU processes
result_np = result.cpu().numpy()  # no copy, just pointer aliasing
```

---

## Peripheral Interfaces on Jetson

Jetson Orin exposes many hardware interfaces, accessible via Linux device files:

```
/dev/spidev1.0     → SPI1 chip-select 0 (our STM32 link)
/dev/i2c-0         → I2C bus 0
/dev/ttyTHS0       → UART0 (UART High Speed = UARTTHS)
/dev/can0          → CAN bus (if MCP2518FD HAT or direct)
/dev/video0        → Camera (CSI or USB)
/dev/nvme0n1       → NVMe SSD
```

Configure pins in Jetson's device tree or via the **Jetson-IO tool**:

```bash
sudo /opt/nvidia/jetson-io/jetson-io.py
# GUI tool to enable/configure SPI, I2C, UART, CAN on 40-pin header
```

40-pin GPIO header (like Raspberry Pi):

```
Pin 19: SPI1_MOSI   ← connects to STM32 SPI_MOSI
Pin 21: SPI1_MISO   ← connects to STM32 SPI_MISO
Pin 23: SPI1_SCK    ← connects to STM32 SPI_SCK
Pin 24: SPI1_CS0    ← connects to STM32 SPI_CS
```

---

## JetPack SDK

NVIDIA's software stack for Jetson:

```
JetPack 6.x
├── L4T (Linux for Tegra) — Ubuntu 22.04 base
├── CUDA 12.x              — GPU computation
├── cuDNN                  — Neural network primitives
├── TensorRT               — Inference optimizer + runtime
├── VPI (Vision Programming Interface) — Computer vision
├── Multimedia API         — Camera, video encode/decode (NVENC/NVDEC)
└── CUDA-X libraries       — cuBLAS, cuSPARSE, etc.
```

Install:

```bash
# Flash with SDK Manager (from NVIDIA host PC)
# Or via apt on a running Jetson:
sudo apt install nvidia-jetpack
```

---

## Power Modes

Jetson has configurable power modes:

```bash
sudo nvpmodel -m 0   # MAXN mode — all cores, maximum performance
sudo nvpmodel -m 2   # 10W mode — save battery
sudo nvpmodel -q     # query current mode
sudo jetson_clocks   # lock clocks to max (prevent throttling)
```

For AMR robots with large battery: use MAXN. For battery-limited drones: use power-saving mode.

---

## PREEMPT-RT for Hard Real-Time ROS2

Standard Linux is **not real-time** — the kernel can delay your process for milliseconds (OS housekeeping, memory management, etc.). At 100Hz, every 10ms matters.

PREEMPT-RT patches the Linux kernel to be fully preemptible:

```bash
# NVidia provides PREEMPT-RT kernel for Jetson
uname -r   # check current kernel
# Example: 5.15.148-rt76-tegra  ← rt = PREEMPT-RT

# Check if RT is active
cat /sys/kernel/realtime   # 1 = RT kernel active
```

```python
# In ROS2 node: configure thread priority for real-time
import os
import ctypes

SCHED_FIFO = 1
libc = ctypes.CDLL('libc.so.6', use_errno=True)

class sched_param(ctypes.Structure):
    _fields_ = [("sched_priority", ctypes.c_int)]

param = sched_param(sched_priority=90)
libc.sched_setscheduler(0, SCHED_FIFO, ctypes.byref(param))
# Now this thread has SCHED_FIFO priority 90 — won't be preempted by other threads
```

---

## Jetson vs Raspberry Pi 5 vs Intel NUC

| | Jetson Orin NX | Raspberry Pi 5 | Intel NUC i7 |
|---|---|---|---|
| Neural network inference | ★★★★★ (GPU+DLA+TRT) | ★★ (CPU only) | ★★★ (GPU via OpenCL) |
| Power efficiency | ★★★★★ (designed for robots) | ★★★★ | ★★ (desktop chip) |
| GPIO/peripherals | ★★★★★ (UART/I2C/SPI/CAN/MIPI) | ★★★★ (UART/I2C/SPI/GPIO) | ★ (USB only, no GPIO header) |
| ROS2 support | ★★★★★ | ★★★★ | ★★★★★ |
| Camera input | ★★★★★ (MIPI CSI-2 ×6) | ★★★ (MIPI CSI-2 ×2) | ★★★ (USB only) |
| Cost | $$$$ | $ | $$$ |
| Use in warehouse robot | ✅ (current choice) | ✗ too slow | ✗ no GPIO, power hungry |

---

## Key Commands

```bash
# System info
jetson_release -v              # JetPack version, L4T version
tegrastats                     # live CPU/GPU/memory/temperature stats
sudo nvpmodel -q               # power mode
nvidia-smi                     # GPU utilization (like desktop)

# Camera
v4l2-ctl --list-devices        # list cameras
gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! autovideosink  # CSI camera preview

# AI inference benchmark
/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --fp16 --useDLACore=0

# ROS2
source /opt/ros/humble/setup.bash
ros2 node list
ros2 topic hz /imu/raw
```
