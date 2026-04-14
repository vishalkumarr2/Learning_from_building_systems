# Zephyr RTOS — Learning Notes

Documented from a learning session on April 13, 2026.

## Topic: STM32 → Jetson Orin sensor pipeline at 100Hz

A real-world embedded architecture where:
- **STM32** runs Zephyr RTOS, reads sensors (IMU/CAN/I2C), packs data with nanopb, sends over SPI
- **Jetson Orin** runs Linux + ROS2, receives SPI frames, deserializes, publishes at 100Hz

---

## Documents in this folder

### Foundation (start here)

| File | What it covers |
|---|---|
| [01_zephyr_intro.md](01_zephyr_intro.md) | What Zephyr is, key features, west build system, prj.conf + app.overlay, comparison with RTOSes |
| [02_pipeline_architecture.md](02_pipeline_architecture.md) | Full STM32→Jetson pipeline overview, data flow diagram, 10ms timing budget, design rationale |

### STM32 Zephyr middleware

| File | What it covers |
|---|---|
| [03_zbus.md](03_zbus.md) | ZBus message bus — channels, publishers, subscribers, listeners, pub/sub internals |
| [04_nanopb.md](04_nanopb.md) | nanopb / Protocol Buffers — wire format, code generation, pb_encode, Python decode |
| [12_zephyr_threads.md](12_zephyr_threads.md) | Threads — K_THREAD_DEFINE, priorities, preemption, stack sizing, lifecycle |
| [13_zephyr_kernel_services.md](13_zephyr_kernel_services.md) | Sync primitives — k_sem, k_mutex, k_msgq, k_timer, k_event, k_work |
| [14_zephyr_devicetree.md](14_zephyr_devicetree.md) | Devicetree + Kconfig — app.overlay syntax, DT macros in C, bindings YAML, prj.conf |
| [15_zephyr_interrupts_gpio.md](15_zephyr_interrupts_gpio.md) | Interrupts + GPIO — ISR rules, gpio_dt_spec, output/input/interrupt, IMU DRDY pattern |

### Communication protocols

| File | What it covers |
|---|---|
| [08_spi_protocol.md](08_spi_protocol.md) | SPI protocol — 4 wires, 4 CPOL/CPHA modes, multi-slave, Zephyr master + slave API |
| [05_spi_slave_dma.md](05_spi_slave_dma.md) | SPI slave on STM32 — DMA, double buffering, CS interrupt, framing protocol |
| [06_dma.md](06_dma.md) | DMA deep dive — modes, streams, interrupts, circular, double buffer, cache coherency bugs |
| [09_i2c.md](09_i2c.md) | I2C protocol — 2-wire addressing, ACK/NACK, burst read, Zephyr API, ICM-42688 example |
| [10_can_bus.md](10_can_bus.md) | CAN bus — differential signaling, arbitration, frame format, Zephyr API, error handling |
| [11_uart.md](11_uart.md) | UART — async serial, baud rate, ring buffer pattern, GPS NMEA parsing, debug console |

### Jetson / ROS2 side

| File | What it covers |
|---|---|
| [16_ros2_fundamentals.md](16_ros2_fundamentals.md) | ROS2 — nodes, topics, services, actions, QoS, launch files, colcon build, debug tools |
| [07_jetson_ros2_bridge.md](07_jetson_ros2_bridge.md) | Jetson ROS2 node — spidev, 100Hz timer, nanopb decode, sync recovery, error handling |
| [17_jetson_orin_hardware.md](17_jetson_orin_hardware.md) | Jetson Orin hardware — GPU, DLA, UMA, SPI/GPIO interfaces, JetPack, power modes |

---

## Learning path (recommended order)

**Full system understanding:**
```
01 → 02 → 08 → 05 → 06 → 09 → 10 → 11 → 03 → 04 → 12 → 13 → 14 → 15 → 16 → 07 → 17
```

**Just the embedded side (STM32):**
```
01 → 14 → 12 → 13 → 15 → 08 → 05 → 06 → 09 → 10 → 11 → 03 → 04
```

**Just the Jetson/ROS2 side:**
```
16 → 17 → 07
```

**Just DMA / SPI:** go straight to **06** then **05**.

---

## Key mental models

- **ZBus** = typed thread-safe mailbox between threads on STM32
- **nanopb** = compact serialization that survives firmware version mismatches
- **DMA** = a DMA controller moves data while the CPU does other things
- **Double buffer** = two buckets — fill one while emptying the other, no gaps ever
- **SPI master on Jetson** = Jetson drives the clock → deterministic 100Hz on ROS2 side
- **I2C** = 2-wire addressed bus, up to 127 sensors, simpler but slower than SPI
- **CAN** = differential bus, long cables, built-in arbitration — for motors and encoders
- **Preemptive threads** = higher-priority thread interrupts lower-priority at any time
- **ISR rule** = do minimum in ISR (give semaphore), do real work in a thread
- **Devicetree** = hardware description in `.dts`, code never has pin numbers hardcoded
