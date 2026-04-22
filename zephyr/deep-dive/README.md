# Zephyr Deep-Dive — Folder Overview
### Hardware-gated content. Add files here as hardware arrives.

---

## Status: 🟡 Blocked on hardware

Per `zephyr/00-mastery-plan.md` Section 1, you need:
- STM32 Nucleo-H743ZI2 (~$60)
- Saleae Logic 8 logic analyzer (~$160)
- ICM-42688-P IMU breakout (~$25)

Until then: keep working through `zephyr/study-notes/` and `zephyr/exercises/` — the theory
will make the hands-on sessions much faster when hardware arrives.

---

## Session Files (11 hardware-gated sessions plus 2 reference guides)

| File | Status | What It Covers |
|------|--------|----------------|
| `01-first-build-flash-debug.md` | 🟡 HW-gated | Blinky → Zephyr shell → GDB breakpoints |
| `02-spi-slave-first-frame.md` | 🟡 HW-gated | Wire up LA, capture first SPI byte; CPOL/CPHA theory |
| `03-dma-cache-gotchas.md` | 🟡 HW-gated | Reproduce D-cache coherency bug; DTCM vs SRAM fix |
| `04-imu-i2c-reads.md` | 🟡 HW-gated | ICM-42688-P register reads; stuck-bus recovery |
| `05-100hz-spi-bridge.md` | 🟡 HW-gated | Full pipeline: STM32 SPI slave DMA → Jetson @ 100Hz |
| `06-ros2-publisher.md` | 🟡 HW-gated | sensor_msgs/Imu publisher; CRC validation; RViz2 verify |
| `07-can-bus-encoder.md` | 🟡 HW-gated | SN65HVD230 wiring; Zephyr CAN API; encoder velocity frames; bus-off handling |
| `08-uart-gps-nmea.md` | 🟡 HW-gated | u-blox NEO-M8N; ring-buffer NMEA accumulation; split-sentence handling |
| `09-zbus-nanopb-bridge.md` | 🟡 HW-gated | ZBus channels; nanopb proto3 encoding; length-prefix framing for DMA |
| `10-jetson-rt-setup.md` | 🟡 HW-gated | Jetson Orin NX: nvpmodel, irqbalance, isolcpus, spidev pinmux, cyclictest |
| `11-ekf-integration.md` | 🟡 HW-gated | robot_localization EKF; TF2 frame tree; covariance tuning; rviz2 validation |

## Reference Files (read these before your first hardware session)

| File | Status | What It Covers |
|------|--------|----------------|
| `00-logic-analyzer-guide.md` | ✅ Ready | Saleae Logic 2 trigger discipline, decoder config, SPI/I2C/CAN/UART capture workflows |
| `12-hardfault-decode.md` | ✅ Ready | CFSR/HFSR/MMFAR bit decoding, GDB addr2line, 5 common Zephyr HardFault patterns |

---

## Day-Wasters to Re-Read Before Starting

These are pre-documented in `zephyr/00-mastery-plan.md` Section 3.
The most dangerous ones for the deep-dive phase:

1. **D-Cache Coherency** — `SCB_CleanDCache` before TX, `SCB_InvalidateDCache` after RX
2. **SPI Slave Pre-Arming Race** — arm DMA on transfer-complete, not on CS assert
3. **NVIC + Kconfig Trap** — `CONFIG_SPI_STM32_DMA=y` is not implied by `CONFIG_SPI=y`
4. **Stuck I2C Bus** — power-cycle leaves SDA held low; 9-clock recovery in init
