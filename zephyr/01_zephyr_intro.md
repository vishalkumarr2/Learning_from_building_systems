# Zephyr RTOS — Introduction

## What is Zephyr?

Zephyr is an **open-source, real-time operating system (RTOS)** hosted by the Linux Foundation,
designed for resource-constrained embedded and IoT devices.

| Property | Detail |
|---|---|
| License | Apache 2.0 |
| Founded | 2016 (Intel donated) |
| Target | Microcontrollers with as little as 4KB RAM |
| Kernel type | Monolithic with modules |
| Supported boards | 700+ (STM32, nRF52/53, ESP32, RISC-V, ARM Cortex-M/R/A, x86) |

---

## Key Features

- **Preemptive multithreading** — priority-based scheduler with time-slicing
- **Memory protection** — optional MPU/MMU support
- **Device driver model** — unified API across hardware vendors
- **Networking** — built-in TCP/IP, Bluetooth, 802.15.4, CAN, USB
- **Power management** — fine-grained sleep states
- **Devicetree** — hardware description borrowed from Linux
- **ZBus** — built-in publish/subscribe message bus (see `03_zbus.md`)

---

## Build System — west + CMake

Zephyr uses **CMake** for building and **west** as its meta-tool (workspace manager + flash tool).

```bash
# Install west
pip install west

# Initialize a workspace
west init my_workspace
cd my_workspace
west update          # pulls Zephyr + all modules

# Build a sample for nRF52840 DK
west build -b nrf52840dk/nrf52840 samples/hello_world

# Flash to board
west flash

# Open serial monitor
west espressif monitor   # or: screen /dev/ttyACM0 115200
```

### Project structure

```
my_workspace/
├── zephyr/               ← Zephyr kernel source (managed by west)
├── modules/              ← HAL drivers, nanopb, mbedTLS, etc.
├── bootloader/
└── my_app/               ← Your application
    ├── CMakeLists.txt
    ├── prj.conf          ← Kconfig — enable/disable features
    ├── app.overlay       ← Devicetree overlay — your board wiring
    └── src/
        └── main.c
```

### prj.conf — Kconfig

This is how you turn features on/off — no #ifdefs in code:

```kconfig
# prj.conf
CONFIG_ZBUS=y              # enable ZBus
CONFIG_SPI=y               # enable SPI driver
CONFIG_I2C=y               # enable I2C driver
CONFIG_CAN=y               # enable CAN driver
CONFIG_NANOPB=y            # enable nanopb serialization
CONFIG_MAIN_STACK_SIZE=2048
CONFIG_THREAD_STACK_INFO=y # debug: show stack usage
```

### app.overlay — Devicetree

Describes how YOUR hardware is wired — separate from board definition:

```dts
/* app.overlay */
&spi1 {
    status = "okay";
    pinctrl-0 = <&spi1_default>;
    cs-gpios = <&gpioa 4 GPIO_ACTIVE_LOW>;
};

&i2c1 {
    status = "okay";
    imu: icm42688@68 {
        compatible = "invensense,icm42688";
        reg = <0x68>;
    };
};
```

---

## How Zephyr compares

| | Zephyr | FreeRTOS | bare-metal |
|---|---|---|---|
| Ecosystem | Large, Linux Foundation backed | Large, AWS backed | None |
| Networking | Built-in | Add-on (FreeRTOS+TCP) | Fully manual |
| Security | Actively maintained CVEs, PSA certified | Variable | You own everything |
| Complexity | Higher — more to learn | Lower | Lowest |
| Message bus | ZBus (built-in) | None (use queues manually) | None |
| Devicetree | Yes (like Linux) | No | No |
| Best for | Complex IoT, multi-peripheral systems | Simple tasks, small MCUs | Ultra-low latency, tiny flash |

---

## Common Use Cases

- **IoT sensors** — BLE devices, wearables, environmental monitors
- **Industrial** — motor controllers, fieldbus gateways (CAN, RS-485)
- **Automotive** — ECUs, CAN nodes, AUTOSAR alternatives
- **Robotics** — sensor hubs, actuator controllers (our STM32 use case)
- **Security-critical** — MCUs needing FIPS/PSA compliance

---

## Good Starting Points

- Official docs: https://docs.zephyrproject.org
- Samples: `zephyr/samples/` in the repo
  - `samples/hello_world` — basic threading
  - `samples/drivers/spi/` — SPI
  - `samples/subsys/zbus/` — ZBus
  - `samples/bluetooth/beacon/` — BLE
