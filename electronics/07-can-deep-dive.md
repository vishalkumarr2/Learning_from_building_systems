# 07 — CAN Bus Deep Dive
### From differential signaling to arbitration, error handling, and SocketCAN
**Prerequisites:** Chapter 01 (basic electronics), Chapter 02 (transistors for transceivers)
**Unlocks:** Vehicle/robot communication, industrial automation, battery management, OBD-II

---

## Why Should I Care? (STM32/Jetson Project Context)

- CAN is the backbone of automotive and industrial robot communication
- OKS AMR robots may use CAN for motor controllers, battery management (BMS), and sensor modules
- CAN is designed for harsh electrical environments: factory floors, vehicles, long cable runs
- Unlike I2C/SPI (board-level buses), CAN is a multi-node fieldbus for distances up to 1km
- Understanding CAN is essential if you interface with motor drivers, BMS, or industrial IO

---

# PART 1 — THE PHYSICS: DIFFERENTIAL SIGNALING

---

## 1.1 Why Differential? — Noise Immunity

I2C, SPI, and UART use "single-ended" signaling: voltage is measured relative to a shared GND wire. Any noise on GND shifts all signals.

CAN uses "differential" signaling: the information is in the VOLTAGE DIFFERENCE between two wires (CANH and CANL). If noise hits both wires equally (which it does when they're twisted together), the difference stays the same.

```
    Single-ended (I2C, SPI):             Differential (CAN):
    
    Signal ──────────────→               CANH ──────────────→
    GND ─────────────────→               CANL ──────────────→

    Noise adds +0.5V to both:            Noise adds +0.5V to both:
    Signal: 3.3V → 3.8V                  CANH: 3.5V → 4.0V
    GND:    0V   → 0.5V                  CANL: 1.5V → 2.0V
    Diff:   3.3V → 3.3V (wrong!)         Diff: 2.0V → 2.0V ✓
    But GND shift means receiver sees     Receiver still sees
    3.3V (unchanged) relative to new      correct 2.0V difference!
    GND → COULD be OK, but cumulative     → Immune to common-mode noise
    GND noise causes errors
```

**ELI15 analogy — the seesaw:**

Imagine two kids on a seesaw. The "signal" is which side is up. If you raise the entire playground by 2 feet (noise), the seesaw still tips the same direction. The receiver only cares about the *difference* between the two ends, not their absolute height.

---

## 1.2 CAN Bus Voltage Levels

CAN defines two states: **dominant** (logic 0) and **recessive** (logic 1).

```
    RECESSIVE (logic 1):      DOMINANT (logic 0):
    Both wires at 2.5V        CANH to ~3.5V, CANL to ~1.5V
    Vdiff = 0V                Vdiff = ~2V
    
    Voltage waveform for a few bits: 1  0  1  1  0  0
    
    CANH: ~~2.5~~|~3.5~|~~2.5~~|~~2.5~~|~3.5~|~3.5~|
                 |     |       |       |     |     |
    CANL: ~~2.5~~|~1.5~|~~2.5~~|~~2.5~~|~1.5~|~1.5~|
                 |     |       |       |     |     |
    Vdiff:  ~0V  | ~2V |  ~0V  |  ~0V  | ~2V | ~2V |
    State:  REC  | DOM |  REC  |  REC  | DOM | DOM |
    Logic:   1   |  0  |   1   |   1   |  0  |  0  |
```

**Why these names?**
- **Dominant** (0): The transceiver actively drives CANH high and CANL low — creating a voltage difference. This state always "wins" over recessive.
- **Recessive** (1): The transceiver does nothing — both lines float to 2.5V through the termination resistors. If ANY node drives dominant, the recessive state is overridden.

This is the CAN version of I2C's wired-AND: multiple nodes can transmit simultaneously, and 0 (dominant) always wins over 1 (recessive). This enables non-destructive arbitration.

---

## 1.3 The CAN Transceiver

Your microcontroller's CAN peripheral (called FDCAN on STM32H7, bxCAN on STM32F4) generates logical 0s and 1s on CANTX/CANRX pins. The **transceiver** converts these to/from differential signals.

```
    STM32                CAN Transceiver              CAN Bus
    ┌──────┐            ┌──────────────┐         ┌─────────────┐
    │      │  CANTX ──→ │              │  CANH ──┤             │
    │ FDCAN│            │  SN65HVD230  │         │   Twisted   │
    │      │  CANRX ←── │  or TJA1050  │  CANL ──┤    Pair     │
    └──────┘            └──────────────┘         └─────────────┘
                              │
                             GND
                             
    Common transceivers:
    ┌────────────┬────────────────────────────────────────────┐
    │ SN65HVD230 │ 3.3V, 1Mbps, standby mode, cheap          │
    │ TJA1050    │ 5V, 1Mbps, very common, automotive        │
    │ TJA1051    │ 3.3V/5V, automotive, VIO for 3.3V logic   │
    │ MCP2551    │ 5V, 1Mbps, Microchip, very common         │
    │ MCP2542    │ CAN FD up to 8Mbps, automotive            │
    └────────────┴────────────────────────────────────────────┘
```

---

## 1.4 Bus Termination — The 120Ω Resistors

CAN bus REQUIRES 120Ω termination resistors at each end of the bus to prevent signal reflections.

```
    Node 1       Node 2       Node 3       Node 4
      │            │            │            │
      │            │            │            │
    ──┴──────────────────────────────────────┴──── CANH
      │                                      │
     120Ω                                   120Ω
      │                                      │
    ──┴──────────────────────────────────────┴──── CANL
    
    ↑ END of bus                        END of bus ↑
    
    Only at the two PHYSICAL ENDS of the bus!
    Middle nodes do NOT get termination resistors!
```

**Why 120Ω?**

CAN uses a 120Ω characteristic impedance cable (twisted pair). To prevent reflections, the termination resistance must match the cable impedance. Two 120Ω resistors in parallel = 60Ω total impedance (this is the specified bus impedance).

**How to verify:** Measure resistance between CANH and CANL with the bus powered off:
- ~60Ω: correct (two 120Ω in parallel)
- ~120Ω: one termination missing
- Open circuit: no termination → bus won't work reliably
- ~40Ω: three termination resistors → remove one

**What happens without termination:**
- Reflections cause ringing on the bus edges
- At low speeds (125kbps), might work anyway (dangerous — passes testing, fails in the field)
- At 1Mbps, the bus will have constant bit errors

---

# PART 2 — CAN FRAME FORMAT

---

## 2.1 Standard CAN Frame (CAN 2.0A — 11-bit ID)

```
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │ SOF │ Identifier │ RTR │ IDE │ r0 │ DLC │ Data     │ CRC     │ ACK │ EOF │ IFS │
    │ (1) │   (11)     │ (1) │ (1) │(1) │ (4) │ (0-64)   │ (15+1)  │(1+1)│ (7) │ (3) │
    └─────────────────────────────────────────────────────────────────────────────────┘
           ←── Arbitration ──→ ←Control→  ← Data field → ← CRC ──→
           
    Total: 44 + 8×N bits (N = data bytes, 0-8) + stuff bits
```

**Field-by-field breakdown:**

| Field | Bits | Purpose |
|-------|------|---------|
| **SOF** | 1 | Start of frame — always dominant (0). All nodes sync to this edge. |
| **Identifier** | 11 | Message ID and priority (lower number = higher priority). |
| **RTR** | 1 | Remote Transmission Request: 0 = data frame, 1 = request frame. |
| **IDE** | 1 | ID Extension: 0 = 11-bit standard, 1 = 29-bit extended. |
| **r0** | 1 | Reserved: must be 0 (dominant). |
| **DLC** | 4 | Data Length Code: 0–8 bytes in CAN 2.0, up to 64 bytes in CAN FD. |
| **Data** | 0–64 | Payload: 0 to 8 bytes (CAN 2.0) or 0 to 64 bytes (CAN FD). |
| **CRC** | 15+1 | 15-bit CRC + 1 delimiter bit. Hardware verifies automatically. |
| **ACK** | 1+1 | ACK slot + delimiter. Receiver pulls ACK slot dominant → "I heard you." |
| **EOF** | 7 | End of frame: 7 recessive bits. |
| **IFS** | 3 | Interframe space: minimum 3 recessive bits between frames. |

**ASCII Diagram — One CAN Frame (ID=0x123, Data=[0xDE, 0xAD]):**

```
    Bit positions (simplified, no stuff bits):

    SOF  ID[10:0]       RTR IDE r0  DLC[3:0]  DATA[0]   DATA[1]   CRC           ACK  EOF
     │   │          │    │   │   │  │       │ │       │ │       │ │             │ │ │ │       │
     0   0 0 1 0 0 1 0 0 0 1 1  0   0   0  0 0 1 0  1 1 0 1 1 1 1 0 1 0 1 0 1 1 0 1 0 ...  1111111
     │   ←── 0x123 = 0b00100100011 ──→            ←0xDE=11011110→ ←0xAD=10101101→
     
     SOF = dominant (0) — all nodes sync
     ID = 0x123 = 0b00100100011 (11 bits, MSB first)
     RTR = 0 (this is a data frame, not a request)
     DLC = 0b0010 = 2 bytes
```

---

## 2.2 Extended CAN Frame (CAN 2.0B — 29-bit ID)

```
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │ SOF │ Base ID │ SRR │ IDE │ Extended ID │ RTR │ r1 │ r0 │ DLC │ Data │ ... │
    │ (1) │  (11)   │ (1) │ (1) │    (18)     │ (1) │(1) │(1) │ (4) │      │     │
    └──────────────────────────────────────────────────────────────────────────────┘
    
    Total 29-bit ID = 11-bit base + 18-bit extension
    SRR = Substitute Remote Request (always recessive)
    IDE = 1 (extended frame flag)
```

Extended frames are common in J1939 (heavy-duty vehicles), CANopen, and DeviceNet.

---

## 2.3 Bit Stuffing

After 5 consecutive bits of the same polarity, the transmitter inserts an opposite bit. The receiver removes it.

```
    Without stuffing: 1 0 0 0 0 0 0 1 1 1 1 1 1 0
    With stuffing:    1 0 0 0 0 0 [1] 0 1 1 1 1 1 [0] 1 0
                              ↑ inserted          ↑ inserted
                         (5 zeros → stuff 1)   (5 ones → stuff 0)
```

**Why?** Without stuffing, long runs of the same bit would make it impossible for receivers to stay synchronized (the clock recovery mechanism needs regular edges). Bit stuffing guarantees an edge at least every 6 bit times.

**Impact:** Bit stuffing increases the actual frame length by up to ~20%. Worst case: every 5th bit is stuffed. When calculating bus throughput, account for stuff bits.

---

# PART 3 — ARBITRATION — THE KILLER FEATURE

---

## 3.1 How Non-Destructive Arbitration Works

Any node can start transmitting when the bus is idle. If two nodes start at the same time:

1. Both transmit their SOF (dominant) — bus sees dominant ✓
2. Both start transmitting their ID bits simultaneously
3. Each node reads back the bus after sending each bit
4. The node with the LOWER ID wins (0 is dominant, wins over 1)

**Walkthrough — Node A sends ID=0x100, Node B sends ID=0x080:**

```
    ID bit:    10   9   8   7   6   5   4   3   2   1   0
    
    0x100 =   0    1   0   0   0   0   0   0   0   0   0
    0x080 =   0    0   1   0   0   0   0   0   0   0   0
    Bus:      0    0   ← B wins here
                   ↑
              Node A sent 1 (recessive)
              Node B sent 0 (dominant)
              Bus = 0 (dominant wins)
              
              Node A reads back 0, but it sent 1 → LOST!
              Node A stops transmitting immediately.
              Node B continues, unaware of the conflict.
```

**Key insight:** The node with the lower ID (numerically) wins. This means:
- ID 0x000 is highest priority
- ID 0x7FF is lowest priority (11-bit)
- **Priority is designed into the ID assignment**, not a runtime decision

---

## 3.2 Practical ID Assignment Strategy

```
    Priority Assignment (example for a robot):
    
    ID Range        Purpose                    Priority
    0x000 – 0x00F   Emergency / E-stop         Highest ← these always get through
    0x010 – 0x0FF   Motor commands             High    ← real-time critical
    0x100 – 0x1FF   Sensor data (IMU, odom)    Medium  ← regular data
    0x200 – 0x3FF   Battery management         Normal  ← periodic status
    0x400 – 0x5FF   Diagnostics / debug        Low     ← non-critical
    0x600 – 0x7FF   Configuration / heartbeat  Lowest  ← infrequent
```

---

# PART 4 — ERROR HANDLING — CAN'S DEFENSIVE SYSTEM

---

## 4.1 Five Types of Errors

CAN has **five** error detection mechanisms, checked in hardware by every node:

| Error Type | What It Checks | How It's Detected |
|------------|---------------|-------------------|
| **Bit Error** | Transmitter reads back each bit | Sent dominant, read recessive (or vice versa in non-arbitration fields) |
| **Stuff Error** | Bit stuffing rule | 6 consecutive same-polarity bits detected |
| **CRC Error** | Data integrity | Received CRC doesn't match calculated CRC |
| **Form Error** | Fixed-format fields | SOF, EOF, ACK delimiter, CRC delimiter not at expected value |
| **ACK Error** | Message received by someone | ACK slot stays recessive (nobody acknowledged) |

When any node detects an error, it transmits an **Error Frame**: 6 dominant bits (violates bit stuffing → ALL nodes detect it → everyone discards the corrupted frame).

---

## 4.2 Error Counters — TEC and REC

Every CAN node has two error counters:
- **TEC** (Transmit Error Counter): incremented when the node's transmission causes errors
- **REC** (Receive Error Counter): incremented when the node detects errors in received frames

```
    Error State Machine:
    
    ┌─────────────┐   TEC > 127     ┌──────────────┐   TEC > 255      ┌───────────┐
    │ ERROR ACTIVE │   or REC > 127  │ ERROR PASSIVE │                  │  BUS OFF  │
    │ (normal)     │ ───────────────→│ (reduced)     │ ────────────────→│ (dead)    │
    │              │                 │               │                  │           │
    │ Sends ACTIVE │                 │ Sends PASSIVE │                  │ Cannot    │
    │ Error Frames │                 │ Error Frames  │                  │ transmit  │
    │ (6 dominant) │                 │ (6 recessive) │                  │ or receive│
    └──────────────┘ ←───────────────┘               │                  └───────────┘
                       TEC, REC < 128                 │               ↓ 128 × 11 recessive
                       (successful msgs)              │               ↓ bits detected
                                                      │               ↓
                                                      └───────────────┘
                                                        Auto-recovery
```

**Why Error Active vs Error Passive?**

- **Error Active** (TEC, REC < 128): Node is healthy. When it detects an error, it sends an Active Error Frame (6 dominant bits → immediately disrupts the bus → ALL nodes see the error). This is aggressive but correct — one bad frame shouldn't be received by anyone.

- **Error Passive** (TEC or REC ≥ 128): Node is having too many errors. Something might be wrong with THIS node (loose connector, bad transceiver). It can still send Error Frames, but they're passive (6 recessive bits → only visible if no one else is transmitting). This prevents a faulty node from constantly disrupting the bus with error frames.

- **Bus-Off** (TEC ≥ 256): Node is removed from the bus completely. It can't transmit or receive. Recovery requires detecting 128 occurrences of 11 consecutive recessive bits (bus idle), then re-initializing. Some systems auto-recover; critical safety systems require manual reset.

**Why this matters for robot debugging:**
- A motor controller with a loose CAN connector will gradually go Error Passive → Bus Off
- Other nodes will see ACK errors (the failing node stops acknowledging)
- Check TEC/REC values in your CAN controller registers for diagnostic info

---

## 4.3 Error Frame Format

```
    Normal frame:   [...data...]  [CRC] [ACK] [EOF: 1111111] [IFS: 111]
    
    Error detected mid-frame:
    [...data...]  ← ERROR DETECTED HERE
    [Error Flag: 000000] [Error Delimiter: 11111111] [IFS: 111]
     6 dominant bits       8 recessive bits
     (Active Error Flag)
    
    This destroys the current frame. All nodes discard it.
    The transmitter AUTOMATICALLY retransmits the frame.
```

**Automatic retransmission** is a key CAN feature: when a frame is corrupted, the transmitter retries without any software intervention. The hardware handles it.

---

# PART 5 — CAN FD (FLEXIBLE DATA-RATE)

---

CAN FD (introduced ~2012) extends classic CAN with:

| Feature | Classic CAN 2.0 | CAN FD |
|---------|-----------------|--------|
| Max data | 8 bytes | 64 bytes |
| Max speed | 1 Mbps | 8 Mbps (data phase) |
| Arbitration speed | Up to 1 Mbps | Up to 1 Mbps (same) |
| Data phase speed | Same as arbitration | Up to 8 Mbps (faster!) |
| CRC | 15-bit | 17-bit (≤16B) or 21-bit (>16B) |

```
    CAN FD Frame:
    
    ┌──────────────────────────────────────────────────────────────────────┐
    │ SOF │ ID │ FDF │ BRS │ DLC │     Data (0-64 bytes)     │ CRC │ ... │
    └──────────────────────────────────────────────────────────────────────┘
                  ↑      ↑
                  │      └── Bit Rate Switch: 1 = switch to fast rate for data
                  └── FD Format: 1 = CAN FD frame (0 = classic CAN)
    
    Speed profile:
    ←── Arbitration at 500kbps ──→←── Data at 4Mbps ──→←── 500kbps ──→
    [SOF][ID][control]              [data payload]        [CRC][ACK][EOF]
    
    Arbitration MUST be at the slower rate (all nodes must participate).
    Data phase CAN be faster (only the recipient needs to keep up).
```

**DLC encoding for CAN FD (data length > 8):**

| DLC value | Bytes (Classic CAN) | Bytes (CAN FD) |
|-----------|--------------------:|----------------:|
| 0–8       | 0–8                | 0–8             |
| 9         | 8                  | 12              |
| 10        | 8                  | 16              |
| 11        | 8                  | 20              |
| 12        | 8                  | 24              |
| 13        | 8                  | 32              |
| 14        | 8                  | 48              |
| 15        | 8                  | 64              |

---

# PART 6 — SOCKETCAN ON LINUX

---

## 6.1 SocketCAN Architecture

Linux treats CAN interfaces like network sockets. You can use standard socket APIs or command-line tools.

```
    ┌──────────────────────────────────────────────────┐
    │                User Space                         │
    │  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
    │  │ cansend  │  │ candump  │  │ Your C/Python  │  │
    │  │ canutils │  │ canutils │  │ application    │  │
    │  └────┬─────┘  └────┬─────┘  └───────┬───────┘  │
    │       │              │                │          │
    │  ═════╪══════════════╪════════════════╪═════════ │
    │       │        socket interface       │          │
    │  ─────┼──────────────┼────────────────┼───────── │
    │                Kernel Space                       │
    │  ┌─────────────────────────────────────────────┐ │
    │  │           SocketCAN Framework               │ │
    │  │  (can.ko, can-raw.ko, can-gw.ko)            │ │
    │  └────────────────┬────────────────────────────┘ │
    │                   │                              │
    │  ┌────────────────┼──────────────────┐          │
    │  │          CAN Driver               │          │
    │  │  (mcp251x, peak_pci, vcan, slcan) │          │
    │  └────────────────┬──────────────────┘          │
    └───────────────────┼──────────────────────────────┘
                        │
                  CAN Hardware / vcan
```

## 6.2 Essential CLI Commands

```bash
# === Setup ===

# Load kernel modules
sudo modprobe can
sudo modprobe can-raw
sudo modprobe vcan              # Virtual CAN for testing

# Create virtual CAN interface (for testing without hardware)
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0

# Configure hardware CAN interface
sudo ip link set can0 type can bitrate 500000     # 500kbps
sudo ip link set can0 type can bitrate 500000 dbitrate 2000000 fd on  # CAN FD
sudo ip link set up can0

# === Sending ===

# Send standard frame: ID=0x123, data=0xDE 0xAD 0xBE 0xEF
cansend can0 123#DEADBEEF

# Send extended frame (29-bit ID): prefix with 8 hex digits
cansend can0 1234ABCD#11223344

# Send CAN FD frame (up to 64 bytes):
cansend can0 123##1.DEADBEEF...  # ## for FD, 1=BRS flag

# === Receiving ===

# Dump all frames on can0 (like tcpdump for CAN):
candump can0

# Dump with timestamp:
candump -ta can0

# Filter: only show ID 0x100–0x1FF:
candump can0,100:700     # mask=0x700, match=0x100

# === Monitoring ===

# Show CAN interface statistics:
ip -details -statistics link show can0

# Show bus error counters (TEC/REC):
ip -details link show can0 | grep -i 'state\|restarts\|error'
```

## 6.3 Python SocketCAN

```python
import socket
import struct

# Create raw CAN socket
sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
sock.bind(('can0',))

# Send a frame
# struct format: arbitration ID (4B, big-endian) + DLC (1B) + padding (3B) + data (8B)
can_id = 0x123
data = bytes([0xDE, 0xAD, 0xBE, 0xEF])
dlc = len(data)
can_frame = struct.pack('=IB3x8s', can_id, dlc, data.ljust(8, b'\x00'))
sock.send(can_frame)

# Receive a frame
frame = sock.recv(16)
can_id, dlc, raw_data = struct.unpack('=IB3x8s', frame)
can_id &= 0x1FFFFFFF  # Mask out flags
data = raw_data[:dlc]
print(f"ID: 0x{can_id:03X}, DLC: {dlc}, Data: {data.hex()}")
```

For production code, use the `python-can` library:

```python
import can

bus = can.interface.Bus(channel='can0', interface='socketcan')

# Send
msg = can.Message(arbitration_id=0x123, data=[0xDE, 0xAD, 0xBE, 0xEF])
bus.send(msg)

# Receive (blocks until a message arrives or timeout)
msg = bus.recv(timeout=1.0)
if msg:
    print(f"ID: 0x{msg.arbitration_id:03X}, Data: {msg.data.hex()}")

# Receive with filtering
bus.set_filters([
    {"can_id": 0x100, "can_mask": 0x700, "extended": False}
])
```

---

# PART 7 — STM32 FDCAN PERIPHERAL

---

## 7.1 STM32H7 FDCAN Configuration (Zephyr)

```c
/* Zephyr CAN API example for STM32H7 FDCAN */
#include <zephyr/drivers/can.h>

const struct device *can_dev = DEVICE_DT_GET(DT_NODELABEL(fdcan1));

/* Initialize */
int ret = can_start(can_dev);
if (ret != 0) {
    printk("CAN start failed: %d\n", ret);
    return;
}

/* Send a frame */
struct can_frame tx_frame = {
    .id = 0x123,
    .dlc = 4,
    .data = {0xDE, 0xAD, 0xBE, 0xEF}
};
ret = can_send(can_dev, &tx_frame, K_MSEC(100), NULL, NULL);

/* Set up RX filter + callback */
struct can_filter rx_filter = {
    .id = 0x200,
    .mask = CAN_STD_ID_MASK,  /* exact match */
    .flags = 0
};

void rx_callback(const struct device *dev, struct can_frame *frame, void *user_data)
{
    printk("Received: ID=0x%03X DLC=%d Data=", frame->id, frame->dlc);
    for (int i = 0; i < frame->dlc; i++) {
        printk("%02X ", frame->data[i]);
    }
    printk("\n");
}

int filter_id = can_add_rx_filter(can_dev, rx_callback, NULL, &rx_filter);
```

## 7.2 DeviceTree overlay for STM32H7 FDCAN

```dts
/* boards/nucleo_h743zi.overlay */
&fdcan1 {
    status = "okay";
    pinctrl-0 = <&fdcan1_rx_pd0 &fdcan1_tx_pd1>;
    pinctrl-names = "default";
    bus-speed = <500000>;        /* 500kbps nominal */
    bus-speed-data = <2000000>;  /* 2Mbps data phase (FD) */
    sample-point = <875>;       /* 87.5% sample point */
    sample-point-data = <750>;
};
```

---

# PART 8 — HIGHER-LAYER PROTOCOLS

---

CAN only defines the physical and data-link layers. Higher-layer protocols add meaning to the raw ID+data frames:

| Protocol | Domain | ID Usage |
|----------|--------|----------|
| **CANopen** | Industrial automation | 7-bit node ID + 4-bit function code. PDO, SDO, NMT, SYNC, EMCY. |
| **J1939** | Heavy vehicles | 29-bit ID encodes PGN (parameter group), source, priority. |
| **DeviceNet** | Factory automation | 11-bit ID with MAC ID + message group. |
| **OBD-II** | Automotive diagnostics | ID 0x7DF (broadcast), 0x7E0–0x7E7 (specific ECU). |
| **UAVCAN (DroneCAN)** | Drones / UAVs | 29-bit ID with priority, type, source/destination. |
| **ISO-TP (ISO 15765-2)** | Transport layer | Multi-frame segmentation for >8 byte messages over CAN 2.0. |

**Which one for robots?** Most custom robot systems use raw CAN with a project-specific protocol. For ROS integration, `ros_canopen` or `socketcan_bridge` packages are common.

---

# PART 9 — CAN BUS DESIGN RULES

---

## 9.1 Physical Layout

```
    CORRECT — Linear bus topology (daisy-chain):
    
    ┌──┐    ┌──┐    ┌──┐    ┌──┐
    │N1├────┤N2├────┤N3├────┤N4│
    └──┘    └──┘    └──┘    └──┘
    120Ω                    120Ω
    (end)                   (end)
    
    WRONG — Star topology (creates reflections):
    
                ┌──┐
                │N3│
                └┬─┘
    ┌──┐    ┌───┴───┐    ┌──┐
    │N1├────┤  HUB  ├────┤N2│       ← DON'T DO THIS!
    └──┘    └───┬───┘    └──┘
                ┌┴─┐
                │N4│
                └──┘
```

## 9.2 Speed vs Distance

| Bit Rate | Max Bus Length | Use Case |
|----------|---------------|----------|
| 1 Mbps | 40m | Short bus, fast response |
| 500 kbps | 100m | Automotive standard |
| 250 kbps | 250m | Trucks, industrial |
| 125 kbps | 500m | Light industrial |
| 50 kbps | 1000m | Building automation |
| 10 kbps | 5km+ | Very long runs |

**Rule:** Speed × distance ≈ constant. Faster = shorter max bus length.

## 9.3 Wiring

- Use **twisted pair** cable, 120Ω characteristic impedance
- Shield the cable in noisy environments (ground shield at one end only → avoid ground loops)
- Keep stub lengths (from bus to node connector) under 30cm at 1Mbps
- Separate CAN wiring from power wiring (at least 10cm distance)

---

# GOTCHA TABLE

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|-------------|-----------------|-----|
| No communication at all | Missing termination resistors | Measure 60Ω between CANH and CANL (powered off) | Add 120Ω at each end of bus |
| Communication at low speed but not high speed | Termination or cable issues | Try lower bitrate. Scope the bus for ringing. | Check termination. Use proper twisted pair. Shorten stubs. |
| Node goes Bus-Off | High TEC (>255). Wiring fault, wrong bitrate, or transceiver failure. | Read TEC/REC from CAN controller registers | Check wiring. Verify all nodes have same bitrate. Replace transceiver. |
| ACK errors (transmitter) | No other node on the bus, or all receivers have wrong bitrate | Need at least 2 nodes. Check all nodes' bitrate matches. | Add a second node or fix bitrate mismatch. |
| Frames received but data is garbled | Bitrate mismatch between nodes | Check configured bitrate on ALL nodes. Even 0.1% off can cause issues. | Use the same crystal/oscillator frequency. Match bitrate exactly. |
| Intermittent errors under vibration | Loose connector or cracked solder joint | Wiggle cables while monitoring bus errors | Re-crimp connectors. Re-solder joints. |
| One node works, another doesn't | Wrong transceiver voltage level (3.3V vs 5V) | Check transceiver T×/R× pins with scope. Check VCC. | Use correct transceiver for your MCU's logic level. |
| Bus error rate increases over time | EMI from nearby power cables or motors | Move CAN cable away from noise sources. Add shielding. | Route CAN cables separately. Use shielded twisted pair. |
| CAN FD frames rejected by some nodes | Classic CAN nodes on FD-enabled bus | Error frames when FD frame is sent | Either upgrade all nodes to FD or keep bus at classic CAN |
| Star topology causes random errors | Reflections from un-terminated branch ends | Scope shows ringing on edges. Works at low speed. | Convert to linear (daisy-chain) topology |

---

# QUICK REFERENCE CARD

```
┌──────────────────────────────── CAN BUS CHEAT SHEET ──────────────────────────────────────┐
│                                                                                            │
│  CORE:  Differential signaling (CANH, CANL), multi-master, broadcast, half-duplex         │
│         Dominant (0) = actively driven. Recessive (1) = passive.  0 wins over 1.          │
│                                                                                            │
│  FRAME STRUCTURE (11-bit ID):                                                              │
│    [SOF][11-bit ID][RTR][IDE][r0][DLC 4b][Data 0-8B][CRC 15b+1][ACK 1+1][EOF 7][IFS 3]   │
│                                                                                            │
│  TERMINATION:  120Ω at each physical END of bus. Net = 60Ω between CANH/CANL.             │
│                                                                                            │
│  SPEED:  1Mbps/40m   500kbps/100m   250kbps/250m   125kbps/500m   10kbps/5km              │
│                                                                                            │
│  ARBITRATION:  Lower ID = higher priority. Dominant (0) wins. Non-destructive.             │
│                                                                                            │
│  ERROR COUNTERS:  TEC/REC                                                                  │
│    Error Active (<128): Normal, sends active error frames                                  │
│    Error Passive (≥128): Sending passive error frames, reduced disruption                  │
│    Bus-Off (TEC≥256): Node removed, requires recovery                                     │
│                                                                                            │
│  5 ERROR TYPES:  Bit, Stuff (6 same bits), CRC, Form, ACK                                 │
│                                                                                            │
│  CAN FD:  Up to 64 bytes data, up to 8Mbps data phase, 1Mbps arbitration                  │
│                                                                                            │
│  LINUX:  ip link set can0 type can bitrate 500000 && ip link set up can0                  │
│          cansend can0 123#DEADBEEF                                                         │
│          candump can0                                                                      │
│                                                                                            │
│  TOPOLOGY:  Linear bus (daisy-chain) ONLY. No star. Stubs < 30cm at 1Mbps.                │
│                                                                                            │
│  TRANSCEIVERS:  SN65HVD230 (3.3V), TJA1050 (5V), MCP2542 (FD)                            │
│                                                                                            │
│  COMMON PROTOCOLS:  CANopen (industrial), J1939 (trucks), OBD-II (cars), DroneCAN (UAV)   │
│                                                                                            │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```
