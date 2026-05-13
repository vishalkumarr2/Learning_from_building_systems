# Exercises: CAN Bus

### Chapter 07: From differential signaling to arbitration, error handling, and SocketCAN

**Self-assessment guide:** Write your answer before expanding the details block. CAN is the backbone of automotive and industrial robot communication — understanding its error handling and arbitration mechanisms is essential for debugging motor controllers, BMS, and safety systems on OKS robots.

**Project context:** OKS AMR robots may use CAN for motor controllers, battery management (BMS), and sensor modules. Unlike I2C/SPI (board-level), CAN is a fieldbus designed for harsh environments: factory floors, long cable runs, and electrical noise. Your STM32H7 has an FDCAN peripheral, and the Jetson runs SocketCAN for ROS integration.

---

## Section A — Conceptual Questions

**A1.** Explain why CAN uses differential signaling instead of single-ended. Give a specific example of a noise event that would corrupt a single-ended signal but leave a differential signal intact.

<details><summary>Answer</summary>

Differential signaling encodes information as the **voltage difference** between two wires (CANH and CANL), rather than the voltage of one wire relative to ground.

**Why this matters:** When noise couples into a cable (e.g., EMI from a motor), it affects **both** wires equally (common-mode noise) because the wires are twisted together and physically close. The receiver measures only the difference, so the noise cancels out.

**Specific example:** A VFD (variable-frequency drive) motor controller on the factory floor produces a 2V EMI spike. What happens:

**Single-ended (like I2C):**
- Signal wire: 3.3V → 5.3V
- GND wire: 0V → 2.0V
- Receiver sees: 5.3V - 2.0V = 3.3V (might be OK, but GND shift causes cumulative errors)
- If the spike is uneven (3V on signal, 1V on GND): receiver sees 6.3V - 1.0V = 5.3V → could read the wrong logic level

**Differential (CAN):**
- CANH: 3.5V → 5.5V (dominant state + 2V noise)
- CANL: 1.5V → 3.5V (dominant state + 2V noise)
- Vdiff: 5.5V - 3.5V = **2.0V** → correct dominant reading ✓
- The noise is identical on both wires → completely canceled

This is the "seesaw analogy": raising the whole playground (common-mode noise) doesn't change which side of the seesaw is up.

</details>

---

**A2.** What do "dominant" and "recessive" mean in CAN, and why are they named that way? What are the voltage levels for each state?

<details><summary>Answer</summary>

| State | Logic | CANH | CANL | Vdiff | Action |
|-------|-------|------|------|-------|--------|
| **Dominant** | 0 | ~3.5V | ~1.5V | ~2.0V | Transceiver actively drives CANH high, CANL low |
| **Recessive** | 1 | ~2.5V | ~2.5V | ~0V | Transceiver does nothing; lines float to 2.5V via termination |

**Why "dominant" and "recessive"?** The names reflect what happens when two nodes transmit simultaneously:
- Dominant (0) ALWAYS wins — it physically pulls the bus away from the recessive state
- Recessive (1) only exists when NO node is driving dominant — any single node asserting dominant overrides all recessive states

This is CAN's equivalent of I2C's "wired-AND": 0 always wins over 1. This property is what makes **non-destructive arbitration** possible — two nodes can transmit at the same time, and the one with the lower ID (more dominant bits) wins without corrupting either message.

The naming convention is borrowed from genetics: a dominant allele always expresses over a recessive one. On CAN, a dominant bit always expresses over a recessive one.

</details>

---

**A3.** Why does CAN require exactly two 120Ω termination resistors, and why must they be placed at the physical ends of the bus? What happens with one, three, or zero termination resistors?

<details><summary>Answer</summary>

CAN uses a twisted pair cable with ~120Ω characteristic impedance. To prevent **signal reflections** at the cable ends, the termination resistance must match the cable impedance.

Two 120Ω resistors at each end → 60Ω total parallel impedance → matches the CAN spec.

**Where they go:** At the two **physical endpoints** of the linear bus. NOT at middle nodes — middle nodes are on a continuous transmission line and don't create reflections.

**What happens with incorrect termination:**

| Scenario | CANH-CANL Resistance | Effect |
|----------|---------------------|--------|
| 0 resistors | Open circuit | Severe ringing, bus unreliable at any speed |
| 1 resistor | ~120Ω | Reflections from unterminated end. Works at low speed (125 kbps), fails at 500+ kbps |
| 2 resistors (correct) | ~60Ω | Clean signal edges, reliable at rated speed |
| 3 resistors | ~40Ω | Over-terminated: excessive current draw, reduced voltage swing, stressed transceivers |

**Diagnostic:** Measure resistance between CANH and CANL with bus powered off:
- ~60Ω = correct (two 120Ω in parallel)
- ~120Ω = one missing
- Open = no termination
- ~40Ω = one too many → find and remove the extra

</details>

---

**A4.** Explain CAN arbitration with a concrete example: Node A sends ID=0x200 and Node B sends ID=0x100 at the same time. Walk through the bit-by-bit arbitration process. Who wins and why?

<details><summary>Answer</summary>

Both nodes detect bus idle and start transmitting simultaneously. They compare the bus state bit-by-bit during the 11-bit ID field:

```
ID bit:     10    9    8    7    6    5    4    3    2    1    0

0x200 =     0    1    0    0    0    0    0    0    0    0    0
0x100 =     0    0    1    0    0    0    0    0    0    0    0
Bus:        0    0    ← Node B wins here!
                 ↑
            Bit 9: Node A sends 1 (recessive)
                   Node B sends 0 (dominant)
                   Bus reads 0 (dominant wins)
```

**At bit 9:**
- Node A sent 1 (recessive) → released the bus
- Node B sent 0 (dominant) → pulled the bus dominant
- Node A reads back the bus and sees 0 — but it sent 1 → **arbitration lost**
- Node A immediately stops transmitting and becomes a receiver
- Node B continues, **completely unaware** there was a conflict

**Node B (ID=0x100) wins because it has the lower ID number.** Lower ID = more dominant bits = higher priority.

This is **non-destructive arbitration**: Node B's frame is transmitted perfectly. Node A will retry after the current frame ends. No data is lost or corrupted.

**Design implication:** Priority is built into the ID assignment. Safety-critical messages (E-stop, motor fault) should use the lowest IDs (0x000–0x00F) to guarantee they always win arbitration.

</details>

---

**A5.** CAN has five error detection mechanisms. Name all five and briefly explain what each checks.

<details><summary>Answer</summary>

| # | Error Type | What It Checks | How Detected |
|---|------------|---------------|--------------|
| 1 | **Bit Error** | Transmitter reads back each bit from the bus | Node sent dominant but reads recessive (or vice versa, outside arbitration field). During arbitration, this is normal (arbitration loss), not an error. |
| 2 | **Stuff Error** | Bit stuffing rule: no more than 5 consecutive same-polarity bits | Receiver detects 6 consecutive bits of the same value → stuffing violation |
| 3 | **CRC Error** | Data integrity of the frame payload | Receiver's calculated CRC doesn't match the received CRC field |
| 4 | **Form Error** | Fixed-format fields have correct values | SOF not dominant, or EOF/ACK delimiter/CRC delimiter not recessive |
| 5 | **ACK Error** | At least one node received the frame | Transmitter reads the ACK slot — if it's still recessive (nobody pulled dominant), no node acknowledged the frame |

When **any** node detects **any** error, it immediately transmits an **Error Frame** (6 dominant bits), which violates the bit stuffing rule and forces ALL nodes to discard the current frame. The original transmitter then **automatically retransmits** — this happens in hardware with zero software intervention.

This five-layer error detection catches virtually all corruption: the Hamming distance of 6 means it can detect up to 5 bit errors in a single frame.

</details>

---

**A6.** What happens when a CAN node's TEC (Transmit Error Counter) crosses 128 and then 256? Describe the three error states and why this design prevents a faulty node from disrupting the entire bus.

<details><summary>Answer</summary>

**Three error states:**

| State | Condition | Behavior |
|-------|-----------|----------|
| **Error Active** | TEC < 128 AND REC < 128 | Normal operation. Sends **active error frames** (6 dominant bits → immediately disrupts bus to flag corruption). This is correct — a healthy node that detects bad data should loudly flag it. |
| **Error Passive** | TEC ≥ 128 OR REC ≥ 128 | Suspected faulty node. Sends **passive error frames** (6 recessive bits → only visible if no one else is transmitting). Must also wait extra time before retransmitting. |
| **Bus-Off** | TEC ≥ 256 | Removed from bus completely. Cannot transmit or receive. Must detect 128 sequences of 11 recessive bits (bus idle) before recovering. |

**Why this protects the bus:**

If a node has a hardware fault (loose connector, dying transceiver), it starts generating errors. Each error increments its TEC/REC counters:
- First few errors: Error Active — the node can still flag genuinely bad data ✓
- After many errors (≥128): The node itself is probably the problem. It goes Error Passive — its error frames are recessive and don't disrupt other nodes' communication
- After extreme errors (≥256): The node is bus-offed — completely silenced. The remaining healthy nodes continue communicating without interference.

**Successful transmissions decrement the counters** (TEC -1 per success), so a node that has a brief fault (e.g., loose wire that's re-seated) gradually recovers from Error Passive back to Error Active.

This is a **democratic fault isolation system**: the more errors a node causes, the more it's silenced, until it's completely removed. No single node can permanently disrupt the bus.

</details>

---

**A7.** What is bit stuffing and why is it necessary for CAN? What's the worst-case overhead from bit stuffing?

<details><summary>Answer</summary>

**Bit stuffing:** After every 5 consecutive bits of the same polarity, the transmitter inserts one bit of the **opposite** polarity. The receiver removes these "stuff bits" automatically.

**Example:**
```
Original:  1 0 0 0 0 0 0 1
Stuffed:   1 0 0 0 0 0 [1] 0 1    ← [1] inserted after 5 zeros
                     ↑ stuff bit
```

**Why it's necessary:** CAN uses NRZ (Non-Return-to-Zero) encoding — the signal level stays constant during consecutive same-value bits. Without edges, the receiver's clock recovery mechanism loses synchronization. Bit stuffing guarantees a signal edge at least every 6 bit-times, keeping receivers synchronized.

This is analogous to UART's start bit providing synchronization per byte, but CAN does it within the frame at the bit level.

**Worst-case overhead:** If every 5th bit triggers a stuff bit:
- For every 5 data bits, 1 stuff bit is added → 5/6 data efficiency → **20% overhead**
- A standard CAN frame with 8 bytes of data: 111 bits → worst case ~133 bits = ~20% inflation
- Typical overhead is much lower because random data doesn't have runs of 5 identical bits on every boundary

**When calculating bus throughput, account for stuff bits.** The actual bit rate on the wire is higher than the data rate by up to 20%.

</details>

---

**A8.** On your Jetson Orin, you type `candump can0` and see frames. Explain what the SocketCAN architecture is and why CAN is treated like a network interface rather than a serial port on Linux.

<details><summary>Answer</summary>

**SocketCAN** is Linux's native CAN subsystem. It treats CAN interfaces like **network sockets** (similar to TCP/UDP), not serial ports. You interact via:

```
socket(PF_CAN, SOCK_RAW, CAN_RAW)  // Like AF_INET for TCP/IP
```

**Architecture layers:**
1. **User space:** `cansend`, `candump`, `python-can`, or your C/Python application
2. **Socket interface:** Standard POSIX socket API (`socket`, `bind`, `send`, `recv`)
3. **Kernel CAN framework:** `can.ko`, `can-raw.ko` — protocol handling, filtering, timestamping
4. **Hardware driver:** `mcp251x`, `peak_pci`, `slcan`, or `vcan` (virtual CAN for testing)
5. **CAN hardware:** Transceiver + controller

**Why network sockets, not serial ports:**

1. **CAN is a multi-access bus**, not point-to-point. Like Ethernet, any node can send to any other. Serial ports (`/dev/ttyS0`) are point-to-point abstractions.

2. **CAN has built-in addressing (IDs)** and filtering — similar to IP addresses and port numbers. This maps naturally to socket `bind` and `setsockopt(CAN_RAW_FILTER)`.

3. **CAN frames are structured** (ID + DLC + data), not byte streams. Socket APIs preserve message boundaries; serial APIs don't.

4. **Standard tools work:** `ip link show can0` shows interface status/errors, `tcpdump` can capture CAN frames, and `wireshark` can decode them.

5. **Multiple applications can read from the same bus** simultaneously using separate sockets, just like multiple UDP listeners. With a serial port, only one reader can access the device.

</details>

---

**A9.** What is CAN FD, and what problem does it solve? Why does CAN FD use different speeds for the arbitration phase and the data phase?

<details><summary>Answer</summary>

**CAN FD (Flexible Data-rate)** extends classic CAN with:
- **Up to 64 bytes per frame** (vs. 8 bytes in classic CAN)
- **Up to 8 Mbps data rate** (vs. 1 Mbps in classic CAN)

**What problem it solves:** Classic CAN's 8-byte payload is limiting. Sending a 32-byte parameter requires 4 frames → more bus overhead, more latency, more arbitration contention. CAN FD sends it in a single frame.

**Why two speeds:**

The **arbitration phase** (SOF + ID field) must run at the slower speed (≤1 Mbps) because **all nodes** participate:
- Every node monitors the bus during arbitration
- Every node reads back its own bits to detect arbitration loss
- The propagation delay between the farthest nodes on a long bus limits the maximum speed — all nodes must agree on the bus state within one bit time
- The entire bus length determines the timing budget

The **data phase** (payload + CRC) can run faster because only the **transmitter and one receiver** need to keep up:
- After arbitration is won, only one node is transmitting
- The receiver at the other end samples data bits — no read-back comparison needed
- Timing constraints are relaxed because it's unidirectional

```
Speed profile:
←── 500 kbps (arbitration) ──→←── 4 Mbps (data) ──→←── 500 kbps ──→
[SOF][ID][control]               [64 bytes data]      [CRC][ACK][EOF]
```

This dual-speed approach maintains backward compatibility (arbitration works the same as classic CAN) while dramatically increasing data throughput.

</details>

---

## Section B — Spot the Bug

**B1.** A robot CAN bus has 4 nodes: motor controller, BMS, IMU, and main computer. The bus is wired in a star topology from a central junction box. Each arm is about 30 cm. There are 120Ω termination resistors at the junction box and at the motor controller. At 250 kbps everything works. At 500 kbps, intermittent frame errors appear. At 1 Mbps, the bus is completely unreliable. The engineer says "we need better transceivers." What's actually wrong?

<details><summary>Answer</summary>

**Bug:** The bus uses a **star topology** instead of the required **linear (daisy-chain) topology.**

In a star topology, each arm creates an unterminated stub that reflects signals back to the junction. At low speeds (250 kbps, bit time = 4 µs), the reflections have time to settle before the sampling point. At higher speeds (1 Mbps, bit time = 1 µs), reflections arrive during the sampling window and corrupt the data.

Additionally, the termination is wrong: it should be at the two **physical ends** of a linear bus, not at the junction and one arm. In a star, there's no clear "end" — every arm endpoint is an unterminated stub causing reflections.

**Fix:**
1. Rewire the bus as a **linear daisy-chain**: Main → Motor → BMS → IMU (or any order)
2. Place 120Ω termination at the **first** and **last** node only
3. Keep stub lengths (branch from main bus to node connector) under 30 cm at 1 Mbps, ideally under 10 cm

The transceivers are fine — the problem is physical topology and signal integrity.

</details>

---

**B2.** A developer sets up a CAN bus for testing with only ONE node (their STM32H7). They send a frame and immediately get an ACK error. They check the code, transceiver wiring, and termination — everything looks correct. Why does the transmission always fail?

<details><summary>Answer</summary>

**Bug:** CAN requires **at least two nodes** on the bus for successful communication. The ACK mechanism is mandatory:

After the transmitter sends a frame, it checks the ACK slot. **At least one other node** must pull the ACK slot dominant (0) to acknowledge receipt. With only one node on the bus, nobody acknowledges → the ACK slot stays recessive → **ACK error** every time.

This is by design: CAN assumes that if nobody heard your message, something is wrong. The transmitter will retry, increment its TEC, and eventually go Error Passive → Bus-Off if it keeps failing.

**Fix for testing:**
1. Add a **second node** — even a simple CAN-to-USB adapter (like PEAK PCAN-USB) counts
2. Use **loopback mode** in the STM32 FDCAN peripheral — the controller internally acknowledges its own frames without needing a second node on the wire. This is for code testing only, not bus validation.
3. Use the **`vcan`** (virtual CAN) interface on Linux for pure software testing: `ip link add dev vcan0 type vcan`

</details>

---

**B3.** An engineer assigns CAN IDs as follows: E-stop command = 0x7FF, motor commands = 0x001–0x010, sensor data = 0x100–0x1FF. During heavy bus load, the E-stop command is delayed by up to 15 ms because motor and sensor frames keep winning arbitration. The engineer is confused because "the E-stop should have highest priority." What's wrong?

<details><summary>Answer</summary>

**Bug:** CAN priority is **inverted from what the engineer assumed.** In CAN arbitration, **lower ID = higher priority** because 0 (dominant) wins over 1 (recessive).

Their assignment:
- E-stop: **0x7FF** (0b111_1111_1111) = **lowest** possible priority!
- Motor: 0x001–0x010 = **very high** priority
- Sensor: 0x100–0x1FF = medium priority

During arbitration, the E-stop's ID (0x7FF) has recessive bits in positions where motor IDs (0x001) have dominant bits. The motor frame ALWAYS wins.

**Fix:** Invert the priority assignment:

| Message | ID Range | Priority |
|---------|----------|----------|
| E-stop | **0x000–0x00F** | **Highest** (dominant bits win) |
| Motor commands | 0x010–0x0FF | High |
| Sensor data | 0x100–0x1FF | Medium |
| Diagnostics | 0x400–0x5FF | Low |
| Heartbeat | 0x600–0x7FF | Lowest |

Now E-stop at ID 0x001 will **always** win arbitration against any other message on the bus — guaranteed by hardware.

</details>

---

**B4.** A CAN bus has 3 nodes and works perfectly. A fourth node is added (a new sensor module). Immediately, all communication on the bus fails — every frame produces errors. The new node's transceiver is verified to be correctly connected (CANH to CANH, CANL to CANL). Removing the new node restores communication. What's wrong with the new node?

<details><summary>Answer</summary>

**Bug:** The new node is configured with a **different bitrate** than the other three.

Even a slight bitrate mismatch (e.g., node 4 at 500 kbps, others at 250 kbps) causes the new node to interpret frames incorrectly. It sees what it thinks are errors (stuff errors, CRC errors, form errors) and transmits **Error Frames** — 6 dominant bits that destroy the current frame on the bus.

Since the new node is Error Active, its error frames are dominant and forcefully corrupt legitimately good frames. The other three nodes see constant errors and eventually also start sending error frames. The bus descends into a storm of error frames.

**Another possibility:** The new node has a 120Ω termination resistor built-in. If the bus already has two terminations, the third creates ~40Ω bus impedance (over-termination), reducing voltage swing and causing bit errors.

**Diagnosis:**
1. Check bitrate configuration on ALL nodes (must match exactly)
2. Measure CANH-CANL resistance with bus powered off — should be ~60Ω (not ~40Ω)
3. Read the new node's TEC/REC counters — if TEC is rapidly increasing, it's causing errors

**Fix:** Configure all nodes to the same bitrate. Remove any extra termination.

</details>

---

**B5.** A developer reads motor current values from a CAN bus. The motor controller sends 4-byte current data at ID 0x201 at 1000 Hz. The developer's `candump` output shows some frames arriving with 5 bytes (DLC=5) instead of 4. They suspect a firmware bug in the motor controller. What else could cause this?

<details><summary>Answer</summary>

**Bug:** This is **not** necessarily a motor controller bug. More likely causes:

1. **CAN FD vs Classic CAN mismatch:** If the motor controller sends classic CAN with DLC=4, but the receiver's CAN driver is configured for CAN FD mode, the DLC encoding might be misinterpreted. In CAN FD, DLC values 9–15 map to 12–64 bytes (non-linear). A DLC of 5 in classic CAN is literally 5 bytes, but check if any translation layer is corrupting the DLC.

2. **EMI-induced bit flip in the DLC field:** The DLC is only 4 bits. A single bit flip in the DLC (e.g., 0100 → 0101 = DLC 4 → DLC 5) would pass CRC validation only 1 in 2¹⁵ times — but CRC doesn't cover stuff bits, and a bit flip that's also a valid stuffed sequence could theoretically pass. More realistically, if the DLC bit flip is consistent with a correct CRC, it points to a firmware bug.

3. **Multiple senders on the same ID:** If another device on the bus also transmits on ID 0x201 with DLC=5, `candump` would show frames from both senders interleaved. Use `candump -ta can0,201:7FF` to filter and look at the data content — are the 5-byte frames from a different source?

**Diagnosis:** Check whether the extra byte on 5-byte frames has meaningful data or is always the same/zero. If zero-padded, it's likely a sender bug. If different data, it might be a different transmitter.

</details>

---

## Section C — Fill in the Blank / From Memory

### C1: CAN Voltage Levels

Fill in the CAN bus voltage levels:

| State | Logic | CANH | CANL | Vdiff | Active/Passive |
|-------|-------|------|------|-------|----------------|
| Dominant | ___ | ___ | ___ | ___ | ___ |
| Recessive | ___ | ___ | ___ | ___ | ___ |

<details><summary>Answer</summary>

| State | Logic | CANH | CANL | Vdiff | Active/Passive |
|-------|-------|------|------|-------|----------------|
| Dominant | **0** | **~3.5V** | **~1.5V** | **~2.0V** | **Actively driven by transceiver** |
| Recessive | **1** | **~2.5V** | **~2.5V** | **~0V** | **Passive — lines float via termination resistors** |

Key rule: **Dominant (0) always wins over recessive (1).** Any node driving dominant overrides all recessive states. This is the foundation of CAN arbitration.

</details>

---

### C2: CAN Error States

Fill in the error state machine:

| State | TEC/REC Condition | Error Frame Type | Can Transmit? | Recovery |
|-------|-------------------|------------------|---------------|----------|
| Error Active | ___ | ___ | ___ | ___ |
| Error Passive | ___ | ___ | ___ | ___ |
| Bus-Off | ___ | ___ | ___ | ___ |

<details><summary>Answer</summary>

| State | TEC/REC Condition | Error Frame Type | Can Transmit? | Recovery |
|-------|-------------------|------------------|---------------|----------|
| Error Active | **TEC < 128 AND REC < 128** | **Active (6 dominant bits)** | **Yes — normal operation** | **Successful transmissions decrement counters** |
| Error Passive | **TEC ≥ 128 OR REC ≥ 128** | **Passive (6 recessive bits)** | **Yes, but with extra delay** | **Successful comms → TEC/REC drop below 128** |
| Bus-Off | **TEC ≥ 256** | **None — cannot transmit** | **No** | **128 × 11 recessive bits detected → re-initialize** |

</details>

---

### C3: CAN Bus Speed vs Distance

Fill in the maximum bus length:

| Bit Rate | Max Bus Length | Common Use Case |
|----------|---------------|-----------------|
| 1 Mbps | ___ | ___ |
| 500 kbps | ___ | ___ |
| 250 kbps | ___ | ___ |
| 125 kbps | ___ | ___ |
| 10 kbps | ___ | ___ |

<details><summary>Answer</summary>

| Bit Rate | Max Bus Length | Common Use Case |
|----------|---------------|-----------------|
| 1 Mbps | **40 m** | **Short bus, fast response (robot internal)** |
| 500 kbps | **100 m** | **Automotive standard** |
| 250 kbps | **250 m** | **Trucks (J1939), industrial** |
| 125 kbps | **500 m** | **Light industrial automation** |
| 10 kbps | **5+ km** | **Building automation, very long runs** |

**Rule of thumb:** speed × distance ≈ constant. Faster = shorter bus.

</details>

---

### C4: CAN Frame Fields

Fill in the standard CAN frame structure:

| Field | Bits | Purpose |
|-------|------|---------|
| SOF | ___ | ___ |
| Identifier | ___ | ___ |
| RTR | ___ | ___ |
| IDE | ___ | ___ |
| DLC | ___ | ___ |
| Data | ___ | ___ |
| CRC | ___ | ___ |
| ACK | ___ | ___ |
| EOF | ___ | ___ |

<details><summary>Answer</summary>

| Field | Bits | Purpose |
|-------|------|---------|
| SOF | **1** | **Start of frame — always dominant (0). All nodes sync to this edge.** |
| Identifier | **11** | **Message ID and priority (lower = higher priority)** |
| RTR | **1** | **Remote Transmission Request: 0 = data frame, 1 = request** |
| IDE | **1** | **ID Extension: 0 = standard (11-bit), 1 = extended (29-bit)** |
| DLC | **4** | **Data Length Code: 0–8 bytes (classic CAN), up to 64 (CAN FD)** |
| Data | **0–64** | **Payload: 0 to 8 bytes (classic) or 0 to 64 bytes (CAN FD)** |
| CRC | **15+1** | **15-bit CRC + 1 delimiter bit. Hardware verifies automatically.** |
| ACK | **1+1** | **ACK slot + delimiter. Any receiver pulls ACK slot dominant = "I heard you."** |
| EOF | **7** | **End of frame: 7 recessive bits** |

Total minimum frame: 44 + 8×N bits (N data bytes) + stuff bits.

</details>

---

## Section D — Lab / Calculation Tasks

**D1.** Your robot has a CAN bus at 500 kbps. Motor commands (8 bytes, ID 0x010) are sent at 1000 Hz. IMU data (8 bytes, ID 0x100) at 200 Hz. BMS status (4 bytes, ID 0x200) at 10 Hz. E-stop (2 bytes, ID 0x001) at 50 Hz. Calculate the total bus utilization. Assume an average of 5% bit-stuffing overhead per frame.

<details><summary>Answer</summary>

**Step 1: Calculate bits per frame for each message type**

Standard CAN frame overhead (excluding data): SOF(1) + ID(11) + RTR(1) + IDE(1) + r0(1) + DLC(4) + CRC(15+1) + ACK(1+1) + EOF(7) + IFS(3) = **47 bits** overhead

| Message | Data Bytes | Data Bits | Total Bits | With 5% Stuffing |
|---------|-----------|-----------|------------|-------------------|
| Motor (8B) | 8 | 64 | 47 + 64 = 111 | 111 × 1.05 = **116.6** |
| IMU (8B) | 8 | 64 | 111 | **116.6** |
| BMS (4B) | 4 | 32 | 47 + 32 = 79 | 79 × 1.05 = **83.0** |
| E-stop (2B) | 2 | 16 | 47 + 16 = 63 | 63 × 1.05 = **66.2** |

**Step 2: Calculate bits per second**

| Message | Bits/frame | Rate | Bits/sec |
|---------|-----------|------|----------|
| Motor | 116.6 | 1000 Hz | 116,600 |
| IMU | 116.6 | 200 Hz | 23,320 |
| BMS | 83.0 | 10 Hz | 830 |
| E-stop | 66.2 | 50 Hz | 3,310 |
| **Total** | | | **144,060 bits/sec** |

**Step 3: Calculate utilization**

Bus utilization = 144,060 / 500,000 = **28.8%**

This is healthy. CAN best practices recommend keeping utilization under **50%** to allow headroom for retransmissions, error frames, and arbitration delays. At 28.8%, you have **21.2%** margin.

**Bonus:** The E-stop at ID 0x001 always wins arbitration, even at 28.8% utilization. Its worst-case latency is one maximum-length frame time = 111 bits / 500,000 = **222 µs** — well within real-time requirements.

</details>

---

**D2.** You receive a CAN frame with ID=0x123 and data bytes [0xDE, 0xAD]. Write the `cansend` command to transmit this frame on the `can0` interface. Then write the Python code using the `python-can` library to send the same frame and receive one frame with a timeout of 1 second.

<details><summary>Answer</summary>

**CLI command:**
```bash
cansend can0 123#DEAD
```

Format: `<ID>#<hex data bytes>`. No spaces, hex digits concatenated.

**Python code:**
```python
import can

# Create bus interface
bus = can.interface.Bus(channel='can0', interface='socketcan')

# Send frame
tx_msg = can.Message(
    arbitration_id=0x123,
    data=[0xDE, 0xAD],
    is_extended_id=False  # 11-bit standard ID
)
bus.send(tx_msg)
print(f"Sent: ID=0x{tx_msg.arbitration_id:03X} Data={tx_msg.data.hex()}")

# Receive one frame (1 second timeout)
rx_msg = bus.recv(timeout=1.0)
if rx_msg:
    print(f"Received: ID=0x{rx_msg.arbitration_id:03X} "
          f"DLC={rx_msg.dlc} Data={rx_msg.data.hex()}")
else:
    print("No frame received within timeout")

bus.shutdown()
```

**For testing without hardware:**
```bash
# Terminal 1: Create virtual CAN
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0

# Terminal 2: Listen
candump vcan0

# Terminal 3: Send
cansend vcan0 123#DEAD
```

</details>

---

**D3.** Your STM32H7 FDCAN peripheral needs to be configured for CAN FD at 500 kbps arbitration / 2 Mbps data phase. The FDCAN kernel clock is 80 MHz. Calculate the prescaler, number of time quanta per bit, and sample point for both the arbitration and data phases. Aim for 87.5% sample point.

<details><summary>Answer</summary>

**CAN bit timing fundamentals:**
- 1 bit = N time quanta (TQ)
- TQ = Prescaler / f_clock
- bit_rate = f_clock / (Prescaler × N)

**Arbitration phase (500 kbps):**

Target: 80 MHz / (Prescaler × N) = 500,000
→ Prescaler × N = 160

Good choices:
- Prescaler = 1, N = 160 TQ (too many — typical CAN controllers support 8–25 TQ)
- Prescaler = 8, N = 20 TQ ← good
- Prescaler = 10, N = 16 TQ ← also good

**Choose Prescaler = 10, N = 16 TQ:**
- TQ = 10 / 80 MHz = 125 ns
- Bit time = 16 × 125 ns = 2.0 µs → 500 kbps ✓

Sample point at 87.5%:
- SYNC_SEG = 1 TQ (always)
- TSEG1 = 16 × 0.875 - 1 = 13 TQ
- TSEG2 = 16 - 1 - 13 = 2 TQ
- Sample point = (1 + 13) / 16 = **87.5%** ✓

**Data phase (2 Mbps):**

Target: 80 MHz / (Prescaler × N) = 2,000,000
→ Prescaler × N = 40

**Choose Prescaler = 2, N = 20 TQ:**
- TQ = 2 / 80 MHz = 25 ns
- Bit time = 20 × 25 ns = 500 ns → 2 Mbps ✓

Sample point at 75% (slightly lower for data phase — more margin for transceiver delays):
- TSEG1 = 20 × 0.75 - 1 = 14 TQ
- TSEG2 = 20 - 1 - 14 = 5 TQ
- Sample point = (1 + 14) / 20 = **75%** ✓

**Zephyr DeviceTree:**
```
bus-speed = <500000>;        // 500 kbps
bus-speed-data = <2000000>;  // 2 Mbps
sample-point = <875>;        // 87.5%
sample-point-data = <750>;   // 75%
```

</details>

---

**D4.** Two CAN nodes arbitrate simultaneously. Node A sends ID 0x2A5 and Node B sends ID 0x2A3. Convert both to binary, perform the bit-by-bit arbitration, and determine: (a) which node wins, (b) at which bit position the loser detects arbitration loss, and (c) the winning frame continues unharmed.

<details><summary>Answer</summary>

**Binary conversion (11 bits, MSB first):**
- 0x2A5 = 0b010_1010_0101
- 0x2A3 = 0b010_1010_0011

**Bit-by-bit arbitration:**

| Bit Pos | 10 | 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
|---------|----|----|----|----|----|----|----|----|----|----|----| 
| 0x2A5 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | **1** | 0 | 1 |
| 0x2A3 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | **0** | 1 | 1 |
| Bus | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | **0** | | |

**(a) Node B (0x2A3) wins** — it has the lower numeric ID.

**(b) Arbitration loss detected at bit position 2:**
- Bits 10–3: both nodes send identical values → no conflict
- **Bit 2:** Node A sends 1 (recessive), Node B sends 0 (dominant)
- Bus = 0 (dominant wins)
- Node A reads back 0 but it sent 1 → **arbitration lost at bit 2**
- Node A immediately stops transmitting

**(c)** Node B continues transmitting bits 1 and 0, completely unaware there was a conflict. The frame is transmitted perfectly. Node A will retry after the frame ends and the bus returns to idle.

**Numeric rule confirmed:** 0x2A3 (675) < 0x2A5 (677) → the lower ID always wins.

</details>

---

**D5.** A CAN frame with ID=0x7E0 is used for OBD-II diagnostic requests. The data field contains [0x02, 0x01, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00] — this is a request for engine RPM. After the DLC field, draw the bit stream for the first 3 data bytes and mark where bit stuffing would be required (assume prior bits didn't leave a pending run).

<details><summary>Answer</summary>

**Data bytes in binary:**
- 0x02 = 0000_0010
- 0x01 = 0000_0001
- 0x0C = 0000_1100

**Bit stream (MSB first, data bytes concatenated):**

```
Byte 0x02:  0 0 0 0 0 0 1 0
Byte 0x01:  0 0 0 0 0 0 0 1
Byte 0x0C:  0 0 0 0 1 1 0 0
```

**Checking for stuffing (5 consecutive same-polarity bits):**

```
Position:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
Data bits: 0  0  0  0  0  0  1  0  0  0  0  0  0  0  1  0  0  0  0  1  1  0  0 ...
                    ↑                       ↑
                 bit 5: 5 zeros!         bit 13: 5 zeros again!
              stuff a 1 after bit 5     stuff a 1 after bit 13

After stuffing:
0 0 0 0 0 [1] 0 1 0 0 0 0 0 [1] 0 0 1 0 0 0 0 1 1 0 0 ...
          ↑ S              ↑ S
          stuff bit        stuff bit
```

Two stuff bits are inserted in just the first 3 bytes. The long run of zeros in 0x02 and 0x01 triggers stuffing twice.

**Observation:** Data with leading zeros (common in small values and zero-padded CAN frames) produces heavy bit stuffing — this is the worst case for CAN throughput overhead.

</details>

---

## Section E — Deeper Thinking

**E1.** Your OKS robot uses SPI for the IMU bridge (STM32↔Jetson, short distance, high speed) and CAN for motor controllers (longer cables, harsh environment). Both protocols need error detection, but they handle it completely differently. Compare the error handling philosophies of SPI (no built-in detection) vs CAN (5 hardware error mechanisms + automatic retransmission). Why is each approach appropriate for its use case?

<details><summary>Answer</summary>

**SPI's "no protection" approach:**
- SPI has zero built-in error detection — no CRC, no ACK, no parity
- Error detection is delegated to the **application protocol** layer (your CRC-16 in the frame header)
- Retransmission is handled in **software** (master detects CRC failure, re-requests data)
- No automatic retry — the application decides whether to retry, use stale data, or raise a fault

**CAN's "belt and suspenders" approach:**
- 5 hardware error detection mechanisms check every frame bit-by-bit
- Automatic retransmission in hardware — the transmitter retries without any software involvement
- Error counters (TEC/REC) automatically isolate faulty nodes
- Every node polices the bus — one bad frame is caught by ALL receivers

**Why each is appropriate:**

**SPI (board-level, short distance, controlled environment):**
- The master and slave are on the same PCB or <15 cm apart → error rate is extremely low (<<1 in 10⁶)
- Adding hardware CRC to SPI would increase silicon complexity for a vanishingly rare event
- The application-level CRC is cheap (2 bytes) and catches the rare corruption
- If a frame is corrupted, the master simply reads again at the next 100 Hz cycle — 10 ms latency is acceptable for IMU data
- There's only 2 devices — no need for bus arbitration, node isolation, or multi-node error policing

**CAN (fieldbus, 40+ m, factory floor with EMI):**
- Long wires act as antennas — EMI from motors, VFDs, and switching power supplies is constant
- Error rates can be 1 in 10³–10⁴ without protection — hardware detection is essential to maintain reliability
- Multiple nodes (motor, BMS, IMU, E-stop) share the bus — a faulty node MUST be automatically isolated
- Automatic retransmission ensures motor commands arrive even during brief EMI bursts — critical for safety
- The TEC/REC system prevents a single broken node from killing communication for all other nodes

**Summary:** SPI trusts the physical layer (short, clean) and delegates error handling to software. CAN distrusts the physical layer (long, noisy) and builds error handling into hardware. Both are correct for their respective environments.

</details>

---

**E2.** A safety engineer asks: "Can we use CAN for our E-stop system?" Analyze this from first principles. What are the failure modes of CAN that could prevent an E-stop message from being delivered? What are the guarantees CAN provides? Would you recommend CAN for safety-critical E-stop, and if so, what additional measures would you add?

<details><summary>Answer</summary>

**CAN's guarantees for E-stop:**
1. **Priority:** With the lowest ID (e.g., 0x001), the E-stop always wins arbitration — even under 100% bus load
2. **Automatic retransmission:** If the frame is corrupted, hardware retries immediately
3. **Error detection:** 5-layer detection catches virtually all corruption
4. **Worst-case latency:** At 500 kbps, one maximum-length frame = 111 bits / 500,000 = 222 µs. E-stop worst-case delay = this + arbitration time ≈ **~0.5 ms** — acceptable for most safety applications

**Failure modes that COULD block E-stop:**

| Failure Mode | Probability | Impact | Mitigation |
|-------------|------------|--------|------------|
| Bus-Off of E-stop node | Low | E-stop node can't transmit | Monitor TEC, redundant E-stop path |
| Bus wire broken | Low | Complete communication loss | Hardwired E-stop relay in parallel |
| EMI sustained for >100 ms | Very low | Continuous retransmissions fail | Redundant bus, hardwired backup |
| All other nodes Bus-Off | Very low | Nobody to ACK | Use loopback mode or dedicated listener |
| Software bug: wrong priority | Medium | E-stop loses arbitration | Verify ID assignment in code review |

**Recommendation: Yes, CAN is appropriate for E-stop, BUT with these additions:**

1. **Hardwired E-stop in parallel:** A physical relay or contactor that disconnects motor power when the E-stop button is pressed. This works even if the CAN bus, MCU, and all software are dead. This is mandatory for ISO 13849 / IEC 62443 compliance.

2. **Heartbeat monitoring:** The E-stop node sends periodic heartbeats (e.g., 10 Hz). If the motor controller doesn't receive a heartbeat within 200 ms, it assumes communication failure and enters safe state.

3. **Redundant CAN bus:** For SIL-2 or higher, use two independent CAN buses. E-stop message must arrive on at least one.

4. **Never rely solely on software E-stop over any communication bus** — a hardwired hardware interlock is the ultimate safety layer.

CAN provides excellent E-stop delivery guarantees (priority, retransmission, error detection), but ISO safety standards require defense-in-depth: the CAN E-stop is one layer, with hardwired backup as the fail-safe.

</details>

---

**E3.** You're debugging a CAN bus issue on a robot in the factory. The bus has 6 nodes. `ip -details link show can0` on the Jetson shows: state=ERROR-ACTIVE, restarts=0, bus-errors=847, error-passive=2, bus-off=0. Error breakdown: stuff=312, form=0, CRC=535, bit=0, ack=0. What can you deduce from these statistics? What is your diagnosis and next step?

<details><summary>Answer</summary>

**Analysis of the error counters:**

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| state=ERROR-ACTIVE | — | This node (Jetson) is healthy — its TEC/REC are both <128 |
| restarts=0 | — | Has never gone Bus-Off ✓ |
| bus-errors=847 | HIGH | 847 total errors is significant — something is wrong on the bus |
| error-passive=2 | — | Briefly went Error Passive twice, recovered → transient issue |
| stuff errors=312 | 37% | 5+ consecutive same-polarity bits detected — clock sync issue |
| CRC errors=535 | 63% | Data corruption — bits are flipping during transmission |
| bit errors=0 | 0% | This node's own transmissions read back correctly |
| ack errors=0 | 0% | Other nodes are acknowledging → they're alive |
| form errors=0 | 0% | Frame structure is correct → not a bitrate mismatch |

**Diagnosis:**

The combination of **high stuff errors + high CRC errors, but zero bit errors and zero form errors** points to:

1. **Signal integrity issue on the bus** — not a bitrate mismatch (form errors would be non-zero) and not this node's fault (bit errors = 0).

2. A remote node has **marginal signal quality**: its transmissions arrive at the Jetson with borderline voltage levels, causing some bits to be read incorrectly. This corrupts both the data (CRC error) and the stuff bit pattern (stuff error).

3. The error-passive=2 events suggest the problem is **intermittent** — possibly EMI-correlated (motor activity) or temperature-dependent (thermal drift in a transceiver).

**Next steps:**

1. **Scope the bus at the Jetson's transceiver input** — look at CANH-CANL differential waveform. Check for reduced voltage swing, slow edges, or ringing.

2. **Measure termination:** Expect ~60Ω between CANH and CANL (powered off). If ~120Ω, a termination resistor is missing.

3. **Tap the bus at different physical locations** — if signal quality degrades toward one end, that end's termination or wiring is suspect.

4. **Check each node's TEC/REC individually** — the node with the highest TEC is most likely the source. If one node is Error Passive, its transceiver may be failing.

5. **Correlate errors with motor activity** — if errors spike during motor acceleration, EMI from the motor driver is coupling into the CAN wiring.

</details>
