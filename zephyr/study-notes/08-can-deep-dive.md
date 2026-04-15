# CAN Bus Deep Dive — Study Notes
### Wire-level differential signaling, frame anatomy, arbitration, error state machine, CAN FD
**Context:** OKS robot uses raw CAN between MCU and motor controllers · STM32H7 bxCAN/FDCAN peripheral · 120Ω terminated bus

---

## PART 1 — THE PHYSICAL LAYER (ELECTRONS ON WIRE)

---

### 1.1 Differential Signaling — Why Two Wires Carry the Same Signal

**The analogy: noise-cancelling headphones**

Noise-cancelling headphones work by recording ambient noise with a microphone and playing back the INVERSE. The real audio + the inverse-noise cancel out the noise, leaving only the signal. CAN does the exact same thing with two wires.

CAN uses TWO wires:
- **CAN_H** (CAN High)
- **CAN_L** (CAN Low)

They carry the SAME signal, but mirrored. When CAN_H goes up, CAN_L goes down by the same amount. The receiver doesn't look at either wire individually — it reads the **difference** between them: `V_diff = CAN_H − CAN_L`.

**Why this kills noise:**

Electromagnetic interference (from motors, relays, power supplies) couples into BOTH wires equally — this is called **common-mode noise**. If a motor spike adds +0.5V to CAN_H, it also adds +0.5V to CAN_L (they're physically next to each other in the same cable). The difference is unchanged:

```
Without noise:  V_diff = 3.5V - 1.5V = 2.0V  ✓
With +0.5V EMI: V_diff = 4.0V - 2.0V = 2.0V  ✓  (same!)
```

This is why CAN works in cars (surrounded by ignition coils, alternators, solenoids) and in the robot (next to brushless motor drivers and power converters). A single-ended signal (like UART) would be destroyed by this noise. CAN doesn't even notice it.

**The actual voltage levels:**

```
Bus State     CAN_H    CAN_L    V_diff    Logic
─────────────────────────────────────────────────
Recessive     2.5V     2.5V     0.0V      1
Dominant      3.5V     1.5V     2.0V      0
```

Both wires idle at 2.5V (the "recessive" state). To transmit a dominant bit, the transceiver pushes CAN_H UP to ~3.5V and CAN_L DOWN to ~1.5V, creating a 2.0V differential. The receiver threshold is typically around 0.9V — anything above that reads as dominant.

**Voltage diagram (one dominant bit surrounded by recessive):**

```
Voltage (V)
    3.5 ─────┐ CAN_H           ┌───── CAN_H
             │                  │
    2.5 ─────┘──── idle ────────┘───── idle     ← both wires sit here
             │                  │
    1.5 ─────┘ CAN_L           └───── CAN_L
             │                  │
         ─────┬──────────┬──────┬──────────
              │ dominant │      │
              │  bit (0) │      │
```

> **Key insight for Python people:** In your ROS world, you never see voltages — you see `can_msgs/Frame` objects. But at the wire level, every `0` bit in that frame is a 2V differential pulse, and every `1` is the absence of a pulse. The entire CAN protocol — arbitration, error detection, everything — is implemented in these voltage transitions.

---

### 1.2 Why Dominant = 0 and Recessive = 1 — The Wired-AND Trick

This naming feels backwards until you understand the trick it enables.

**The wired-AND principle:**

```
If ANY node drives dominant (0) → bus is dominant (0)
Only when ALL nodes are recessive (1) → bus is recessive (1)

This is logical AND:  0 AND 1 = 0   (dominant wins)
                      1 AND 1 = 1   (recessive only if both recessive)
                      0 AND 0 = 0   (dominant wins)
```

**How it works electrically:**

Each CAN transceiver has two output transistors:
- One pulls CAN_H toward VCC (3.5V)
- One pulls CAN_L toward GND (1.5V)

To send **dominant (0)**: both transistors turn ON → CAN_H goes high, CAN_L goes low → 2V differential.
To send **recessive (1)**: both transistors turn OFF → bus floats to 2.5V on both wires (pulled by termination resistors) → 0V differential.

Because dominant requires ACTIVE driving and recessive is PASSIVE floating, a dominant bit from any single node overrides recessive bits from all other nodes. This is identical to I2C's open-drain SDA line — but implemented with differential signaling for noise immunity.

**Why this enables collision-free arbitration:**

When two nodes transmit simultaneously, they're both driving the bus. At each bit position:
- If both send 0 (dominant): bus = dominant. Both see what they sent. No conflict.
- If both send 1 (recessive): bus = recessive. Both see what they sent. No conflict.
- If one sends 0 and the other sends 1: bus = dominant (0 wins). The node that sent 1 reads back 0 — it KNOWS it lost. It backs off.

**No data is destroyed.** The winning node's frame continues uninterrupted. The losing node retries after the current frame ends. This is the core of CAN arbitration (explained in detail in §4).

> **Compare to Ethernet:** Ethernet also has collisions, but it uses a destructive collision mechanism — both frames are garbled, both nodes must wait a random time and retry. CAN's wired-AND means the higher-priority frame ALWAYS gets through on the first attempt.

---

### 1.3 Bus Topology — Why CAN is a True Bus

**CAN uses a linear bus topology:**

```
    120Ω                                                  120Ω
    ┌──┐    ┌───────┐    ┌───────┐    ┌───────┐         ┌──┐
────┤  ├────┤Node A ├────┤Node B ├────┤Node C ├── ∙∙∙ ──┤  ├──
────┤  ├────┤       ├────┤       ├────┤       ├── ∙∙∙ ──┤  ├──
    └──┘    └───────┘    └───────┘    └───────┘         └──┘
  terminator                                          terminator
  (bus end)                                           (bus end)
```

Each node connects to the main bus via a **short stub** (ideally < 0.3m at 1 Mbps). The bus itself is a linear backbone with termination resistors at each end.

**Why NOT a star topology:**

```
                ┌───────┐
                │Node A │
                └───┬───┘
                    │  ← long cable (problem!)
        ┌───────┐   │   ┌───────┐
        │Node B ├───┼───┤Node C │
        └───────┘   │   └───────┘
                    │
                ┌───┴───┐
                │Node D │
                └───────┘
```

A star topology creates long stubs. At high speeds, signal reflections from the unterminated stub ends interfere with the main signal. At 125 kbps you might get away with it. At 1 Mbps, the reflections corrupt bits and cause intermittent errors that are VERY hard to debug (because they depend on which combination of nodes is transmitting).

**Maximum stub length by bitrate:**

| Bitrate   | Max stub length | Max bus length |
|-----------|-----------------|----------------|
| 1 Mbps    | 0.3 m           | 40 m           |
| 500 kbps  | 1.0 m           | 100 m          |
| 250 kbps  | 2.0 m           | 250 m          |
| 125 kbps  | 5.0 m           | 500 m          |
| 10 kbps   | —               | 5 km           |

**In the robot:** The CAN bus runs from the main MCU board to motor controllers, with total cable length well under 2m. At 500 kbps or 1 Mbps, this is comfortably within spec. The concern is not cable length but proper termination and grounding.

---

### 1.4 Termination — The 120Ω Resistors

**Where they go:** One 120Ω resistor at EACH END of the bus. Not in the middle. Not at every node. Exactly two.

```
    120Ω                                         120Ω
    ┌──┐                                         ┌──┐
────┤  ├──── CAN_H ──────────────────── CAN_H ──┤  ├──
    │  │                                         │  │
────┤  ├──── CAN_L ──────────────────── CAN_L ──┤  ├──
    └──┘                                         └──┘
  End A                                        End B
```

The resistor connects CAN_H to CAN_L at each bus end. During recessive state, these resistors pull both lines to 2.5V (the bias voltage from the transceivers).

**Why 120Ω:** This matches the **characteristic impedance** of a standard twisted pair cable used for CAN (~120Ω). When the impedance of the termination matches the cable, signals reaching the bus end are absorbed instead of reflected back. It's the same principle as impedance matching in audio (50Ω antennas, 75Ω coax).

**What happens without termination:**

| Scenario | Bitrate | Symptom |
|----------|---------|---------|
| No termination at all | 125 kbps | May work intermittently with occasional CRC errors |
| No termination at all | 1 Mbps | Total failure — reflections corrupt most frames, 50%+ error rate |
| One terminator only | 500 kbps | Works sometimes, fails when bus is loaded (many nodes talking at once) |
| Terminator in middle of bus | Any | Reflections from both unterminated ends, unreliable |
| 3+ terminators | Any | Bus impedance drops too low, transceivers struggle to drive dominant |

**How to verify termination (power OFF the bus first):**

```
Measurement:  Multimeter between CAN_H and CAN_L
Expected:     ~60Ω  (two 120Ω resistors in parallel)

If you read:
  ∞ (open)    → No termination at all. Add two 120Ω resistors.
  120Ω        → Only one terminator present. Find the missing end.
  40Ω         → Three terminators. Remove one.
  30Ω or less → Multiple nodes have built-in termination enabled.
               Some transceiver modules (like the MCP2515 breakouts)
               have onboard 120Ω that you need to cut a trace to disable.
```

> **Robot gotcha:** Some CAN transceiver modules sold on Amazon/AliExpress come with a 120Ω resistor soldered on. If you put two of these on a bus that already has external terminators, you now have four resistors → ~30Ω → bus barely works. Always check.

---

### 1.5 The CAN Transceiver — Bridge Between MCU and Bus

The MCU's CAN peripheral speaks **logic-level** signals: TX (output, 0V or 3.3V) and RX (input, 0V or 3.3V). These CANNOT drive the differential CAN bus directly. The **transceiver** chip converts between the two worlds:

```
  ┌──────────┐              ┌──────────────┐              ╔══════════╗
  │   MCU    │    TX ──────→│              │── CAN_H ────→║          ║
  │ (STM32)  │              │  Transceiver │              ║ CAN Bus  ║
  │          │    RX ←──────│  (SN65HVD230)│←─ CAN_L ────║          ║
  └──────────┘              └──────────────┘              ╚══════════╝
    3.3V logic                 differential                  120Ω
    domain                     domain                      terminated
```

**Common transceiver chips:**

| Chip | VCC | Speed | Notes |
|------|-----|-------|-------|
| SN65HVD230 | 3.3V | 1 Mbps | Best for 3.3V MCUs (STM32). Most common in hobby/robot use. |
| MCP2551 | 5V | 1 Mbps | Classic 5V transceiver. Needs level shifting with 3.3V MCU. |
| TJA1050 | 5V | 1 Mbps | Very common in automotive. TX needs 0.7×VCC for high → fails at 3.3V! |
| TJA1051 | 5V | 5 Mbps | CAN FD capable. Has 3.3V-tolerant inputs. |
| MCP2562FD | 3.3V–5V | 8 Mbps | CAN FD. Works with 3.3V MCU AND 5V bus. Great choice. |

**The STBY (Standby) pin trap:**

Many transceivers have a STBY or S pin. If left floating or pulled HIGH, the transceiver enters standby mode — it appears dead (no TX, no RX). This is the #1 cause of "my CAN bus doesn't work" for beginners.

```
STBY pin states:
  LOW (GND)   → Normal operation (what you want)
  HIGH (VCC)  → Standby mode (transceiver disabled — bus looks dead)
  FLOATING    → Undefined. Some chips default to standby. Always tie to GND.
```

Fix: tie STBY to GND with a 0Ω resistor or solder bridge. Check your module's schematic.

---

### 1.6 Cable Requirements

**Twisted pair** is strongly recommended. The twisting ensures that EMI from nearby sources couples equally into both wires, maintaining common-mode rejection.

| Environment | Cable recommendation |
|-------------|---------------------|
| Bench / prototype | Any two wires work at 125 kbps. For 500+ kbps, use twisted pair. |
| Robot / industrial | Shielded twisted pair (STP). Shield connected to chassis ground at ONE end only (to avoid ground loops). |
| Vehicle / high EMI | Shielded twisted pair with drain wire. ISO 11898 compliant cable. |

**Bus length vs speed (ISO 11898):**

| Bitrate | Max bus length | Bit time |
|---------|---------------|----------|
| 1 Mbps  | 40 m          | 1 μs     |
| 500 kbps | 100 m        | 2 μs     |
| 250 kbps | 250 m        | 4 μs     |
| 125 kbps | 500 m        | 8 μs     |
| 50 kbps  | 1 km         | 20 μs    |
| 10 kbps  | 5 km         | 100 μs   |

The speed limit is set by signal propagation time — the signal must reach the farthest node AND the acknowledgment must return WITHIN one bit time. At 1 Mbps, one bit is 1 μs. Light in copper travels ~200m/μs, so round-trip in 1 μs limits you to ~40m.

---

## PART 2 — CAN FRAME ANATOMY (BIT BY BIT)

---

### 2.1 The Complete CAN Standard Frame (CAN 2.0A)

Every CAN frame has a fixed structure. Here is the COMPLETE standard (11-bit ID) frame, bit by bit:

```
┌─────┬────────────────┬─────────────┬──────────────┬─────────────┬──────────┬──────────┐
│ SOF │  Arbitration   │   Control   │    Data      │    CRC      │   ACK    │   EOF    │
│     │    Field       │   Field     │   Field      │   Field     │  Field   │          │
│ 1b  │  11+1 = 12b    │  1+1+4 = 6b │ 0-64 bits    │  15+1 = 16b │  1+1 = 2b│  7b      │
└─────┴────────────────┴─────────────┴──────────────┴─────────────┴──────────┴──────────┘
  │         │                │              │              │           │          │
  │    ID[10:0] + RTR    IDE+r0+DLC     payload bytes    CRC-15 +   ACK slot  7 recessive
  │                                                      delim      + delim    bits
  │
  1 dominant bit
```

**Bit-level detail for a frame with ID=0x123, DLC=2, Data=0xAB 0xCD:**

```
Bit #   Field           Value   Bus Level    Notes
──────────────────────────────────────────────────────────────────
  1     SOF             0       Dominant     Falling edge synchronizes receivers
  2     ID[10]          0       Dominant     ← MSB of 0x123 = 0b001_0010_0011
  3     ID[9]           0       Dominant
  4     ID[8]           1       Recessive
  5     ID[7]           0       Dominant
  6     ID[6]           0       Dominant
  7     ID[5]           1       Recessive
  8     ID[4]           0       Dominant
  9     ID[3]           0       Dominant
 10     ID[2]           0       Dominant     ← after 5 dominant (bits 2,3,5,6,8),
                                               stuff bit would be inserted if needed
 11     ID[1]           1       Recessive
 12     ID[0]           1       Recessive
 13     RTR             0       Dominant     0 = Data frame, 1 = Remote frame
 14     IDE             0       Dominant     0 = Standard (11-bit ID)
 15     r0              0       Dominant     Reserved, always dominant
 16     DLC[3]          0       Dominant     DLC = 2 = 0b0010
 17     DLC[2]          0       Dominant
 18     DLC[1]          1       Recessive
 19     DLC[0]          0       Dominant
 20-27  Data byte 0     0xAB    Mixed       1010_1011
 28-35  Data byte 1     0xCD    Mixed       1100_1101
 36-50  CRC[14:0]       varies  Mixed       15-bit CRC polynomial
 51     CRC delim       1       Recessive   Always recessive
 52     ACK slot        0*      Dominant     *Transmitter sends recessive;
                                             receiver(s) drive dominant
 53     ACK delim       1       Recessive   Always recessive
 54-60  EOF             1111111 Recessive   7 recessive bits
 61-63  IFS             111     Recessive   3-bit inter-frame space
```

Total frame length: 44 + (8 × DLC) bits PLUS stuff bits. For DLC=0: 44 bits minimum. For DLC=8: 108 bits minimum, up to ~130 with worst-case bit stuffing.

---

### 2.2 SOF — Start of Frame

One dominant bit. Seems trivial, but it serves a critical purpose: **hard synchronization**.

CAN uses NRZ (Non-Return-to-Zero) encoding — there's no separate clock line. All nodes must derive timing from the data transitions. When the bus goes from recessive (idle) to dominant (SOF), every receiver snaps its internal bit-timing counter to this edge. This is the only "hard sync" in the frame — all subsequent timing adjustments are "soft syncs" on data edges.

---

### 2.3 Arbitration Field — ID + RTR

**The 11-bit identifier:**

The ID is transmitted MSB first (most significant bit first). This is what makes priority-based arbitration work:

```
ID value    Binary (MSB first)      Priority
─────────────────────────────────────────────
0x000       000 0000 0000           HIGHEST (all dominant)
0x001       000 0000 0001           Very high
0x100       001 0000 0000           Medium
0x7FE       111 1111 1110           Very low
0x7FF       111 1111 1111           LOWEST (all recessive)
```

Lower ID number = higher priority because more dominant bits means it wins arbitration sooner.

**Design rule:** Assign the lowest IDs to the most critical messages.

```
Robot CAN ID assignment example:
  0x010  Emergency stop command        ← highest priority
  0x020  Motor torque command
  0x080  Encoder feedback
  0x100  Sensorbar data
  0x200  Diagnostics/heartbeat
  0x300  Firmware update data          ← lowest priority
```

**RTR (Remote Transmission Request) bit:**

- **0 (dominant)**: This is a data frame (has payload).
- **1 (recessive)**: This is a remote frame (requesting data from another node with this ID).

Because dominant wins: a data frame with the same ID always wins over a remote request for that ID. This ensures the actual data gets through before the request does.

> **In practice:** Remote frames are rarely used in modern CAN systems. Most systems just have nodes transmit data periodically. The robot doesn't use remote frames at all.

---

### 2.4 Control Field — IDE + r0 + DLC

| Bit | Name | Value | Meaning |
|-----|------|-------|---------|
| IDE | Identifier Extension | 0 = standard (11-bit), 1 = extended (29-bit) |
| r0 | Reserved | 0 (dominant) | Always 0 in CAN 2.0A |
| DLC[3:0] | Data Length Code | 0–8 | Number of data bytes |

DLC values 9–15 are technically allowed by the spec but mean "8 bytes" in classic CAN. In CAN FD, they map to longer payloads (see §6).

---

### 2.5 CRC Field — 15-bit Protection

Every transmitter calculates a 15-bit CRC (Cyclic Redundancy Check) over the SOF, arbitration, control, and data fields (with stuff bits REMOVED before CRC calculation). The polynomial is:

```
x^15 + x^14 + x^10 + x^8 + x^7 + x^4 + x^3 + 1
```

Every receiver independently calculates the same CRC and compares. If they don't match → CRC error → error flag → frame is discarded.

The CRC delimiter is always recessive (1). If a receiver sees dominant here → form error.

**Detection capability:** The 15-bit CRC detects:
- All single-bit errors
- All double-bit errors
- All odd-number-of-bit errors
- Any burst error up to 15 bits long
- 99.997% of all other error patterns

---

### 2.6 ACK Field — Every Receiver Votes

This is a clever two-bit slot:

```
Bit 1 (ACK slot):  Transmitter sends RECESSIVE (1)
                   Any receiver that got the frame correctly
                   drives DOMINANT (0) during this same bit time
                   → bus reads dominant → transmitter knows frame was received

Bit 2 (ACK delimiter): Always recessive (1)
```

**What the transmitter sees:**
- Dominant ACK slot → at least one node received the frame correctly → success
- Recessive ACK slot → nobody acknowledged → ACK error → transmit error counter +8

**What this means:** Even if the bus has 50 nodes and 49 of them had CRC errors, if ONE node received correctly and drove ACK, the transmitter considers it successful. The 49 nodes that had errors will discard the frame silently.

**What if there are NO other nodes on the bus?** The transmitter never gets an ACK → every frame triggers an ACK error → TEC increases +8 per frame → after 32 frames, TEC hits 256 → Bus Off. This is why a single CAN node on an unterminated bus quickly goes Bus Off.

---

### 2.7 EOF and Inter-Frame Space

**EOF:** 7 recessive bits. If any node sees a dominant bit during EOF → form error.

**Inter-Frame Space (IFS):** 3 recessive bits after EOF. During IFS, no node may start transmitting (except for overload frames, which are rare). After IFS, the bus is idle and any node may begin a new frame.

---

## PART 3 — BIT STUFFING

---

### 3.1 Why Bit Stuffing Exists

CAN uses **NRZ (Non-Return-to-Zero)** encoding. Each bit is simply a HIGH or LOW voltage that persists for one bit time. There's no transition between identical bits:

```
NRZ encoding of 0b11100001:
     ┌───┐┌───┐┌───┐               ┌───┐
     │   ││   ││   │               │   │
─────┘   └┘   └┘   └───┘───┘───┘───┘   └───
  1    1    1    1    0    0    0    0    1

        ↑ no transitions between identical bits!
        These long runs would lose clock sync.
```

The problem: receivers need transitions (edges) to keep their internal clocks synchronized with the transmitter. If 20 identical bits are sent in a row, the receiver's clock drifts and it mistakes bit boundaries. This is called **clock drift**.

**The solution:** After 5 consecutive bits of the same polarity, the transmitter AUTOMATICALLY inserts one bit of the OPPOSITE polarity. The receiver knows the rule and automatically removes these stuff bits.

---

### 3.2 Bit Stuffing Rules

1. Bit stuffing applies to SOF through CRC (inclusive)
2. CRC delimiter, ACK field, and EOF are NOT stuffed (they have fixed format)
3. After 5 consecutive identical bits, one complementary bit is inserted
4. Stuff bits participate in the bit count for further stuffing
5. If a receiver sees 6 consecutive identical bits → **stuff error**

---

### 3.3 Worked Example — Data Byte 0x00

Data byte 0x00 = `00000000` (eight zeros). Let's trace the bit stuffing:

```
Original:    0 0 0 0 0  0 0 0
             ─────────  ─────
             5 zeros    ← three more zeros

After 5 zeros, insert a stuff bit (1):

Stuffed:     0 0 0 0 0 [1] 0 0 0 0 0 [1] ...
                        ↑               ↑
                     stuff bit       stuff bit

Wait — after the stuff bit (1), we continue counting zeros. After the next 5 zeros, another stuff bit:

Input:   0 0 0 0 0 | 0 0 0
Output:  0 0 0 0 0 [1] 0 0 0 0 0 [1] 0 0 0

8 data bits → 8 + 2 = 10 transmitted bits (25% overhead)
```

**Worked example — Data byte 0xAA:**

0xAA = `10101010` (alternating). Maximum transitions, no consecutive runs > 2.

```
Original:    1 0 1 0 1 0 1 0
             No run exceeds 2 bits → zero stuff bits inserted
Stuffed:     1 0 1 0 1 0 1 0

8 data bits → 8 transmitted bits (0% overhead)
```

**Worst case for an entire frame:** A frame with DLC=8 and all-zero data can have up to ~24 additional stuff bits, making the frame ~130 bits instead of ~108. This variability means CAN frame timing is NOT constant — something to account for when calculating bus utilization.

---

### 3.4 Worked Example — Full Byte 0x1F = 0b00011111

```
Original bits:  0 0 0 1 1 1 1 1
                ─────           
                no stuffing (only 3 zeros before a one)
                      ─────────
                      5 ones → need a stuff bit!

Stuffed:        0 0 0 1 1 1 1 1 [0]
                                 ↑
                              stuff bit

8 data bits → 9 transmitted bits (12.5% overhead)
```

---

## PART 4 — ARBITRATION — THE COLLISION-FREE MAGIC

---

### 4.1 How Two Nodes Compete Without Destroying Data

This is CAN's killer feature. It's what makes CAN fundamentally different from Ethernet, RS-485, or any other multi-master bus.

**Setup:** Two nodes (A and B) both have frames to send. The bus is idle (recessive).

**Step 1:** Both nodes detect bus idle and begin transmitting simultaneously. Neither knows the other is transmitting.

**Step 2:** Both transmit SOF (one dominant bit). Both drive the bus dominant. Both READ the bus and see dominant. Since they both sent dominant and read dominant, no conflict. Both continue.

**Step 3:** Both transmit their ID, one bit at a time, MSB first. At each bit, each node:
1. Drives its bit onto the bus
2. Reads the bus back
3. Compares what it sent to what it read

If it sent RECESSIVE (1) but reads DOMINANT (0) → some other node is sending a dominant bit → that node has higher priority → **this node immediately stops transmitting and becomes a receiver**.

If it sent DOMINANT (0) and reads DOMINANT (0) → no conflict at this bit. Continue.

**Step 4:** The winning node continues transmitting its frame, completely unaware that arbitration happened. No data was lost. No time was wasted. No collision recovery needed.

---

### 4.2 Complete Worked Example

**Node A** wants to send ID = 0x64A = `0b 110 0100 1010`
**Node B** wants to send ID = 0x649 = `0b 110 0100 1001`

```
Bit#  ID bit  Node A sends  Node B sends  Bus result   Who wins?
──────────────────────────────────────────────────────────────────
 1    SOF     0 (dom)       0 (dom)       0 (dom)      tie
 2    ID[10]  1 (rec)       1 (rec)       1 (rec)      tie
 3    ID[9]   1 (rec)       1 (rec)       1 (rec)      tie
 4    ID[8]   0 (dom)       0 (dom)       0 (dom)      tie
 5    ID[7]   0 (dom)       0 (dom)       0 (dom)      tie
 6    ID[6]   1 (rec)       1 (rec)       1 (rec)      tie
 7    ID[5]   0 (dom)       0 (dom)       0 (dom)      tie
 8    ID[4]   0 (dom)       0 (dom)       0 (dom)      tie
 9    ID[3]   1 (rec)       1 (rec)       1 (rec)      tie
10    ID[2]   0 (dom)       0 (dom)       0 (dom)      tie
11    ID[1]   1 (rec) ←     0 (dom) ←     0 (dom)      NODE B WINS!
12    ID[0]   — backs off   1 (rec)       1 (rec)      B continues
```

**At bit 11:** Node A sent recessive (1) but read back dominant (0). Node A knows it lost. It immediately stops transmitting and begins receiving Node B's frame.

Node B sent dominant (0) and read back dominant (0). Node B doesn't even know Node A existed — it just continued transmitting normally.

**Notice:** Node B wins because ID 0x649 < 0x64A. The LOWER ID always wins because its bits are more dominant (more zeros in the MSB positions).

**After B's frame completes:** Node A detects bus idle and retransmits its frame. If no other node is competing, A's frame goes through on the very next attempt. Total overhead: just the time of one frame (B's frame). No random backoff, no exponential delay.

---

### 4.3 Priority Rules Summary

```
Rule 1:  Lower ID = Higher priority
         0x000 beats everything. 0x7FF loses to everything.

Rule 2:  Data frame beats Remote frame (same ID)
         Because RTR=0 (dom) for data, RTR=1 (rec) for remote.

Rule 3:  Standard frame beats Extended frame (same base ID)
         Because IDE=0 (dom) for standard, IDE=1 (rec) for extended.

Rule 4:  If two nodes have the SAME ID → bus error
         CAN spec says each ID must be unique to one producer.
         Two nodes transmitting the same ID will pass arbitration
         but corrupt each other's control/data fields.
```

---

## PART 5 — CAN ERROR HANDLING STATE MACHINE

---

### 5.1 The Five Error Types

CAN defines exactly five error conditions. Each one triggers a specific detection mechanism:

```
Error Type     Detected By        Condition
──────────────────────────────────────────────────────────────────
Bit Error      Transmitter        Sent bit X, read back bit Y
                                  (NOT during arbitration or ACK)
Stuff Error    Any node           6+ consecutive same bits
                                  (in stuffed portion of frame)
CRC Error      Receiver           Calculated CRC ≠ received CRC
Form Error     Any node           Fixed-form bit has wrong value
                                  (CRC delim, ACK delim, or EOF
                                   has dominant instead of recessive)
ACK Error      Transmitter        ACK slot remained recessive
                                  (nobody acknowledged the frame)
```

**When a node detects ANY error:**
1. It stops processing the current frame
2. It transmits an **error flag** (6 bits)
3. This destroys the frame for ALL nodes (on purpose)
4. All nodes discard the frame
5. It increments its error counter

---

### 5.2 Error Flags — Active vs Passive

**Active Error Flag:** 6 dominant bits.

This deliberately violates the bit-stuffing rule (max 5 consecutive same bits). Every other node on the bus ALSO detects this as a stuff error and discards the frame. The 6 dominant bits are LOUD — they drown out any ongoing transmission.

**Passive Error Flag:** 6 recessive bits.

Because recessive is passive (floating), this flag doesn't disturb the bus if nobody else is transmitting. It's a "quiet" error flag. Other nodes may or may not notice it.

**Why two types?** Error-active nodes are "trusted" — they're healthy and their error flags should be taken seriously. Error-passive nodes may be faulty (they've had many errors) and shouldn't be able to disrupt the bus for everyone else. A broken node that keeps erroring out will transition from active to passive, reducing its ability to disturb healthy nodes.

---

### 5.3 Error Counters — TEC and REC

Every CAN node maintains two counters:

```
TEC (Transmit Error Counter):
  +8  on transmit error
  −1  on successful transmit

REC (Receive Error Counter):
  +1  on receive error (first error in frame)
  +8  on receive error (if error flag detected after sending own error flag)
  −1  on successful receive
  Minimum value: 0 (never goes negative)
```

**Why asymmetric (TEC increments by 8, REC by 1):**

A faulty transmitter is FAR more dangerous than a faulty receiver:
- A faulty transmitter sends broken frames that ALL nodes see → network-wide disruption
- A faulty receiver only affects itself — it misses frames but doesn't prevent others from communicating

By punishing transmit errors 8× harder, a broken transmitter hits Bus Off after just 32 failed frames (`32 × 8 = 256`). A receiver with problems takes much longer to reach Bus Off, giving it more chances to recover.

---

### 5.4 The Three Error States

```
                    ┌──────────────┐
                    │ Error Active │   TEC < 128 AND REC < 128
                    │              │   • Can transmit immediately
                    │              │   • Sends ACTIVE error flags (6 dominant)
                    └──────┬───────┘
                           │ TEC ≥ 128 OR REC ≥ 128
                           ▼
                    ┌──────────────┐
                    │Error Passive │   128 ≤ TEC < 256, or REC ≥ 128
                    │              │   • Must wait 8 extra bit times
                    │              │     after bus idle before transmitting
                    │              │   • Sends PASSIVE error flags (6 recessive)
                    └──────┬───────┘
                           │ TEC ≥ 256
                           ▼
                    ┌──────────────┐
                    │   Bus Off    │   TEC ≥ 256
                    │              │   • Cannot transmit
                    │              │   • Cannot receive
                    │              │   • Node is DEAD on the bus
                    │              │   • Recovery: 128 × 11 recessive bits
                    └──────────────┘
```

**Transitions back:**
- Bus Off → Error Active: after seeing 128 occurrences of 11 consecutive recessive bits (= 1408 bit times). Both TEC and REC reset to 0.
- Error Passive → Error Active: when BOTH TEC < 128 AND REC < 128 (successful transmissions bring TEC down by 1 each time).

---

### 5.5 Worked Example — Error Counter Progression

A node transmits 10 frames. Frames 3, 4, and 7 fail (CRC error detected by receiver → error flag from receiver → transmitter sees bit error):

```
Frame  Result    TEC change   TEC value   REC change   REC value   State
──────────────────────────────────────────────────────────────────────────
  1    Success      −1           0*          −1           0*       Active
  2    Success      −1           0*          −1           0*       Active
  3    FAIL         +8           8            0           0        Active
  4    FAIL         +8          16            0           0        Active
  5    Success      −1          15            0           0        Active
  6    Success      −1          14            0           0        Active
  7    FAIL         +8          22            0           0        Active
  8    Success      −1          21            0           0        Active
  9    Success      −1          20            0           0        Active
 10    Success      −1          19            0           0        Active

* TEC cannot go below 0, so successful transmits at TEC=0 stay at 0.
```

After 10 frames (3 failures), TEC = 19. Still comfortably Error Active.

**How fast to Bus Off?** If EVERY frame fails: `256 ÷ 8 = 32` consecutive failed transmissions → Bus Off. This is by design — a consistently broken transmitter is silenced quickly.

---

### 5.6 Why Bus Off Exists

Consider a node with a hardware defect — maybe its transceiver is damaged and drives CAN_H permanently high (dominant). Without Bus Off:
- Every frame from any node would be corrupted (the stuck dominant overrides everything)
- The bus would be 100% dead — no communication at all
- Every other node keeps trying, all fail, nobody can communicate

With Bus Off:
- The faulty node's TEC rapidly climbs to 256
- It enters Bus Off and disconnects from the bus
- All other nodes continue communicating normally
- The faulty node can only recover by detecting 1408 consecutive recessive bits (which it can only see if its TX hardware stops driving the bus)

**Bus Off recovery strategies:**

| Strategy | Description | When to use |
|----------|-------------|-------------|
| Auto-recovery | MCU automatically attempts recovery after Bus Off | Most applications. The FDCAN peripheral has a register bit for this. |
| Manual recovery | Software must explicitly trigger recovery (usually after a delay) | Safety-critical: you want to diagnose WHY the node went Bus Off before putting it back. |
| Power cycle | Only way if the transceiver hardware is stuck | Last resort. If auto-recovery fails repeatedly, the hardware is broken. |

---

### 5.7 Practical Implications of the Error State Machine

**Scenario 1: Loose connector**

A node's CAN connector is making intermittent contact. Every few seconds, the connection is lost for a few milliseconds.

- During disconnection: the node's transmissions get ACK errors (→ TEC +8) and receives get CRC errors (→ REC +1)
- TEC climbs faster than REC
- If the intermittent connection causes ~16 consecutive transmit failures, TEC = 128 → Error Passive
- The node now waits 8 extra bit times before transmitting → lower bus load → more room for other nodes
- If the connector recovers, successful transmissions bring TEC back down by 1 per frame
- If it doesn't recover within another 16 failures, TEC = 256 → Bus Off

**Scenario 2: Bitrate mismatch**

One node is configured for 500 kbps, all others are at 1 Mbps. The mismatched node samples bits at the wrong time → every frame it receives has CRC errors → REC climbs. Every frame it transmits looks corrupted to everyone else → nobody ACKs → TEC climbs. TEC reaches 256 first (8 per failure vs 1) → Bus Off after 32 frames.

**Scenario 3: Terminated bus with only one node**

A developer connects a single node to a properly terminated bus. The node transmits but nobody ACKs → ACK error → TEC +8 → after 32 frames → Bus Off. This is normal and expected. The node recovers when another node is connected.

---

## PART 6 — CAN FD (FLEXIBLE DATA RATE)

---

### 6.1 What Changed in CAN FD

Classic CAN has two hard limits:
1. **Max data length:** 8 bytes per frame
2. **Max bitrate:** 1 Mbps

CAN FD (ISO 11898-1:2015) removes both limits:

```
                    Classic CAN         CAN FD
──────────────────────────────────────────────────────
Max data bytes      8                   64
Max bitrate         1 Mbps              8 Mbps (data phase)
CRC                 15-bit              17-bit (≤16 B) or 21-bit (>16 B)
Stuff bits in CRC   No                  Yes (fixed stuffing)
```

---

### 6.2 The BRS (Bit Rate Switch) Concept

CAN FD uses TWO bitrates within a single frame:

```
┌────────────────────────┬──────────────────────────────┬──────────────────┐
│  Arbitration phase     │  Data phase                  │  Back to nominal │
│  (nominal bitrate)     │  (higher bitrate)            │                  │
│  SOF → BRS             │  BRS → CRC delim             │  ACK → EOF       │
│  e.g. 500 kbps         │  e.g. 4 Mbps                 │  e.g. 500 kbps   │
└────────────────────────┴──────────────────────────────┴──────────────────┘
      ← all nodes can participate →   ← only FD nodes →     ← all nodes →
           in arbitration                  understand
```

**Why two rates?**
- Arbitration MUST run at the nominal rate so ALL nodes (including slow ones) can participate. The bit-by-bit comparison requires all nodes to agree on bit boundaries.
- Data phase only involves the transmitter and the intended receiver(s). It can run faster because timing precision between two nodes is easier than network-wide synchronization.

**The BRS bit:** 0 = no rate switch (data at nominal rate). 1 = data phase switches to the higher rate. The ESI (Error State Indicator) bit follows.

---

### 6.3 CAN FD Data Length Codes

CAN FD DLC values 0–8 work the same as classic CAN. Values 9–15 are redefined:

```
DLC value   Classic CAN bytes   CAN FD bytes
─────────────────────────────────────────────
  0             0                   0
  1             1                   1
  2             2                   2
  3             3                   3
  4             4                   4
  5             5                   5
  6             6                   6
  7             7                   7
  8             8                   8
  9             8                  12
 10             8                  16
 11             8                  20
 12             8                  24
 13             8                  32
 14             8                  48
 15             8                  64
```

Note: values are NOT arbitrary. You can't send 10 bytes in CAN FD — you must round up to 12. The controller pads unused bytes (typically with 0xCC or 0x00).

---

### 6.4 CRC Changes in CAN FD

Classic CAN uses a 15-bit CRC for all frames. CAN FD uses:
- **17-bit CRC** for frames with ≤16 data bytes
- **21-bit CRC** for frames with >16 data bytes

Additionally, CAN FD uses **fixed stuff bits** in the CRC field (one stuff bit after every 4 bits), unlike classic CAN which does NOT stuff the CRC field. This improves error detection at the longer data lengths.

---

### 6.5 Backwards Compatibility — The Trap

CAN FD frames have a different format than classic CAN frames (the FDF bit replaces the reserved r0 bit). What happens when classic and FD nodes share a bus:

```
FD node transmits FD frame:
  → Classic node sees FDF=1 where it expects r0=0 → form error
  → Classic node sends active error flag
  → Frame is destroyed for EVERYONE

Result: you CANNOT have classic and FD frames coexist on the same bus
        unless the FD nodes ALSO send classic frames for messages
        that classic nodes need to see.
```

**Migration strategy:** If you need classic nodes to receive some messages, have FD nodes send those specific IDs as classic CAN frames (DLC ≤ 8, no BRS). Use CAN FD frames only for messages that classic nodes don't need.

---

### 6.6 FDCAN on STM32H7

The STM32H743 has an **FDCAN** peripheral (not bxCAN like older STM32s). Key differences:

| Feature | bxCAN (STM32F4) | FDCAN (STM32H7) |
|---------|-----------------|-----------------|
| CAN FD support | No | Yes |
| Max data bytes | 8 | 64 |
| Message RAM | 3 TX mailboxes, 2 RX FIFOs | Configurable in dedicated RAM (up to 4K words) |
| TX buffers | 3 fixed mailboxes | Up to 32 dedicated buffers + TX FIFO + TX queue |
| RX buffers | 2 FIFOs × 3 deep | 2 FIFOs × 64 deep + 64 dedicated buffers |
| Filter type | Mask or list | Mask, range, exact, or classic filter |

**Message RAM:** FDCAN uses a dedicated static RAM block for storing frames. You configure how much RAM to allocate to TX buffers, RX FIFOs, filters, etc. The total is limited, and getting the allocation wrong causes silent failures (frames dropped without error).

In Zephyr, the FDCAN is accessed through the standard CAN API:

```c
#include <zephyr/drivers/can.h>

const struct device *can_dev = DEVICE_DT_GET(DT_NODELABEL(fdcan1));

struct can_frame frame = {
    .id = 0x123,
    .dlc = 8,
    .data = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08},
};

can_send(can_dev, &frame, K_MSEC(100), NULL, NULL);
```

---

## PART 7 — HIGHER-LAYER PROTOCOLS

---

### 7.1 CANopen

CANopen defines a standardized application layer on top of CAN:
- **Object Dictionary (OD):** Every node has a dictionary of parameters (index + subindex). Read/write any parameter remotely.
- **SDO (Service Data Object):** Request/response protocol for reading/writing OD entries. Slow but reliable (confirmed).
- **PDO (Process Data Object):** Broadcast data mapped from OD entries. Fast (unconfirmed). Used for real-time control data like motor speed, sensor values.
- **NMT (Network Management):** Start, stop, reset nodes. Node sends heartbeat to prove it's alive.
- **Emergency (EMCY):** Error reporting. Node broadcasts error codes when faults occur.

Used in: industrial robots, CNC machines, medical devices, some mobile robots.

---

### 7.2 J1939 (SAE)

The dominant CAN protocol in heavy vehicles (trucks, buses, agricultural, construction):
- Uses **extended CAN** (29-bit ID) exclusively
- The 29-bit ID encodes: priority (3 bits), PGN (18 bits), source address (8 bits)
- **PGN (Parameter Group Number):** defines the message type (e.g., PGN 65262 = engine temperature)
- **Transport Protocol:** for messages >8 bytes — splits into multiple CAN frames with reassembly
- **Address claiming:** nodes negotiate unique source addresses at bus startup

```
29-bit ID structure:
  ┌───────┬──────────────────┬────────────┐
  │ P (3) │    PGN (18)      │   SA (8)   │
  └───────┴──────────────────┴────────────┘
  Priority  Parameter Group    Source Addr
```

---

### 7.3 Raw CAN (What the Robot Uses)

No application-layer protocol — just raw IDs and data bytes, with meaning defined by the application.

```
Example robot CAN messages:
  ID 0x201, DLC=8: motor command
    byte[0:1] = left motor RPM (int16, big-endian)
    byte[2:3] = right motor RPM (int16, big-endian)
    byte[4:7] = reserved

  ID 0x301, DLC=8: encoder feedback
    byte[0:3] = left encoder count (int32, little-endian)
    byte[4:7] = right encoder count (int32, little-endian)
```

Advantages: simple, no overhead, full control.
Disadvantages: no standard, no remote configuration, no auto-discovery.

---

## PART 8 — CAN DEBUGGING

---

### 8.1 USB-CAN Adapters

| Adapter | Interface | OS support | Price | CAN FD |
|---------|-----------|------------|-------|--------|
| **CANable** (canable.io) | USB, socketcan | Linux (native), Windows (slcan) | ~$25 | v2.0 yes |
| **PCAN-USB** (Peak) | USB, socketcan | Linux, Windows (proprietary driver) | ~$200 | FD version ~$300 |
| **Kvaser Leaf Light** | USB | Linux, Windows (proprietary) | ~$150 | v2 yes |
| **USBtin** | USB, serial | Linux (slcan), Windows | ~$30 | No |

For most development work, a CANable ($25) is more than sufficient.

---

### 8.2 Linux SocketCAN Tools

Linux treats CAN as a network interface (like `eth0`). The `can-utils` package provides command-line tools:

```bash
# Set up CAN interface
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up

# Monitor all CAN traffic
candump can0
# Output:  can0  123   [8]  01 02 03 04 05 06 07 08
#          iface  ID   DLC  data bytes

# Send a frame
cansend can0 123#0102030405060708
#             ID#data (hex)

# Send a remote frame
cansend can0 123#R

# Monitor with timestamps
candump -ta can0
# Output: (1713168000.123456)  can0  123   [8]  01 02 03 04 05 06 07 08

# Show error frames
candump -e can0
# Error frames show up with ID starting with 0x20000000

# Show bus statistics
canbusload can0@500000
# Shows utilization percentage
```

**From Python (for ROS integration):**

```python
import can

bus = can.interface.Bus(channel='can0', interface='socketcan')

# Receive
msg = bus.recv(timeout=1.0)
if msg:
    print(f"ID: 0x{msg.arbitration_id:03X}, Data: {msg.data.hex()}")

# Transmit
msg = can.Message(arbitration_id=0x123, data=[0x01, 0x02, 0x03], is_extended_id=False)
bus.send(msg)
```

---

### 8.3 What to Look For When Debugging

**Error frames in candump:**

```bash
candump -e can0
# Error frames show as:  can0  20000004   [8]  00 08 00 00 00 00 60 00
#                              ^^^^^^^^
#                              Error frame flag + error type bits
```

Error flag bits in the arbitration ID:
- Bit 29: TX timeout
- Bit 28: Arbitration lost
- Bit 27: Controller error (stuff, form, CRC, bit, ACK)
- Bit 26: Transceiver error
- Bit 25: No ACK received
- Bit 24: Bus Off

**Common debug scenarios:**

| Symptom | Likely cause | Check |
|---------|-------------|-------|
| No frames at all | Termination, bitrate, transceiver, wiring | Resistance between CAN_H and CAN_L (should be ~60Ω). Check STBY pin. |
| Frames from only one direction | TX/RX swapped on one node | Swap CAN_H/CAN_L at one end |
| Frequent error frames | Bitrate mismatch, intermittent connection | Verify all nodes use same bitrate. Check connectors. |
| Node goes Bus Off | Hardware fault, stuck dominant | Disconnect suspect node. Does bus recover? |
| CRC errors only | Noise, marginal termination, long stubs | Add termination. Shorten stubs. Add shielding. |
| Frames received but data wrong | Endianness, byte order | Compare with spec. CAN doesn't define byte order — application must. |

---

### 8.4 The "Silent Bus" Checklist

When you connect a CAN adapter and `candump` shows absolutely nothing:

```
□ 1. Is the bus terminated? Measure: should read ~60Ω between CAN_H and CAN_L
□ 2. Are ALL nodes using the same bitrate? Even one mismatch kills the bus
□ 3. Is the transceiver powered? Check VCC on the transceiver chip
□ 4. Is the STBY pin LOW? Check with multimeter (some modules default to standby)
□ 5. Are TX and RX swapped? Try crossing CAN_H and CAN_L at one end
□ 6. Is any node actually TRANSMITTING? At least one node must send for traffic to exist
□ 7. Is the CAN peripheral enabled in your MCU firmware? Check Kconfig / HAL init
□ 8. Are you using the right CAN pins? STM32 has multiple CAN pin options (remapping)
□ 9. Is the adapter working? Try a loopback test (short TX to RX on the adapter)
□ 10. Ground connection? CAN_GND must be connected between nodes (not just H and L)
```

---

## GOTCHA TABLE

| # | Gotcha | What Goes Wrong | Fix |
|---|--------|----------------|-----|
| 1 | STBY pin floating | Transceiver in standby mode. Zero frames. | Tie STBY to GND |
| 2 | Only one terminator | Works at 125 kbps, fails intermittently at 500+ | Add second 120Ω at far end |
| 3 | Module has built-in 120Ω | Bus over-terminated (3+ resistors). Marginal signal. | Cut PCB trace or remove module's resistor |
| 4 | TJA1050 with 3.3V MCU | TX logic threshold needs 3.5V (0.7×5V). Never triggers. | Use SN65HVD230 (3.3V native) or level shifter |
| 5 | Bitrate mismatch | All frames CRC-error out. Node goes Bus Off in seconds. | Set identical bitrate on every node |
| 6 | TX/RX wiring swapped | Node sends but never receives, or vice versa | Swap CAN_H ↔ CAN_L or check pin mapping |
| 7 | Missing CAN_GND | Works on bench, fails when nodes are on different boards | Always connect CAN_GND between nodes |
| 8 | Same ID on two producers | Arbitration succeeds but data field is corrupted | Every producing node must have a unique ID |
| 9 | DLC says 8 but only 4 bytes meaningful | Receiver reads stale/padding bytes as real data | Document DLC semantics. Pad with 0x00 or 0xCC. |
| 10 | Assuming byte order | Motor does big-endian, MCU does little-endian → values are gibberish | Explicitly define and document byte order per CAN ID |
| 11 | Bus Off after power-on (single node) | No ACK because no other node → TEC → 256 → Bus Off | Expected behavior. Node recovers when peer joins. |
| 12 | STM32 CAN pin remap | Using wrong alternate function → TX pin just toggles, no CAN | Check AF number in datasheet. STM32H7: AF9 for FDCAN1. |
| 13 | CAN FD frame on classic bus | Classic node sees form error → active error flag → every FD frame killed | Don't mix FD and classic unless FD nodes also send classic frames |
| 14 | Forgotten bit stuffing in timing | Calculated frame time ignores stuff bits → bus overload | Add 20-25% margin for worst-case stuff bits |
| 15 | Error Passive node hogs bus time | Error Passive waits 8 extra bits → slows down, triggers timeouts | Diagnose root cause. Replace faulty node. |

---

## QUICK REFERENCE CARD

```
┌─────────────────────── CAN Quick Reference ────────────────────────┐
│                                                                     │
│  VOLTAGE LEVELS                                                     │
│    Recessive (1): CAN_H = 2.5V, CAN_L = 2.5V, diff = 0V           │
│    Dominant  (0): CAN_H = 3.5V, CAN_L = 1.5V, diff = 2.0V         │
│                                                                     │
│  FRAME STRUCTURE (Standard)                                         │
│    SOF(1) + ID(11) + RTR(1) + IDE(1) + r0(1) + DLC(4)             │
│    + DATA(0-64) + CRC(15) + DEL(1) + ACK(1) + DEL(1)              │
│    + EOF(7) + IFS(3)                                                │
│                                                                     │
│  PRIORITY                                                           │
│    Lower ID number = higher priority                                │
│    0x000 = highest,  0x7FF = lowest                                 │
│                                                                     │
│  BIT STUFFING                                                       │
│    After 5 consecutive same bits → insert 1 opposite bit            │
│    Applies SOF through CRC. Not in ACK/EOF.                        │
│                                                                     │
│  ERROR COUNTERS                                                     │
│    TEC: +8 on TX error, −1 on TX success                           │
│    REC: +1 on RX error, −1 on RX success                           │
│                                                                     │
│  ERROR STATES                                                       │
│    Error Active:   TEC < 128 AND REC < 128                         │
│    Error Passive:  TEC ≥ 128 OR REC ≥ 128                          │
│    Bus Off:        TEC ≥ 256 → recovery: 128 × 11 recessive bits   │
│                                                                     │
│  TERMINATION                                                        │
│    120Ω at each end. Measure powered off: ~60Ω between H and L.    │
│                                                                     │
│  SPEED / LENGTH                                                     │
│    1M → 40m   500k → 100m   250k → 250m   125k → 500m             │
│                                                                     │
│  CAN FD                                                             │
│    Data field: up to 64 bytes                                       │
│    Data phase: up to 8 Mbps                                         │
│    DLC 9-15 → 12,16,20,24,32,48,64 bytes                          │
│                                                                     │
│  LINUX SOCKETCAN                                                    │
│    ip link set can0 type can bitrate 500000 && ip link set can0 up │
│    candump can0            # receive                                │
│    cansend can0 123#AABB   # transmit                               │
│    candump -e can0         # show error frames                      │
│                                                                     │
│  TRANSCEIVER SELECTION                                              │
│    3.3V MCU → SN65HVD230 or MCP2562FD                              │
│    5V MCU  → TJA1050 or MCP2551                                    │
│    CAN FD  → MCP2562FD or TJA1051                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

*Next: CAN on Zephyr RTOS — devicetree overlay, Kconfig, filters, mailboxes, and ISR-safe queuing.*
