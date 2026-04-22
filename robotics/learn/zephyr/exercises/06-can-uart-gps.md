# Exercises: CAN Bus + UART GPS NMEA Parser
### Covers: Projects 5–6 — CAN encoder receive, UART ring buffer, NMEA sentence parsing
**These exercises assume you have read `deep-dive/07-can-bus-encoder.md` and `deep-dive/08-uart-gps-nmea.md`.**

---

## Section A — CAN Bus Conceptual Questions

Answer in your own words. If you can explain it simply, you understand it.

---

**A1.** You wire up a CAN node using a TJA1050 transceiver and a 3.3V STM32. Describe the exact
failure mode that will occur. Why does the TJA1050 not work here? What chip should you use instead,
and why?

---

**A2.** A technician measures 112Ω between CANH and CANL on a running 2-node CAN bus. Is this
correct, too high, or too low? Where should the two 120Ω termination resistors be placed on the
bus, and why does placement matter (not just total resistance)?

*(Hint: think about signal reflection.)*

---

**A3.** You connect a USB-CAN adapter to the bus and run `candump can0`. The terminal is completely
silent — no frames appear, no errors. Meanwhile your STM32 firmware calls `can_send()` every
100ms and returns 0 (success). List the three most likely causes in order of probability. How
do you confirm which one is the culprit?

---

**A4.** Explain "bus-off" in your own words:
- What sequence of events puts a CAN node into bus-off state?
- What is the error counter threshold at which bus-off is entered?
- What happens to a node in bus-off state (can it send? receive? does it affect other nodes)?
- How do you recover from bus-off in Zephyr?
- Why might a bitrate mismatch produce intermittent bus-off rather than immediate hard failure?

---

**A5.** Your CAN encoder sends this frame:

```
ID: 0x201
DLC: 8
Data: F4 01 0C FE 39 30 00 00
```

Using the encoder frame struct definition:
```c
struct encoder_frame {
    int16_t  left_vel_mms;    // bytes 0-1, little-endian
    int16_t  right_vel_mms;   // bytes 2-3, little-endian
    uint32_t timestamp_ms;    // bytes 4-7, little-endian
};
```

Calculate: what are `left_vel_mms`, `right_vel_mms`, and `timestamp_ms`?  
Convert `left_vel_mms` and `right_vel_mms` to m/s.

---

**A6.** In Zephyr, `can_add_rx_filter_msgq()` takes a `can_filter` struct. You want to receive:
- All frames with IDs 0x200–0x20F (encoder frames from any of 16 wheels)

Write the `can_filter` struct initialization to match exactly this range using a mask filter.
Explain what the mask field means (which bits does `1` match against?).

---

**A7.** Explain why you must call `can_set_state_change_callback()` in a production CAN receiver,
not just `can_add_rx_filter_msgq()`. What will happen without it if CANH is shorted to ground
mid-operation?

---

## Section B — UART / GPS Conceptual Questions

---

**B1.** Explain why you cannot process a UART receive callback as if it contains a complete NMEA
sentence. What causes NMEA sentences to arrive fragmented? How does the baud rate and sentence
length interact to determine the number of callbacks needed for one sentence?

*(Show the calculation for `$GNGGA` at 9600 baud: how many bytes per sentence, how many
milliseconds to transmit, how many 1ms timer ticks could theoretically fire during one sentence?)*

---

**B2.** The Zephyr ring buffer API uses `ring_buf_put()` and `ring_buf_get()`. Explain what
happens when `ring_buf_put()` is called from UART IRQ context and the buffer is full. Is data
dropped or does it block? How should you handle this case in a GPS parser?

---

**B3.** Validate this NMEA sentence checksum:

```
$GNGGA,120000.00,3540.123456,N,13940.654321,E,1,08,0.9,45.2,M,36.7,M,,*XX
```

The checksum is the XOR of all characters between `$` and `*` (exclusive). What is `XX` in hex?
*(You need to calculate this by hand — step through the XOR character by character or write a
5-line Python script to verify.)*

---

**B4.** Convert these two raw NMEA coordinate values to decimal degrees:

- Latitude: `3540.123456,N`
- Longitude: `13940.654321,E`

Show the formula and intermediate steps. Why does NMEA use `DDMM.MMMMMM` format instead of
decimal degrees, and what is the classic off-by-one mistake beginners make when parsing it?

---

**B5.** Your GPS loopback injection test works perfectly (`inject_nmea` produces `lat=35.67`),
but when you connect the real u-blox M8N module, you see `lat=0.0` and `fix=0`. List four
plausible causes in order of likelihood. Which one is most common indoors? What is the minimum
satellite count for a valid GGA fix?

---

**B6.** Write pseudocode for a `parse_nmea_line()` function that:
1. Validates the checksum
2. Identifies the sentence type (`$GNGGA`, `$GNRMC`, etc.)
3. Returns `false` for unrecognized types without crashing

Then explain: your buffer contains `"$GNGGA,12000\r\n$GNRM"` — two partial sentences in one
string. Describe exactly how your ring-buffer accumulation logic handles this without state
corruption.

---

## Section C — Hardware Verification Drills

These are "write the exact command" questions — muscle memory for hardware debugging.

---

**C1.** Before writing any Zephyr CAN code, you should verify the physical bus with a USB-CAN
adapter. Write the exact Linux commands to:
1. Bring up `can0` at 500kbps
2. Start capturing all frames to terminal
3. Send a test frame with ID 0x123, DLC 8, data `DE AD BE EF 00 01 02 03`
4. Verify the bitrate setting was applied correctly

---

**C2.** You have a u-blox M8N connected to a Linux laptop via USB-UART adapter at `/dev/ttyUSB0`.
Write the exact command to see raw NMEA output at 9600 baud without any installation.

---

**C3.** You need to simulate a UART GPS input for testing without real hardware. Your Zephyr shell
has an `inject_nmea` command. Write the full `$GNGGA` sentence you would inject for:
- Time: 10:30:00 UTC
- Position: 35°40.123456'N, 139°40.654321'E
- Fix type: 1 (GPS fix)
- Satellites: 8
- Altitude: 45.2m

*(Include a correct checksum.)*

---

## Section D — Integration: Both Together

---

**D1.** In your ZBus+nanopb bridge (session 09), the packer thread reads from three ZBus channels:
IMU, CAN odometry, and GPS. The GPS updates at 1Hz, IMU at 100Hz. In a typical 10ms frame:
- Which channels will have new data?
- What does your packer thread do for the GPS channel when there's no new message?
- Specifically, what ZBus function should it use to get the last-known value without blocking?

---

**D2.** After receiving a CAN encoder frame, you compute a forward velocity and angular velocity and
publish them on ZBus. Then the packer thread reads them. If the CAN bus goes silent for 500ms
(cable disconnected), what does the Jetson see in the `odom.vel` field of decoded frames? Is this
safe default behavior? What should you add?

---

## Answers

<details>
<summary>A1</summary>

TJA1050 requires 5V VCC and has a CANH/CANL input threshold of ~3.5V. A 3.3V STM32 TX drives
only 3.3V, which falls within the TJA1050's "undefined" region. The transceiver will either not
drive the bus at all or produce erratic dominant/recessive levels. The SN65HVD230 is 3.3V VCC
compatible and its logic thresholds are matched to 3.3V CMOS outputs.

</details>

<details>
<summary>A2</summary>

112Ω is approximately correct (slightly low due to measurement tolerance). Ideally you measure
60Ω (two 120Ω in parallel) with power off. The two terminators belong at the physical ends of
the bus cable — one at each extreme. Placement matters because reflections occur at impedance
discontinuities; a terminator in the middle leaves one end unterminated, causing ringing and
bit errors at high speed.

</details>

<details>
<summary>A5</summary>

Data bytes: `F4 01 0C FE 39 30 00 00`
- left:  bytes [0,1] = 0xF4, 0x01 → int16 little-endian = 0x01F4 = **+500 mm/s = +0.500 m/s**
- right: bytes [2,3] = 0x0C, 0xFE → int16 little-endian = 0xFE0C = 65036 unsigned → 65036 - 65536 = **-500 mm/s = -0.500 m/s**
- timestamp: bytes [4,7] = 0x39, 0x30, 0x00, 0x00 → uint32 LE = 0x00003039 = **12345 ms**

The robot is spinning in place (left wheel forward, right wheel backward at equal speed).
A typical AMR AMR maximum speed is ≈ 2000 mm/s; 500 mm/s is a normal operating speed.

</details>

<details>
<summary>B3</summary>

XOR of all chars between $ and * in `GNGGA,120000.00,3540.123456,N,13940.654321,E,1,08,0.9,45.2,M,36.7,M,,`
This is tedious by hand — use Python: `reduce(xor, map(ord, s))`. The checksum will be a
two-digit hex value. The key skill is knowing the rule: XOR of every byte between `$` and `*`,
exclusive.

</details>

<details>
<summary>B4</summary>

Latitude `3540.123456,N`: degrees = 35, minutes = 40.123456
Decimal degrees = 35 + (40.123456 / 60) = 35 + 0.668724... = 35.668724° N

Longitude `13940.654321,E`: degrees = 139, minutes = 40.654321
Decimal degrees = 139 + (40.654321 / 60) = 139 + 0.677572... = 139.677572° E

Classic mistake: treating `3540.123456` as `35.40123456` (treating the decimal point as the
degrees/minutes boundary) instead of correctly splitting at the last two digits before the
decimal point.

</details>
