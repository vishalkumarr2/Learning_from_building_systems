# Exercises: CAN Bus — Encoder Velocity Frames
### Covers: Deep-dive session 07 — SN65HVD230 wiring, Zephyr CAN API, encoder frames, bus-off recovery

---

## Section A — Conceptual Questions

**A1.** Before writing a single line of firmware, you want to verify the physical CAN bus is healthy.
Describe the exact sequence of three checks: one multimeter check (power off), one `cansend`/`candump`
verification (two-terminal test), and what each result tells you.

<details><summary>Answer</summary>

**1. Multimeter check (power off):**
Place probes across CANH and CANL. Measure resistance. Expected: ~60Ω (two 120Ω termination
resistors in parallel). If you read >200Ω, one or both terminators are missing or open. If you
read <30Ω, there is a short between the lines. The SN65HVD230 is 3.3V-compatible — do NOT use
TJA1050 (5V only).

**2. candump + cansend (two terminals, host Linux):**
Terminal 1: `candump can0`
Terminal 2: `cansend can0 123#DEADBEEF`
Terminal 1 should echo the frame back (including its own frame as reflected by the bus). If nothing
appears, the interface is off — run `sudo ip link set can0 up type can bitrate 500000` first.

**3. What each result tells you:**
- 60Ω + candump echo → physical layer is healthy before any firmware is involved.
- No echo despite 60Ω → check D/R pin logic level on SN65HVD230 (must be driven low to enable Rx).
- Frame CRC errors in `dmesg | grep can` → wiring capacitance or missing stub termination.

</details>

---

**A2.** The `can_add_rx_filter_msgq()` API takes a `struct can_filter`. Explain the difference between
setting `filter.mask = 0x7FF` versus `filter.mask = 0x000`. Which do you use to receive **only**
the encoder frame with ID `0x101`, and why?

<details><summary>Answer</summary>

`filter.mask` is the bits that **must match** the `filter.id`. A `1` in a mask bit means "this bit
is checked"; a `0` means "don't care".

- `filter.mask = 0x7FF` → all 11 standard ID bits are checked → only frames with exactly `filter.id`
  pass through.
- `filter.mask = 0x000` → no bits are checked → every frame passes through (promiscuous).

To receive only `0x101`:
```c
struct can_filter enc_filter = {
    .flags = 0,
    .id    = 0x101,
    .mask  = CAN_STD_ID_MASK   /* 0x7FF */
};
can_add_rx_filter_msgq(dev, &enc_msgq, &enc_filter);
```

Using `0x000` as mask would also deliver encoder frames, but the node would receive every frame
on the bus (motor commands, battery status, etc.) and the message queue would fill immediately at
any bus activity.

</details>

---

**A3.** Your CAN bus enters BUS-OFF state. Explain:
1. What sequence of hardware events caused BUS-OFF (transmit error counter)?
2. What does the Zephyr `can_set_state_change_callback()` callback receive?
3. Why is calling `can_recover()` directly inside the state-change callback dangerous?

<details><summary>Answer</summary>

**1. Sequence causing BUS-OFF:**
The CAN controller maintains a transmit error counter (TEC). Each failed transmission increments TEC
by 8. When TEC exceeds 255, the controller enters BUS-OFF and disconnects itself from the bus
(stops both transmitting and receiving) to protect other nodes. Common causes: missing or incorrect
termination, wrong bitrate, loose wiring, or the STM32 transmitting while the transceiver's RS/S
pin (SN65HVD230) is in standby mode.

**2. Callback receives:**
`struct can_bus_err_cnt` with `tx_err_cnt` and `rx_err_cnt`, plus a `state` parameter of type
`enum can_state` (one of: `CAN_STATE_ERROR_ACTIVE`, `CAN_STATE_ERROR_PASSIVE`,
`CAN_STATE_BUS_OFF`, `CAN_STATE_STOPPED`).

**3. Danger of calling `can_recover()` in the callback:**
`can_set_state_change_callback()` fires from **interrupt context**. `can_recover()` is a blocking
call that initiates a hardware recovery sequence and may sleep. Calling any blocking function from
interrupt context causes undefined behaviour (often a kernel panic or immediate crash on
Zephyr/Cortex-M). The fix: use `k_work_submit()` inside the callback to schedule recovery
in a workqueue thread.

</details>

---

**A4.** You are parsing the encoder velocity frame:
```c
struct encoder_frame {
    int16_t left_vel;   /* mm/s */
    int16_t right_vel;  /* mm/s */
    uint32_t timestamp_ms;
};
```
The frame arrives as 8 CAN bytes. Write the exact `memcpy` call to unpack it safely from
`can_frame.data[]`, and explain why a direct cast (`(struct encoder_frame *)&frame.data`) is
incorrect on Cortex-M7.

<details><summary>Answer</summary>

**Correct:**
```c
struct encoder_frame enc;
memcpy(&enc, frame.data, sizeof(enc));  /* 8 bytes */
```

**Why cast is wrong:**
`can_frame.data` is a `uint8_t[]` array which may be aligned to 1 byte. `struct encoder_frame`
requires at minimum 2-byte alignment for `int16_t` fields (and 4-byte alignment for `uint32_t` on
Cortex-M7 with strict alignment). Casting the address directly gives a pointer with insufficient
alignment. Accessing it triggers a **UsageFault** (unaligned access fault) on Cortex-M7 when the
UNALIGN_TRP bit is set in CCR, or produces silently wrong results otherwise. `memcpy` is always
safe because it operates byte-by-byte.

</details>

---

**A5.** Your encoder thread calls `k_msgq_get(&enc_msgq, &frame, K_FOREVER)`. The CAN bus goes silent
(robot is lifted). What happens to the thread? How would you modify the call to detect CAN silence
within 500ms and log a warning?

<details><summary>Answer</summary>

With `K_FOREVER`, the thread **blocks indefinitely** until a message arrives. It consumes no CPU
(Zephyr puts it into `pend` state) but it also never runs — silent failure with no warning.

To detect 500ms silence:
```c
int ret = k_msgq_get(&enc_msgq, &frame, K_MSEC(500));
if (ret == -EAGAIN) {
    LOG_WRN("CAN encoder: no frame in 500ms — bus silent or robot stopped");
    /* increment a watchdog counter; trigger safe stop if count exceeds threshold */
    continue;
}
```

In production code you would also publish a "CAN timeout" event to ZBus so the state machine can
transition to degraded mode rather than continuing to use stale velocity data.

</details>

---

## Section B — Practical / Debug Scenarios

**B1.** You flash your firmware and run `candump can0` on the host. You see no frames, but your
Zephyr code calls `can_send()` every 100ms and returns 0 (success). The logic analyzer shows
valid CAN waveform on the CANH/CANL wires. What is the most likely cause?

<details><summary>Answer</summary>

The host Linux interface `can0` is **down** or configured with the wrong bitrate. `can_send()`
returning 0 means the STM32 successfully placed the frame on the bus and got an ACK (from the
SN65HVD230 transceiver's internal ACK or another node). The Jetson is not listening.

Diagnosis:
```bash
ip -details link show can0   # check if state is ERROR-ACTIVE or UP
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 500000
candump can0
```

If after bringing can0 up you immediately see `can0: arbitration lost` or `error frame` in candump
output, you have a bitrate mismatch between STM32 and host (check your `CONFIG_CAN_DEFAULT_BITRATE`
in Kconfig vs `500000` in the ip command).

</details>

---

**B2.** Your encoder velocity reading shows intermittent spikes: `left_vel` suddenly reads +32767 or
-32768 for one frame, then returns to normal. The logic analyzer shows the CAN frame is arriving
intact. What is the likely firmware bug?

<details><summary>Answer</summary>

Endianness mismatch. CAN data bytes default to big-endian (Motorola byte order) in automotive
standards, but `struct encoder_frame` is being `memcpy`d assuming little-endian (x86/ARM default).

When the high byte and low byte of `left_vel` are swapped, the value `0x0100` (256 mm/s) is read
as `0x0001` (1 mm/s), and near the int16 boundary values like `0x0080` (128 mm/s) become
`0x8000` (-32768 mm/s) — a sign-extension artifact of treating the MSB as the high byte.

Fix: either pack the struct in big-endian order on the STM32 transmitter, or swap bytes on the
receiver:
```c
enc.left_vel  = __builtin_bswap16(enc.left_vel);
enc.right_vel = __builtin_bswap16(enc.right_vel);
```
Always document the byte order in the CAN DBC or header file comment.

</details>

---

**B3.** You connect a second STM32 to the same CAN bus for testing. Now your first STM32 enters
BUS-OFF immediately on power-up. The second STM32 is also in BUS-OFF. What physical condition
causes this?

<details><summary>Answer</summary>

**Missing termination.** Without the 120Ω resistors at each end of the bus, every transmitted
dominant bit (0V differential) reflects back as an error. Both nodes see their own transmission as
an error, increment their TEC, and rapidly reach BUS-OFF.

Secondary possibility: both nodes are transmitting simultaneously with different data on the same
frame ID (`0x101`), causing arbitration loss to loop — but this produces `CAN_STATE_ERROR_ACTIVE`
with incremental TEC, not instant BUS-OFF.

Verify: measure CANH–CANL resistance with both nodes powered off → should be ~60Ω. If you read
open circuit (>10kΩ), terminators are missing. Add one 120Ω resistor across CANH/CANL at each
physical end of the cable.

</details>

---

## Section C — Code Reading

**C1.** This code is supposed to receive encoder frames and publish them to ZBus. Find the bug:

```c
K_MSGQ_DEFINE(enc_msgq, sizeof(struct can_frame), 4, 4);

void can_rx_thread(void *a, void *b, void *c) {
    struct can_frame frame;
    struct encoder_frame enc;
    while (1) {
        k_msgq_get(&enc_msgq, &frame, K_FOREVER);
        memcpy(&enc, frame.data, sizeof(enc));
        zbus_chan_pub(&encoder_chan, &enc, K_NO_WAIT);
    }
}

/* In main: */
can_add_rx_filter_msgq(can_dev, &enc_msgq, NULL);
```

<details><summary>Answer</summary>

**Two bugs:**

1. `can_add_rx_filter_msgq(can_dev, &enc_msgq, NULL)` — passing `NULL` as the filter means the
   kernel will accept the `NULL` pointer (behaviour is driver-defined, often crashes or matches
   nothing). A valid `struct can_filter *` must be passed. The promiscuous equivalent is a filter
   with `mask = 0x000`, not NULL.

2. `memcpy(&enc, frame.data, sizeof(enc))` — `sizeof(struct encoder_frame)` is 8 bytes, but
   `sizeof(struct can_frame)` is larger than `frame.data` alone (the `can_frame` struct also
   contains the frame ID, DLC, flags). This specific line is fine as written because `frame.data`
   is the data array start, but the preceding `k_msgq_get` copies the **entire** `can_frame` struct
   into `frame`, which is correct. However: there is no DLC check. If a different node sends a
   4-byte frame on the same ID, `memcpy` reads 8 bytes, silently copying 4 bytes of `frame.flags`
   into `enc.timestamp_ms`.

Fix:
```c
if (frame.dlc < sizeof(struct encoder_frame)) {
    LOG_WRN("Short encoder frame: dlc=%d", frame.dlc);
    continue;
}
```

</details>

---
