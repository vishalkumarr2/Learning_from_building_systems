# 03 — Op-Amps, ADC, and Sampling Theory
### Signal conditioning: getting clean numbers from messy analog signals
**Prerequisites:** Chapters 01 (R, C, filters) and 02 (diodes, transistors)
**Unlocks:** Understanding IMU data paths, ADC configuration on STM32, anti-aliasing, why your accelerometer readings are noisy

---

## Why Should I Care? (STM32/Jetson Project Context)

Your ICM-42688 IMU has an internal ADC that samples the MEMS accelerometer/gyroscope elements. Understanding how ADCs work tells you:
- Why the IMU has configurable ODR (Output Data Rate) and why higher isn't always better
- What the noise floor is (quantization + thermal noise) and when you've hit the hardware limit
- Why reading the IMU at 100Hz via SPI when its internal ODR is 1kHz is decimation (good, not aliasing)
- Why the MPU-6050's DLPF (Digital Low-Pass Filter) setting matters — it's the anti-alias filter

Op-amps matter because the analog signal chain *before* any ADC determines the quality of the digital data you get out. Garbage in = garbage out.

---

# PART 1 — OP-AMPS (OPERATIONAL AMPLIFIERS)

---

## 1.1 What an Op-Amp IS

An op-amp is a differential amplifier with:
- **Two inputs:** V+ (non-inverting) and V- (inverting)
- **One output:** tries to make Vout = A × (V+ - V-), where A is the open-loop gain (~100,000 to 1,000,000)
- **Two power supply pins:** Vcc+ and Vcc- (or Vcc and GND for single-supply)

```
    Schematic symbol:

            Vcc+
             │
         V+ ─┤\
              │  >── Vout
         V- ─┤/
             │
            Vcc-
            (or GND)
```

**ELI15 analogy — the differential thermostat:**

Imagine a thermostat that compares two thermometers:
- Thermometer V+ is in the room
- Thermometer V- is on the thermostat dial (your setpoint)
- The heater blasts at 100,000× the temperature difference

If V+ is even 0.01° above V-: heater goes full blast → room gets very hot. If V+ is even 0.01° below V-: AC goes full blast → room gets very cold.

This is useless on its own — the room either freezes or burns. You need **negative feedback** to make it useful: route the output back to the input to create self-regulation. This is exactly what happens when you connect the output back to V-.

---

## 1.2 The Two Golden Rules (with Negative Feedback)

When negative feedback is present (output connected to V- through some network):

```
    RULE 1: No current flows into V+ or V-
            (input impedance is > 1TΩ)

    RULE 2: V+ = V-
            (the op-amp adjusts Vout to force its inputs equal)
```

These two rules let you analyze ANY op-amp circuit by:
1. Writing V+ = V- (rule 2)
2. Writing KCL (current equations) assuming no current into the inputs (rule 1)
3. Solving for Vout

**Where the rules break down:**
- Without feedback (comparator mode): V+ ≠ V-, output slams to rail
- At very high frequencies: the op-amp can't respond fast enough → gain drops (GBW product)
- Near the power rails: output can't go above Vcc or below GND (unless rail-to-rail op-amp)
- Rule 1 is approximate: bias currents of a few nA flow in practice (matters for precision circuits)

---

## 1.3 Inverting Amplifier

```
            Rf
    Vin ─┤Rin├──┬──┤Rf├──┬── Vout
                │         │
               V- ─┤\     │
                   │  >───┘
               V+ ─┤/
                │
               GND
```

**Analysis using the golden rules:**

1. V+ = 0V (connected to GND)
2. V- = V+ = 0V (Rule 2 — the "virtual ground")
3. Current through Rin: I = (Vin - 0) / Rin = Vin / Rin
4. No current into V- (Rule 1), so all of I flows through Rf
5. Voltage at Vout: 0V - I × Rf = -Vin × (Rf / Rin)

```
    Gain = Vout/Vin = -Rf/Rin

    The negative sign means the output is INVERTED
    (positive input → negative output, and vice versa)
```

### Worked Example: Amplify a microphone signal by 10×

```
    Rin = 10kΩ,  Rf = 100kΩ
    Gain = -100k/10k = -10

    Input: 50mV peak sine wave (from microphone)
    Output: 500mV peak sine wave, inverted

    Check: 500mV is well within a ±12V power supply → won't clip ✓
```

---

## 1.4 Non-Inverting Amplifier

```
    Vin ──── V+ ─┤\
                  │  >──┬── Vout
            V- ──┤/    │
             │          │
            ┌┴┐    ┌───┘
            │Rin│   │
            └┬┘   ┌┴┐
             │    │Rf│
            GND   └┬┘
                   │
                  GND ← wait, this connects Vout → Rf → Rin → GND
```

Let me redraw more clearly:

```
                   ┌──────────────────────┐
                   │          Rf          │
    Vin ── V+ ─┤\  │                      │
                │  >──┬───────────────────┘─── Vout
           V- ─┤/    │
            │         │
            └──┤ Rin ├┘
            │
           GND
```

**Analysis:**

1. V+ = Vin (directly connected)
2. V- = V+ = Vin (Rule 2)
3. Current through Rin: I = Vin / Rin (from V- to GND)
4. Same current flows through Rf (Rule 1, no current into V-)
5. Vout = V- + I × Rf = Vin + (Vin/Rin) × Rf = Vin × (1 + Rf/Rin)

```
    Gain = Vout/Vin = 1 + Rf/Rin

    Always ≥ 1, non-inverting (same polarity as input)
```

### Worked Example: Gain of 5

```
    Rf = 40kΩ,  Rin = 10kΩ
    Gain = 1 + 40k/10k = 5

    Input: 0.5V from sensor
    Output: 2.5V → nice range for a 3.3V ADC
```

---

## 1.5 Voltage Follower (Buffer) — Gain = 1 but WHY It's Useful

```
    Vin ── V+ ─┤\
                │  >──┬── Vout = Vin
           V- ─┤/    │
            │         │
            └─────────┘    ← direct feedback, Rf = 0, Rin = ∞

    Gain = 1 + 0/∞ = 1
```

**"Why would I want gain = 1? I already have the voltage!"**

The answer is **impedance matching**. The op-amp has:
- **Extremely high input impedance** (~TΩ): it draws NO current from the source
- **Very low output impedance** (~Ω): it can drive significant current to the load

**Real-world problem it solves:**

```
    PROBLEM:                           SOLUTION:

    Sensor                              Sensor
    (high impedance,    → ADC           (high impedance)  → Buffer → ADC
     can only supply       (low                              (draws     (gets
     10µA without          impedance,                        nothing    driven
     voltage sagging)      demands                           from       properly)
                           100µA                             sensor)
                           during
                           sampling)

    Voltage SAGS!                       Voltage STABLE!
```

Remember the voltage divider loading problem in chapter 01? A buffer op-amp eliminates it entirely.

**When to use it:**
- Thermistor (100kΩ) → ADC: buffer needed (thermistor can't source sampling current)
- Voltage divider → ADC: buffer needed if divider uses high-value resistors
- IMU digital output → MCU: buffer NOT needed (digital signals are driven hard)

---

## 1.6 Summing Amplifier

```
    V1 ─┤R1├──┐
               │      Rf
    V2 ─┤R2├──┼──┤Rf├──┬── Vout
               │         │
    V3 ─┤R3├──┤   V- ─┤\ │
               │       │ >┘
              GND  V+ ─┤/
                    │
                   GND
```

```
    Vout = -Rf × (V1/R1 + V2/R2 + V3/R3)
```

If all R values are equal: Vout = -(V1 + V2 + V3). It adds analog signals. Used in audio mixers, DAC circuits, and sensor fusion.

---

## 1.7 Differential Amplifier

```
    V1 ─┤R1├──┬── V- ─┤\
               │        │  >── Vout
              ┌┴┐       │
              │Rf│  V+ ─┤/
              └┬┘   │
               │    ├┤R1├── V2
              Vout  │
                   ┌┴┐
                   │Rf│
                   └┬┘
                    │
                   GND
```

With matched resistors (R1 = R1, Rf = Rf):

```
    Vout = (Rf/R1) × (V2 - V1)
```

**Why should I care?** This is the core of:
- CAN bus receivers (measure differential signal)
- Current sense amplifiers (measure voltage across a shunt resistor)
- Bridge sensor amplifiers (Wheatstone bridge output)

---

## 1.8 Comparator — Op-Amp Without Feedback

Remove the feedback path. Now V+ and V- can be different, and the enormous gain makes the output slam to one rail or the other:

```
    If V+ > V-:   Vout = Vcc  (positive rail)
    If V+ < V-:   Vout = GND  (or negative rail)
```

Used for: zero-crossing detection, threshold detection, analog-to-digital conversion (SAR ADC uses a comparator internally!).

**Practical tip:** Don't use a regular op-amp as a comparator in high-speed circuits — dedicated comparator ICs (LM339, TLV3501) have faster response and proper output stages (often open-drain for logic-level compatibility).

---

## 1.9 Common Op-Amp ICs

| IC | Supply | Rail-to-Rail? | Bandwidth | Use Case |
|----|--------|--------------|-----------|----------|
| LM358 | 3-32V | Output only | 1MHz | General purpose, cheap |
| MCP6002 | 1.8-6V | Input + Output | 1MHz | Single-supply, battery circuits |
| OPA2340 | 2.5-5.5V | Input + Output | 5.5MHz | Precision ADC buffering |
| AD8605 | 2.7-5.5V | Input + Output | 10MHz | Low noise, precision |
| LMV324 | 2.7-5.5V | Output only | 1MHz | Quad op-amp, general |

**"Rail-to-rail" means:** output can swing to within ~10mV of the supply rails. A non-rail-to-rail op-amp on a 3.3V supply might only output 0.2V to 3.1V. For single-supply embedded circuits, always use rail-to-rail.

---

## 1.10 Practical Signal Chain: Sensor → ADC

```
    Analog Sensor ──→ [Anti-Alias Filter] ──→ [Buffer] ──→ [ADC]
    (high impedance)    (RC low-pass)          (op-amp)     (STM32)
    (noisy, out-of-     (removes freqs          (impedance   (needs stable,
     range voltage)      above Nyquist/2)        matching)    low-Z input)
```

**Example for a force sensor (0-5V output, read by 3.3V STM32 ADC):**

```
    Sensor ─┤10k├──┬──┤10k├──┬──┤ 3.3kΩ ├──┬── STM32 ADC pin
     (0-5V)        │          │              │
                  ┌┴┐        ┌┴┐           ┌┴┐
                  │20k│      op-amp        │100n│   (anti-alias RC filter)
                  └┬┘  (buffer)           └┬┘     f_c = 1/(2π×3.3k×100n) ≈ 480Hz
                   │                        │
                  GND        Divider:      GND
                             5V × 20k/(10k+20k) = 3.33V max ✓
```

---

# PART 2 — ADC (ANALOG-TO-DIGITAL CONVERTER)

---

## 2.1 What an ADC Does

An ADC takes a continuous analog voltage (e.g., 1.234V) and converts it to a discrete integer (e.g., 1530 out of 4095).

```
    Analog voltage                    Digital number
    0V ──────── 3.3V     →      0 ──────── 4095
    (continuous)                 (discrete, 12-bit)
```

The ADC is the bridge between the analog world (sensors, physics) and the digital world (MCU, math, software).

---

## 2.2 Resolution and LSB

**Resolution** is the number of bits in the ADC output.

```
    Number of steps = 2^N     (N = number of bits)
    LSB (Least Significant Bit) = Vref / 2^N
```

| Resolution | Steps | LSB at 3.3V Vref | Comment |
|-----------|-------|-------------------|---------|
| 8-bit | 256 | 12.9mV | Coarse — suitable for temperature, battery level |
| 10-bit | 1024 | 3.22mV | Moderate — many Arduino ADCs |
| 12-bit | 4096 | 0.806mV | Good — most STM32 ADCs |
| 16-bit | 65536 | 0.050mV | High precision — external ADCs like ADS1115 |
| 24-bit | 16,777,216 | 0.197µV | Extreme precision — load cells, strain gauges |

### Worked Example

STM32H743 has a 16-bit ADC (but in 12-bit mode typically). With Vref = 3.3V and 12-bit mode:

```
    ADC reads 2048.
    Voltage = 2048 × (3.3V / 4096) = 2048 × 0.000806V = 1.650V
```

If the actual voltage is 1.645V, the error is 5mV — less than 1 LSB. **This is the best the ADC can do.** You cannot measure more precisely than 1 LSB without oversampling or a higher-resolution ADC.

---

## 2.3 ADC Architectures

### SAR (Successive Approximation Register) — Most Common in MCUs

**How it works — binary search:**

```
    Step 1: Is the input > Vref/2?
            YES → MSB = 1, subtract Vref/2
    Step 2: Is the remainder > Vref/4?
            YES → next bit = 1, subtract
            NO  → next bit = 0
    Step 3: Continue for each bit...
    
    12-bit SAR needs 12 comparisons → 12 clock cycles per conversion
```

**ELI15:** You're guessing a number between 0 and 4095. You ask "is it above 2048?" Then "is it above 3072 (or 1024)?" Each question halves the range. After 12 questions, you know the exact number. This is exactly how a SAR ADC works, using a DAC + comparator internally.

```
    SAR ADC block diagram:

    Vin ──→ [Sample & Hold] ──→ [Comparator] ──→ [SAR Logic] ──→ Digital Output
                                    ↑
                                [Internal DAC]
                                    ↑
                              [SAR Register]
                              (binary search)
```

- Speed: 100ksps to 10Msps
- Resolution: 8-18 bits
- Power: moderate
- **This is what's inside your STM32H7**

### Sigma-Delta (ΔΣ) ADC — High Resolution, Slow

Uses oversampling and noise shaping. Samples at many MHz but outputs at kHz rates. Achieves 24-bit resolution.

- Speed: 10sps to 100ksps
- Resolution: 16-24 bits
- Power: low
- Used in: precision measurement (load cells, thermocouples, strain gauges)

### Flash ADC — Fast, Expensive

Uses 2^N comparators in parallel (a 8-bit flash ADC needs 255 comparators!). Converts in one clock cycle.

- Speed: 100Msps to 10Gsps
- Resolution: 6-10 bits
- Power: very high
- Used in: oscilloscopes, software-defined radio, high-speed data acquisition

---

## 2.4 Reference Voltage — Often More Important Than ADC Bits

Every ADC compares the input to a reference voltage. If the reference drifts, ALL measurements drift.

```
    True voltage = ADC_reading × (Vref / 2^N)
```

If Vref has 1% error, ALL your readings have 1% error — no matter if you use 8-bit or 24-bit ADC. A perfect 24-bit ADC with a noisy Vref is worse than a 12-bit ADC with a stable Vref.

**Sources of Vref:**
- **Internal reference:** STM32 has internal 1.2V reference. Accurate to ±1%. Convenient but limited precision.
- **VDDA pin:** Many MCUs use the analog supply voltage as Vref. If VDDA = 3.3V ±5%, your readings are ±5%.
- **External reference IC:** REF3030 provides 3.000V ±0.1%. Use for precision.

---

## 2.5 Input Impedance and the Buffer Problem

SAR ADCs have a sample-and-hold circuit: a switch + small capacitor (~5pF internally). When the switch closes, the capacitor must charge to the input voltage within the sampling time (~100ns).

```
    [Source] ──┤R_source├── Switch ──┤ C_sample ├── GND
                              │           (5pF)
                              │
                        Sampling happens here

    The source must charge 5pF through R_source in ~100ns
    If R_source is too high, the capacitor doesn't fully charge → reading is WRONG
```

**STM32 ADC rule:** Source impedance must be < 50kΩ for accurate readings at full speed.

- **Low-impedance source (voltage divider with 1kΩ resistors):** ✓ Works directly
- **High-impedance source (100kΩ thermistor):** ✗ Need a buffer op-amp (voltage follower) between sensor and ADC
- **Workaround:** Increase sampling time in ADC configuration (STM32 lets you set 3 to 810.5 ADC clock cycles per sample). Longer time = more time for capacitor to charge = tolerates higher source impedance.

---

## 2.6 INL and DNL — Nonlinearity

**DNL (Differential Nonlinearity):** Each step should be exactly 1 LSB wide. DNL tells you how much each step deviates. DNL = ±0.5 LSB means each step is between 0.5 and 1.5 LSB wide.

**INL (Integral Nonlinearity):** Over the full range, the transfer function should be a straight line. INL tells you the maximum deviation from ideal. INL = ±2 LSB means your readings might be off by 2 LSB from the "true" value.

**Practical impact:** A 12-bit ADC with ±3 LSB INL effectively has ~10-bit accuracy. The last 2 bits are meaningless.

---

## 2.7 STM32 ADC Quick Reference

```c
/* Zephyr ADC configuration (devicetree overlay): */
&adc1 {
    status = "okay";
    #address-cells = <1>;
    #size-cells = <0>;

    channel@0 {
        reg = <0>;
        zephyr,gain = "ADC_GAIN_1";
        zephyr,reference = "ADC_REF_INTERNAL";
        zephyr,acquisition-time = <ADC_ACQ_TIME(ADC_ACQ_TIME_TICKS, 480)>;
        zephyr,resolution = <12>;
    };
};
```

STM32H7 ADC features:
- **DMA mode:** DMA transfers ADC results to RAM without CPU intervention. Essential for continuous sampling.
- **Scan mode:** Convert multiple channels in sequence automatically.
- **Injected channels:** High-priority channels that interrupt ongoing scan (used for motor control).
- **Oversampling:** Hardware 16× oversampling → 16-bit effective resolution from 12-bit ADC.

---

# PART 3 — SAMPLING THEORY

---

## 3.1 Nyquist Theorem — The Most Important Theory You'll Use

```
    To accurately digitize a signal, you must sample at
    LEAST 2× the highest frequency component in the signal.

    f_sample ≥ 2 × f_max
```

**ELI15 — the dot-to-dot drawing:**

Imagine drawing a sine wave using dots. If you place dots very close together (high sample rate), you can clearly see the wave shape. If you place dots far apart (low sample rate), you might connect the dots and see a completely different wave — or even a straight line.

```
    Original signal (10Hz sine):

    ╱╲    ╱╲    ╱╲    ╱╲    ╱╲
   ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲
  ╱    ╲╱    ╲╱    ╲╱    ╲╱    ╲╱

    Sampled at 30Hz (3× — well above Nyquist):
    •   •   •   •   •   •   •   •   •
    Clear wave shape visible when connecting dots ✓

    Sampled at 12Hz (1.2× — barely above Nyquist):
    •       •       •       •
    Shape is ambiguous, edges are mushy

    Sampled at 8Hz (0.8× — BELOW Nyquist):
    •           •           •
    Connecting these dots shows a DIFFERENT, SLOWER wave!
    This is ALIASING — you see a ghost frequency that doesn't exist.
```

---

## 3.2 Aliasing — When You Violate Nyquist

When you sample below 2× the signal frequency, high-frequency content "folds" down into lower frequencies. The aliased frequency is:

```
    f_aliased = |f_signal - N × f_sample|

    Where N is chosen to minimize f_aliased
```

### Worked Example: The Wagon Wheel Effect

A 150Hz signal sampled at 200Hz:
```
    f_aliased = |150 - 1 × 200| = 50Hz

    You see a 50Hz signal that DOES NOT EXIST in the real world!
```

This is the same effect as a helicopter rotor appearing to spin backwards in video (frame rate < 2× rotor frequency).

```
    Frequency domain view:

    Real spectrum:                    Seen by ADC:
    
    │                                 │
    │       ╱╲                        │ ╱╲     ╱╲
    │      ╱  ╲                       │╱  ╲   ╱  ╲
    ├─────╱────╲──────────            ├────╲─╱────╲──
    0   100  150  200  300   Hz       0  50 100 150   Hz
           real signal                 alias! real
                                      (ghost)

    Nyquist frequency = f_sample/2 = 100Hz
    Everything above 100Hz "folds" back below 100Hz
```

**This is IRREVERSIBLE.** Once aliased, you cannot recover the original signal. No amount of digital filtering can undo it. You MUST filter BEFORE the ADC.

---

## 3.3 Anti-Aliasing Filter — MUST Come Before the ADC

An anti-aliasing filter is a low-pass filter placed between the analog signal and the ADC input. It removes all frequency content above f_sample/2.

**The simplest: an RC low-pass filter:**

```
    Sensor ──┤ R ├──┬── ADC input
                    │
                   ┌┴┐
                   │C│
                   └┬┘
                    │
                   GND

    f_cutoff = 1 / (2π × R × C)
```

### Worked Example: Anti-alias for 1kHz sampling

```
    f_sample = 1000Hz → Nyquist = 500Hz
    Target cutoff: 500Hz (or slightly below for margin — say 450Hz)

    Choose C = 100nF:
    R = 1 / (2π × 450 × 100e-9) = 3.54kΩ → use 3.3kΩ

    Actual cutoff = 1/(2π × 3300 × 100e-9) = 482Hz ✓
```

A single RC filter has -20dB/decade rolloff. At 1kHz (twice the cutoff), attenuation is only about -6dB (signal is halved). For stronger rejection, cascade two RC stages (-40dB/decade) or use a second-order active filter with an op-amp.

---

## 3.4 Oversampling — Trading Speed for Resolution

**Key insight:** If you sample much faster than Nyquist and average, you reduce noise and gain effective resolution.

```
    Extra bits = log₂(√(oversampling_ratio))

    Oversample by 4× → √4 = 2 → log₂(2) = 1 extra bit
    Oversample by 16× → √16 = 4 → log₂(4) = 2 extra bits
    Oversample by 64× → √64 = 8 → log₂(8) = 3 extra bits
```

### Worked Example

STM32 12-bit ADC, need 14-bit effective resolution:
```
    Extra bits needed = 2
    Oversampling ratio = (2^2)² = 16×

    Sample at 16× desired rate, average every 16 samples
    If target output rate = 1kHz → sample at 16kHz
```

STM32H7 has this built in hardware (the oversampling engine in the ADC peripheral), so it's free in terms of CPU time.

---

## 3.5 Quantization Noise

Even a perfect ADC has irreducible noise: the act of mapping a continuous voltage to a discrete integer loses information. The maximum error is ±0.5 LSB.

```
    Quantization noise RMS = LSB / √12

    For 12-bit, 3.3V:
    LSB = 3.3/4096 = 0.806mV
    Noise RMS = 0.806mV / 3.464 = 0.233mV
```

This sets the fundamental noise floor. Your signal-to-noise ratio can never exceed:

```
    SNR_max = 6.02 × N + 1.76 dB

    12-bit: SNR_max = 74dB
    16-bit: SNR_max = 98dB
```

In practice, thermal noise and power supply noise push the actual SNR well below this.

---

## 3.6 Your IMU Context: ICM-42688 Sampling

The ICM-42688 has:
- Internal 16-bit ADC for accel and gyro
- Configurable ODR: 1.6Hz to 32kHz
- Internal digital low-pass filter (DLPF) — acts as anti-alias filter

When you read it via SPI at 100Hz:

```
    IMU internal:   MEMS → [Anti-alias] → [16-bit ADC @ 32kHz] → [DLPF] → [FIFO]
    You via SPI:    [FIFO] → SPI read at 100Hz → Jetson

    This is DECIMATION (reading every Nth sample from a faster stream).
    It's safe because the DLPF already removed frequencies above your 50Hz Nyquist.
    
    Configure ODR=1kHz, DLPF bandwidth=100Hz → 
    The IMU gives you 1kHz data, anti-aliased at 100Hz.
    You read every 10th sample → 100Hz output, no aliasing. ✓
```

---

# PART 4 — DIGITAL BUILDING BLOCKS

---

## 4.1 Multiplexer (MUX)

N inputs → 1 output, selected by address bits.

```
    Input 0 ──┐
    Input 1 ──┤
    Input 2 ──┤ MUX ──── Output
    Input 3 ──┘
              ↑  ↑
             A1  A0  (address: 2 bits → select 1 of 4)
```

**Use cases:**
- Sharing one ADC across 8 analog sensors (read one at a time via address)
- GPIO expansion: 4 buttons → 1 GPIO pin + 2 address pins
- Audio routing: select one of N microphones

**Common IC:** CD4051 (8-channel analog MUX, I2C/GPIO-selectable)

---

## 4.2 Serializer and Deserializer

These are the core building blocks inside EVERY serial interface (SPI, UART, I2C):

```
    SERIALIZER (parallel-to-serial):

    8 parallel bits:  D7 D6 D5 D4 D3 D2 D1 D0
                       │  │  │  │  │  │  │  │
                       ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼
                      ┌───────────────────────┐
               CLK →  │    SHIFT REGISTER     │ → Serial out
                      └───────────────────────┘

    On each clock pulse, one bit shifts out on the serial line.
    After 8 clocks, all bits have been transmitted.

    This is EXACTLY what the SPI TX path does internally.
```

```
    DESERIALIZER (serial-to-parallel):

    Serial in → ┌───────────────────────┐
          CLK → │    SHIFT REGISTER     │
                └───────────────────────┘
                 │  │  │  │  │  │  │  │
                 ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼
                D7 D6 D5 D4 D3 D2 D1 D0

    On each clock pulse, one bit shifts in from the serial line.
    After 8 clocks, all 8 bits are available in parallel.

    This is EXACTLY what the SPI RX path does internally.
```

---

## 4.3 Shift Register — The Core Building Block

The 74HC595 (serial-in, parallel-out) is the simplest example. You can chain them to get unlimited output pins from just 3 GPIO lines:

```
    MCU GPIO (Data) ──→ 74HC595 ──→ 74HC595 ──→ ... (daisy-chained)
    MCU GPIO (Clock) ─→    ↑           ↑
    MCU GPIO (Latch) ─→    ↑           ↑

    8 outputs per chip. 3 chips = 24 outputs from 3 MCU pins.
```

The 74HC165 is the input equivalent (parallel-in, serial-out) — read 8 buttons from 3 pins.

**This is exactly how SPI daisy-chain mode works:** each slave device's shift register output feeds into the next slave's input.

---

## 4.4 Integrated Circuits (ICs) — Reading Datasheets

An IC is any complex circuit packaged into a single component. When you encounter a new IC:

**Read page 1 of the datasheet first.** It tells you:
1. **What it does** (op-amp, ADC, voltage regulator, IMU, etc.)
2. **Supply voltage range** (e.g., 2.7V–5.5V → works on 3.3V ✓)
3. **Key parameter** (bandwidth, resolution, Rds_on, etc.)
4. **Package** (DIP, SOIC, QFP, BGA — determines how you solder/socket it)
5. **Pin count** and **pinout diagram**

**Common packages:**

```
    DIP (Dual In-line Package) — through-hole, breadboard-friendly
    ┌──────────────┐
    │  1 ●         │ 8
    │  2           │ 7
    │  3    IC     │ 6
    │  4           │ 5
    └──────────────┘

    SOIC (Small Outline IC) — SMD, same pinout as DIP but flat
    QFP (Quad Flat Package) — pins on all 4 sides, used for MCUs (STM32)
    BGA (Ball Grid Array) — solder balls on bottom, high pin count (Jetson Orin)
```

---

# GOTCHA TABLE

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|-------------|-----------------|-----|
| ADC reads 0 or 4095 always | Input out of range, or pin not configured | Measure actual voltage on ADC pin with multimeter | Check voltage divider scaling, ADC channel config |
| ADC readings are noisy (±10 LSB jitter) | Poor decoupling on VDDA, no anti-alias filter, high-impedance source | Scope VDDA for noise. Add 100nF+10µF. Check source impedance | Decouple VDDA, add RC filter, add buffer op-amp |
| ADC reads wrong voltage | Wrong Vref assumption in software | Measure actual Vref with multimeter. Check if using internal or external ref | Use correct Vref in calculation |
| See 50Hz in data (mains hum) | Electromagnetic interference, poor grounding | FFT the ADC data, look for 50/60Hz peak | Shield sensor wires, improve grounding, add common-mode filter |
| See unexpected low-frequency oscillation | Aliasing — sampling below Nyquist | Compare sample rate to anti-alias filter cutoff | Add or fix anti-alias filter before ADC |
| Op-amp output clips at ~1V below VCC | Not a rail-to-rail op-amp | Check datasheet for output swing specification | Use rail-to-rail op-amp (MCP6002, OPA2340) |
| Op-amp oscillates (squeals) | Long feedback path, phase margin issue | Scope the output — see oscillation at high freq | Add 10-100pF cap across Rf. Shorter PCB traces. |
| Buffer reads wrong voltage | Op-amp offset voltage | Measure with no input — should be 0V but shows a few mV | Use precision op-amp with lower Vos |
| Oversampled reading not improving | Noise is not random (systematic offset) | Averaging removes random noise, not offset | Calibrate out the offset, check for systematic error source |

---

# QUICK REFERENCE CARD

```
┌──────────────────────────── OP-AMP / ADC / SAMPLING CHEAT SHEET ─────────────────────────────┐
│                                                                                               │
│  OP-AMP GOLDEN RULES (with negative feedback):                                               │
│    1. No current flows into V+ or V-                                                         │
│    2. V+ = V-                                                                                │
│                                                                                               │
│  GAINS:   Inverting:      Av = -Rf/Rin                                                       │
│           Non-inverting:  Av = 1 + Rf/Rin                                                    │
│           Buffer:         Av = 1  (but transforms impedance!)                                │
│                                                                                               │
│  ADC:     Resolution:     LSB = Vref / 2^N                                                  │
│           Voltage:        V = ADC_reading × Vref / 2^N                                      │
│           STM32 12-bit:   LSB = 3.3V/4096 = 0.806mV                                        │
│                                                                                               │
│  NYQUIST:  f_sample ≥ 2 × f_max                                                             │
│  ALIASED:  f_alias = |f_signal - N × f_sample|                                              │
│                                                                                               │
│  ANTI-ALIAS FILTER:  f_c = 1/(2πRC)  (place BEFORE ADC, always)                             │
│  OVERSAMPLING BITS:  extra = log₂(√(OSR))   (4× = +1 bit, 16× = +2 bits)                  │
│                                                                                               │
│  SNR:     SNR_max = 6.02N + 1.76 dB   (theoretical best for N-bit ADC)                     │
│  QUANT NOISE:  σ = LSB/√12                                                                  │
│                                                                                               │
│  COMMON OP-AMPS:  LM358 (cheap), MCP6002 (rail-to-rail), OPA2340 (precision)               │
│  ADC TYPES:  SAR (fast, moderate res) | ΔΣ (slow, high res) | Flash (very fast, low res)    │
│                                                                                               │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```
