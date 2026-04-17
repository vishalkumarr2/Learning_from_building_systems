# Exercises: Op-Amps, ADC, and Sampling Theory

### Chapter 3: Signal conditioning — getting clean numbers from messy analog signals

**Self-assessment guide:** Write your answer before expanding the details block. If you can answer 80% without peeking, you've mastered the analog signal chain.

---

## Section A — Conceptual Questions

**A1.** Your colleague says "the STM32 has a 12-bit ADC, so we get 12 bits of accuracy." You check the datasheet and see INL = ±3 LSB. How many bits of *effective* accuracy do you really have?

<details><summary>Answer</summary>

INL = ±3 LSB means the transfer function deviates from the ideal straight line by up to 3 LSB. This means the last ~2 bits of the 12-bit reading are unreliable — they're within the error band. The effective accuracy is roughly **10 bits** (2^10 = 1024 steps, each ~3.2mV at 3.3V Vref).

To put it concretely: if the ADC reads 2048, the actual voltage could be anywhere from (2048 - 3) to (2048 + 3) LSBs = 2045 to 2051, which maps to 1.649V to 1.654V. The bottom 2 bits are noise/error, not signal. If you need true 12-bit accuracy, you'd need an external ADC with INL ≤ ±0.5 LSB.

</details>

**A2.** A teammate reads the ICM-42688 IMU at 100Hz via SPI, and notices that accelerometer data sometimes has strange low-frequency oscillations that don't correspond to robot movement. The IMU's internal ODR is set to 100Hz with DLPF disabled. What's likely causing this?

<details><summary>Answer</summary>

With the DLPF **disabled** and ODR at 100Hz, there's no anti-alias filter before the IMU's internal decimation to 100Hz output. The IMU's internal ADC samples at a much higher rate (32kHz), but the MEMS accelerometer picks up **high-frequency vibration** from motors, wheels, and structural resonances (e.g., 200Hz, 500Hz). 

Without the DLPF, these frequencies above 50Hz (Nyquist for 100Hz ODR) **alias** into the 0–50Hz band. A 150Hz vibration appears as a phantom |150 - 100| = **50Hz** signal. A 200Hz vibration appears at |200 - 200| = **0Hz** (DC offset shift!). These aliased signals look like real low-frequency motion but are ghosts.

**Fix:** Enable the DLPF with bandwidth ≤ 50Hz (the Nyquist frequency for 100Hz ODR). Or increase ODR to 1kHz with DLPF at 100Hz bandwidth, then decimate in software — this is the recommended approach for your SPI bridge.

</details>

**A3.** Your colleague designs a non-inverting amplifier with gain = 100 using an LM358 op-amp (GBW = 1MHz). The sensor signal is 500Hz. They say "gain is 100, frequency is only 500Hz, well below 1MHz, so it should work." What's the actual bandwidth at gain = 100?

<details><summary>Answer</summary>

The **gain-bandwidth product (GBW)** is constant for a given op-amp. Available bandwidth = GBW / Gain = 1MHz / 100 = **10kHz**. 

At 500Hz the amplifier should work (500Hz << 10kHz), so the colleague is right for pure sine wave amplification. BUT: if the sensor signal has sharp edges or harmonics above 10kHz, those components will be attenuated, distorting the waveform. And at 10kHz, the gain has already dropped to 1/√2 of its DC value (-3dB point).

More critically: with gain = 100, the closed-loop bandwidth is only 10kHz, but the phase shift accumulates rapidly near the bandwidth limit. If there's any stray capacitance in the feedback network, the amplifier could oscillate. For gain = 100, consider a higher-GBW op-amp (AD8605 at 10MHz gives 100kHz bandwidth at gain = 100) or split the gain across two stages (10× each, each with 100kHz bandwidth).

</details>

**A4.** You need to measure a 100kΩ thermistor with the STM32 ADC. A colleague suggests connecting the thermistor directly to the ADC pin (in a voltage divider with a 100kΩ reference resistor). The ADC is configured for maximum speed (shortest sampling time). Will this work?

<details><summary>Answer</summary>

No. The STM32 ADC has an internal sample-and-hold capacitor (~5pF). At the shortest sampling time (~100ns), this capacitor must charge through the source impedance. The source impedance here is the parallel combination of the two 100kΩ resistors = **50kΩ**. 

Time constant: τ = 50kΩ × 5pF = 250ns. In 100ns of sampling time, the capacitor charges to only 1 - e^(-100/250) = 1 - e^(-0.4) ≈ **33%** of the actual voltage — the reading will be completely wrong.

**Fix options:**
1. **Increase sampling time** to 810.5 ADC clock cycles (~27µs at 30MHz ADCCLK). In 27µs, the cap charges to 1 - e^(-27000/250) ≈ 100% — works fine.
2. **Add a buffer op-amp** (voltage follower) between the divider and ADC. Output impedance ~10Ω. τ = 10Ω × 5pF = 50ps — charges instantly at any sampling time.
3. **Lower the divider resistor values** (10kΩ + thermistor), but this changes the thermistor's effective sensitivity and self-heating.

</details>

**A5.** An op-amp is wired as a voltage follower but the output oscillates wildly at ~5MHz. The circuit looks correct (V+ from sensor, V- connected to Vout, supply pins decoupled). What's likely causing the oscillation?

<details><summary>Answer</summary>

The most common cause of voltage follower oscillation is **insufficient phase margin** due to capacitive loading or long PCB traces in the feedback path. A voltage follower has 100% feedback (β = 1), which demands the full bandwidth of the op-amp. If there's:

1. **Capacitive load** (e.g., a long cable or trace to the ADC adds 50–100pF) — this adds a pole in the feedback loop, reducing phase margin.
2. **Long feedback trace** — adds inductance/capacitance, creating a parasitic delay in the feedback path.
3. **Stray capacitance on V- pin** — similar effect.

**Fixes:**
- Add a small **isolation resistor** (47–100Ω) between the op-amp output and the capacitive load.
- Add a **compensation capacitor** (10–100pF) directly from output to V- input.
- Use shorter PCB traces for the feedback connection.
- Choose an op-amp specified as "unity-gain stable" with higher phase margin (many precision op-amps are only stable at gain ≥ 2).
- Add a small capacitor (10pF) across a feedback resistor if using a non-inverting configuration.

</details>

**A6.** Your ADC reference voltage comes from the STM32's VDDA pin, which is connected to the 3.3V rail through a ferrite bead. The 3.3V rail comes from a switching regulator. Your colleague says "the 12-bit ADC gives us 0.806mV resolution, so we can detect signal changes smaller than 1mV." What's the flaw in this reasoning?

<details><summary>Answer</summary>

The 0.806mV LSB is the **quantization step**, not the measurement accuracy. The actual accuracy is limited by the **Vref stability**. If the 3.3V rail has ±1% ripple (from the switching regulator), then Vref varies by ±33mV. Every ADC reading has this uncertainty: True_voltage = ADC_reading × (3.3V ± 33mV) / 4096. A 1% Vref error at mid-range (ADC = 2048) means ±33mV × (2048/4096) = ±16.5mV uncertainty.

So the true measurement accuracy is dominated by Vref noise (~33mV), not quantization step (0.8mV). The bottom 5 bits of the 12-bit reading are essentially meaningless.

**Fix:** Use an **external precision voltage reference** (e.g., REF3030, 3.000V ±0.1%) for VREF+ pin. Or use the STM32's **internal 1.2V reference channel** to calibrate VDDA in software (measure the known reference, calculate the actual VDDA, correct all readings).

</details>

**A7.** A 1kHz signal is sampled at 1.5kHz (below the 2kHz Nyquist requirement). At what frequency does the aliased ghost appear? Can any digital filter applied *after* sampling remove it?

<details><summary>Answer</summary>

Aliased frequency: f_alias = |f_signal - N × f_sample| where N is chosen to minimize the result.
f_alias = |1000 - 1 × 1500| = **500Hz**.

You see a 500Hz signal that does not exist in the real world.

**No digital filter can remove it.** Once the ADC has sampled, the 500Hz alias is indistinguishable from a real 500Hz signal in the digital domain — they have identical sample values. This is fundamentally irreversible. The information about "was this originally 1kHz or 500Hz?" is permanently lost at the sampling instant.

This is why the anti-alias filter must be **analog** and placed **before** the ADC. It's the one filter that absolutely cannot be done in software.

</details>

**A8.** An inverting amplifier has Rin = 10kΩ and Rf = 100kΩ. The input is a 0.2V signal from a temperature sensor. What is the output voltage? Is there a practical problem with using this circuit on a single-supply 3.3V system?

<details><summary>Answer</summary>

Gain = -Rf/Rin = -100k/10k = **-10**
Vout = -10 × 0.2V = **-2.0V**

**Problem:** On a single 3.3V supply (VCC = 3.3V, VCC- = GND = 0V), the op-amp output cannot go below 0V (or even close to it for non-rail-to-rail types). The -2.0V output **clips at 0V** (or ~0.1V for rail-to-rail). You get no useful output.

**Fix options:**
1. Use a **non-inverting amplifier** instead: Gain = 1 + 100k/10k = +11. Vout = 11 × 0.2V = 2.2V ✓ (positive, within 0–3.3V range).
2. If inversion is required, bias the non-inverting input to a DC offset (e.g., 1.65V = mid-rail) so the output swings around mid-rail instead of around 0V.
3. Use a dual-supply op-amp (±5V or ±12V) to allow negative outputs — but this adds complexity inappropriate for battery/embedded systems.

</details>

**A9.** Your STM32 12-bit ADC gives noisy readings (±8 LSB jitter) when measuring a stable DC voltage. You enable 16× hardware oversampling. What effective resolution do you now have? What's the output rate if the base conversion rate was 1Msps?

<details><summary>Answer</summary>

Extra bits from oversampling = log₂(√(oversampling_ratio)) = log₂(√16) = log₂(4) = **2 extra bits**.

Effective resolution = 12 + 2 = **14 bits** (16,384 steps, ~0.201mV LSB at 3.3V).

The ±8 LSB jitter at 12-bit corresponds to ±8 × 0.806mV = ±6.4mV. After 16× averaging, the noise is reduced by √16 = 4×, giving ±6.4mV/4 = **±1.6mV** jitter, which is ±2 LSB at 14-bit resolution — a significant improvement.

Output rate: 1Msps / 16 = **62.5ksps**. For a 100Hz SPI bridge reading, this is vastly more than needed (you'd only take ~625 conversions per 100Hz frame). The STM32's hardware oversampler handles this without CPU overhead.

</details>

**A10.** A comparator (op-amp without feedback) is used to detect when a battery voltage drops below 3.0V. V+ is connected to the battery through a voltage divider, V- is connected to a 1.5V reference. The system works but triggers repeatedly when the battery is near 3.0V. What's happening and how do you fix it?

<details><summary>Answer</summary>

This is **comparator chatter** (also called "bouncing"). When the battery voltage is near the threshold, noise (a few mV of ripple, EMI, or thermal noise) causes the divided voltage to repeatedly cross V-, making the output rapidly toggle between HIGH and LOW.

**Fix: Add hysteresis** (positive feedback). Connect a resistor from the output back to V+ (not V-). This creates two thresholds:
- **Upper threshold** (battery rising): slightly above 3.0V (e.g., 3.05V)
- **Lower threshold** (battery falling): slightly below 3.0V (e.g., 2.95V)

Once the output trips, the threshold shifts, requiring the input to move by a significant amount before it can trip back. With 100mV of hysteresis, the noise must exceed 100mV to cause chatter — far more than typical noise levels.

Alternatively, use a dedicated **voltage supervisor IC** (like TPS3839) that has built-in hysteresis and is designed for exactly this purpose.

</details>

---

## Section B — Spot the Bug

**B1.** Non-inverting amplifier gain calculation:

> "I need gain = 5. Using the non-inverting formula: Gain = Rf/Rin = 5. So Rf = 50kΩ, Rin = 10kΩ."

What's wrong?

<details><summary>Answer</summary>

The non-inverting amplifier gain formula is **Gain = 1 + Rf/Rin**, not Rf/Rin. That's the inverting formula (without the negative sign). With Rf = 50kΩ and Rin = 10kΩ: Gain = 1 + 50k/10k = **6, not 5**.

To get gain = 5: 5 = 1 + Rf/Rin → Rf/Rin = 4 → Rf = **40kΩ**, Rin = 10kΩ. Check: 1 + 40k/10k = 5 ✓.

</details>

**B2.** Anti-alias filter design for a 500Hz ADC sample rate:

> "Nyquist frequency = 500Hz. Anti-alias cutoff = 500Hz. Using R = 3.3kΩ, C = 100nF. f_c = 1/(2π × 3300 × 100e-9) = 482Hz. Close enough to 500Hz."

Two errors. Find them.

<details><summary>Answer</summary>

**Error 1:** Nyquist frequency = f_sample / 2 = 500 / 2 = **250Hz**, not 500Hz. The engineer confused sample rate with Nyquist frequency.

**Error 2:** Even with the corrected Nyquist of 250Hz, setting the filter cutoff AT Nyquist is too aggressive. At f_c, the filter attenuates by only -3dB (70% passes through). Frequencies between f_c and f_sample/2 still alias with significant amplitude. A first-order RC filter rolls off at only -20dB/decade, so at 500Hz (2× the cutoff) the attenuation is only about -6dB — half the signal still gets through.

**Correct design:** Target f_c ≈ 200Hz (below 250Hz with margin). R = 1/(2π × 200 × 100e-9) ≈ 8kΩ → use 8.2kΩ. Better yet, use a second-order filter for -40dB/decade rolloff.

</details>

**B3.** ADC voltage calculation:

> STM32 12-bit ADC, Vref = 3.3V. ADC reads 3000.
> "Voltage = 3000 / 4096 = 0.732V"

Find the error.

<details><summary>Answer</summary>

The formula is missing the Vref multiplier. The correct calculation is:

Voltage = ADC_reading × (Vref / 2^N) = 3000 × (3.3V / 4096) = 3000 × 0.000806 = **2.42V**

The engineer divided by 4096 but forgot to multiply by Vref (3.3V). Their result of 0.732 is just the ratio (3000/4096 = 0.732), which represents 73.2% of full-scale — the fractional position. The actual voltage requires scaling by Vref.

</details>

**B4.** Oversampling calculation:

> "I need 16-bit effective resolution from my 12-bit ADC. That's 4 extra bits. Oversampling ratio = 4² = 16×. Sample at 16× the output rate."

Find the error.

<details><summary>Answer</summary>

The formula for extra bits from oversampling is: extra_bits = log₂(√(OSR)), which means OSR = (2^extra_bits)² = 4^extra_bits.

For 4 extra bits: OSR = 4⁴ = **256×**, not 16×.

The engineer used OSR = extra_bits² = 4² = 16, which would only give log₂(√16) = log₂(4) = **2 extra bits** (14-bit effective), not 4.

The oversampling cost grows **exponentially** with extra bits:
- +1 bit: 4× oversample
- +2 bits: 16× oversample  
- +3 bits: 64× oversample
- +4 bits: 256× oversample

At 256× with a target output rate of 100Hz, you need to sample at 25.6kHz — feasible for the STM32 ADC but a significant computation burden for software averaging.

</details>

**B5.** Sensor signal conditioning chain:

> "I have a 0–5V sensor feeding the STM32's 3.3V ADC. I'll add an RC anti-alias filter (R=3.3kΩ, C=100nF) directly on the sensor output, then it goes straight to the ADC pin."

Two problems. Find them.

<details><summary>Answer</summary>

**Problem 1: No voltage scaling.** The sensor outputs 0–5V but the ADC accepts 0–3.3V max. At 5V output, the ADC pin sees 5V, which exceeds the absolute maximum rating (typically VDD+0.3V = 3.6V). This will **damage the ADC input** and potentially the entire MCU. Need a voltage divider (or level-shifting clamp) before the ADC to scale 5V → 3.3V max.

**Problem 2: No buffer.** The RC filter has an output impedance equal to R = 3.3kΩ. If placed directly before the ADC, this source impedance interacts with the ADC's internal sample-and-hold capacitor (~5pF). τ = 3.3kΩ × 5pF = 16.5ns — this is actually marginal for fast sampling times but probably OK for longer sampling periods. However, the real issue is that a voltage divider BEFORE the RC filter would have even higher output impedance (depending on resistor values), and the ADC sampling would load the divider. A buffer op-amp between the divider and the RC filter (or between filter and ADC) is the proper solution.

</details>

**B6.** A differential amplifier is used to measure current through a 0.1Ω shunt resistor carrying up to 10A:

> "Voltage across shunt = 10A × 0.1Ω = 1V. Using matched resistors R1 = Rf = 10kΩ, gain = Rf/R1 = 1. Output = 1V. Feeding directly to 3.3V ADC."

This gives wildly inaccurate readings. Why?

<details><summary>Answer</summary>

The shunt resistor is in the **high-side** (positive rail) of the motor power path. The voltage across the shunt is indeed 1V, BUT this 1V sits on top of a **high common-mode voltage** (e.g., 12V or 24V for a motor supply). The differential amplifier inputs see V1 = 24V and V2 = 23V (difference = 1V).

A standard op-amp powered from 3.3V or 5V **cannot handle 24V at its inputs** — the inputs exceed the supply voltage, destroying the op-amp or at minimum clamping to the rails (giving garbage readings). Even if the op-amp survived, its common-mode rejection ratio (CMRR) at unity gain with standard resistor tolerances would make the measurement very noisy.

**Fix:** Use a dedicated **high-side current sense amplifier** (INA219, INA226) designed to handle high common-mode voltages (up to 26V) and provide a ground-referenced output suitable for an ADC. These ICs have precision internal resistors and high CMRR. Alternatively, move the shunt to the **low side** (between load and GND) where common-mode voltage is near 0V — but this disrupts the load's ground return.

</details>

---

## Section C — Fill in the Blank / From Memory

### C1. Op-Amp Gain Formulas

| Configuration | Gain Formula | Input at | Feedback to |
|---------------|-------------|----------|-------------|
| Inverting | Av = ______ | V- (through Rin) | ______ |
| Non-Inverting | Av = ______ | ______ | V- (through Rin) |
| Voltage Follower | Av = ______ | V+ | ______ (direct) |
| Summing (equal R) | Vout = ______ | V- (through R each) | V- |
| Differential (matched) | Vout = ______ | Both inputs | Both |

<details><summary>Answer</summary>

| Configuration | Gain Formula | Input at | Feedback to |
|---------------|-------------|----------|-------------|
| Inverting | Av = **-Rf/Rin** | V- (through Rin) | **V-** |
| Non-Inverting | Av = **1 + Rf/Rin** | **V+** | V- (through Rin) |
| Voltage Follower | Av = **1** | V+ | **V-** (direct, Rf=0) |
| Summing (equal R) | Vout = **-(V1 + V2 + V3)** | V- (through R each) | V- |
| Differential (matched) | Vout = **(Rf/R1)(V2 - V1)** | Both inputs | Both |

</details>

### C2. Op-Amp Golden Rules

When **negative feedback** is present:

- Rule 1: ____________________________________
- Rule 2: ____________________________________

When do these rules break down? (Name at least 3 cases):
1. ____________________________________________
2. ____________________________________________
3. ____________________________________________

<details><summary>Answer</summary>

- Rule 1: **No current flows into V+ or V-** (input impedance > 1TΩ)
- Rule 2: **V+ = V-** (the op-amp adjusts Vout to force inputs equal)

Breakdown cases:
1. **Without feedback** (comparator mode) — V+ ≠ V-, output slams to rail
2. **At very high frequencies** — gain drops (GBW product), op-amp can't respond fast enough
3. **Near power rails** — output can't exceed VCC or go below GND (unless rail-to-rail)
4. **Bias currents** — a few nA actually do flow into inputs (matters for precision)

</details>

### C3. ADC Key Values at 3.3V Vref

| Resolution | Number of Steps | LSB Value | SNR_max |
|-----------|-----------------|-----------|---------|
| 8-bit | ______ | ______mV | ______dB |
| 10-bit | ______ | ______mV | ______dB |
| 12-bit | ______ | ______mV | ______dB |
| 16-bit | ______ | ______mV | ______dB |

Formula: LSB = ______  |  SNR_max = ______

<details><summary>Answer</summary>

| Resolution | Number of Steps | LSB Value | SNR_max |
|-----------|-----------------|-----------|---------|
| 8-bit | 256 | 12.9mV | 50dB |
| 10-bit | 1024 | 3.22mV | 62dB |
| 12-bit | 4096 | 0.806mV | 74dB |
| 16-bit | 65,536 | 0.050mV | 98dB |

LSB = **Vref / 2^N**
SNR_max = **6.02 × N + 1.76 dB**

</details>

### C4. Sampling Theory Formulas

1. Nyquist theorem: f_sample ≥ ______
2. Aliased frequency: f_alias = ______
3. Anti-alias filter cutoff target: f_c ≤ ______
4. Oversampling extra bits: extra = ______
5. Quantization noise RMS: σ = ______

<details><summary>Answer</summary>

1. f_sample ≥ **2 × f_max**
2. f_alias = **|f_signal − N × f_sample|** (N chosen to minimize result)
3. f_c ≤ **f_sample / 2** (at or below Nyquist, ideally with margin)
4. extra = **log₂(√(OSR))** — equivalently, OSR = 4^(extra_bits)
5. σ = **LSB / √12**

</details>

### C5. ADC Architecture Comparison

| Architecture | Speed | Resolution | Power | Used In |
|-------------|-------|------------|-------|---------|
| SAR | ______ | ______ bits | ______ | ______ |
| Sigma-Delta | ______ | ______ bits | ______ | ______ |
| Flash | ______ | ______ bits | ______ | ______ |

<details><summary>Answer</summary>

| Architecture | Speed | Resolution | Power | Used In |
|-------------|-------|------------|-------|---------|
| SAR | 100ksps–10Msps | 8–18 bits | Moderate | STM32 MCUs, general embedded |
| Sigma-Delta | 10sps–100ksps | 16–24 bits | Low | Load cells, thermocouples, precision measurement |
| Flash | 100Msps–10Gsps | 6–10 bits | Very high | Oscilloscopes, SDR, high-speed data acquisition |

</details>

---

## Section D — Lab / Calculation Tasks

**D1. Non-Inverting Amplifier Design**

A force sensor outputs 0–200mV. You need to amplify this to 0–3.0V for the STM32 12-bit ADC (Vref = 3.3V).

a) What gain do you need?
b) Choose Rin and Rf (use standard E12 values).
c) Verify the gain with your chosen values.
d) What percentage of the ADC range does 3.0V use?
e) What force sensor voltage change corresponds to 1 LSB?

<details><summary>Answer</summary>

a) Gain = 3.0V / 0.2V = **15**

b) Non-inverting: Gain = 1 + Rf/Rin → 15 = 1 + Rf/Rin → Rf/Rin = 14.
Choose Rin = 10kΩ: Rf = 14 × 10kΩ = 140kΩ. Not a standard value.
Try Rin = 6.8kΩ (E12): Rf = 14 × 6.8kΩ = 95.2kΩ → use 100kΩ (E12).
Or: Rin = 10kΩ, Rf = 150kΩ (E12). Gain = 1 + 150k/10k = 16 (slightly high).
Better: **Rin = 10kΩ, Rf = 150kΩ** (gain = 16, output tops at 3.2V ≈ within 3.3V range).
Or: **Rin = 1kΩ, Rf = 15kΩ** (E12 value, gain = 1 + 15k/1k = 16). Pin down the ratio.

Best practical choice: **Rin = 10kΩ, Rf = 150kΩ** → Gain = 16.

c) Gain = 1 + 150k/10k = **16**. Output at 200mV input: 200mV × 16 = 3.2V. Within 3.3V supply ✓. At 0V input: 0V × 16 = 0V ✓.

d) % of ADC range = 3.0V / 3.3V × 100 = **90.9%** — excellent utilization.
With gain=16: 3.2V/3.3V = 97.0% — even better.

e) LSB = 3.3V / 4096 = 0.806mV at the ADC. Referred back to the sensor: 0.806mV / 16 = **0.050mV = 50µV**. Each ADC step corresponds to 50µV change at the sensor.

</details>

**D2. Anti-Alias Filter Design for IMU**

Your SPI bridge reads the IMU at 100Hz. The IMU internal ODR is 1kHz. You want to add an external anti-alias filter on the analog accelerometer backup channel that samples at 500Hz.

a) What is the Nyquist frequency?
b) Design a single-stage RC filter. Choose R and C values.
c) At 1kHz (a common motor vibration frequency), what attenuation does this filter provide?
d) Is single-stage enough, or do you need second-order? Justify with numbers.

<details><summary>Answer</summary>

a) Nyquist = f_sample / 2 = 500 / 2 = **250Hz**

b) Target f_c ≈ 200Hz (below Nyquist with margin).
Choose C = 100nF: R = 1/(2π × 200 × 100e-9) = 1/(125.7e-6) = **7.96kΩ → use 8.2kΩ** (E12).
Check: f_c = 1/(2π × 8200 × 100e-9) = **194Hz** ✓

c) At 1kHz, the ratio f/f_c = 1000/194 = 5.15. For first-order: |H| = 1/√(1 + (f/f_c)²) = 1/√(1 + 26.5) = 1/5.24 = **0.191 = -14.4dB**. About 19% of the 1kHz noise passes through and aliases.

d) -14.4dB means motor vibrations at 1kHz are reduced to ~19% amplitude — they'll alias into the 0–250Hz band at 19% strength. For accelerometer data used in navigation estimation, this is **not enough**. A 0.1g motor vibration at 1kHz becomes a 0.019g phantom at |1000 - 500| = 500Hz... but wait, 500Hz = Nyquist, so it aliases to 0Hz (a DC offset shift of 0.019g). This corrupts the gravity vector measurement.

**Second-order filter** (-40dB/decade): at 1kHz, attenuation = -14.4dB × 2 = **-28.8dB** → 3.6% passes through. Much better. Use two cascaded RC stages (R₁=8.2kΩ, C₁=100nF → R₂=8.2kΩ, C₂=100nF) or an active second-order Butterworth with an op-amp.

</details>

**D3. Oversampling vs. External ADC**

You need 16-bit effective resolution at 100Hz output rate. Your STM32 has a 12-bit ADC capable of 2Msps.

a) How many extra bits do you need? What oversampling ratio?
b) At what rate must the ADC sample?
c) Is this feasible with the 2Msps ADC?
d) An alternative: the ADS1115 is a 16-bit external ADC at 860sps over I2C. Compare the approaches.

<details><summary>Answer</summary>

a) Extra bits needed: 16 - 12 = **4 bits**.
OSR = 4^(extra_bits) = 4⁴ = **256×**

b) Sample rate = 256 × 100Hz = **25.6kHz**

c) 25.6kHz << 2Msps. **Absolutely feasible** — the ADC is running at only 1.3% of its maximum rate. DMA can handle continuous conversion with zero CPU overhead. The STM32 hardware oversampler supports up to 1024× in hardware, so 256× is natively supported — the ADC outputs pre-averaged 16-bit values with no software processing needed.

d) **Comparison:**

| Factor | Oversampling (internal) | ADS1115 (external) |
|--------|------------------------|---------------------|
| Resolution | 16-bit **effective** (noise may limit to ~14-bit) | 16-bit **native** (better INL/DNL) |
| Speed | 100Hz easily | 860sps max → 100Hz ✓ |
| Noise floor | Limited by 12-bit quantization + averaging | Lower — designed for precision |
| Extra components | None | ADS1115 + I2C pull-ups + decoupling |
| CPU overhead | None (hardware) | I2C driver, interrupt handling |
| Accuracy | Depends on Vref quality, limited ENOB | Better — has internal precision Vref |

**Recommendation:** For robotics applications needing "good enough" 14-bit-class readings at 100Hz, the internal oversampling is simpler and free. For true precision measurement (strain gauges, precision temperature), the ADS1115 provides genuinely better noise performance because its ΔΣ architecture was designed from the ground up for high resolution.

</details>

**D4. Aliasing in Practice**

A robot wheel encoder produces a 450Hz signal at top speed. Your ADC samples at 1000Hz.

a) Does this violate Nyquist? By how much?
b) Now the robot speeds up and the encoder reaches 600Hz (you didn't expect this). Where does the aliased signal appear?
c) The aliased signal looks like what real speed to the software? How would this affect the robot's odometry?

<details><summary>Answer</summary>

a) Nyquist requires f_sample ≥ 2 × f_max = 2 × 450 = 900Hz. Your 1000Hz sample rate exceeds this. Nyquist frequency = 500Hz. The 450Hz signal is **below** 500Hz, so **no violation**. Margin: 500Hz - 450Hz = 50Hz of headroom. Barely safe.

b) At 600Hz: f_alias = |600 - 1 × 1000| = **400Hz**. The 600Hz signal appears as a phantom 400Hz signal.

c) If 450Hz corresponds to top speed, then 400Hz corresponds to about 400/450 × top_speed = **88.9%** of top speed. But the actual speed is 600/450 × top_speed = **133.3%** of expected top speed. The software sees the robot SLOWING DOWN (88.9% of top speed) when it's actually going FASTER (133.3%). This would cause:
- Odometry underestimating distance traveled
- Speed controller increasing motor power (it thinks the robot is too slow) → robot accelerates further
- Potential positive feedback loop → runaway robot!

This is a critical safety issue. The anti-alias filter must be designed for the maximum possible encoder frequency (including fault conditions), or use a digital encoder interface (counter/timer) instead of ADC sampling.

</details>

**D5. Complete Signal Chain Design**

Design the analog conditioning chain for reading a 0–10V industrial pressure sensor with the STM32 12-bit ADC (Vref = 3.3V, 100Hz output needed, 1kHz sample rate).

Specify each stage with component values:
a) Input protection
b) Voltage scaling
c) Buffering (if needed — justify)
d) Anti-alias filter
e) Calculate the sensor voltage resolution per LSB

<details><summary>Answer</summary>

a) **Input protection:** TVS diode (SMBJ10A, clamps at ~17V) from signal to GND at the connector. Protects against transients on the industrial sensor cable.

b) **Voltage scaling:** Voltage divider to map 10V → 3.0V (leave 0.3V headroom below 3.3V).
Ratio: 3.0/10 = 0.3. Vout = Vin × R2/(R1+R2) = 0.3.
Choose R2 = 3.3kΩ, solve: 0.3 = 3.3k/(R1+3.3k) → R1 = 7.7kΩ → **use 8.2kΩ** (E12).
Check: Vout = 10 × 3.3k/(8.2k+3.3k) = 10 × 0.287 = **2.87V**. A bit low.
Adjust: R1 = 6.8kΩ: Vout = 10 × 3.3k/(6.8k+3.3k) = 10 × 0.327 = **3.27V** ✓. Close to 3.3V.

c) **Buffering:** YES — the divider output impedance = R1 ∥ R2 = (6.8k × 3.3k)/(6.8k + 3.3k) = **2.22kΩ**. This is acceptable for slower sampling times, but for reliable operation across all MCU configurations, add a **MCP6002 voltage follower**. It also isolates the RC filter impedance from the divider loading issue. Decouple with 100nF on VCC.

d) **Anti-alias filter:** f_sample = 1kHz, Nyquist = 500Hz. Target f_c ≈ 400Hz.
R = 1/(2π × 400 × 100e-9) = **3.98kΩ → use 3.9kΩ** (E12 value), C = 100nF.
f_c = 1/(2π × 3900 × 100e-9) = **408Hz** ✓. Place after the buffer.

e) **Resolution:** Full-scale voltage at ADC = 3.27V. LSB at ADC = 3.3V/4096 = 0.806mV.
Usable range: 0 to 3.27V = 0 to 4059 ADC counts.
Sensor voltage per LSB = 10V / 4059 = **2.46mV per LSB** at the sensor.
So the system resolves pressure changes equivalent to 2.46mV out of 10V = **0.025% of full scale**.

</details>

---

## Section E — Deeper Thinking

**E1.** In your STM32H7 ↔ Jetson Orin SPI bridge project, the ICM-42688 IMU has its own internal ADC (16-bit ΔΣ) sampling at 32kHz, an internal DLPF (anti-alias), and outputs digital data over SPI. You read it at 100Hz. Explain why this architecture (analog sensing → high-rate ADC → digital filter → decimation → SPI read) is fundamentally better than if you tried to read an analog accelerometer with the STM32's 12-bit SAR ADC at 100Hz. Cover at least: aliasing, resolution, noise, and system complexity.

<details><summary>Answer</summary>

**Aliasing protection:** The ICM-42688 samples the analog MEMS at 32kHz — its Nyquist frequency is 16kHz, far above any mechanical vibration the accelerometer sees. The internal DLPF then attenuates everything above the output bandwidth (e.g., 100Hz) BEFORE decimating to 1kHz ODR. When you read at 100Hz, you get **doubly-filtered** data: first by the DLPF, then by the 1kHz→100Hz decimation that only works because the DLPF already removed content above 500Hz. No aliasing at any stage.

With a direct analog accelerometer at the STM32, you'd need to design your own anti-alias filter for the 50Hz Nyquist (at 100Hz sampling). A single RC filter provides only -20dB/decade — motor vibrations at 500Hz would still alias significantly. You'd need a multi-stage active filter to match what the IMU does digitally.

**Resolution:** The IMU's internal ΔΣ ADC provides 16-bit resolution. Oversampling from 32kHz to 1kHz (32× ratio) adds log₂(√32) ≈ 2.5 extra bits of effective resolution through the noise shaping inherent in ΔΣ architecture. The STM32's 12-bit SAR ADC would need 256× oversampling to reach 16-bit effective resolution.

**Noise:** The IMU's MEMS element, ADC, and analog front-end are co-designed and physically co-located on a single die. The analog signal path is millimeters long, fully shielded by the IC package. A discrete analog accelerometer → wire → STM32 ADC path has centimeters of trace/wire acting as an antenna for EMI, motor noise, and switching regulator noise. The noise floor of the integrated solution is dramatically lower.

**System complexity:** The IMU handles the entire analog signal chain internally. You get clean, calibrated, filtered digital data over SPI. A discrete solution requires: precision voltage reference, anti-alias filter (multi-stage), impedance-matched buffer op-amp, shielded routing, independent analog power supply with extra filtering — adding 10+ components and significant PCB area versus one SPI read.

</details>

**E2.** The Nyquist theorem says you must sample at ≥ 2× the signal frequency. But in practice, audio CDs sample at 44.1kHz for a 20kHz audio bandwidth (2.205×, barely above 2×). Professional audio uses 96kHz or 192kHz (4.8× to 9.6× Nyquist). Why sample so much faster than Nyquist? Relate this to the practical limitations of anti-alias filters and reconstruction filters.

<details><summary>Answer</summary>

Nyquist gives the **theoretical minimum**, but it assumes a **perfect brick-wall anti-alias filter** — one that passes everything below f_Nyquist with zero attenuation and blocks everything above with infinite attenuation. Such a filter doesn't exist physically.

**Real filters roll off gradually.** A practical anti-alias filter needs a **transition band** between the passband (signals we want) and the stopband (signals we must block). The wider the ratio between f_sample and f_max, the wider this transition band can be, and the simpler (cheaper, lower-order) the filter can be.

**At 44.1kHz for 20kHz audio:** Nyquist = 22.05kHz. The transition band is only 22.05 - 20 = **2.05kHz** (from 20kHz to 22.05kHz). To attenuate signals by >90dB within this narrow band requires a very steep filter — typically a 9th-order elliptic filter with precisely matched components. This was the state of the art when CDs were designed (1982).

**At 192kHz for 20kHz audio:** Nyquist = 96kHz. Transition band = 96 - 20 = **76kHz**. A simple 3rd-order Butterworth does the job. The filter is cheaper, introduces less phase distortion, and has better transient response.

The same applies to **reconstruction** on playback: converting digital back to analog requires a smoothing filter that removes the staircase artifacts. Higher sample rate = simpler reconstruction filter.

**For your SPI bridge:** The IMU samples internally at 32kHz for signals up to ~500Hz. That's 32× oversampling — a massive margin that allows a very gentle internal digital DLPF with excellent passband flatness and linear phase. This is why the IMU's digital filter outperforms anything you could build with discrete analog components.

</details>

**E3.** You're debugging a mysterious 2Hz oscillation in the robot's position estimate. The IMU data looks clean (no 2Hz content when logged directly). The ADC on a separate analog sensor channel shows the 2Hz oscillation clearly. The analog sensor itself is DC-coupled with a bandwidth of 500Hz. The ADC is sampled at 100Hz. Your colleague says "the sensor must be picking up 2Hz vibration from somewhere." Propose an alternative hypothesis involving aliasing. How would you test each hypothesis?

<details><summary>Answer</summary>

**Alternative hypothesis:** The analog sensor is picking up a high-frequency signal that **aliases** to 2Hz at the 100Hz sample rate.

For a signal to alias to 2Hz at 100Hz sampling: f_real = N × f_sample ± 2Hz. Possible source frequencies:
- 98Hz (|98 - 100| = 2Hz) — structural resonance of the robot frame
- 102Hz (|102 - 100| = 2Hz) — motor running at ~6000 RPM
- 198Hz (|198 - 200| = 2Hz) — second harmonic
- 202Hz, 298Hz, 302Hz, etc.

Motor electrical frequencies are prime suspects — a motor spinning at varying speeds near 6000 RPM would produce mechanical vibration near 100Hz that sweeps through the alias zone.

**Test 1 — Is it aliasing?** Temporarily increase the ADC sampling rate to 1kHz (Nyquist = 500Hz). If the 2Hz oscillation disappears and is replaced by a signal at 98Hz or 102Hz, it's aliasing. If the 2Hz persists at 1kHz sampling, it's a real 2Hz signal.

**Test 2 — Add anti-alias filter.** Add an RC filter (f_c = 40Hz) before the ADC input. If the 2Hz disappears, the source was a high-frequency component above 50Hz that got aliased. If it persists, it's a real 2Hz signal passing through the filter.

**Test 3 — FFT at high sample rate.** Sample at 10kHz and perform FFT. Look for peaks near 98, 102, 198, or 202Hz. If found, those are the aliasing source. Correlate with motor speed.

The key insight: the IMU doesn't show 2Hz because its internal DLPF acts as anti-alias filter. The analog sensor channel has no such filter — any high-frequency content aliases freely.

</details>

**E4.** Connect the op-amp voltage follower concept (gain = 1, impedance transformation) to a software analogy you already know. Think about a function that takes an input and returns the same value but changes some property of how downstream consumers interact with it. Then explain why this matters for the signal chain from voltage divider → buffer → RC filter → ADC in your robot.

<details><summary>Answer</summary>

**Software analogy:** A voltage follower is like a **caching proxy** or **adapter** in software design patterns.

Consider a database query that returns a record. The raw database connection is fragile — only one consumer can read at a time, and each read takes variable latency. You add a cache layer in front:
```python
def get_sensor_value():          # Voltage divider (high impedance source)
    return database.query(...)   # Slow, one reader, breaks under load

def buffered_get():              # Op-amp voltage follower
    value = get_sensor_value()   # Reads once (no loading)
    return value                 # Returns same value, but fast,
                                 # multiple readers OK
```

The cache returns the **same data** (gain = 1) but transforms the access characteristics:
- **Input:** high latency, limited concurrent readers (= high impedance, can't source current)
- **Output:** low latency, many concurrent readers (= low impedance, can drive multiple loads)

In your signal chain: The voltage divider (10kΩ + 20kΩ) has ~14kΩ output impedance. Without a buffer, every downstream component "loads" it — the RC filter draws current, the ADC's sample capacitor draws more. Each load shifts the voltage. With a buffer: the divider sees ~TΩ load (nothing), the RC filter and ADC see ~10Ω source (everything they need). The **value is identical** but the delivery characteristics are transformed.

This is exactly the **adapter pattern**: same interface, same semantics, different non-functional properties (current capability, impedance, charging speed). The voltage follower adapts a high-impedance source into a low-impedance source, just as a cache adapts a slow data source into a fast one.

</details>
