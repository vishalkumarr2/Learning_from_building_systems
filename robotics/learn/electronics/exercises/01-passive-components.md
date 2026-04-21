# Exercises: Passive Components

### Chapter 1: Resistors, Capacitors, Inductors

**Self-assessment guide:** Write your answer before expanding the details block. If you can answer 80% without peeking, you're ready for chapter 02.

---

## Section A — Conceptual Questions

**A1.** Your colleague places a 100nF decoupling cap on the STM32H7 VCC pin, but 15mm away from the pin on the opposite side of the PCB. They say "it's the right value so it should work fine." What's wrong with this reasoning?

<details><summary>Answer</summary>

The **distance** matters as much as the value. PCB traces have inductance (~1nH per mm). At 15mm, the trace adds ~15nH of inductance between the cap and the VCC pin. This inductance opposes the fast current transients the decoupling cap is supposed to supply. At high frequencies (tens of MHz), the trace inductance dominates and the cap can't respond fast enough — it's as if it isn't there. The cap must be **within 3mm** of the VCC pin with short, wide traces and a direct via to the ground plane.

</details>

**A2.** You measure 2.1V at the output of a voltage divider designed to give 3.3V from a 5V input. The divider uses R1=10kΩ and R2=20kΩ. Without the load, it measured 3.33V. What's happening?

<details><summary>Answer</summary>

The load connected to the divider output is drawing current, which effectively appears in **parallel** with R2, reducing R2_effective. This is the **loading problem**. The load impedance is comparable to R2 (20kΩ). For example, if the load is ~20kΩ, then R2_eff = 20k ∥ 20k = 10kΩ, so Vout = 5 × 10k/(10k+10k) = 2.5V. To fix this: use lower-value resistors in the divider (so the load current is negligible), or place a **buffer op-amp** (voltage follower) between the divider output and the load.

</details>

**A3.** An intern suggests using a 1MΩ pull-up resistor on the SPI chip-select line "to save power." Is this a good idea?

<details><summary>Answer</summary>

No. A 1MΩ pull-up is far too weak. The RC time constant with even small parasitic capacitance (~10pF on a trace) would be τ = 1MΩ × 10pF = 10µs. At 100Hz SPI transactions that's marginal, but the real problem is **noise immunity** — the weak pull-up can't overcome crosstalk, EMI, or leakage from adjacent traces, leaving the CS line susceptible to false assertions. Additionally, during boot (before the GPIO is configured), a 1MΩ pull-up may not reliably hold CS HIGH, causing the slave to interpret garbage on the bus. Use **10kΩ** for SPI CS — the steady-state current is only 3.3V/10kΩ = 330µA, which is negligible.

</details>

**A4.** You're debugging an STM32 that randomly resets when the robot's motors activate. No decoupling caps were placed on the board. Your colleague says "the power supply is rated for 3A, so it's fine." What's the actual problem?

<details><summary>Answer</summary>

The power supply being high-current doesn't help because the problem is **inductance of the power delivery path**, not the supply's capacity. The PCB traces from the supply to the IC have parasitic inductance (L). When the motors switch on, they cause a sudden current draw. The trace inductance resists this change (V = L × dI/dt), causing the VCC voltage at the IC to momentarily **sag** below the minimum operating voltage (~2.7V for the STM32), triggering a brownout reset. Decoupling caps (100nF + 10µF) placed directly at the IC pins act as local energy reservoirs, supplying the instantaneous current demand while the supply catches up.

</details>

**A5.** Your 12V relay driver circuit uses a 100Ω current-sensing resistor in series with the relay coil. After running for 30 minutes, the resistor is too hot to touch. The relay coil draws 100mA. Should you worry?

<details><summary>Answer</summary>

Yes! Power dissipation: P = I² × R = (0.1)² × 100 = **1.0W**. A standard through-hole resistor is rated at 0.25W. You're dissipating **4× its rating** — it will overheat, drift in value, discolor, and eventually fail (open circuit). Fix: use a lower value resistor (10Ω: P = 100mW, safe), or use a higher-power rated resistor (2W wire-wound). Also, 100Ω drops V = IR = 10V across the resistor, leaving only 2V for the relay coil — the relay probably isn't even operating correctly.

</details>

**A6.** Why does the STM32's internal ~40kΩ pull-up work fine for reading a button, but NOT for I2C SCL/SDA lines at 400kHz?

<details><summary>Answer</summary>

It's about **rise time**. The pull-up must charge the bus capacitance (typically 50–400pF for I2C with multiple devices) fast enough for the clock frequency. With R_pull = 40kΩ and C_bus = 100pF: τ = 40kΩ × 100pF = 4µs. The I2C 400kHz spec requires rise time < 300ns (~0.7τ). With τ = 4µs, the rise time would be ~2.8µs — nearly 10× too slow. The signal edges would be rounded, the receiver wouldn't reliably detect transitions, and communication would fail. A button doesn't care — it only needs to settle within human reaction time (milliseconds). External **4.7kΩ** pull-ups give τ = 470ns, meeting the spec.

</details>

**A7.** In an RC low-pass filter with R = 3.3kΩ and C = 100nF, what happens to a 10kHz noise signal riding on a DC sensor output? What about the DC component?

<details><summary>Answer</summary>

Cutoff frequency: f_c = 1/(2π × 3300 × 100e-9) = **482Hz**. The 10kHz noise is ~20× above f_c. For a first-order RC filter, attenuation is -20dB/decade, so at 10kHz (~1.3 decades above 482Hz) the noise is attenuated by ~26dB, meaning it's reduced to about **5%** of its original amplitude. The DC component (0Hz) passes through completely unattenuated — a capacitor blocks DC (open circuit), so no current flows to ground through C, and no voltage drops across R for a steady voltage. The filter cleans the noise while preserving the signal.

</details>

**A8.** You measure a component with a multimeter and get 4.7kΩ. The SMD marking on the component reads "472." Is your measurement correct?

<details><summary>Answer</summary>

Yes. SMD 3-digit codes work as: first two digits are significant figures, third digit is the power-of-10 multiplier. "472" = 47 × 10² = 47 × 100 = **4700Ω = 4.7kΩ**. Your measurement matches perfectly.

</details>

**A9.** A capacitor is marked "10µF, X5R, 6.3V." You plan to use it on a 5V rail for decoupling. A colleague says "6.3V rating, 5V rail — we have 1.3V of margin, that's plenty." What's the hidden problem?

<details><summary>Answer</summary>

X5R and X7R ceramic capacitors suffer **DC bias derating** — the actual capacitance decreases significantly as the applied DC voltage approaches the rated voltage. A "10µF" X5R cap rated at 6.3V may only provide **2–4µF** when biased at 5V DC. You could lose 50–70% of the stated capacitance. Check the manufacturer's DC bias derating curve in the datasheet. Fix: use a cap rated for at least **2× your operating voltage** (e.g., 16V or 25V rated cap for a 5V rail), or use a physically larger package that derates less.

</details>

**A10.** Why is hot-plugging a board with large (100µF+) capacitors into a powered bus dangerous?

<details><summary>Answer</summary>

From I = C × dV/dt: when you connect an uncharged cap to a powered bus, dV/dt is enormous (voltage goes from 0 to bus voltage almost instantly). The capacitor looks momentarily like a **short circuit**, drawing a huge inrush current limited only by the trace and connector resistance. For a 100µF cap on a 5V bus with 0.1Ω total path resistance: I_peak = 5V/0.1Ω = **50A** for a brief instant. This can weld connector pins, damage the power source, blow traces, or trigger the source's overcurrent protection (shutting down the entire bus and affecting other devices).

</details>

---

## Section B — Spot the Bug

**B1.** An engineer designs an LED circuit from a 3.3V GPIO:

> R = 3.3V / 10mA = 330Ω. Using 330Ω resistor with a red LED (Vf = 2.0V).

What's the error?

<details><summary>Answer</summary>

The engineer forgot to **subtract the LED forward voltage** before applying Ohm's law. The voltage across the resistor is NOT 3.3V — it's 3.3V - 2.0V = 1.3V (the LED "uses" 2V). Correct calculation: R = 1.3V / 10mA = 130Ω → use 150Ω standard value. With 330Ω, the actual current is I = 1.3V / 330Ω = **3.9mA**, which will be quite dim. The LED works, but at less than half the intended brightness.

</details>

**B2.** A voltage divider is designed to scale 5V to 3.3V:

> R1 = 20kΩ (top), R2 = 10kΩ (bottom). Vout = 5 × 10k/(20k+10k) = 1.67V. "Close enough to 1.65V for our purposes."

The target was 3.3V. What went wrong?

<details><summary>Answer</summary>

The resistor positions are **swapped**. To get 3.3V from 5V, you need the larger resistor on the bottom (R2) and the smaller on top (R1). The ratio needed is Vout/Vin = 3.3/5 = 0.66. Using R2=20kΩ (bottom) and R1=10kΩ (top): Vout = 5 × 20k/(10k+20k) = 3.33V ✓. The engineer put R1=20kΩ on top and R2=10kΩ on bottom, which gives 5 × 10k/(20k+10k) = 1.67V — the complement of what they wanted.

</details>

**B3.** A filter design for a 1kHz ADC sampling rate:

> "Anti-alias cutoff should be 1kHz. Using R=1.6kΩ, C=100nF. f_c = 1/(2π × 1600 × 100e-9) = 995Hz. Perfect."

What's the conceptual error?

<details><summary>Answer</summary>

The cutoff should be at the **Nyquist frequency** (f_sample/2 = 500Hz), not at f_sample. A first-order RC filter at 995Hz only attenuates signals at 500Hz by ~3dB — essentially no protection against aliasing for frequencies between 500Hz and 1kHz. Worse, signals just above Nyquist get folded back with barely any attenuation. The filter should be designed for f_c ≈ **450–500Hz** (below Nyquist with some margin). Using C=100nF: R = 1/(2π × 500 × 100e-9) ≈ 3.2kΩ → use 3.3kΩ.

</details>

**B4.** Parallel resistor calculation:

> "I need 7.5kΩ. I'll put 10kΩ and 5kΩ in parallel: R_total = 10k + 5k = 15kΩ, then... wait, that's series. For parallel: R_total = 10k + 5k / 2 = 7.5kΩ."

Find the error.

<details><summary>Answer</summary>

The formula is completely wrong. Parallel resistance is NOT the average. The correct formula is R_total = (R1 × R2)/(R1 + R2) = (10k × 5k)/(10k + 5k) = 50M/15k = **3.33kΩ**. A parallel combination is ALWAYS less than the smallest resistor. To get 7.5kΩ from available parts, you could use 10kΩ ∥ 30kΩ = (10k × 30k)/(10k + 30k) = 7.5kΩ, or 15kΩ ∥ 15kΩ = 7.5kΩ.

</details>

**B5.** RC time constant calculation:

> "R = 10kΩ, C = 10µF. τ = R × C = 10,000 × 10 = 100,000 seconds. Wow, that's a long time!"

What went wrong?

<details><summary>Answer</summary>

Unit error. 10µF = 10 × 10⁻⁶ F = 0.00001 F, not 10. The correct calculation is: τ = 10,000Ω × 0.00001F = **0.1 seconds = 100ms**. The engineer used µF as if it were Farads. Always convert to base SI units (Ω, F, H) before multiplying. Quick mental check: τ = 10kΩ × 10µF. The k (10³) and µ (10⁻⁶) partially cancel: 10 × 10³ × 10 × 10⁻⁶ = 100 × 10⁻³ = 0.1s.

</details>

**B6.** An engineer places a tantalum 47µF capacitor in a circuit. During assembly testing, the capacitor catches fire.

> "I used the correct value and voltage rating (10V cap on a 5V rail). It must be a defective component."

What likely happened?

<details><summary>Answer</summary>

Tantalum capacitors are **polarized**. The most likely cause is that it was soldered **with reversed polarity**. Tantalum caps fail catastrophically (short circuit → fire) when reverse-biased. Unlike electrolytics that may slowly leak or bulge, tantalums fail violently and immediately. The engineer should check the polarity marking (stripe/dot indicates positive terminal on tantalum, vs. stripe = negative on electrolytics — a common confusion). Also check the schematic symbol matches the physical orientation.

</details>

---

## Section C — Fill in the Blank / From Memory

### C1. Complete the Passive Components Comparison Table

Fill in without looking, then check.

| Property | Resistor | Capacitor | Inductor |
|----------|----------|-----------|----------|
| Unit | ______ | ______ | ______ |
| DC behavior | ______ | ______ | ______ |
| AC behavior | ______ | ______ | ______ |
| Energy storage | ______ | ______ | ______ |
| Key equation | V = _____ | I = _____ | V = _____ |
| Impedance formula | ______ | ______ | ______ |
| Can't change instantly | N/A | ______ | ______ |

<details><summary>Answer</summary>

| Property | Resistor | Capacitor | Inductor |
|----------|----------|-----------|----------|
| Unit | Ohm (Ω) | Farad (F) | Henry (H) |
| DC behavior | Conducts (with V drop) | Open circuit (blocks) | Short circuit (passes) |
| AC behavior | Same impedance at all freq | Lower Z at higher freq | Higher Z at higher freq |
| Energy storage | None (dissipates as heat) | Electric field | Magnetic field |
| Key equation | V = IR | I = C × dV/dt | V = L × dI/dt |
| Impedance formula | R (constant) | Xc = 1/(2πfC) | XL = 2πfL |
| Can't change instantly | N/A | Voltage | Current |

</details>

### C2. Key Formulas — Write From Memory

Fill in the blank formulas:

1. Ohm's law: V = ______
2. Power (3 forms): P = ______, P = ______, P = ______
3. Voltage divider: Vout = ______
4. Two resistors in parallel: R_total = ______
5. RC time constant: τ = ______
6. RC low-pass cutoff: f_c = ______
7. Capacitive reactance: Xc = ______
8. Inductive reactance: XL = ______
9. Resonant frequency (LC): f = ______
10. RC charging voltage at time t: V(t) = ______

<details><summary>Answer</summary>

1. V = I × R
2. P = V × I, P = I²R, P = V²/R
3. Vout = Vin × R2/(R1 + R2)
4. R_total = (R1 × R2)/(R1 + R2)
5. τ = R × C
6. f_c = 1/(2πRC)
7. Xc = 1/(2πfC)
8. XL = 2πfL
9. f = 1/(2π√(LC))
10. V(t) = Vmax × (1 − e^(−t/τ))

</details>

### C3. RC Charging Milestones

| Time | % of Vmax |
|------|-----------|
| 1τ | ______% |
| 2τ | ______% |
| 3τ | ______% |
| 5τ | ______% |

<details><summary>Answer</summary>

| Time | % of Vmax |
|------|-----------|
| 1τ | 63.2% |
| 2τ | 86.5% |
| 3τ | 95.0% |
| 5τ | 99.3% ("fully charged") |

</details>

### C4. E12 Standard Values

Write the 12 "magic numbers" of the E12 series (the base values from which all standard resistors are derived):

______, ______, ______, ______, ______, ______, ______, ______, ______, ______, ______, ______

<details><summary>Answer</summary>

10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82

Every standard resistor/capacitor value is one of these multiplied by a power of 10.

</details>

---

## Section D — Lab / Calculation Tasks

**D1. Voltage Divider with Loading**

You need to scale a 0–5V sensor output to 0–3.3V for the STM32 ADC. You use R1 = 10kΩ (top) and R2 = 20kΩ (bottom).

a) What is the unloaded output voltage?
b) The STM32 ADC has ~50kΩ input impedance during sampling. What is the actual output voltage with this load?
c) Suggest two ways to fix the discrepancy.

<details><summary>Answer</summary>

a) Vout = 5 × 20k/(10k + 20k) = 5 × 0.667 = **3.33V** ✓

b) R2_eff = R2 ∥ R_load = (20k × 50k)/(20k + 50k) = 1,000,000k/70k = **14.3kΩ**
   Vout = 5 × 14.3k/(10k + 14.3k) = 5 × 0.588 = **2.94V** — significantly lower than expected!

c) **Fix 1:** Use lower-value resistors. R1=1kΩ, R2=2kΩ. R2_eff = 2k ∥ 50k = 1.92kΩ. Vout = 5 × 1.92/(1+1.92) = 3.29V — much closer. Penalty: higher current draw (5V/(1k+2k) = 1.67mA vs. 0.17mA).

**Fix 2:** Add a buffer op-amp (voltage follower) between the divider and the ADC. The op-amp has TΩ input impedance (no loading) and low output impedance (can drive the ADC easily). Adds cost and a component but is the proper solution for precise readings.

</details>

**D2. Decoupling Capacitor Selection**

Your STM32H7 has VCC, VDDA, and VDDUSB power pins.

a) What is the minimum decoupling you should place at each pin?
b) Why do you need both 100nF and 10µF caps?
c) The 100nF cap has a self-resonant frequency of ~29MHz. What happens above this frequency?

<details><summary>Answer</summary>

a) Each power pin needs **at minimum** a 100nF ceramic capacitor within 3mm, plus a shared 10µF bulk cap nearby. VDDA (analog supply) should additionally have a **ferrite bead** between it and the digital 3.3V rail to filter switching noise.

b) Different cap values are effective at different frequency ranges due to parasitic inductance (ESL). The 100nF (low ESL, typically in 0402 package) resonates around 29MHz — excellent for high-frequency switching transients. The 10µF handles slower transients (µs to ms range, ~1–2MHz effective range). Together they provide wideband decoupling. This is the "bulk + bypass" strategy.

c) Above self-resonance, the capacitor's parasitic inductance dominates. It stops behaving like a capacitor and starts behaving like an **inductor** — impedance increases with frequency instead of decreasing. It becomes ineffective at filtering noise above ~29MHz. This is why ultra-high-frequency designs add even smaller caps (10nF in 0201) for content above 100MHz.

</details>

**D3. Pull-Up Resistor and Rise Time**

An I2C bus has 200pF of total capacitance (traces + device input caps). The bus runs at 400kHz. The I2C spec requires rise time < 300ns.

a) Calculate the RC time constant with a 10kΩ pull-up.
b) Rise time from 0.3×VDD to 0.7×VDD ≈ 0.8τ. Does 10kΩ meet the spec?
c) What pull-up value would you use? Show your calculation.

<details><summary>Answer</summary>

a) τ = R × C = 10kΩ × 200pF = 10,000 × 200×10⁻¹² = **2µs**

b) Rise time ≈ 0.8 × τ = 0.8 × 2µs = **1.6µs**. The spec requires < 300ns. 1.6µs is **5.3× too slow**. 10kΩ fails.

c) Required: 0.8 × τ < 300ns → τ < 375ns → R < τ/C = 375ns/200pF = **1875Ω**. Use **1.8kΩ** (standard E12 value). Check: τ = 1.8kΩ × 200pF = 360ns. Rise time ≈ 0.8 × 360ns = 288ns < 300ns ✓. Current when driven low: 3.3V/1.8kΩ = 1.83mA — well within GPIO sink specs.

</details>

**D4. Power Dissipation Safety**

You're designing a current-sense resistor for motor current monitoring. The motor draws up to 2A. You choose a 0.1Ω sense resistor.

a) What voltage will the ADC see at maximum current?
b) What power does the resistor dissipate?
c) What power rating does the resistor need? (Apply the 50% derating rule.)

<details><summary>Answer</summary>

a) V = I × R = 2A × 0.1Ω = **0.2V** (200mV)

b) P = I² × R = 2² × 0.1 = **0.4W** (400mW)

c) With 50% derating: rated power must be ≥ 0.4W / 0.5 = **0.8W minimum**. Use a 1W resistor. A standard 1/4W (0.25W) resistor would be operating at **160% of its rating** — it would overheat. A 1W wire-wound or thick-film SMD resistor (2512 package) is appropriate.

</details>

**D5. Low-Pass Filter Design**

Your analog pressure sensor outputs up to 50Hz of useful signal plus noise at 1kHz and higher. You're sampling with the STM32 ADC at 200Hz.

a) What is the Nyquist frequency?
b) Design an RC filter to attenuate the noise. Choose R and C values.
c) At the 1kHz noise frequency, how much attenuation does your single-stage RC filter provide? Is this enough?

<details><summary>Answer</summary>

a) Nyquist = f_sample/2 = 200/2 = **100Hz**

b) Cutoff should be at or below Nyquist. Targeting f_c ≈ 80Hz (some margin below 100Hz):
   Choose C = 100nF: R = 1/(2π × 80 × 100e-9) = 1/(50.3e-6) = **19.9kΩ → use 20kΩ** (or 18kΩ standard).
   Check with 18kΩ: f_c = 1/(2π × 18000 × 100e-9) = 88Hz ✓

c) At 1kHz with f_c = 88Hz, the signal is 1000/88 = 11.4× above cutoff (~1.06 decades). A first-order filter gives -20dB/decade, so attenuation ≈ -20 × 1.06 = **-21.2dB** (signal reduced to ~8.7% of original). This provides decent reduction, but not much. Some noise will still alias. For critical applications, cascade two RC stages for **-40dB/decade** rolloff, or use a second-order active filter with an op-amp.

</details>

**D6. Back-EMF Voltage Spike**

A relay coil has L = 50mH and carries 200mA. The driving MOSFET turns off in 500ns (typical for a logic-level FET). There is no flyback diode.

a) Calculate the voltage spike across the MOSFET.
b) If the MOSFET is rated for Vds_max = 30V and the supply is 12V, will it survive?
c) What component must be added, and where?

<details><summary>Answer</summary>

a) V = L × dI/dt = 0.050H × (0.200A / 500×10⁻⁹s) = 0.050 × 400,000 = **20,000V** (20kV theoretical). In practice, arcing and component breakdown clamp it to a few hundred volts, but the math shows the enormous stress.

b) The MOSFET sees V_supply + V_spike. Even if the real spike is "only" 100V due to parasitic capacitance absorbing some energy, 12V + 100V = 112V >> 30V. The MOSFET's drain-source junction will **avalanche breakdown and be destroyed**.

c) Add a **flyback diode** (e.g., 1N4007 for power, or BAT54 Schottky for faster decay) across the relay coil, with cathode toward the positive supply and anode toward drain. When the MOSFET turns off, the collapsing magnetic field forward-biases the diode, clamping the spike to ~0.7V (or ~0.3V for Schottky) above the supply rail. Total Vds = 12V + 0.7V = 12.7V << 30V ✓.

</details>

---

## Section E — Deeper Thinking

**E1.** In your STM32H7 ↔ Jetson Orin SPI bridge at 100Hz: the STM32 runs its DMA+SPI at 10MHz clock. Each 100Hz transaction transfers a burst of bytes. Explain why decoupling caps are critical specifically *during* these SPI bursts, even though 100Hz sounds slow. Connect this to the concepts of I = C × dV/dt and trace inductance.

<details><summary>Answer</summary>

Although the transaction *rate* is only 100Hz, each transaction involves the SPI peripheral clocking at **10MHz** — toggling output drivers, shifting data, running DMA. During a burst, the STM32 draws rapid current pulses at the 10MHz SPI clock rate. Each clock edge involves switching CMOS output stages that draw transient current from VCC. 

The trace inductance from the power supply opposes these fast current changes (V_droop = L_trace × dI/dt). At 10MHz, dI/dt is large, so the voltage droop is significant. Without the 100nF decoupling cap right at the VCC pin, the supply voltage would sag on every SPI clock edge, potentially corrupting the data being shifted out or causing the MCU to brownout.

The 100nF cap (effective up to ~29MHz) is loaded by the burst, and the 10µF bulk cap replenishes it between bursts (100Hz = 10ms between transactions — plenty of time for the bulk cap to recharge the bypass cap). This is why both cap values are needed: the 100nF handles the MHz-rate switching *within* a burst; the 10µF handles the burst-level energy demand at 100Hz.

</details>

**E2.** A teammate is choosing between two approaches for reading a thermistor (100kΩ at 25°C) with the STM32 ADC: (A) a direct voltage divider with a 100kΩ fixed resistor, or (B) the same divider but with an op-amp buffer in between. They're leaning toward (A) to save BOM cost. Make the case for or against each option, considering ADC input impedance, sampling time configuration, and measurement accuracy.

<details><summary>Answer</summary>

**Option A (direct divider):** The source impedance seen by the ADC is R_thermistor ∥ R_fixed ≈ 50kΩ. The STM32 ADC rule-of-thumb requires source impedance < 50kΩ for accurate readings at default sampling time. A 50kΩ source is borderline — the internal sample capacitor (~5pF) needs τ = 50kΩ × 5pF = 250ns to charge. At the fastest sampling time (~100ns), it won't fully settle, causing reading errors.

**Workaround for A:** Increase the ADC sampling time to 480.5 or 810.5 ADC clock cycles. At 30MHz ADC clock, 810.5 cycles = 27µs — far more than 5τ = 1.25µs. This works but reduces maximum throughput.

**Option B (with buffer):** The op-amp input impedance (~TΩ) doesn't load the divider at all, so the divider output is accurate. The op-amp output impedance (~10Ω) easily charges the ADC sample cap in nanoseconds, so any sampling time setting works. Cost: one MCP6002 (~$0.30) and two 100nF decoupling caps.

**Recommendation:** If this is the only ADC channel and you can afford slow sampling time, Option A works with the sampling time increase. If you're scanning multiple channels rapidly, the slower sampling time per channel becomes a bottleneck — Option B is worth the $0.30.

</details>

**E3.** Explain the relationship between inductors in DC-DC converters and the noise you might see on SPI signals. How do ferrite beads and decoupling caps form a multi-stage defense? Trace the noise path from source to victim.

<details><summary>Answer</summary>

**Noise source:** The Jetson Orin's DC-DC buck converter switches at ~1MHz. The switch node (between the high-side FET and inductor) has a rectangular voltage waveform swinging between 0V and Vin, with very fast edges (~5ns). These edges contain harmonics up to hundreds of MHz.

**Noise path:** (1) The switching current loop (FET → inductor → output cap → FET) radiates electromagnetically. (2) The pulsating current travels through the power plane, causing voltage ripple on the 3.3V rail shared with the STM32. (3) Ground return currents cause voltage differences across the ground plane that couple into signal traces.

**Multi-stage defense:**
- **Stage 1 — Ferrite bead** on the 3.3V line feeding the STM32: Acts as a lossy inductor. Passes DC freely (just wire) but presents increasing impedance at MHz frequencies, converting noise to heat. Typical FB: 600Ω @ 100MHz.
- **Stage 2 — Bulk 10µF ceramic cap** after the ferrite bead: Forms a pi-filter with the FB. Absorbs the mid-frequency noise that gets through.
- **Stage 3 — 100nF ceramic cap** at each VCC pin: Absorbs the high-frequency remnants (10–100MHz).

Together: the FB + caps form a CLC pi-filter. Each stage handles a different frequency band, providing wideband noise rejection from the DC-DC converter's switching noise to the sensitive SPI peripheral.

</details>

**E4.** The "no instantaneous voltage change" rule for capacitors and the "no instantaneous current change" rule for inductors are duals of each other. For each component, give one real-world failure mode in your robot that would result from violating (trying to violate) these rules, and one design technique that exploits these rules constructively.

<details><summary>Answer</summary>

**Capacitor — voltage can't change instantly:**
- *Failure mode:* Hot-plugging a sensor module with 100µF input caps into the robot's powered 5V bus. The caps try to charge instantly, drawing an enormous inrush current that can blow the bus fuse, damage the connector, or cause a voltage dip that resets other devices on the same rail.
- *Constructive use:* Decoupling — when the STM32 demands a sudden current burst (DMA SPI transfer), the 100nF cap's voltage can't change instantly, so it supplies the current from its stored charge, keeping VCC stable.

**Inductor — current can't change instantly:**
- *Failure mode:* Turning off a relay MOSFET without a flyback diode. The inductor (relay coil) demands that its current keep flowing. Since current can't stop instantly, the inductor generates hundreds of volts of back-EMF trying to maintain it, destroying the MOSFET.
- *Constructive use:* The buck converter inductor smooths the pulsating switch output into steady DC. When the switch opens, the inductor's "refusal" to let current stop forces current to continue flowing through the freewheeling diode into the load, maintaining continuous output current and smooth voltage.

</details>
