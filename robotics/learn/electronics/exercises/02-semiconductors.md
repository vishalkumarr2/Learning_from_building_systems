# Exercises: Semiconductors

### Chapter 2: Diodes, BJTs, MOSFETs

**Self-assessment guide:** Write your answer before expanding the details block. If you can answer 80% without peeking, you're ready for chapter 03.

---

## Section A — Conceptual Questions

**A1.** Your colleague picks an IRF540N MOSFET to switch a 12V LED strip from a 3.3V STM32 GPIO. They say "it handles 33A, way more than we need." You check the datasheet and see Rds(on) = 44mΩ at Vgs=10V, but no Rds(on) spec at Vgs=3.3V. What's the problem?

<details><summary>Answer</summary>

The IRF540N is a **standard** (not logic-level) MOSFET. Its Vgs(th) is 2–4V, meaning at 3.3V gate drive it's only partially turned on — operating in the linear region, not fully saturated. The real Rds(on) at Vgs=3.3V is likely **500mΩ or more** (over 10× higher than the spec'd 44mΩ at 10V). If the LED strip draws 5A: P = 5² × 0.5 = **12.5W** — the MOSFET will overheat and potentially catch fire. Even if it doesn't fail immediately, it'll be extremely hot and waste power.

Fix: Use a **logic-level MOSFET** (like IRLZ44N or Si2302) that specifies Rds(on) at Vgs = 2.5V or 3.3V in the datasheet.

</details>

**A2.** A team member is debugging an I2C level shifter using a BSS138 circuit. The 3.3V side can pull the 5V side low, but the 5V side cannot pull the 3.3V side low. What's the most likely cause?

<details><summary>Answer</summary>

The most likely cause is a **missing or disconnected pull-up resistor on one side**, or the BSS138 is oriented incorrectly. The BSS138 level shifter works bidirectionally but relies on specific connections:
- **Source** must be on the low-voltage (3.3V) side
- **Drain** must be on the high-voltage (5V) side
- **Gate** is tied to the low voltage rail (3.3V)

If Source and Drain are swapped, Case 3 (5V side driving low) fails: the body diode conducts from Source to Drain (3.3V→5V), but the MOSFET can't turn on to pull the 3.3V side low because Vgs would be negative. Also check that the 10kΩ pull-up to 3.3V is present on the Source side — without it, the 3.3V side has no path to go high.

</details>

**A3.** You're choosing between a silicon rectifier diode (1N4007, Vf=0.7V) and a Schottky diode (1N5819, Vf=0.3V) for a flyback diode across a relay coil that switches at ~10Hz. Your colleague says "use the cheaper silicon one, speed doesn't matter at 10Hz." Are they right?

<details><summary>Answer</summary>

They're mostly right but for the wrong reason. Speed matters even at 10Hz because the switching **event** happens in nanoseconds (when the MOSFET turns off). The silicon diode's **reverse recovery time** (~1µs) means that during turn-off transients, it briefly conducts backwards before clamping properly, allowing a small voltage spike to pass through. For a 12V relay at 10Hz, this is unlikely to damage a 30V-rated MOSFET, so the 1N4007 is acceptable.

However, if this were a PWM application (kHz switching) or if the MOSFET had a tight voltage margin, the Schottky's zero reverse recovery time would be important. For the relay case, the **real** advantage of Schottky would be 0.3V vs 0.7V forward drop — the relay current decays faster through the Schottky, meaning shorter turn-off time (solenoid disengages faster). For a relay that just needs to de-energize, either works fine.

</details>

**A4.** A GPIO pin on the STM32 is rated for 20mA maximum sink/source. The relay you need to drive requires 80mA. Your colleague suggests "just connect 4 GPIO pins in parallel to get 80mA." What's wrong with this approach?

<details><summary>Answer</summary>

GPIO pins cannot be reliably paralleled for more current. Due to manufacturing variations, each pin has slightly different output impedance. The pin with the lowest impedance will carry **most** of the current, easily exceeding its 20mA limit while the others contribute very little. There's no mechanism to current-share between GPIOs. This will damage the overloaded pin.

The correct solution is to use the GPIO to drive a **transistor** (BJT or MOSFET) that handles the relay's 80mA. A 2N2222 NPN BJT with an appropriate base resistor (Rb = (3.3V - 0.7V)/(2 × 80mA/100) ≈ 1.6kΩ) uses only ~1.6mA from the GPIO while switching 80mA through the relay. Don't forget the flyback diode across the relay coil.

</details>

**A5.** Why does the BJT relay driver circuit in the chapter include a flyback diode, but the MOSFET LED strip driver circuit does not?

<details><summary>Answer</summary>

The relay coil is an **inductive** load — it stores energy in a magnetic field that generates back-EMF when current is cut off. Without a flyback diode, this back-EMF produces a voltage spike that can destroy the transistor.

An LED strip is a **resistive** load (LEDs + current-limiting resistors). It stores no magnetic energy and produces no back-EMF when switched off. There's nothing to clamp. 

The rule is simple: **any inductive load** (relay, solenoid, motor, speaker) needs a flyback diode. **Resistive loads** (LEDs, heaters, incandescent lamps) do not.

</details>

**A6.** A P-channel MOSFET high-side switch has its gate connected directly to a 3.3V GPIO (VCC rail is also 3.3V). The engineer expects GPIO HIGH = load off, GPIO LOW = load on. But the load never turns on. Why?

<details><summary>Answer</summary>

For a P-channel MOSFET, the gate voltage is referenced to the **Source** pin, which is connected to VCC (3.3V). To turn it ON, Vgs must be sufficiently negative (Vgs < -Vth, typically -1V to -2V).

- GPIO LOW (0V): Vgs = 0V - 3.3V = **-3.3V** → ON ✓
- GPIO HIGH (3.3V): Vgs = 3.3V - 3.3V = **0V** → OFF ✓

This should work... if the GPIO can actually drive to 0V. The most likely issue is that the GPIO is configured as **open-drain** (not push-pull) or is not configured as output at all. If open-drain with no pull-down, the pin floats when "low" rather than driving to 0V. Another possibility: the MOSFET's Vth is very high (e.g., -3V), requiring Vgs more negative than -3V, which 3.3V GPIO can't provide. Check the MOSFET datasheet for Vgs(th) and ensure it turns on with the available gate drive.

</details>

**A7.** Your colleague says "MOSFETs are always better than BJTs for switching because they don't waste power at the gate." Is this correct in all cases?

<details><summary>Answer</summary>

It's correct for **DC steady-state** — a MOSFET gate draws zero DC current (it's capacitive) while a BJT base draws continuous current (Ib = Ic/β). However, MOSFETs are NOT zero-power at **switching time**. 

The gate is a capacitor (gate charge Qg, typically a few nC). Every on/off transition requires charging and discharging this capacitor. Gate power = Qg × Vgs × f_switch. At high switching frequencies (100kHz+ in DC-DC converters), this becomes significant and requires a dedicated gate driver.

Also, BJTs have lower saturation voltage (Vce_sat ≈ 0.2V) for small load currents compared to MOSFET Rds_on × I. For very low-current loads (<10mA), a BJT may actually dissipate less in the load path. So for the 100Hz switching in your SPI bridge: MOSFET gate loss is negligible, and a MOSFET is clearly better. For a 2MHz DC-DC converter: gate drive loss is a real design consideration.

</details>

**A8.** ESD protection: your SPI data line runs through a cable between the STM32 board and a sensor module. A TVS diode (SMBJ3.3A) is placed on the sensor board. Your colleague says "we're protected." What scenario are they NOT protected from?

<details><summary>Answer</summary>

The TVS is on the **sensor board**, but ESD events can occur at **either end** of the cable. If ESD enters at the STM32 end (e.g., someone touches the connector), the high-voltage spike travels through the STM32's SPI pins *before* reaching the TVS on the sensor board. The STM32 GPIO may already be damaged.

TVS protection must be placed **at the point of entry** — right where the cable connects to each board. Best practice: TVS on both boards, positioned as close to the connectors as possible. Also, TVS clamp voltage (5.5V for SMBJ3.3A) must be safe for the IC. The TVS clamp is below the STM32's absolute maximum input voltage (typically 4.0V for -0.3V rated) — actually, 5.5V clamp on a 3.3V pin rated for max 4V could still cause damage! You'd need a TVS with a lower clamp voltage, or additional series resistance to limit current.

</details>

**A9.** In the H-bridge diagram, why is turning on Q1 + Q2 (both left-side transistors) simultaneously catastrophic? Use Ohm's law to explain what happens.

<details><summary>Answer</summary>

Q1 is the high-side P-channel MOSFET connecting VCC to the left motor terminal. Q2 is the low-side N-channel MOSFET connecting the left motor terminal to GND. If both are ON simultaneously:

VCC → Q1 (Rds_on ≈ 20mΩ) → Q2 (Rds_on ≈ 20mΩ) → GND

Total resistance = Rds_on(Q1) + Rds_on(Q2) ≈ **40mΩ**. The motor is bypassed entirely. By Ohm's law: I = V/R = 12V / 0.04Ω = **300A**. This is a direct short circuit through both MOSFETs. The I²R power in each MOSFET = 300² × 0.02 = **1800W**. Both MOSFETs explode within microseconds, the PCB trace melts, and you may get an arc/fire. This is called **"shoot-through"** and is why H-bridge ICs have built-in **dead-time** — a brief period where both FETs are off during switching transitions.

</details>

---

## Section B — Spot the Bug

**B1.** A BJT relay driver calculation:

> Relay current: 100mA. BJT β = 200. Base resistor: Rb = (3.3V - 0.7V) / (100mA / 200) = 2.6V / 0.5mA = 5.2kΩ. Using 4.7kΩ.

The relay doesn't switch on. What's the error?

<details><summary>Answer</summary>

The calculation gives the **minimum** base current (Ib = Ic/β = 0.5mA) assuming the typical β of 200. But BJT datasheets specify β_min, which could be 75–100 for a 2N2222. At β_min = 100: Ib_needed = 100mA/100 = 1mA. With Rb = 4.7kΩ: Ib = 2.6V/4.7kΩ = 0.55mA — not enough for saturation with worst-case β.

The rule is to **overdrive by 2–3×** for reliable saturation. Use Ib = 2–3 × Ic/β_min = 2–3mA. Rb = 2.6V / 2.5mA = **1kΩ** would be a reliable choice. The original design works with a "typical" transistor but fails on the production line when you get a BJT with β on the low end.

</details>

**B2.** An LED driver circuit:

> 12V supply, blue LED (Vf = 3.0V), desired current 15mA.
> R = (12V - 3.0V) / 15mA = 600Ω. Using 560Ω (nearest standard).
> Connecting it to a 3.3V GPIO output.

The LED barely lights. What went wrong?

<details><summary>Answer</summary>

The resistor was calculated for a **12V supply**, but it's connected to a **3.3V GPIO**. With a 3.3V source and blue LED (Vf = 3.0V):
V across resistor = 3.3V - 3.0V = **0.3V**.
I = 0.3V / 560Ω = **0.54mA** — barely enough to see.

Worse: 3.3V is barely above the LED's forward voltage. The LED may not turn on at all with a blue/white LED. For 3.3V GPIOs with blue LEDs, the voltage margin is too thin for practical use. Solutions: (1) Use a red LED (Vf ≈ 2.0V) for 3.3V direct drive, (2) Use an NPN BJT to switch the LED from a 5V or 12V supply, or (3) Use a boost LED driver.

</details>

**B3.** An engineer adds ESD protection to a 3.3V I2C bus:

> "I placed a standard 1N4007 silicon diode from each signal line to GND. Cathode on signal, anode on GND. This will clamp any negative ESD spike."

Two problems exist. Find them.

<details><summary>Answer</summary>

**Problem 1: Direction handles only negative spikes.** With cathode on signal and anode on GND, the diode only conducts when the signal goes **below GND** (negative ESD). It provides zero protection against **positive** ESD spikes above 3.3V, which are equally common and equally destructive. Need protection in both directions — either a TVS diode or a pair of diodes (one to VCC, one to GND).

**Problem 2: The 1N4007 is far too slow for ESD.** ESD events have rise times of ~1ns and durations of ~100ns. The 1N4007 has a reverse recovery time of ~30µs — thousands of times too slow. By the time it responds, the ESD has already passed through and damaged the IC. Use a **TVS diode** (response time in picoseconds) specifically designed for ESD protection, like the SMBJ3.3A or a dedicated ESD array like the TPD4E05U06.

</details>

**B4.** A MOSFET power switch design:

> "Using Si2302 N-channel MOSFET to switch power to a 5V sensor from the robot's 5V rail. Drain connected to 5V rail, Source connected to sensor VCC, Gate driven by 3.3V GPIO. Sensor GND to common ground."

The sensor powers on but reads incorrectly. Why?

<details><summary>Answer</summary>

This is a **high-side switch using an N-channel MOSFET** — it doesn't work properly because of the **source follower** effect. When the MOSFET turns on, current flows from Drain (5V) to Source (sensor VCC). As the Source voltage rises toward 5V, Vgs = Vgate - Vsource = 3.3V - Vsource. When Vsource reaches ~2V, Vgs drops to 1.3V — below Vth — and the MOSFET turns off. The Source can never reach 5V; it stabilizes around **Vgate - Vth ≈ 3.3V - 1.0V = 2.3V**. The sensor gets only ~2.3V instead of 5V.

**Fix:** Use a **P-channel** MOSFET for high-side switching (Source to 5V, Drain to sensor, Gate pulled low to turn on). Or use the N-channel as a **low-side switch** (between sensor GND and ground), which sidesteps the source-follower problem because Source is always at GND.

</details>

**B5.** An engineer measures a diode with a multimeter in "diode test" mode:

> "I measure 0.7V in one direction and 0.3V in the other. This diode is broken — it should show 0V in forward and OL in reverse."

What's the actual measurement showing?

<details><summary>Answer</summary>

The engineer's understanding of "diode test" mode is wrong. In diode mode, the multimeter applies a small current and measures the **forward voltage drop**. A healthy diode shows:
- Forward: voltage drop (0.5–0.8V for silicon, 0.2–0.4V for Schottky)
- Reverse: "OL" (over-limit/open — no current flows)

The measurement of 0.7V in one direction and 0.3V in the other suggests the "diode" is actually **two diodes back-to-back** in the same package, or more likely this is a **TVS diode** or **bidirectional Zener**. A single regular diode should show OL in reverse, not 0.3V. If it really is a standard diode showing 0.3V in reverse, the diode IS damaged (partial reverse breakdown/leakage). But the expected behavior stated by the engineer ("0V forward, OL reverse") is also wrong — forward should show ~0.7V, not 0V.

</details>

**B6.** BSS138 level shifter for I2C, linking a 3.3V STM32 to a 5V EEPROM:

> "I connected Source to the 5V side, Drain to the 3.3V side, Gate to 3.3V. Pull-ups: 10kΩ to each rail."

It doesn't shift levels at all. Why?

<details><summary>Answer</summary>

Source and Drain are **swapped**. The BSS138 level shifter requires:
- **Source** → low-voltage side (3.3V)
- **Drain** → high-voltage side (5V)
- **Gate** → low-voltage rail (3.3V)

With Source on the 5V side: when the 3.3V side pulls low, Vgs = 3.3V(gate) - 5V(source) = **-1.7V**, which is below Vth for an N-channel MOSFET — it never turns on. The body diode also faces the wrong direction (conducts 3.3V → 5V instead of routing the low signal properly). The entire bidirectional mechanism fails. Swap Source and Drain connections: Source to 3.3V side, Drain to 5V side.

</details>

---

## Section C — Fill in the Blank / From Memory

### C1. Diode Forward Voltage by Type

| Diode Type | Typical Vf | Speed | Typical Use |
|-----------|-----------|-------|-------------|
| Silicon rectifier (1N4007) | ______V | ______ | ____________ |
| Schottky (BAT54) | ______V | ______ | ____________ |
| Red LED | ______V | N/A | ____________ |
| Blue/White LED | ______V | N/A | ____________ |
| Zener | ______V fwd, Vz rev | ______ | ____________ |
| TVS (SMBJ3.3A) | ______V | ______ | ____________ |

<details><summary>Answer</summary>

| Diode Type | Typical Vf | Speed | Typical Use |
|-----------|-----------|-------|-------------|
| Silicon rectifier (1N4007) | 0.7V | Slow (µs) | Power rectification, flyback |
| Schottky (BAT54) | 0.2–0.4V | Fast (ns) | Power supply, OR-ing, high-freq rectification |
| Red LED | 1.8–2.2V | N/A | Indication |
| Blue/White LED | 2.8–3.5V | N/A | Indication, illumination |
| Zener | 0.7V fwd, Vz rev | Medium | Voltage regulation/reference |
| TVS (SMBJ3.3A) | 0.7V | Very fast (ps) | ESD/transient protection |

</details>

### C2. BJT vs MOSFET Comparison

| Property | BJT | MOSFET |
|----------|-----|--------|
| Control signal | ______ (Ib controls Ic) | ______ (Vgs controls Id) |
| Gate/Base current (DC) | ______ | ______ |
| Input impedance | ______ | ______ |
| ON voltage drop | Vce_sat ≈ ______V | ______ × Id |
| Thermal behavior | ______ temp coeff → runaway | ______ temp coeff → self-limiting |
| Modern primary use | ______ | ______ |

<details><summary>Answer</summary>

| Property | BJT | MOSFET |
|----------|-----|--------|
| Control signal | Current (Ib controls Ic) | Voltage (Vgs controls Id) |
| Gate/Base current (DC) | Continuous (Ib flows always) | Zero (gate is capacitive) |
| Input impedance | Low (~kΩ) | Extremely high (~TΩ) |
| ON voltage drop | Vce_sat ≈ 0.2V | Rds_on × Id |
| Thermal behavior | Positive temp coeff → runaway | Negative temp coeff → self-limiting |
| Modern primary use | Analog circuits, current sources | Digital switching, power management |

</details>

### C3. Key Formulas — Write From Memory

1. BJT base resistor: Rb = ______
2. LED current-limiting resistor: R = ______
3. MOSFET power dissipation (when on): P = ______
4. Zener bias resistor: R = ______
5. BJT saturation rule for base current: Ib ≥ ______

<details><summary>Answer</summary>

1. Rb = (Vgpio − Vbe) / Ib where Ib = 2–3 × Ic/β_min (overdrive for saturation)
2. R = (Vcc − Vf) / I_desired
3. P = Id² × Rds_on
4. R = (Vin − Vz) / Iz
5. Ib ≥ 2 × Ic / β_min (2× minimum for reliable saturation)

</details>

### C4. N-Channel vs P-Channel — Where They Go

Fill in the blanks:

- N-channel: ____-side switch. Gate referenced to ____. GPIO ____  = ON.
- P-channel: ____-side switch. Gate must be pulled ____ relative to ____. GPIO ____ = ON.
- Rule of thumb: Prefer __-channel because ______.

<details><summary>Answer</summary>

- N-channel: **Low**-side switch. Gate referenced to **GND**. GPIO **HIGH** = ON.
- P-channel: **High**-side switch. Gate must be pulled **LOW** relative to **Source (VCC)**. GPIO **LOW** = ON.
- Rule of thumb: Prefer **N**-channel because they're **cheaper, lower Rds_on, and more widely available**.

</details>

---

## Section D — Lab / Calculation Tasks

**D1. BJT Relay Driver — Full Design**

Design a BJT driver for a 24V relay coil (R_coil = 480Ω) from a 3.3V GPIO that can source max 4mA. Use a 2N2222 (β_min = 100, Vbe = 0.7V, Vce_sat = 0.2V).

a) What is the relay coil current?
b) What minimum base current is needed for saturation (with 2× overdrive)?
c) Calculate the base resistor value.
d) Verify the GPIO can supply this current.
e) What happens if you forget the flyback diode?

<details><summary>Answer</summary>

a) Ic = V_relay / R_coil = 24V / 480Ω = **50mA** (the relay is powered by 24V through the coil, the BJT just sinks the return current to GND)

Actually, with the BJT: the relay sees 24V - Vce_sat = 24V - 0.2V = 23.8V. Ic = 23.8V / 480Ω = **49.6mA ≈ 50mA** ✓

b) Ib_min = Ic / β_min = 50mA / 100 = 0.5mA. With 2× overdrive: Ib = **1.0mA**

c) Rb = (Vgpio - Vbe) / Ib = (3.3V - 0.7V) / 1.0mA = 2.6V / 0.001A = **2.6kΩ → use 2.7kΩ** (standard E12 value)

d) GPIO current = Ib = 2.6V / 2700Ω = **0.96mA** < 4mA max ✓ (well within GPIO spec)

e) The relay coil (L ≈ R × τ, significant inductance) generates back-EMF when the BJT turns off. V_spike = L × dI/dt. With fast BJT turn-off (~100ns), the spike easily exceeds the 2N2222's Vce_max of 40V. The collector voltage shoots up to (24V + spike), exceeding breakdown → BJT destroyed. Add a 1N4007 flyback diode (cathode to +24V, anode to collector/relay junction).

</details>

**D2. Logic-Level MOSFET Selection**

You need to switch a 5A 12V DC motor from a 3.3V GPIO. Compare these two MOSFETs:

| Parameter | IRLZ44N | Si2302 |
|-----------|---------|--------|
| Vgs(th) | 1–2V | 0.5–1V |
| Rds(on) @ Vgs=2.5V | 0.05Ω | 0.1Ω |
| Id_max | 47A | 2.6A |

a) Which MOSFET can handle this load?
b) Calculate power dissipation for the suitable MOSFET.
c) Does it need a heatsink? (Assume TO-220 package can dissipate 2W without heatsink.)

<details><summary>Answer</summary>

a) The Si2302 has Id_max = 2.6A — **cannot** handle 5A. It would be destroyed immediately. The IRLZ44N has Id_max = 47A and Vgs(th) = 1–2V, so 3.3V gate drive will fully saturate it. **IRLZ44N** is the correct choice.

b) At Vgs = 3.3V (slightly above the 2.5V spec point), Rds_on ≈ 0.05Ω (may be slightly higher, ~0.06Ω).
P = Id² × Rds_on = 5² × 0.05 = **1.25W**

c) 1.25W < 2W (TO-220 free-air limit), so **no heatsink** needed for ambient temperature ≤ 40°C. But with 50% derating (good practice): 2W × 0.5 = 1W. At 1.25W, you're slightly above the derated limit. A small heatsink or PCB copper pour for thermal relief would be wise for reliability, especially in an enclosed robot chassis.

</details>

**D3. Zener Voltage Regulator**

Design a Zener regulator to provide a 5.1V reference from a 12V supply. The load draws 5mA.

a) Calculate the series resistor value.
b) Calculate power dissipated in the resistor and Zener.
c) Why would you NOT use this to power a microcontroller?

<details><summary>Answer</summary>

a) Total current through R must be Zener current + load current. Zener needs ~10–20mA for good regulation. Use Iz = 15mA. Total: 15mA + 5mA = 20mA.
R = (Vin - Vz) / I_total = (12V - 5.1V) / 20mA = 6.9V / 0.02A = **345Ω → use 330Ω**

Check: I = 6.9V / 330Ω = 20.9mA. Iz = 20.9 - 5 = 15.9mA ✓

b) Resistor: P_R = I² × R = 0.0209² × 330 = **0.144W** (safe for 1/4W resistor)
Zener: P_Z = Vz × Iz = 5.1V × 15.9mA = **0.081W** (safe for 500mW Zener)

c) This is a terrible MCU power supply because: (1) **Regulation is poor** — if load current changes (MCU draws 10mA during sleep, 100mA during SPI burst), Vz shifts. (2) **Efficiency is awful** — most power (12V × 20mA = 240mW) is wasted as heat in R and the Zener, with only 5.1V × 5mA = 25.5mW reaching the load (~10% efficiency). (3) **No current limiting** — if the MCU draws more than Iz + I_load, the output voltage collapses. Use a proper voltage regulator (LDO or buck converter) for powering active circuits. Zener regulators are only suitable for low-current references.

</details>

**D4. BSS138 Level Shifter Analysis**

Walk through the BSS138 level shifter for a single I2C SDA line (3.3V ↔ 5V). BSS138 specs: Vth = 1.5V, body diode Vf = 0.7V.

a) 3.3V device drives LOW (0V). Calculate Vgs. Does the MOSFET turn on? What voltage does the 5V side see?
b) Neither side drives (both released). What voltages appear on each side? What is Vgs?
c) 5V device drives LOW (0V). Trace the body diode path. What voltage does the 3.3V side initially see? Then what happens to the MOSFET?

<details><summary>Answer</summary>

Setup: Gate tied to 3.3V. Source = 3.3V side. Drain = 5V side. 10kΩ pull-ups to each rail.

a) SDA_3V3 = 0V. Gate = 3.3V. Source = 0V (pulled low by 3.3V device).
Vgs = 3.3V - 0V = **3.3V > Vth (1.5V) → MOSFET ON**.
Drain (5V side) is pulled low through the MOSFET's channel. SDA_5V = **0V** (≈ Rds_on × I, negligible). ✓

b) Both sides released. R_pull on 3.3V side pulls SDA_3V3 to **3.3V**. R_pull on 5V side pulls SDA_5V to **5V**.
Source = 3.3V. Gate = 3.3V. Vgs = 3.3V - 3.3V = **0V → MOSFET OFF**.
Both sides sit at their respective rail voltages. ✓

c) 5V device pulls SDA_5V to 0V. Drain = 0V. Source is at 3.3V (from pull-up).
**Body diode:** conducts from Source (3.3V) to Drain (0V) since Source > Drain by more than Vf.
SDA_3V3 drops to approximately Vf = **0.7V** (through the body diode).
Now: Vgs = 3.3V(gate) - 0.7V(source) = **2.6V > Vth → MOSFET turns ON**.
With MOSFET channel conducting, SDA_3V3 is pulled fully to ≈ **0V** (channel Rds_on is much lower than body diode). ✓

</details>

**D5. Shoot-Through Energy Calculation**

An H-bridge uses MOSFETs with Rds_on = 25mΩ each at full enhancement. VCC = 24V. If shoot-through occurs (both high-side and low-side ON simultaneously) for 100ns:

a) What is the short-circuit current?
b) How much energy is dissipated in those 100ns?
c) If the MOSFET junction can absorb 10µJ before failure, does it survive?

<details><summary>Answer</summary>

a) Total resistance = Rds_on(high) + Rds_on(low) = 25mΩ + 25mΩ = 50mΩ.
I = V/R = 24V / 0.05Ω = **480A** (!)

b) Power = I² × R_total = 480² × 0.05 = **11,520W** per MOSFET pair.
Energy in 100ns: E = P × t = 11,520 × 100×10⁻⁹ = **1.152mJ = 1152µJ**

c) 1152µJ >> 10µJ. The MOSFET is destroyed roughly **100× over** in just 100ns of shoot-through. Even a 10ns glitch would dissipate ~115µJ — still 11× the failure threshold. This is why H-bridge ICs implement dead-time (typically 50–200ns of all-FETs-off between transitions) and why discrete H-bridge designs require careful gate driver timing.

</details>

---

## Section E — Deeper Thinking

**E1.** The BSS138 level shifter is described as "one of the most elegant circuits in electronics." Explain why it needs **open-drain** outputs on both sides to work bidirectionally. What would happen if a push-pull (totem-pole) output drove one side HIGH to its rail while the other side was also driving HIGH to a different rail?

<details><summary>Answer</summary>

The BSS138 circuit works because both sides use **open-drain** (or open-collector) outputs with pull-up resistors. Open-drain means the device can only actively drive the line LOW (by sinking current to ground) or release it (high impedance, letting the pull-up bring it HIGH).

This is critical because the two sides have **different voltage rails** (3.3V and 5V). If a push-pull output on the 5V side drove SDA_5V to 5V, and the MOSFET happened to be on, then 5V would be forced onto the 3.3V side through the MOSFET channel — destroying the 3.3V device(s) connected there.

With open-drain: when a device releases the line, the **local pull-up** sets the voltage to the **local rail**. The 3.3V side floats to 3.3V; the 5V side floats to 5V. No cross-rail voltage forcing occurs. The MOSFET only conducts when one side is actively pulling LOW, which is safe because LOW is 0V on both sides.

This is exactly why I2C uses open-drain: it enables multi-master operation, multi-voltage bus segments, and wired-AND logic where any device can pull the bus low.

</details>

**E2.** In your robot, motor driver MOSFETs have body diodes. During PWM operation of an H-bridge, explain what role the body diodes play during the dead-time (both high and low FETs off). What happens to the motor current during this period, and why might you add external Schottky diodes in parallel with the body diodes?

<details><summary>Answer</summary>

During motor PWM switching, when one pair of H-bridge FETs turns off and the complementary pair is about to turn on, there's a dead-time (~50–200ns) where ALL four FETs are off. But the motor is an inductive load — its current **cannot stop instantly** (V = L × dI/dt).

During dead-time, the motor current continues flowing through the **body diodes** of the opposite FETs. For example: if current was flowing left-to-right, it continues through the body diode of Q3 (high-side) and Q2 (low-side), creating a freewheeling path. The body diodes act as flyback diodes for the motor inductance.

The problem: MOSFET body diodes are **slow** (reverse recovery time ~100–500ns). When the complementary FETs turn on after dead-time, the body diode takes time to stop conducting (reverse recovery), causing a momentary shoot-through current spike. This:
1. Increases switching losses
2. Creates voltage spikes (EMI)
3. Stresses the FETs

**External Schottky diodes** in parallel have zero reverse recovery time. They conduct the freewheeling current instead of the body diodes. When the complementary FETs turn on, the Schottky stops instantly — no shoot-through. This is critical at high PWM frequencies (>20kHz) used for quiet motor operation.

</details>

**E3.** Consider the full signal chain from a 5V legacy sensor to the STM32's 3.3V ADC on your robot. You need: voltage scaling, ESD protection, and anti-alias filtering. Design the protection and conditioning chain end-to-end, specifying component types and approximate values. Which semiconductor concepts from this chapter do you need, and which passive component concepts from chapter 01?

<details><summary>Answer</summary>

**Signal chain design (sensor → STM32 ADC):**

```
5V Sensor → [TVS] → [Voltage Divider] → [Buffer Op-Amp] → [RC Anti-Alias Filter] → [STM32 ADC]
```

**Stage 1 — ESD Protection (Chapter 02: TVS diodes):**
TVS diode (SMBJ5.0A) from sensor signal to GND. Clamps transients to ~8V (safe for the downstream resistive divider). Place at the connector, before any other components.

**Stage 2 — Voltage Scaling (Chapter 01: voltage divider):**
R1 = 10kΩ (top), R2 = 20kΩ (bottom). Vout = 5 × 20k/(10k+20k) = 3.33V max. Low enough for the 3.3V ADC input.

**Stage 3 — Buffer (Chapter 03 preview: op-amp voltage follower):**
MCP6002 rail-to-rail op-amp in voltage follower configuration. Isolates the high-impedance divider from the ADC's sampling capacitor. Input impedance ~TΩ (no loading), output impedance ~10Ω.

**Stage 4 — Anti-Alias Filter (Chapter 01: RC low-pass):**
R = 3.3kΩ, C = 100nF → f_c ≈ 480Hz. Placed after the buffer, before the ADC pin. Attenuates frequencies above 480Hz before the ADC samples at 1kHz (Nyquist = 500Hz).

**Semiconductor concepts used:** TVS diode behavior, forward voltage drop awareness (for understanding why the TVS clamping voltage matters), op-amp buffer from chapter 03.

**Passive concepts used:** Voltage dividers with loading analysis, RC filter cutoff calculation, decoupling on the op-amp supply pins.

</details>

**E4.** A thermal protection question: BJTs have positive temperature coefficient of gain (β increases with temperature → more current → more heat → more current → thermal runaway). MOSFETs have negative temperature coefficient of Rds_on (resistance increases with temperature → more heat → more resistance → less current → self-limiting). Explain how this difference affects your choice for the motor driver stage in a robot that operates in a 40°C warehouse. What would happen to a BJT Darlington motor driver (like TIP120) vs. a MOSFET (IRLZ44N) if a software bug caused the PWM duty cycle to stay at 100% for 60 seconds?

<details><summary>Answer</summary>

**BJT (TIP120 Darlington) at 100% duty for 60 seconds:**
At 100% duty, the TIP120 carries full motor current continuously. Power dissipation = Ic × Vce_sat. The Darlington's Vce_sat is ~1.2V (two junction drops). At 3A: P = 3A × 1.2V = 3.6W. The junction heats up. As temperature rises, β increases → Vce_sat drops slightly, but the bigger problem is that β of the cascaded pair can increase dramatically (from 1000 to 3000+). If the base drive isn't precisely controlled, the transistor can enter secondary breakdown — localized hotspots where current concentrates, leading to **thermal runaway**. The TIP120 does NOT self-correct; it requires external thermal management, proper operating area derating, and a heatsink. At 40°C ambient, the thermal budget is already reduced.

**MOSFET (IRLZ44N) at 100% duty for 60 seconds:**
P = Id² × Rds_on ≈ 3² × 0.05 = 0.45W. As temperature rises, Rds_on increases (~0.4%/°C), so at 80°C junction: Rds_on ≈ 0.075Ω, P ≈ 0.675W. The increased resistance naturally limits the current (slightly) and distributes heat more evenly since hotter regions carry less current. The MOSFET **self-limits** — it reaches thermal equilibrium rather than running away. With a small heatsink or copper pour, 0.675W is easily managed. 

**Conclusion:** For a robot in a 40°C warehouse where software bugs can cause sustained high current, MOSFETs are dramatically safer than BJTs. The self-limiting thermal behavior provides a built-in safety margin that BJTs lack.

</details>
