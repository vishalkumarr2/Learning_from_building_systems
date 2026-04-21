# 01 — Passive Components: Resistors, Capacitors, Inductors
### The foundation of every circuit you'll ever build or debug
**Prerequisite:** Basic algebra (V=IR, that's it)
**Unlocks:** Pull-up/pull-down resistors, decoupling, voltage dividers, RC filters, power budgeting

---

## Why Should I Care? (STM32/Jetson Project Context)

Every single wire in your SPI bridge has passive components on it:
- **Resistors:** Pull-ups on CS lines keep slaves deselected during boot. Voltage dividers scale 5V signals to 3.3V. Current-limiting resistors protect GPIOs.
- **Capacitors:** 100nF decoupling caps next to the STM32 prevent brownout during DMA bursts. Bulk 10µF caps smooth the 3.3V rail. RC filters on analog sensor inputs prevent aliasing.
- **Inductors:** Ferrite beads on power lines keep switching noise from reaching the MCU. The Jetson's DC-DC converter uses inductors internally.

If you skip this chapter, you cannot debug power-related SPI glitches. Half of the "my SPI drops frames randomly" bugs are really "my power supply sags when the MCU is busy" bugs.

---

# PART 1 — RESISTORS

---

## 1.1 What a Resistor Physically IS

A resistor is a component that opposes the flow of electric current. It converts electrical energy into heat.

**The highway analogy:**
Imagine current (electrons) as cars on a highway. A resistor is a section of road that narrows from 4 lanes to 1 lane. The same number of cars must get through, but they slow down and pile up. The wider the bottleneck (lower resistance), the more cars flow. The narrower (higher resistance), the fewer.

Where the analogy breaks down: in a real highway, cars queue up. In a resistor, electrons don't "wait" — instead, a voltage difference builds up across the resistor (like water pressure building behind a constriction). This voltage drop is what Ohm's law describes.

**Physically, resistors are made from:**
- **Carbon film:** A thin carbon layer on a ceramic rod. Cheap, ±5% tolerance. Brown/black cylinders with color bands.
- **Metal film:** A thin metal (nickel-chromium) spiral on ceramic. Better: ±1% tolerance, lower noise. Blue body.
- **SMD (Surface Mount):** Tiny rectangular chips soldered flat on PCBs. The "0402" and "0603" you see everywhere. Marked with 3-digit codes (e.g., "103" = 10 × 10³ = 10kΩ).
- **Wire-wound:** Actual wire on a bobbin. For high power (5W+). Used in power supplies.

```
Through-hole resistor (axial):

  Color bands
  ↓↓↓↓
  ████████████════════════════████████████
 ─────┤                              ├─────
  Lead                               Lead
  (wire)                             (wire)

SMD resistor (0603 package, 1.6mm × 0.8mm):

  ┌──────┐
  │ 103  │ ← marking = 10 × 10³ = 10kΩ
  └──────┘
```

---

## 1.2 Ohm's Law — The One Equation That Rules Them All

```
    V = I × R

    V = voltage across the resistor (Volts)
    I = current through the resistor (Amps)
    R = resistance (Ohms, Ω)
```

Rearranged: `I = V/R` and `R = V/I`

**Why should I care?** Every time you add a component to a circuit, you need Ohm's law to calculate the current flowing, the voltage dropped, or the resistor value needed. It's the `print()` of electronics.

### Worked Example: LED Current Limiting

**Problem:** You want to light a red LED from a 3.3V GPIO pin. The LED datasheet says: forward voltage Vf = 2.0V, typical current = 10mA.

```
        3.3V GPIO output
            │
            │
           ┌┴┐
           │R│  ← What value?
           │ │
           └┬┘
            │
           ╲│╱
          ──┤├── ← Red LED (Vf = 2.0V)
            │
           GND
```

**Solution:**
1. Voltage across the resistor: Vr = 3.3V - 2.0V = 1.3V (the LED "uses" 2V, the rest falls across R)
2. Desired current: I = 10mA = 0.010A
3. Ohm's law: R = V/I = 1.3V / 0.010A = 130Ω

Standard resistor values don't include 130Ω. Use 150Ω (the next standard value up):
- Actual current: I = 1.3V / 150Ω = 8.7mA ← still bright enough, slightly safer

**Common mistake:** Forgetting to subtract the LED forward voltage. If you use R = 3.3V / 10mA = 330Ω, you get I = 1.3/330 = 3.9mA, which is quite dim. (Though 330Ω is commonly recommended as a conservative "works for most LEDs" value.)

---

## 1.3 Voltage Dividers

A voltage divider takes a high voltage and produces a lower voltage using two resistors.

```
    Vin ──┬──
          │
         ┌┴┐
         │R1│
         └┬┘
          ├──── Vout
         ┌┴┐
         │R2│
         └┬┘
          │
    GND ──┴──
```

**Formula:**

```
    Vout = Vin × R2 / (R1 + R2)
```

**ELI15 intuition:** Think of R1 and R2 as two sections of a garden hose, the same water (current) flows through both. The pressure (voltage) drops proportionally across each section. If both hoses are the same length (R1 = R2), the pressure at the midpoint is exactly half.

### Worked Example: Scaling 5V to 3.3V for an ADC input

A sensor outputs 0-5V, but your STM32 ADC accepts 0-3.3V maximum.

**Target ratio:** Vout/Vin = 3.3/5 = 0.66

**Choose R2 = 20kΩ, solve for R1:**
```
    0.66 = 20k / (R1 + 20k)
    R1 + 20k = 20k / 0.66 = 30.3k
    R1 = 10.3k → use 10kΩ standard value
```

**Check:** Vout = 5 × 20k / (10k + 20k) = 5 × 0.667 = 3.33V ✓

**The loading problem:** A voltage divider assumes no current is drawn from Vout. If you connect a load that draws current, the output voltage drops. For the ADC example above, the STM32 ADC input impedance is ~50kΩ during sampling. This appears in parallel with R2:

```
    R2_effective = R2 ∥ R_load = (20k × 50k) / (20k + 50k) = 14.3kΩ
    Vout_loaded = 5 × 14.3k / (10k + 14.3k) = 2.94V  ← lower than expected!
```

**Fix:** Use lower resistor values (1kΩ + 2kΩ) so the load doesn't matter, OR add a buffer (op-amp voltage follower) between the divider and the ADC. See chapter 03.

---

## 1.4 Pull-Up and Pull-Down Resistors

**Why should I care?** SPI CS lines need pull-ups. I2C SDA/SCL need pull-ups. Reading a button on a GPIO needs a pull-up or pull-down. Missing pull-ups cause ~30% of "my bus doesn't work" bugs.

### What "Floating" Means

A CMOS input (like a GPIO pin configured as input) has extremely high impedance — think 10MΩ to 1TΩ. It draws essentially zero current. This means that if you leave it disconnected, its voltage is determined by whatever stray charge/coupling is around.

```
    What happens to a floating CMOS input:

    Voltage
    3.3V ┤      ╱╲    ╱╲        ← picks up 50/60Hz from nearby
         │    ╱    ╲╱    ╲       power lines, radio, your hand
         │  ╱                    waving nearby, etc.
    1.65V┤ ╱
         │╱          ╲
    0V   ┤            ╲╱╲╱
         └─────────────────── time

    The input randomly oscillates between HIGH and LOW!
```

This is a "floating" input. It reads garbage. MOSFETs inside the IC partially turn on, which wastes power and can damage the chip over time.

### How Pull-Ups Fix It

A pull-up resistor connects the signal to VCC through a resistor. When nothing else drives the line, the resistor gently "pulls" it to VCC (HIGH). When a device wants to drive it LOW, it can easily overpower the weak pull-up.

```
    WITH PULL-UP:                    WITH PULL-DOWN:

    VCC (3.3V)                       Signal ──┬── to GPIO
      │                                       │
     ┌┴┐                                     ┌┴┐
     │R│ 10kΩ (typical)                       │R│ 10kΩ
     └┬┘                                     └┬┘
      ├── to GPIO (reads HIGH                  │
      │    when nothing drives                GND
      │    the line LOW)                 (reads LOW when
      │                                  nothing drives HIGH)
     ╱  ← switch or open-drain device
    GND
```

**Pull-up value selection:**
- **Too high (1MΩ):** Still somewhat floating — slow rise time, susceptible to noise
- **Too low (100Ω):** When the line is driven LOW, current = 3.3V/100Ω = 33mA — wastes power, may exceed GPIO sink current spec (typically 20mA)
- **Sweet spot:** 1kΩ to 100kΩ depending on application
  - I2C: 1kΩ–10kΩ (must charge bus capacitance fast enough — see chapter 06)
  - SPI CS: 10kΩ (just needs to stay high during boot)
  - Button: 10kΩ–100kΩ (no speed requirement)

### Internal Pull-Ups in MCUs

Most MCUs have internal pull-up/pull-down resistors you can enable in software. On STM32, they're typically 30kΩ–50kΩ. This is fine for buttons but TOO WEAK for I2C (can't charge bus capacitance fast enough at 400kHz). Always use external pull-ups for I2C.

```c
/* Zephyr: enable internal pull-up on a GPIO */
gpio_pin_configure(dev, PIN, GPIO_INPUT | GPIO_PULL_UP);
```

---

## 1.5 Power Dissipation — When Resistors Catch Fire

**Formula:**

```
    P = I² × R     (power in watts)
    P = V² / R     (equivalent form)
    P = V × I      (most general form)
```

### Worked Example: Why a 100Ω Resistor on 12V Gets Hot

```
    I = V/R = 12V / 100Ω = 0.12A = 120mA
    P = I² × R = 0.12² × 100 = 1.44W
```

A standard through-hole resistor is rated for **0.25W** (1/4 watt). You're dissipating 1.44W — that's **5.8× the rated power**. The resistor will overheat, the color bands will turn brown/black, and eventually the resistor burns open (fails as open circuit).

**Fix options:**
- Use a higher resistance (1kΩ: P = 144mW ✓)
- Use a higher-power resistor (2W wire-wound)
- Use a different approach entirely (voltage regulator)

**Rule of thumb:** Always calculate power dissipation. Derate by 50% (don't run a 1/4W resistor above 1/8W for long life).

---

## 1.6 Series and Parallel Combinations

### Series: Resistances ADD

```
    ─── R1 ─── R2 ─── R3 ───

    R_total = R1 + R2 + R3
```

**Intuition:** Three narrow sections of highway in a row. Cars must pass through all three. Total narrowing = sum of all narrowing.

### Parallel: Conductances ADD (reciprocals)

```
         ┌── R1 ──┐
    ─────┤        ├─────
         ├── R2 ──┤
         │        │
         └── R3 ──┘

    1/R_total = 1/R1 + 1/R2 + 1/R3
```

**Intuition:** Three highways side by side. Cars can take any of the three. More routes = less total resistance. The result is ALWAYS less than the smallest individual resistor.

**Special case (two resistors in parallel):**
```
    R_total = (R1 × R2) / (R1 + R2)
```

**Quick mental math:** Two equal resistors in parallel = half of one. 10kΩ ∥ 10kΩ = 5kΩ.

---

## 1.7 Tolerance and Markings

**Through-hole color code:**

| Color | Digit | Multiplier | Tolerance |
|-------|-------|------------|-----------|
| Black | 0 | ×1 | — |
| Brown | 1 | ×10 | ±1% |
| Red | 2 | ×100 | ±2% |
| Orange | 3 | ×1k | — |
| Yellow | 4 | ×10k | — |
| Green | 5 | ×100k | ±0.5% |
| Blue | 6 | ×1M | ±0.25% |
| Violet | 7 | — | ±0.1% |
| Grey | 8 | — | — |
| White | 9 | — | — |
| Gold | — | ×0.1 | ±5% |
| Silver | — | ×0.01 | ±10% |

**Example:** Brown-Black-Orange-Gold = 1, 0, ×1000, ±5% = 10kΩ ±5%

**SMD 3-digit code:** First two digits + multiplier (power of 10). Examples:
- 103 = 10 × 10³ = 10kΩ
- 472 = 47 × 10² = 4.7kΩ
- 330 = 33 × 10⁰ = 33Ω

**E12 standard values (the 12 "magic numbers"):** 10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82 — then multiply by powers of 10. Every resistor value you can buy is one of these multiplied by 10ⁿ.

---

# PART 2 — CAPACITORS

---

## 2.1 What a Capacitor Physically IS

A capacitor is two conductive plates separated by an insulator (dielectric). Apply voltage → electrons pile up on one plate and are repelled from the other. This stores energy in the electric field between the plates.

**The water tank analogy (detailed mapping):**

```
    WATER MODEL                         ELECTRICAL MODEL
    ───────────                         ────────────────
    Water tank                    ←→    Capacitor
    Tank width (diameter)         ←→    Capacitance (Farads)
    Water level (height)          ←→    Voltage (Volts)
    Water flow (liters/sec)       ←→    Current (Amps)
    Pipe resistance               ←→    Series resistance (ESR)

    Filling a tank through a narrow pipe:
    - At first, tank is empty, water flows fast     → capacitor charges fast initially
    - As tank fills, back-pressure builds           → voltage rises, current decreases
    - Eventually, pressure equalizes → flow stops   → capacitor fully charged, current = 0

    This is EXACTLY the RC charging curve!
```

```
    Capacitor symbol:

    ──┤├──   Non-polarized (ceramic, film)

    ──┤(──   Polarized (electrolytic, tantalum)
       +     curved plate = negative terminal
```

**Key equations:**
```
    Q = C × V        (charge stored = capacitance × voltage)
    I = C × dV/dt    (current = capacitance × rate of voltage change)
```

That second equation is the important one. It says: **current only flows through a capacitor when its voltage is CHANGING.** A steady (DC) voltage → no current flows. A changing (AC) voltage → current flows proportional to how fast it changes.

---

## 2.2 Why Capacitors Block DC but Pass AC

From `I = C × dV/dt`:
- **DC (constant voltage):** dV/dt = 0 → I = 0. No current flows. The capacitor looks like an open circuit.
- **AC (changing voltage):** dV/dt is nonzero → current flows. Higher frequency = faster change = more current.

**Capacitive reactance (impedance):**
```
    Xc = 1 / (2π × f × C)

    Where:
    Xc = impedance in Ohms (how much the cap "resists" current)
    f  = frequency in Hz
    C  = capacitance in Farads
```

**Real numbers:**
| Capacitor | @ 100Hz | @ 10kHz | @ 1MHz |
|-----------|---------|---------|--------|
| 100nF | 15.9kΩ | 159Ω | 1.6Ω |
| 10µF | 159Ω | 1.6Ω | 0.016Ω |

At DC (f=0), Xc = ∞ (open circuit). At high frequency, Xc → 0 (short circuit). This is why capacitors are AC "pass-through" and DC "blockers."

---

## 2.3 Decoupling Capacitors — The Most Important Capacitor in Your Circuit

**Why should I care?** If your SPI bridge randomly corrupts data or the MCU resets under heavy load, the #1 suspect is inadequate decoupling.

### The Problem

When a digital IC switches its outputs, it draws a burst of current from VCC. On an STM32H7 running at 480MHz, these bursts happen billions of times per second, each lasting nanoseconds.

The power supply is far away (centimeters of PCB trace). Traces have inductance (~1nH per mm). Inductance resists changes in current (V = L × dI/dt). So when the IC demands current NOW, the trace inductance prevents it from arriving instantly. Result: VCC sags for a few nanoseconds.

```
    Without decoupling:

    VCC ──────────────────────────── IC VCC pin
    (10cm of trace = 100nH)        │
                                    └── heavy switching
    VCC at the IC:
    3.3V ┤‾‾‾‾‾‾╲    ╱‾‾‾‾╲    ╱‾‾
         │        ╲  ╱      ╲  ╱    ← voltage dips to 2.8V
    2.8V ┤         ╲╱        ╲╱       during switching events
         └──────────────────────── time
         This can cause the IC to malfunction or reset!
```

### The Solution

Place a 100nF ceramic capacitor **as close as physically possible** to the IC's VCC pin (within 3mm). It acts as a local energy reservoir.

```
    With decoupling:

    VCC ──────────────┬────── IC VCC pin
    (10cm trace)      │
                     ┌┴┐
                     │C│ 100nF ceramic
                     └┬┘      ← within 3mm of VCC pin!
                      │
    GND ──────────────┴────── IC GND pin

    Now when the IC demands current:
    1. The 100nF cap supplies it INSTANTLY (it's right there)
    2. The power supply slowly recharges the cap through the trace
    3. VCC at the IC stays stable
```

### Why 100nF? Why not 1µF or 10µF?

It's about **frequency response**. Capacitors aren't ideal — they have parasitic inductance (ESL) from their leads and internal structure. This creates a self-resonant frequency:

```
    f_resonant = 1 / (2π × √(L × C))
```

| Cap Value | Typical ESL | Resonant Freq | Good For |
|-----------|-------------|---------------|----------|
| 100nF ceramic (0402) | 0.3nH | ~29MHz | High-freq switching noise |
| 1µF ceramic (0603) | 0.5nH | ~7MHz | Mid-freq noise |
| 10µF ceramic (0805) | 1nH | ~1.6MHz | Power supply ripple |
| 100µF electrolytic | 5nH | ~70kHz | Bulk energy storage |

**Best practice:** Use BOTH a 100nF close to the IC (for high-freq) AND a 10µF nearby (for slower transients). This is called "bulk + bypass" or "multi-value decoupling."

```
    The "two cap" recipe per IC:

    VCC ────┬──────┬────── IC VCC
            │      │
           ┌┴┐   ┌┴┐
           │C1│   │C2│
           │10µ│  │100n│   ← C2 as close as possible
           └┬┘   └┬┘
            │      │
    GND ────┴──────┴────── IC GND
```

---

## 2.4 RC Time Constant

When you charge a capacitor through a resistor, the voltage rises exponentially:

```
    V(t) = Vmax × (1 - e^(-t/τ))

    Where τ = R × C   (the "time constant")
```

| Time | Voltage | % of Vmax |
|------|---------|-----------|
| 1τ | 63.2% | Two-thirds charged |
| 2τ | 86.5% | |
| 3τ | 95.0% | |
| 4τ | 98.2% | |
| 5τ | 99.3% | "Fully" charged |

```
    RC Charging Curve:

    Vmax ┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
         │                            ╱‾‾‾‾‾‾‾‾‾‾‾‾
         │                      ╱‾‾‾╱
    63%  ┤─ ─ ─ ─ ─ ─ ─ ╱‾‾╱
         │            ╱╱
         │         ╱╱
         │      ╱╱
         │   ╱╱
         │ ╱╱
    0V   ┤╱
         └────┬─────┬─────┬─────┬─────┬── time
              1τ    2τ    3τ    4τ    5τ
```

### Worked Example

**R = 10kΩ, C = 100nF:**
```
    τ = R × C = 10,000 × 0.0000001 = 0.001 seconds = 1ms
```

The capacitor reaches 63% in 1ms and >99% in 5ms.

**Why should I care?** This is exactly what determines I2C bus rise time! The pull-up resistor (R) charges the bus capacitance (C). If τ is too large, the signal can't rise fast enough for the clock frequency. See chapter 06 for the full calculation.

---

## 2.5 Capacitor Types — When to Use Which

| Type | Capacitance Range | Voltage | Tolerance | Polarized? | Best For |
|------|-------------------|---------|-----------|------------|----------|
| C0G/NP0 ceramic | 1pF – 10nF | up to 50V | ±1% | No | Precision filters, timing circuits, RF |
| X7R/X5R ceramic | 100pF – 100µF | up to 50V | ±10-20% | No | Decoupling, bypass, bulk (small) |
| Electrolytic | 1µF – 10,000µF | up to 450V | ±20% | **YES** | Bulk energy storage, power supply filtering |
| Tantalum | 100nF – 1000µF | up to 50V | ±10% | **YES** | Compact bulk, stable ESR |
| Film (polyester) | 1nF – 10µF | up to 630V | ±5% | No | Audio coupling, high-voltage, long life |

**Critical polarized cap rule:** Electrolytic and tantalum caps have a positive and negative terminal. **Reversing polarity can cause them to explode.** Literally. Tantalums fail short-circuit (fire!).

```
    Electrolytic cap: stripe marks the NEGATIVE terminal

    ┌─────────────────────┐
    │   ─  ─  ─  ─  ─    │ ← stripe on the can
    │                     │
    └──┤              ├───┘
       │(neg)    (pos)│
```

**Ceramic cap gotcha (X7R/X5R):** The capacitance drops significantly with applied DC voltage. A "10µF" X5R cap rated at 6.3V may only be 5µF when you apply 3.3V across it, and 2µF at 5V. Always check the datasheet derating curve. The "no instantaneous voltage change" rule, `I = C × dV/dt`, means the capacitor acts like a short circuit for sudden voltage spikes (dV/dt is enormous → huge current flows). This is exactly what makes decoupling work — the cap absorbs the voltage spike.

---

## 2.6 The "No Instantaneous Voltage Change" Rule

From `I = C × dV/dt`: if voltage changed infinitely fast (dt → 0), current would be infinite. Since infinite current can't exist, the voltage CANNOT change instantaneously.

**Consequences:**
1. Capacitors smooth out voltage spikes (decoupling)
2. You can't teleport a capacitor from 0V to 5V — it charges through whatever resistance is in the path
3. Short-circuiting a charged capacitor → enormous current pulse (the cap tries to maintain its voltage, current shoots up to equalize). This can weld contacts and destroy components.

**Practical rule for your SPI bridge:** Never hot-plug a board with large capacitors into a powered bus. The rush of current charging the caps can damage the source or the connector.

---

# PART 3 — INDUCTORS

---

## 3.1 What an Inductor Physically IS

An inductor is a coil of wire. When current flows through it, it creates a magnetic field. Energy is stored in this magnetic field.

**The flywheel analogy (detailed mapping):**

```
    FLYWHEEL MODEL                      ELECTRICAL MODEL
    ──────────────                      ────────────────
    Flywheel (spinning mass)      ←→    Inductor (coil)
    Rotation speed                ←→    Current (Amps)
    Torque (force to speed up)    ←→    Voltage (Volts)
    Inertia (mass × radius²)     ←→    Inductance (Henrys)

    Key behavior:
    - Heavy flywheel resists speeding up            → inductor resists current increase
    - Heavy flywheel resists slowing down            → inductor resists current decrease
    - You CAN'T instantly stop a spinning flywheel  → you CAN'T instantly stop inductor current
      (it stores kinetic energy)                      (it stores magnetic energy)

    Where the analogy breaks down:
    - A flywheel eventually stops (friction)
    - An ideal inductor would maintain current forever (superconducting coil really does)
    - Real inductors have wire resistance (DCR) → current eventually decays
```

**Key equation:**
```
    V = L × dI/dt

    Where:
    V = voltage across the inductor (Volts)
    L = inductance (Henrys)
    dI/dt = rate of change of current (Amps/second)
```

Compare this with the capacitor equation `I = C × dV/dt`. They're duals:
- Capacitor: resists **voltage** change, stores energy in **electric** field
- Inductor: resists **current** change, stores energy in **magnetic** field

---

## 3.2 DC vs AC Behavior — Opposite of Capacitors

**At DC (steady current):** dI/dt = 0 → V = 0. The inductor looks like a short circuit (just a piece of wire). Current flows freely.

**At AC (changing current):** dI/dt is nonzero → voltage develops across the inductor. Higher frequency = faster change = more opposition.

**Inductive reactance:**
```
    XL = 2π × f × L

    Opposite of capacitive reactance!
```

| Component | DC Behavior | AC Behavior |
|-----------|-------------|------------|
| Capacitor | Open circuit (blocks DC) | Passes AC (lower impedance at higher freq) |
| Inductor | Short circuit (passes DC) | Blocks AC (higher impedance at higher freq) |

This complementary behavior is why LC combinations make such good filters.

---

## 3.3 Back-EMF — Why Switching Off an Inductive Load is Dangerous

**Why should I care?** If your robot drives relays, solenoids, or motors from MCU GPIOs, you MUST understand back-EMF or you WILL destroy transistors.

From `V = L × dI/dt`: if you try to change current VERY FAST (like turning off a switch), dI/dt becomes enormous, so V becomes enormous — potentially hundreds of volts from a 12V relay coil!

```
    Scenario: Switch driving a relay coil (L = 100mH, I = 100mA)

    Switch CLOSED:                  Switch OPENS:
    12V ──┬── Relay ──┬── GND      12V ──┬── Relay ──┬── GND
          │   coil    │                  │   coil    │
          ├── SWITCH ─┘                  X   OPEN    │
          │                              │           │
                                    The inductor DEMANDS that its 100mA
                                    keeps flowing. It generates whatever
                                    voltage is necessary to maintain current
                                    flow through the air gap.

                                    V_spike = L × dI/dt
                                    If dI/dt = 100mA in 1µs:
                                    V = 0.1 × 0.1/0.000001 = 10,000V!!

    (In practice, the arc or component breakdown limits it to ~50-200V,
     but that's still enough to destroy a MOSFET rated for 30V.)
```

### The Fix: Flyback Diode

Place a diode across the coil, reverse-biased during normal operation. When the switch opens, the coil's voltage reverses and the diode conducts, routing the current safely through itself until the magnetic field dissipates.

```
    12V ──┬── Relay ──┬── GND
          │   coil    │
          │    ╱|     │    ← flyback diode (reverse-biased normally,
          │   ╱ |     │       conducts when coil generates back-emf)
          │  ╱──┘     │
          │           │
          └── SWITCH ─┘
```

Now when the switch opens, current circulates through the diode + coil, safely decaying as the energy dissipates in the coil's resistance. The voltage spike is clamped to ~0.7V (the diode forward drop).

---

## 3.4 Inductors in DC-DC Converters (Conceptual)

**Why should I care?** The Jetson Orin and STM32 both use DC-DC converters internally. Understanding why they need inductors helps you understand switching noise — a major source of SPI interference.

A buck converter (step-down) uses a switch + inductor + capacitor to efficiently convert 12V to 3.3V:

```
    12V ──┤SWITCH├──┬── 3.3V out
                    │
                   ┌┴┐
                   │L│  Inductor
                   └┬┘
                    │
                   ┌┴┐
                   │C│  Output capacitor
                   └┬┘
                    │
    GND ────────────┴──
```

The switch rapidly toggles on/off (100kHz–2MHz). The inductor smooths the pulsating current into a steady DC. The capacitor smooths the remaining ripple.

**The noise consequence:** The switch creates high-frequency electromagnetic noise. This is why:
1. DC-DC converters have recommended layouts with specific trace widths and ground planes
2. Sensitive analog circuits (ADCs, sensors) should be kept away from DC-DC inductor loops
3. Ferrite beads (special lossy inductors) are placed on power lines going to sensitive ICs

---

## 3.5 Ferrite Beads — The Inductor You Use Every Day

A ferrite bead is a special inductor designed to be lossy at high frequencies — it converts noise to heat instead of storing and releasing energy.

```
    Schematic symbol:

    ──┤FB├──    or    ──┤LFB├──
```

Used on power rails to block high-frequency switching noise while passing DC freely. In your project, you'll likely see them on the STM32 VDDA (analog power) pin, filtering noise from the main 3.3V rail.

---

# PART 4 — RLC CIRCUITS AND FILTERS

---

## 4.1 The Big Picture: Frequency-Selective Circuits

By combining R, L, and C, you can build circuits that pass some frequencies and block others. These are **filters**, and they're everywhere in electronics.

---

## 4.2 Low-Pass Filter (RC)

Passes low frequencies, blocks high frequencies. The simplest: one resistor + one capacitor.

```
    Input ──┤ R ├──┬── Output
                   │
                  ┌┴┐
                  │C│
                  └┬┘
                   │
    GND ───────────┘

    Cutoff frequency: f_c = 1 / (2π × R × C)
```

**At the cutoff frequency, the output is 70.7% of the input (-3dB).**

### Worked Example: Anti-alias filter for ADC

Your ADC samples at 1kHz. Nyquist says you must filter out everything above 500Hz.

Design for f_c = 500Hz:
```
    Choose C = 100nF (common value)
    R = 1 / (2π × 500 × 100×10⁻⁹) = 1 / (0.000314) = 3.18kΩ → use 3.3kΩ
```

**Check:** f_c = 1 / (2π × 3300 × 100×10⁻⁹) = 482Hz ✓ (close enough)

This is a first-order filter: -20dB/decade rolloff (factor of 10 attenuation per decade of frequency above f_c). For steeper rolloff, cascade multiple stages or use op-amp active filters.

---

## 4.3 High-Pass Filter (RC)

Passes high frequencies, blocks low frequencies. Swap R and C:

```
    Input ──┤ C ├──┬── Output
                   │
                  ┌┴┐
                  │R│
                  └┬┘
                   │
    GND ───────────┘

    Same cutoff formula: f_c = 1 / (2π × R × C)
```

Used for: AC coupling (removing DC offset from audio signals), differentiating signals.

---

## 4.4 Series RLC and Resonance

When L and C are in series, there's a special frequency where their impedances cancel out:

```
    f_resonant = 1 / (2π × √(L × C))
```

At resonance:
- XL = XC (inductive and capacitive reactances are equal)
- They cancel, leaving only R
- Current is maximum (only limited by R)
- This is how radio tuners work — the LC circuit "selects" one frequency

**Quality factor Q:**
```
    Q = 1/R × √(L/C)     (for series RLC)
```

- High Q: sharp, narrow resonance peak (good for selecting one radio station)
- Low Q: broad, gentle peak (good for wideband filters)

---

## 4.5 Pi Filter (CLC) for Power Supply

```
    VCC_noisy ──┤C1├──┤L├──┤C2├── VCC_clean
                 │         │
                GND       GND
```

C1 filters high-frequency noise, L blocks remaining mid-frequency noise, C2 smooths the final output. Used on analog power rails (VDDA, VREF) to keep switching noise from DC-DC converters away from sensitive ADCs.

---

# PART 5 — COMPARISON TABLE

---

## The Three Passives at a Glance

| Property | Resistor | Capacitor | Inductor |
|----------|----------|-----------|----------|
| **Symbol** | ─┤R├─ | ─┤C├─ | ─┤L├─ |
| **Unit** | Ohm (Ω) | Farad (F) | Henry (H) |
| **DC behavior** | Conducts (with V drop) | Open circuit (blocks) | Short circuit (passes) |
| **AC behavior** | Same impedance at all freq | Lower impedance at higher freq | Higher impedance at higher freq |
| **Energy storage** | None (dissipates as heat) | Electric field | Magnetic field |
| **Key equation** | V = IR | I = C × dV/dt | V = L × dI/dt |
| **Impedance** | R (constant) | 1/(2πfC) ↓ with freq | 2πfL ↑ with freq |
| **Can't change instantly** | N/A | Voltage | Current |
| **Common values** | 10Ω – 10MΩ | 1pF – 10,000µF | 1nH – 10mH |
| **Used for** | Current limiting, dividers, pull-ups | Decoupling, filtering, timing, energy storage | Filtering, energy storage, DC-DC, EMI suppression |

---

# GOTCHA TABLE

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|-------------|-----------------|-----|
| LED doesn't light up | Resistor too high, wrong polarity | Measure voltage across LED. Check anode/cathode orientation | Reduce resistance. Flip LED. |
| MCU resets randomly under load | Missing/bad decoupling caps | Oscilloscope on VCC pin — look for dips below min voltage | Add 100nF ceramic + 10µF bulk near every VCC pin |
| Voltage divider output wrong | Load drawing too much current | Measure Vout with and without load connected | Use lower R values, or add a buffer op-amp |
| I2C bus unreliable | Wrong pull-up value | Oscilloscope: look for rounded edges on SCL/SDA, slow rise time | Calculate proper pull-up for bus capacitance (see ch 06) |
| Resistor is hot | Power dissipation exceeds rating | P = V²/R — calculate and compare to resistor wattage rating | Use higher R value or higher-power resistor |
| Electrolytic cap explodes | Reverse polarity | Check + and - markings vs schematic | Always match polarity! Stripe = negative terminal |
| 3.3V rail has 200mV of 500kHz noise | Switching regulator noise | Scope with 20MHz BW limit off. Look at frequency | Add ferrite bead + 100nF before sensitive IC |
| ADC readings noisy | No anti-alias filter, poor decoupling | Scope the ADC input pin. Check VDDA decoupling | Add RC low-pass filter. Add 100nF+10µF on VDDA |
| Relay/motor driver MOSFET dies | Missing flyback diode, back-EMF | Check for diode across inductive load | Add flyback diode (1N4148 for small, 1N4007 for power) |
| Filter doesn't seem to work | Wrong component values | Calculate f_c = 1/(2πRC), compare to target frequency | Recalculate and verify with oscilloscope |

---

# QUICK REFERENCE CARD

```
┌─────────────────────────────── PASSIVE COMPONENTS CHEAT SHEET ──────────────────────────────┐
│                                                                                              │
│  OHM'S LAW:     V = I × R           I = V / R           R = V / I                           │
│  POWER:         P = V × I           P = I²R             P = V²/R                            │
│                                                                                              │
│  VOLTAGE DIVIDER:   Vout = Vin × R2/(R1+R2)                                                │
│  PARALLEL R:        R = (R1 × R2)/(R1 + R2)                                                │
│                                                                                              │
│  RC TIME CONSTANT:  τ = R × C       5τ = "fully charged"     63% at 1τ                      │
│  LOW-PASS CUTOFF:   f_c = 1/(2πRC)                                                         │
│  HIGH-PASS CUTOFF:  f_c = 1/(2πRC)  (same formula, swapped topology)                       │
│                                                                                              │
│  CAPACITOR IMPEDANCE:  Xc = 1/(2πfC)    (decreases with frequency)                         │
│  INDUCTOR IMPEDANCE:   XL = 2πfL        (increases with frequency)                         │
│  RESONANT FREQUENCY:   f = 1/(2π√(LC))                                                     │
│                                                                                              │
│  DECOUPLING RECIPE:  100nF ceramic (≤3mm from VCC pin) + 10µF bulk (nearby)                │
│  LED RESISTOR:        R = (Vcc - Vf) / I_desired                                           │
│  PULL-UP TYPICAL:     4.7kΩ (I2C), 10kΩ (SPI CS, buttons)                                 │
│                                                                                              │
│  STANDARD E12 VALUES: 10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82 × 10^n              │
│                                                                                              │
│  SMD SIZES:  0201 (tiny) → 0402 → 0603 → 0805 → 1206 (large, hand-solderable)             │
│              Size format: length × width in 0.01 inches (0603 = 0.06" × 0.03")             │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```
