# 02 — Semiconductors: Diodes, BJTs, MOSFETs
### The active components that switch, amplify, and protect
**Prerequisites:** Chapter 01 (Ohm's law, voltage dividers, RC time constants)
**Unlocks:** Level shifting (for SPI/I2C between 3.3V and 5V), open-drain buses (I2C), motor/relay drivers, ESD protection, power switching

---

## Why Should I Care? (STM32/Jetson Project Context)

- **Diodes:** TVS diodes protect your SPI lines from ESD. Schottky diodes are in every power supply on both boards. Flyback diodes protect FET drivers on your robot.
- **MOSFETs:** The BSS138 level-shifter circuit converts 3.3V I2C to 5V I2C for legacy sensors. MOSFETs switch power to motor drivers and sensor modules. The I2C bus itself relies on MOSFET-based open-drain outputs.
- **BJTs:** Still used in some relay driver circuits and as current sources in precision analog circuits.

If you can't read a MOSFET datasheet or calculate a base resistor, you can't design or debug the power management around your sensing systems.

---

# PART 1 — DIODES

---

## 1.1 The PN Junction — What a Diode Physically IS

Silicon is a *semiconductor* — it conducts slightly, but not well. By adding impurities (doping), we change its behavior:

- **N-type:** Add phosphorus (5 valence electrons) → extra free electrons → conducts via negative charges
- **P-type:** Add boron (3 valence electrons) → missing electrons ("holes") → conducts via positive charge carriers

When you join P-type and N-type silicon together, something magical happens at the boundary:

```
    P-type          │         N-type
    ○ ○ ○ ○ ○      │      ● ● ● ● ●
    ○ ○ ○ ○ ○      │      ● ● ● ● ●     ○ = hole (positive carrier)
    ○ ○ ○ ○ ○      │      ● ● ● ● ●     ● = electron (negative carrier)
                    │
              DEPLETION REGION
              (no mobile carriers)
```

Electrons from the N-side diffuse into the P-side and fill holes. This creates a thin region with no mobile carriers (the depletion region), and builds up a voltage barrier (~0.7V for silicon).

**The one-way valve analogy:**

Think of the depletion region as a hill. Current (water) can easily flow downhill (forward bias), but needs to be pumped uphill (reverse bias → no flow, unless you push REALLY hard → breakdown).

- **Forward bias (+ to P, - to N):** Applied voltage overcomes the 0.7V barrier → current flows freely
- **Reverse bias (+ to N, - to P):** Applied voltage reinforces the barrier → no current flows (until breakdown)

```
    Diode symbol:

    Anode ──▶|── Cathode
    (P)              (N)

    Current flows from Anode to Cathode (in the direction of the arrow)
    The line (|) is the cathode — the "wall" that blocks reverse current
```

**Where the analogy breaks down:** A real diode isn't a perfect one-way valve:
- Forward: there's a ~0.7V voltage drop (you lose 0.7V across the diode, always)
- Reverse: a tiny leakage current flows (nanoamps — usually ignorable)
- If reverse voltage exceeds the breakdown voltage, current floods through (destructive for regular diodes, intentional for Zener diodes)

---

## 1.2 Forward Voltage Drop by Diode Type

| Diode Type | Forward Voltage (Vf) | Speed | Typical Use |
|-----------|---------------------|-------|-------------|
| Silicon rectifier (1N4007) | 0.7V | Slow (µs) | Power rectification, flyback |
| Schottky (BAT54, 1N5819) | 0.2–0.4V | Fast (ns) | Power supply, OR-ing, high-freq rectification |
| Red LED | 1.8–2.2V | N/A | Indication |
| Green LED | 2.0–3.5V | N/A | Indication |
| Blue/White LED | 2.8–3.5V | N/A | Indication, illumination |
| Zener (various) | Vf forward, Vz reverse | Medium | Voltage regulation/reference |
| TVS (SMBJ3.3A) | 0.7V forward | Very fast (ps) | ESD/transient protection |

**Key insight:** The forward voltage drop is NOT proportional to current — it's nearly constant. A silicon diode drops ~0.7V whether 1mA or 1A flows through it (it varies slightly, but treat it as constant for circuit design).

---

## 1.3 Zener Diodes — Voltage Regulation

A Zener diode is designed to safely conduct in reverse when the reverse voltage reaches a specific value (the Zener voltage, Vz).

```
    Use case: 12V input → stable 5.1V output for a reference

    12V ──┤ R ├──┬── 5.1V output
                 │
                ─┤─    ← Zener diode (5.1V)
                 │       Conducts in reverse at Vz
                 │
    GND ─────────┘

    R limits the current through the Zener.
    R = (Vin - Vz) / Iz = (12 - 5.1) / 0.020 = 345Ω → use 330Ω
```

Not efficient for high current loads (the excess power is wasted as heat in R and the Zener), but excellent for voltage references.

---

## 1.4 Flyback / Freewheeling Diode

Covered in chapter 01 (inductors section), but reinforced here because it's critical:

```
    12V ──┬─── Motor/Relay ───┬── MOSFET to GND
          │                    │
          │    ╱|  flyback     │
          └── ╱ | diode ──────┘
              ╱──┘
```

When the MOSFET turns off, the motor/relay coil generates back-EMF (see chapter 01, section 3.3). The flyback diode gives the decaying current a path to flow, clamping the voltage spike to ~0.7V above the supply rail.

**Use Schottky for flyback if speed matters** — Schottky diodes have no reverse recovery time (they turn on instantly), while silicon diodes have ~1µs of "confused" time where they briefly conduct backwards.

---

## 1.5 TVS Diodes — ESD Protection

**Why should I care?** ESD (Electrostatic Discharge) from your finger can deliver 15kV in nanoseconds. Your signal lines (SPI, I2C, UART) are connected to IC pins rated for perhaps 5V max. A TVS diode clamps the spike before it reaches the IC.

```
    Signal ──┬── to IC pin
             │
            ─┤─  ← TVS diode (e.g., SMBJ3.3A for 3.3V rails)
             │     Clamps voltage to ~5.5V (safe for 3.3V IC)
             │
    GND ─────┘
```

**How it works:** Normal operation: TVS is reverse-biased, essentially invisible. When a transient spike exceeds the breakdown voltage (e.g., 5.5V), the TVS instantly conducts to ground, clamping the voltage. After the spike, it returns to high impedance.

Used on: USB data lines, CAN bus, Ethernet, SPI/I2C lines in harsh environments (robots!).

---

## 1.6 LED as a Diode

An LED (Light-Emitting Diode) IS a diode — it has a PN junction, a forward voltage drop, and conducts only in one direction. The energy released when electrons cross the junction produces photons (light) instead of just heat.

```
    Forward voltage sets the COLOR (it's the bandgap energy):
    - Red:    ~2.0V
    - Yellow: ~2.1V
    - Green:  ~2.2V (or 3.0V for blue-green)
    - Blue:   ~3.0V
    - White:  ~3.2V (blue LED + phosphor coating)

    Current sets the BRIGHTNESS:
    - 1mA:  dim indicator
    - 10mA: standard brightness
    - 20mA: maximum for most indicator LEDs
    - 350mA+: power LEDs (with heatsink!)
```

**ALWAYS use a current-limiting resistor.** Without one, the LED's forward voltage is essentially fixed, so the current is limited only by the power supply's internal resistance → too much current → LED dies.

---

# PART 2 — BJT (BIPOLAR JUNCTION TRANSISTOR)

---

## 2.1 What a BJT IS — Current-Controlled Current Source

A BJT has three terminals: Base (B), Collector (C), Emitter (E).

```
    NPN BJT:                    PNP BJT:

         C                          C
         │                          │
     B──┤│ (arrow out)          B──┤│ (arrow in)
         │↓                         │↑
         E                          E

    NPN: Current flows C→E when base current flows B→E
    PNP: Current flows E→C when base current flows E→B
```

**The water faucet analogy:**

```
    WATER FAUCET                    NPN BJT
    ────────────                    ────────
    Water supply (high pressure) → VCC (through load) to Collector
    Faucet handle (small twist)  → Base current (small, from GPIO)
    Water output (large flow)    → Emitter to GND (large current)

    A tiny twist of the handle (small base current)
    controls a large flow of water (large collector current).
    The ratio is β (current gain, typically 100-300):

    I_collector = β × I_base
```

**Where the analogy breaks down:** A water faucet is proportional — more twist = more water. A BJT used as a switch is either fully OFF (base current = 0, cutoff) or fully ON (base current large enough to saturate it, Vce ≈ 0.2V). We rarely use it in the proportional middle ground (that's the amplifier region, used in analog circuits).

---

## 2.2 BJT Regions of Operation

| Region | Base Current | Collector-Emitter | Behavior | Use |
|--------|-------------|-------------------|----------|-----|
| Cutoff | 0 | Open (high V) | Switch OFF | Digital |
| Active | Moderate | Proportional to Ib | Amplifier | Analog |
| Saturation | Large | ~0.2V (shorted) | Switch ON | Digital |

For digital switching (driving relays, LEDs, motors from GPIOs), we only care about **cutoff** and **saturation**.

---

## 2.3 Worked Example: BJT Relay Driver from 3.3V GPIO

**Problem:** Drive a 12V relay coil (50mA, 240Ω) from a 3.3V GPIO that can only source 4mA.

```
    12V ──┬── Relay Coil ──┐
          │                 │
         ─┤─ flyback diode  │    ← ALWAYS add flyback diode!
          │                 │
          └────────────────┤ C
                      B ──┤│    ← NPN BJT (2N2222 or BC547)
               GPIO ──┤R├─┤│
                            │ E
                            │
    GND ────────────────────┘
```

**Design steps:**

1. **Collector current needed:** Ic = 50mA (the relay needs this)
2. **BJT β (current gain):** datasheet says β_min = 100 for 2N2222
3. **Base current needed:** Ib = Ic / β = 50mA / 100 = 0.5mA minimum
4. **Overdrive for reliable saturation:** Use 2-3× → Ib = 1.5mA
5. **Base resistor:** Rb = (Vgpio - Vbe) / Ib = (3.3V - 0.7V) / 1.5mA = 1.73kΩ → **use 1.8kΩ**
6. **Check GPIO current:** 1.5mA < 4mA GPIO max ✓
7. **Check power:** Relay sees 12V - Vce_sat = 12V - 0.2V = 11.8V → 49mA → works fine ✓

**Why not use the GPIO directly?**
- GPIO can only source 4mA; relay needs 50mA
- GPIO is 3.3V; relay is 12V
- The BJT provides both current amplification AND voltage level translation

---

## 2.4 Darlington Pair — When One BJT Isn't Enough

Two BJTs cascaded: the emitter of the first drives the base of the second. Total gain = β₁ × β₂ (e.g., 100 × 100 = 10,000). But Vce_sat increases to ~1.2V (two Vbe drops). The TIP120 is a common Darlington in one package.

```
         C
         │
    B ──┤│     First BJT
         │↓
         ├──┤│  Second BJT
         │   │↓
              E
```

Used when: the base current available is VERY small (e.g., driving a 1A motor from a logic gate that can only source 100µA).

---

# PART 3 — MOSFET (METAL-OXIDE-SEMICONDUCTOR FIELD-EFFECT TRANSISTOR)

---

## 3.1 How a MOSFET Works — Voltage-Controlled Switch

Unlike a BJT (current-controlled), a MOSFET is controlled by **voltage** at the gate. The gate is insulated from the channel by a thin oxide layer — **no DC current flows into the gate** (negligible, femtoamps).

```
    N-Channel MOSFET:               P-Channel MOSFET:

         D (Drain)                       D (Drain)
         │                               │
    G ──┤│  (no arrow = N-ch)       G ──┤│→ (arrow in = P-ch)
         │                               │
         S (Source)                       S (Source)
```

**How it switches (N-channel enhancement mode):**

```
    Gate Voltage (Vgs) vs Channel:

    Vgs < Vth:    Channel is OFF (no current flows D→S)
                  MOS = open circuit between D and S

    Vgs > Vth:    Channel is ON (current flows D→S)
                  MOS = low resistance between D and S (Rds_on)

    Vth = threshold voltage, typically 1-4V
```

**ELI15:** Imagine a drawbridge:
- Gate voltage = the crank that raises/lowers the bridge
- Drain-Source = the road across the moat
- When Vgs < Vth: bridge is UP, no traffic crosses (OFF)
- When Vgs > Vth: bridge is DOWN, traffic flows freely (ON)
- No one crosses the drawbridge mechanism itself (no gate current)

This is fundamentally different from a BJT, where base current continuously flows. A MOSFET gate is capacitive — you just need to charge/discharge it to switch states (important for gate drivers at high frequencies).

---

## 3.2 Key MOSFET Parameters

### Vgs(th) — Threshold Voltage

The minimum gate-to-source voltage to start turning the MOSFET on.

**Critical for 3.3V MCUs:** If Vgs(th) is 4V and your GPIO outputs 3.3V, the MOSFET **never fully turns on**. It stays in the partially-on region, getting very hot.

| MOSFET Type | Vgs(th) | Works with 3.3V GPIO? |
|-------------|---------|----------------------|
| Standard (IRF540N) | 2-4V | **NO** — partially on, high Rds_on, overheating |
| Logic-level (IRLZ44N) | 1-2V | **YES** — fully on with 3.3V gate drive |
| Sub-logic (Si2302) | 0.4-1V | **YES** — even works with 1.8V logic |

**Rule:** For direct GPIO drive, always use "logic-level" MOSFETs (check that Rds_on is specified at Vgs = 2.5V or 3.3V in the datasheet, not just at Vgs = 10V).

### Rds(on) — On-Resistance

When fully on, the MOSFET behaves like a small resistor between Drain and Source. This is Rds(on).

```
    Power dissipated = I² × Rds_on

    Example: IRLZ44N, Rds_on = 22mΩ at Vgs=5V, switching 5A:
    P = 5² × 0.022 = 0.55W → needs small heatsink

    Example: IRF540N, Rds_on = 77mΩ at Vgs=10V, but 500mΩ at Vgs=3.3V:
    P = 5² × 0.5 = 12.5W → THIS WILL CATCH FIRE without a massive heatsink
    (and it's not even fully on! Hence the "standard vs logic-level" distinction)
```

### Gate Capacitance (Qg)

The gate acts like a small capacitor (a few nanofarads). To switch the MOSFET quickly, you must charge/discharge this capacitor quickly. This requires a gate driver for high-frequency switching (>100kHz).

For your 100Hz SPI bridge, switching a MOSFET at 100Hz from a GPIO is trivially easy — not a concern. But for the DC-DC converters on your boards, this is why they need dedicated driver ICs.

---

## 3.3 N-Channel vs P-Channel — Where They Go

```
    LOW-SIDE SWITCH (N-channel):        HIGH-SIDE SWITCH (P-channel):

    VCC ── Load ──┐                     VCC ──┤ P-ch ├──┬── Load ── GND
                   │                      │    S    D    │
                  ┌┤ N-ch                G ──┤│←         │
             G ──┤│                           │          │
                  │                           │          │
    GND ──────────┘                     GPIO (inverted logic)

    GPIO HIGH → N-ch ON → Load powered   GPIO LOW → P-ch ON → Load powered
    GPIO LOW  → N-ch OFF → Load off      GPIO HIGH → P-ch OFF → Load off
```

**N-channel (low-side):** Easy to drive — Gate referenced to GND (same as MCU). Put the load between VCC and Drain. Current flows from Drain to Source when ON.

**P-channel (high-side):** Harder to drive — Gate must be pulled LOW relative to Source (which is at VCC). Used when you want to switch the positive supply rail to a load. Logic is inverted.

**Rule of thumb:**
- N-channel MOSFETs are cheaper, lower Rds_on, more variety → prefer whenever possible
- P-channel only when you MUST switch the high side (or use an N-channel + gate driver)

---

## 3.4 MOSFET as a Switch — Practical Examples

### Example 1: Switching a 12V LED Strip from a 3.3V GPIO

```
    12V ── LED strip ── (+) ──┐
                               │ D
                          G ──┤│  IRLZ44N (logic-level N-ch)
                               │ S
    GND ───────────────────────┘

    GPIO (3.3V) ──┤ 100Ω ├── G    ← gate resistor (optional, limits inrush)
    GND ─────────────────── S
```

When GPIO = HIGH (3.3V): Vgs = 3.3V > Vth (~1.5V) → MOSFET ON → LED strip lights up
When GPIO = LOW (0V): Vgs = 0V < Vth → MOSFET OFF → LED strip off

**Why 100Ω gate resistor?** The gate is a capacitor. Without a resistor, the GPIO charges it with a current spike limited only by the GPIO output resistance (~25Ω). The 100Ω limits the current spike to 3.3V/125Ω ≈ 26mA, which is within GPIO specs. For 100Hz switching, this is optional but good practice.

### Example 2: High-Side Power Switch with P-Channel

```
    3.3V_SENSOR_RAIL ── S ──┤│← ── D ── Sensor VCC
                        │  P-ch  │
                   GPIO ┤R├ G     │
                        │        │
                       GND    Sensor GND ── GND

    R = 10kΩ (when GPIO is low → Vgs = -3.3V → ON → sensor powered)
    Also add 10kΩ from G to S (pull-up to keep sensor OFF during boot)
```

This is how you power-cycle a sensor from your MCU — useful for resetting a stuck I2C device.

---

## 3.5 Level Shifting with MOSFETs — The BSS138 Circuit

**Why should I care?** Your STM32 runs at 3.3V. Some sensors (legacy IMUs, certain EEPROMs) use 5V I2C. You need to safely connect them.

The BSS138 bidirectional level shifter is the standard circuit for I2C level conversion:

```
    3.3V side                            5V side

    3.3V ──┤ R1 ├──┬─── SDA_3V3        5V ──┤ R2 ├──┬─── SDA_5V
           (10kΩ)  │                         (10kΩ)  │
                   │   ┌────┐                        │
                   └───┤ G  │  BSS138 (N-ch)         │
                       │    ├────────────────────────┘
                       │ S  │  D
                       └────┘
                     Source tied to 3.3V side
```

**How it works (this is clever!):**

**Case 1: 3.3V device drives LOW**
- SDA_3V3 goes to 0V
- Vgs = 3.3V - 0V = 3.3V > Vth → MOSFET ON
- Drain (5V side) pulled LOW through MOSFET
- SDA_5V goes LOW ✓

**Case 2: 3.3V device releases (open-drain high)**
- R1 pulls SDA_3V3 to 3.3V
- Vgs = 3.3V - 3.3V = 0V → MOSFET OFF
- R2 pulls SDA_5V to 5V
- Both sides HIGH at their respective levels ✓

**Case 3: 5V device drives LOW**
- SDA_5V goes to 0V
- MOSFET body diode conducts (Source is now higher than Drain)
- SDA_3V3 drops to ~0.7V, then...
- Vgs = 3.3V - 0.7V = 2.6V > Vth → MOSFET turns ON
- SDA_3V3 pulled fully to 0V through MOSFET
- SDA_3V3 goes LOW ✓

**Case 4: 5V device releases (open-drain high)**
- R2 pulls SDA_5V to 5V
- R1 pulls SDA_3V3 to 3.3V
- Vgs = 3.3V - 3.3V = 0V → MOSFET OFF
- Both sides HIGH ✓

**This is one of the most elegant circuits in electronics.** It's bidirectional, works with open-drain buses, and needs only one MOSFET and two resistors per signal line.

**You need TWO of these** for I2C (one for SDA, one for SCL). Modules like the SparkFun BOB-12009 put it all on one tiny board.

---

## 3.6 Body Diode — The Parasitic Diode

Every MOSFET has a parasitic diode between Source and Drain (formed by the body-to-drain PN junction):

```
    N-channel:          P-channel:

      D                   D
      │                   │
     ┌┤ ←body diode      ┌┤ body diode→
    ─┤│                  ─┤│
     │                    │
      S                   S

    N-ch: body diode conducts from S to D (Source more positive)
    P-ch: body diode conducts from D to S (Drain more positive)
```

**When it matters:**
- In H-bridges (motor drivers): the body diode provides a freewheeling path during dead-time
- In reverse polarity protection: an N-ch MOSFET in the positive supply line uses the body diode to bootstrap itself ON
- The body diode is SLOW compared to a Schottky — in fast-switching applications, add an external Schottky in parallel

---

## 3.7 H-Bridge — Four MOSFETs for Bidirectional Motor Control

```
    VCC ──┬──────────────────────┬
          │                      │
         ┌┤ Q1 (P-ch)    Q3 (P-ch) ├┐
         │                            │
         ├──── MOTOR ─────────────────┤
         │                            │
         ┌┤ Q2 (N-ch)    Q4 (N-ch) ├┐
          │                      │
    GND ──┴──────────────────────┴

    Q1+Q4 ON, Q2+Q3 OFF → Current flows LEFT to RIGHT → Motor forward
    Q3+Q2 ON, Q1+Q4 OFF → Current flows RIGHT to LEFT → Motor reverse
    All OFF → Motor coasts (free-spinning)
    Q2+Q4 ON → Motor brakes (shorted through low-side MOSFETs)

    NEVER turn on Q1+Q2 or Q3+Q4 simultaneously!
    That shorts VCC to GND through two MOSFETs → "shoot-through" → magic smoke.
```

In practice, you use an H-bridge IC (L298N, DRV8871, TB6612) that handles the dead-time and shoot-through protection for you.

---

# PART 4 — BJT vs MOSFET COMPARISON

---

| Property | BJT (NPN/PNP) | MOSFET (N-ch/P-ch) |
|----------|---------------|---------------------|
| **Control signal** | Current (Ib controls Ic) | Voltage (Vgs controls Id) |
| **Gate/Base current** | Continuous (Ib flows always) | Zero DC (gate is capacitive) |
| **Input impedance** | Low (~kΩ) | Extremely high (~TΩ) |
| **ON voltage drop** | Vce_sat ≈ 0.2V (good for low V) | Rds_on × Id (can be <0.01V for large FETs) |
| **Switching speed** | Moderate (µs for power BJTs) | Fast (ns for small FETs) |
| **Thermal behavior** | Positive temp coefficient of β → thermal runaway possible | Negative temp coefficient of Rds_on → naturally self-limiting |
| **Typical use (modern)** | Analog circuits, current sources, voltage references | Digital switching, power management, motor drivers |
| **Cost** | Very cheap | Slightly more expensive |
| **Why still used?** | Better for linear/analog, simpler voltage reference circuits | Dominates digital and power switching |

**Rule of thumb:** For switching loads (motors, LEDs, relays, heaters), use a MOSFET. For analog amplification or current sources, consider a BJT.

---

# GOTCHA TABLE

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|-------------|-----------------|-----|
| MOSFET doesn't turn on from 3.3V GPIO | Vgs(th) too high (standard, not logic-level) | Check datasheet: Rds_on specified at Vgs=10V only, not 3.3V | Use logic-level MOSFET (IRLZ44N, Si2302) |
| MOSFET gets very hot | Not fully on (high Rds_on) or excessive current | Measure Vds when ON — should be <0.5V. Calculate I²×Rds_on | Ensure gate is driven above Vgs_max specified for low Rds_on |
| BJT driver doesn't switch relay | Base resistor too high → insufficient Ib | Calculate Ib = Ic/β, use 2-3× for saturation margin | Lower base resistor value |
| MOSFET driver FET destroyed | Missing flyback diode on inductive load → back-EMF exceeds Vds_max | Check for diode across load. Measure with scope during turn-off | Add flyback Schottky diode |
| Level shifter doesn't work | Wrong MOSFET type (P-ch instead of N-ch), or pull-ups missing | Verify BSS138 (N-ch, Vth~1.5V). Check R1, R2 present | Use N-ch with Vth < 2V, add pull-ups to both voltage rails |
| 5V signal on 3.3V-only input → IC damage | No level shifting or clamping | Dead IC. Measure input voltage with scope | Add voltage divider, BSS138 shifter, or TVS clamp |
| LED flickers or burns out | No current-limiting resistor, or value too low | Measure current with multimeter in series | Calculate R = (Vcc - Vf) / I_desired |
| Open-drain output always low | Pull-up resistor missing | Measure voltage — stuck at 0V or floating | Add pull-up resistor to appropriate VCC rail |
| Diode in reverse → circuit doesn't work | Anode and cathode swapped | Check marking: stripe on component = cathode | Flip diode. Cathode = stripe = the "bar" in the symbol |
| GPIO damaged after driving large load | Sink/source current exceeded (>20mA) | GPIO pin doesn't go to full 0V or 3.3V | Use BJT or MOSFET transistor as a driver |

---

# QUICK REFERENCE CARD

```
┌─────────────────────────────── SEMICONDUCTORS CHEAT SHEET ──────────────────────────────────┐
│                                                                                              │
│  DIODE:    Forward voltage: Silicon=0.7V, Schottky=0.3V, LED=2-3V (color dependent)        │
│            Cathode = stripe on package = bar in schematic symbol                             │
│            Flyback diode: ALWAYS across inductive loads (reverse-biased)                     │
│                                                                                              │
│  BJT:      Ib = Ic / β            (base current = collector current / gain)                 │
│            Vbe ≈ 0.7V              (base-emitter drop, always)                              │
│            Rb = (Vgpio - 0.7V) / (2×Ic/β)   (2× for saturation margin)                    │
│            NPN: C→E current, arrow OUT of emitter                                           │
│            PNP: E→C current, arrow IN to emitter                                            │
│                                                                                              │
│  MOSFET:   Logic-level: check Rds_on at Vgs=2.5V or 3.3V in datasheet                     │
│            Power dissipation: P = Id² × Rds_on                                              │
│            N-ch: low-side switch, gate HIGH = ON                                             │
│            P-ch: high-side switch, gate LOW (relative to source) = ON                        │
│            Gate is capacitive — no DC current into gate                                      │
│                                                                                              │
│  BSS138 LEVEL SHIFTER (per line):                                                           │
│            Vlow ──┤Rpull├──Source   Gate=Source   Drain──┤Rpull├── Vhigh                    │
│            Rpull = 4.7kΩ–10kΩ each side. Bidirectional. Open-drain only.                    │
│                                                                                              │
│  COMMON PARTS:                                                                               │
│    Diode:  1N4148 (signal), 1N4007 (1A power), BAT54 (Schottky)                            │
│    BJT:    2N2222/BC547 (NPN), 2N2907/BC557 (PNP)                                          │
│    MOSFET: BSS138 (level shift), IRLZ44N (logic N-ch power), Si2302 (small N-ch SOT-23)    │
│    TVS:    SMBJ3.3A (3.3V lines), SMBJ5.0A (5V lines)                                     │
│                                                                                              │
│  PACKAGES:  SOT-23 (3-pin tiny), SOT-223 (3-pin medium), TO-220 (3-pin, heatsinkable)     │
│             SO-8 (8-pin SMD), DPAK (power SMD), D2PAK (big power SMD)                      │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```
