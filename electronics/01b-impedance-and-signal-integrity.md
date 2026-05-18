# 01B — Impedance and Signal Integrity: When Theory Meets Wire

## A deep-dive companion to 01A — expand every concept that causes "but WHY?" moments

**Prerequisite:** You have read `01a-rlc-foundations.md` sections 1–6, at minimum. You know what R, C, L are and that impedance is frequency-dependent opposition.

**Goal:** Understand floating pins, pull-up/pull-down resistors, signal ringing, filters, and ADC source impedance deeply enough that you can debug real hardware without guessing.

---

## Why This File Exists

`01A` gives you each concept in 4–8 lines. That is enough to remember the rule. But experience shows that a few sections trigger a chain of "but why does THAT happen?" questions before the idea clicks:

- High impedance / floating inputs
- Pull-up vs pull-down: which one and why
- Ringing and why resistance helps stop it
- Filters: what they physically do
- SPI signal corruption at high speed
- ADC wrong readings from slow source impedance

Each section below starts with the simplest possible version, adds the mechanism, then ends with a practical decision guide. Skip to the section you need.

---

## SECTION 1 — High Impedance: The Feather on a Needle

### 1.1 What High Impedance Actually Means in Plain Language

When a datasheet says a GPIO input has "input impedance of 1 MΩ or more," it means the input pin is almost completely sealed shut. Very little current can flow in or out.

That sounds good. Less current means less power wasted.

But here is the hidden cost: **when almost nothing can flow, almost anything can disturb it.**

Think of two objects on a scale:

```text
High impedance (no resistor):   A feather balanced on a needle tip.
                                 A puff of air tips it instantly.

Low impedance (with resistor):  A 10 kg brick sitting on the floor.
                                 You need real force to move it.
```

The GPIO input is the feather. The electromagnetic noise in any room — from your phone, the power outlet in the wall, nearby switching circuits, even the friction of your clothes — is the puff of air.

### 1.2 Where the "Puff of Air" Comes From

Your room is full of invisible electric fields right now:

- Power outlets radiate a 50 Hz or 60 Hz electrical hum through the walls.
- Your phone and Wi-Fi router broadcast GHz radio waves constantly.
- Every switching component on your own board fires tiny electrical pulses that spray outward.
- Rubbing your socks on carpet generates kilovolts of static charge.

Under normal circumstances, a circuit with a strong signal source shrugs all of this off. The source is driving the wire with real voltage and current. A little noise on top is tiny by comparison.

But a floating GPIO input has no source. It is sealed shut. The only thing sitting on that pin is the stray charge that drifted there from the air, the nearby traces, and the random motion of electrons.

### 1.3 Why Small Charge Means Large Voltage (The Real Mechanism)

This is the part that feels like magic until you see the math.

The input pin has a tiny amount of stray capacitance — typically 2 to 10 picofarads. This is not a capacitor you placed there. It is the natural capacitance of the metal pin itself sitting near the ground plane.

The relationship between charge, capacitance, and voltage is:

```text
V = Q / C
```

At 5 picofarads:

```text
V = Q / 5e-12
```

So to get the voltage to jump by 1 volt, you only need:

```text
Q = 1V * 5e-12 = 5 picocoulombs of charge
```

Five picocoulombs is almost nothing. It is the charge you could get from a single static spark that you barely feel.

And because the gate is sealed shut, that charge has nowhere to drain. It sits there and holds the voltage at whatever it landed at. A millisecond later, a slightly different amount drifts in, and the voltage is somewhere else.

**Summary:** High impedance pin + tiny capacitance + no drain path = random voltage from microscopic stray charge.

### 1.4 The "Why Does V = Q/C Explain Floating Pins" Checklist

| Question | Answer |
|----------|--------|
| Why does a floating pin go random? | Stray charge accumulates on pin capacitance with no drain path |
| Why doesn't grounding the pin fix it? | Grounding IS low impedance — that does fix it (as a pull-down would) |
| Why does adding a resistor fix it? | The resistor gives stray charge a drain path before it builds up |
| Why 10 kΩ and not 1 Ω? | 1 Ω would sink too much current from the signal source. 10 kΩ drains noise without loading the signal. |

---

## SECTION 2 — Pull-Up and Pull-Down Resistors: Anchors for Floating Nodes

### 2.1 The Core Idea

A pull-up or pull-down resistor connects your floating input pin to a known voltage through a resistor. That resistor is the drain pipe. Stray charge cannot accumulate because it immediately bleeds away through the resistor.

```text
PULL-UP:                           PULL-DOWN:

3.3V ──[ 10kΩ ]──── GPIO pin       GPIO pin ──[ 10kΩ ]──── GND
                        │                          │
                    [Switch]                   [Switch]
                        │                          │
                       GND                        3.3V
```

**Pull-up:** Default state is HIGH. The pin is gently held at 3.3 V. Pressing the switch connects it to GND, pulling it LOW.

**Pull-down:** Default state is LOW. The pin is gently held at 0 V. Pressing the switch connects it to 3.3 V, pushing it HIGH.

### 2.2 Why the Resistor Value Matters

If you used a 1 Ω pull-up, the pin would be rock solid — but the switch would draw 3.3 A when pressed. That destroys the switch, fries the trace, and trips the power supply.

If you used a 10 MΩ pull-up, you barely improve on floating. The RC time constant with even 10 pF of stray capacitance is:

```text
tau = 10,000,000 Ω * 10e-12 F = 100 µs
```

A pin that takes 100 µs to settle is dangerously slow for most applications.

The 10 kΩ is a compromise that has been validated across the industry for decades:

```text
tau = 10,000 Ω * 10e-12 F = 100 ns    → fast enough
current = 3.3V / 10,000 Ω = 330 µA    → safe
noise immunity = strong enough for real environments
```

### 2.3 The "Why Does Noise Still Get In?" Sub-Question

Adding a pull-up does not make the pin perfect. It just changes the problem from "completely undefined" to "well-anchored with a small noise band."

Here is what actually happens at the resistor:

The stray charge arrives at the pin and tries to raise the voltage. But now the resistor is providing a constant gentle current pulling back toward 3.3 V. The noise charge must overcome that restoring force to change the voltage significantly.

For 10 kΩ and a voltage disturbance of 1 V, the restoring current is:

```text
I = 1V / 10,000 Ω = 100 µA
```

That is orders of magnitude larger than the stray charge currents from environmental noise, which are typically in the nanoamp range. The noise simply cannot move the pin anymore.

### 2.4 How to Choose: Pull-Up or Pull-Down?

Work through these questions in order:

**Step 1: Does your chip have a built-in option?**

Most modern microcontrollers (Arduino, STM32, ESP32, RP2040) have internal pull-up resistors that you enable in software:

```c
// Arduino
pinMode(2, INPUT_PULLUP);

// STM32 HAL
GPIO_InitStruct.Pull = GPIO_PULLUP;

// Zephyr
gpio_pin_configure(dev, pin, GPIO_INPUT | GPIO_PULL_UP);
```

Internal pull-down is less common (not all chips have it). Use internal pull-up first when available.

**Step 2: What should the pin read when nothing is happening?**

| You want the default state to be... | Use...     |
|-------------------------------------|------------|
| HIGH (1) — switch closes to GND     | Pull-up    |
| LOW (0) — switch closes to 3.3V     | Pull-down  |

**Step 3: Is this an open-drain or open-collector sensor?**

If yes, you **must** use a pull-up. These sensors can only pull the line LOW. They cannot drive it HIGH. Without an external pull-up, the line has no path to reach HIGH on its own.

I2C lines are the most common example. Always pull-up.

**Step 4: Does safety require a specific default state?**

If a wire breaking or becoming unplugged must result in a safe, known state, choose the resistor that puts the pin in the safe position when the wire is gone.

Example: An emergency stop line should read HIGH normally and LOW when triggered. Use a pull-up so a broken wire also reads LOW — triggering a stop automatically. This is called "fail-safe" design.

### 2.5 What "Defined Node" and "Undefined Node" Mean

The phrase **"dangerous when a node is left undefined"** from section 6.6 of 01A means:

```text
Defined node:   the voltage is pinned to a known value by a resistor, power rail, or driver.
Undefined node: no component is responsible for setting the voltage — it is floating.
```

An undefined node is dangerous because:

1. Software cannot predict what it reads.
2. The random HIGH/LOW toggles waste power (CMOS draws current when a gate hovers near the switching threshold).
3. A node near the switching threshold and toggling randomly can trigger interrupts, increment counters, fire actuators.

The fix is always the same: ensure every node is **defined** by connecting it to a known voltage through an appropriate path.

---

## SECTION 3 — Ringing and Damping: Why a Resistor Stops the Bounce

### 3.1 The Problem in One Sentence

When a fast digital signal travels down a wire, it is not a smooth event — it is a physical wave. That wave bounces off the far end and returns. The two waves collide, creating voltage spikes and oscillations. This is ringing.

### 3.2 Why a Wire is Not Just a Wire at High Speed

Below about 1 MHz, a short PCB trace acts like a resistor. Current flows, voltage drops, done.

Above roughly 10–100 MHz (depending on trace length), the wire develops additional personalities:

```text
Wire personality 1 — Resistance:   Still there. Tiny, but real.
Wire personality 2 — Inductance:   The magnetic field around the wire acts like a tiny coil.
Wire personality 3 — Capacitance:  The trace over a ground plane forms a capacitor.
```

Together, these three make the wire a **distributed RLC circuit** — not one component at a point, but R, L, and C sprinkled along every millimeter of the trace.

The key consequence: **electrical signals travel down this wire as a wave, not as a simple voltage level.**

### 3.3 What Causes the Ringing (The Physical Mechanism)

When a microcontroller output switches from LOW to HIGH, it launches an electromagnetic wave down the trace. That wave travels at roughly 60% the speed of light (~15 cm/ns in typical PCB material).

When the wave reaches the far end, it hits the input pin of the receiving device. That pin has high impedance — a locked door.

When a high-speed wave hits a locked door (impedance mismatch), it **reflects**:

```text
Microcontroller ──────────────────────────── Receiver
                ────► (wave travels out)
                               ◄──── (wave bounces back)
```

The reflected wave travels backward, crashes into the outgoing wave, and the two combine. Their voltages add where they overlap, creating spikes that exceed the supply voltage (overshoot) and dips below ground (undershoot).

After the first bounce, the reflected wave hits the microcontroller output, which also has some impedance mismatch, and bounces again forward. This continues until resistance in the system finally absorbs all the energy.

```text
Signal on the line during ringing (simplified):

4.0V ─────────────────────────────────────────────────
      Overshoot ↑
3.3V ──────────────────  ───────────── ─────────────
                       \/       /\   \/
2.0V                            ──────────────────────── (stabilizes here)
                       ↑ Multiple crossings
0.0V ────────────────────
      Undershoot ↓
-0.5V ────────────────────────────────────────────────
```

Each crossing of the logic threshold looks like a valid signal edge. The receiver thinks it saw multiple clock edges from one real edge.

### 3.4 Why Adding a Resistor Stops the Ringing

The resistor is placed in series, right next to the output pin of the driver:

```text
MCU ── [ 22-47Ω ] ── SCLK trace ── Receiver
```

Here is exactly what it does:

Every transmission line (wire) has a characteristic impedance — a property of its physical geometry, typically 50–75 Ω on a PCB trace.

The MCU output transistors have an output impedance of roughly 10–25 Ω.

```text
MCU output alone:   10-25 Ω    (mismatches the 50-75 Ω trace → reflections)
MCU + 33 Ω resistor: 43-58 Ω  (closely matches the trace → reflections absorbed)
```

When the series resistance approximately matches the characteristic impedance of the trace, the reflected wave returning from the far end hits the source impedance and finds a matching load. Instead of bouncing again, it is fully absorbed and converted to heat in the resistor.

```text
Clean signal with series resistor:

3.3V ──────────────────────── ──────────────────────
                            |                        |
                            |                        |
0.0V ──────────────────────   ──────────────────────
     (no overshoot, no ringing, clean transitions)
```

### 3.5 Beginner/Senior Translation Pair

| Level | Translation |
|-------|-------------|
| Beginner | The resistor calms the signal down like a shock absorber on a car. |
| Senior | The series resistor provides source termination, matching the output impedance to the trace's characteristic impedance, absorbing reflections at the source and eliminating standing waves. |

Both are correct. The physics is the same.

### 3.6 Choosing the Right Value

The goal is:

```text
MCU output impedance + series resistor ≈ trace characteristic impedance (~50 Ω)
```

In practice:

- MCU output impedance is typically 10–25 Ω (check the datasheet, or measure it).
- Choose a resistor so the total is 40–60 Ω.
- Standard values: 22 Ω, 33 Ω, 47 Ω.

You are not aiming for an exact match. The real world has tolerances. A range of 22–47 Ω handles the vast majority of common PCB trace geometries.

**Effect of going too high (e.g., 100 Ω):**

```text
Total source impedance = 110-125 Ω → exceeds trace impedance
Signal edge becomes slow and rounded
Overcorrected: the signal rise time is now a meaningful fraction of your bit time
```

At low speeds this is harmless. At 40 MHz SPI, a sluggish edge can cause setup/hold violations.

**Effect of going too low (e.g., 5 Ω):**

```text
Total source impedance = 15-30 Ω → still mismatches, less damping
Some ringing remains, but smaller than without any resistor
```

Better than nothing, but not fully effective.

---

## SECTION 4 — Filters: Sorting Signals by Speed

### 4.1 The Two-Word Rule

Before any math or diagrams:

```text
Capacitors like fast signals.
Inductors hate fast signals.
```

Filters are just these two components placed as gatekeepers. Arrange them cleverly and you can let fast signals through while blocking slow ones, or the reverse.

### 4.2 RC Low-Pass Filter — "The Slow Signals Only" Gate

```text
Input ─── [ R ] ───┬─── Output
                   │
                 [ C ]
                   │
                  GND
```

**ELI5:** Two roads lead away from the input. The straight road goes to the output. The side road leads to a drain (GND) but the entrance to that road is controlled by the capacitor.

- **Slow signals (low frequency):** The capacitor's entrance is almost completely closed (high Xc). Slow signals have nowhere to turn off. They travel straight to the output.
- **Fast signals (high frequency):** The capacitor's entrance is wide open (low Xc). Fast signals see the drain as easier than fighting through the resistor. They short-circuit to ground.

**Result:** Only slow signals reach the output. Fast noise is dumped to ground.

**Cutoff frequency** — the boundary between "mostly passes" and "mostly blocked":

```text
fc = 1 / (2 * pi * R * C)
```

Example:

```text
R = 3.3 kΩ, C = 100 nF
fc = 1 / (2 * pi * 3300 * 100e-9)
fc ≈ 482 Hz
```

Above 482 Hz: attenuated. A 10 kHz noise signal is about 21× above fc, so it is reduced to roughly 5% of its original amplitude.

Below 482 Hz: passes cleanly.

**Common use:** Place before an ADC input to remove switching noise. The ADC measures DC or very slow sensor signals. The filter kills the MHz garbage without affecting the sensor reading.

### 4.3 RC High-Pass Filter — "The Fast Signals Only" Gate

```text
Input ─── [ C ] ───┬─── Output
                   │
                 [ R ]
                   │
                  GND
```

Same components, swapped positions.

- **Slow signals (low frequency, including DC):** The capacitor entrance to the output is nearly closed. The slow signals cannot pass. They get blocked.
- **Fast signals (high frequency):** The capacitor entrance is open. Fast signals zip through to the output. The resistor's path to ground is less attractive.

**Result:** Only fast signals reach the output. Slow signals and DC are blocked.

**Common uses:**
- **Audio coupling:** Strip the DC offset from an audio signal while passing the audio frequencies.
- **Edge detection:** If you want to trigger on transitions (fast events) and ignore the steady state, a high-pass filter passes the edge and ignores the rest.
- **Removing DC drift:** A sensor output drifting slowly (temperature effects, aging) gets blocked. Only the fast, real signal gets through.

### 4.4 LC Low-Pass Filter — The Heavy-Duty Version

```text
Input ─── [ L ] ───┬─── Output
                   │
                 [ C ]
                   │
                  GND
```

Two gatekeepers working together:

- The inductor blocks high-frequency current from even entering the output side (series barrier).
- The capacitor provides an easy path to drain any high-frequency that slips through (shunt drain).

This gives much steeper attenuation above the cutoff than an RC filter with similar component values.

**Why not always use LC?**

Inductors are expensive, large, and can radiate magnetic fields. They also have resonance with the capacitor (see Part 7 of 01A). You have to pick L and C carefully to avoid the filter itself ringing.

For most sensor filtering, RC is simpler and adequate. LC is worth the trouble for power supply filtering where you need to handle large currents.

**Common use:** Smoothing the output of a switching power supply (buck converter). The converter chops at hundreds of kHz. The LC filter removes that high-frequency chopping and delivers clean DC.

### 4.5 Ferrite Bead + Capacitor — The Microscopic Noise Shield

```text
3.3V main ─── [ ferrite bead ] ───┬─── Sensor VDDA (quiet power)
                                  │
                              [ 100nF ]
                                  │
                                 GND
```

A ferrite bead is a small cylinder of iron-powder material on the power line. It is essentially an inductor tuned for a specific frequency range, often 10 MHz to 1 GHz.

Two things happen simultaneously:

1. **The ferrite bead absorbs high-frequency noise** and turns it into microscopic heat rather than reflecting it back. A normal inductor stores energy and returns it. A ferrite bead dissipates it.
2. **The 100 nF capacitor** catches any remaining high-frequency noise that got past the bead and shunts it to ground.

The result is a tiny, isolated, quiet power island for the sensor.

**Why does a sensor need its own quiet power?**

A sensor measuring micro-scale signals (temperature, humidity, acceleration) has its analog circuitry on the same chip as its digital interface. Every time the digital side sends data, it draws current pulses. Those pulses ripple through the shared power supply. The sensor's analog circuits see that ripple as a real signal. It corrupts the reading.

The ferrite bead + capacitor separates the analog supply from the digital noise, giving the sensitive analog circuits clean, stable power.

**Practical rule:** Whenever you see `VDDA`, `AVCC`, `AVDD`, or `VREF` on a datasheet, that pin needs a ferrite bead or at minimum a separate RC filter from the main power rail, plus a local 100 nF decoupling cap.

### 4.6 Quick Filter Decision Guide

| Situation | Filter Type |
|-----------|-------------|
| Noisy sensor reading going into ADC | RC Low-Pass |
| Audio coupling between circuits | RC High-Pass |
| Removing DC drift from a sensor | RC High-Pass |
| Smoothing a switching power supply | LC Low-Pass |
| Sensitive analog sensor on a digital board | Ferrite Bead + 100 nF |
| Clock signal to slave device | Series 22–47 Ω (damping, not filtering) |

---

## SECTION 5 — SPI Clock Ringing: A Full Example

### 5.1 Why SPI Is Sensitive to This

SPI clocks can run at 1 MHz, 10 MHz, 40 MHz, or higher. The higher the clock, the sharper the edges must be, and the more the wire behaves like a transmission line rather than a simple conductor.

At 40 MHz, each clock period is 25 nanoseconds. One nanosecond of ringing is 4% of a clock period. That is real.

### 5.2 The Sequence of Events That Causes Corruption

**Step 1: The MCU fires a clock edge.**

A transistor inside the MCU switches. The output goes from 0 V to 3.3 V in roughly 1–5 ns. This is not gentle. It is a step impulse.

In the frequency domain, a 2 ns edge contains energy at frequencies up to roughly 1/(2 × rise time) = 250 MHz. The 40 MHz clock is carrying RF energy.

**Step 2: The wave travels down the trace.**

The wave propagates at ~200 ps/mm for typical FR4 PCB material. A 10 cm trace means the wave takes 2 ns to travel end to end.

**Step 3: The wave hits the input pin.**

The input pin is high impedance. From the wave's point of view, it has hit a wall.

Physics: when a wave on a transmission line hits a load impedance much higher than the line impedance, the reflection coefficient approaches +1. Almost all of the energy reflects back.

```text
Reflection coefficient:
    Γ = (Z_load - Z_line) / (Z_load + Z_line)

For Z_load = 1 MΩ, Z_line = 50 Ω:
    Γ ≈ (1,000,000 - 50) / (1,000,000 + 50) ≈ +1.0
```

Nearly 100% reflection. The entire wave bounces.

**Step 4: The reflected wave adds to the original.**

At the moment of reflection, the incoming and reflected waves are at the same location. They add together:

```text
Incoming:   +3.3 V step
Reflected:  +3.3 V step (same polarity for high-impedance reflection)
Sum:        +6.6 V momentarily
```

The line overshoots to twice the supply voltage. For a 3.3 V system, that is 6.6 V — exceeding the absolute maximum rating of most logic devices (typically 3.6–4.0 V for 3.3 V chips).

**Step 5: The reflected wave hits the MCU output.**

The MCU output transistor is not a perfect termination either. It reflects some energy forward again, with reduced amplitude.

This forward-backward bouncing continues, shrinking each time due to the losses in the wire, until it dies out. The entire pattern is ringing.

### 5.3 The Fixes and Why Each One Works

| Fix | Why it works |
|-----|--------------|
| **Shorten the wire** | Less surface area means less L and C distributed along the line. Less L and C means the resonant frequency is higher and ringing is smaller. Rule of thumb: below ~5 cm at SPI speeds, ringing is usually not a problem. |
| **Put ground next to the signal** | A close ground return reduces the current loop area. Smaller loop = less inductance. It also defines the characteristic impedance of the trace more precisely. |
| **Use a ground plane** | The ground plane under the trace creates a controlled transmission line with predictable 50 Ω characteristic impedance. Without a ground plane, the impedance is undefined and varies along the trace. |
| **Add 22–47 Ω series resistor at driver** | As explained in Section 3. Absorbs the reflected wave before it bounces again. |
| **Slow the edge rate (slew rate control)** | Some MCUs let you configure the output drive speed: fast (2 ns edges) or slow (10 ns edges). Slower edges have less high-frequency energy content to begin with. The ringing may still occur but is much smaller. |

### 5.4 Why "Works at 1 MHz, Fails at 20 MHz"

At 1 MHz, each bit period is 1000 ns. Even if ringing lasts 20 ns, it is done and the signal is settled long before the receiver samples it. The receiver sees a clean level.

At 20 MHz, each bit period is 50 ns. If ringing lasts 20 ns, the signal is still oscillating at the moment the receiver is supposed to sample. Multiple threshold crossings mean the receiver may count extra edges or miss the real one.

The ringing itself has not changed. What changed is how much of the bit period it consumes.

---

## SECTION 6 — ADC Source Impedance: Why Your Sensor Readings Drift

### 6.1 What an ADC Actually Does During Sampling

Most engineers think of the ADC as a voltmeter that just reads the voltage at the input pin.

In reality, the ADC performs a two-step process:

**Step 1 — Sample:** An internal switch closes, connecting the input to a tiny internal capacitor (the sampling capacitor). Charge flows from the source, through the source resistance, and into the capacitor until the capacitor voltage matches the source voltage.

**Step 2 — Hold and Convert:** The switch opens, isolating the capacitor. The ADC then slowly converts the stored charge into a digital number.

The key insight: **during step 1, the ADC is drawing current from your sensor.** If your sensor cannot supply that current fast enough, the sampling capacitor does not fully charge, and the reading is wrong.

### 6.2 The Bucket Analogy

```text
Your sensor → [ source impedance ] → Internal switch → [ sampling capacitor ]
(water source)      (narrow pipe)       (gate valve)       (small bucket)
```

Imagine:

- The sampling capacitor is a small bucket.
- Your sensor is a water source.
- The source impedance is the pipe connecting them.
- The sampling time is the window during which the gate valve is open.

**Thin pipe (high source impedance):** Water trickles in. If the valve is open for only a short time, the bucket does not fill completely. You measure a fraction of the real water level.

**Wide pipe (low source impedance):** Water rushes in. The bucket fills instantly, no matter how short the window.

### 6.3 The Symptom: Why Previous Channel Affects Current Channel

Most microcontroller ADCs do not have one sampling path per pin. They share a single ADC engine, routing it to different pins through a multiplexer:

```text
ADC engine
   ↑
[ Mux ] ←── Pin 0 (light sensor, was 3.3V)
        ←── Pin 1 (temperature sensor, should be 1.0V)
        ←── Pin 2 (pressure sensor)
```

When the mux switches from Pin 0 to Pin 1:

- The internal sampling capacitor was fully charged to 3.3 V from Pin 0.
- Pin 1 (the temperature sensor) needs to discharge that capacitor from 3.3 V down to 1.0 V.
- If the temperature sensor has high source impedance, it cannot push or pull charge fast enough.
- The sampling window closes before the capacitor reaches 1.0 V.
- You read something between 3.3 V and 1.0 V — corrupted by the previous channel.

**This is why ADC readings appear to depend on which channel you sampled before.** It is not a software bug. It is a physics problem at the hardware level.

### 6.4 The Symptoms Summary

| Symptom | Root cause |
|---------|-----------|
| Reading is higher or lower than the real value | Source impedance too high, sampling time too short |
| Reading depends on which channel you read before | Previous charge left on internal cap, current source cannot drive it to new level fast enough |
| Slower sampling rate improves accuracy | More time for the sampling capacitor to settle to correct voltage |
| Accuracy improves with a smaller voltage divider resistor | Lower source resistance, faster charge transfer |

### 6.5 The Four Fixes in Order of Difficulty

**Fix 1: Increase ADC sampling time (easiest)**

Every MCU has a way to extend the sampling window. The ADC clock is typically divided, and you can add extra sampling cycles. Check your datasheet for the "sampling time" setting.

```c
// STM32 HAL example
ADC_ChannelConfTypeDef sConfig = {0};
sConfig.SamplingTime = ADC_SAMPLETIME_640CYCLES_5;   // was 2.5 cycles
```

More cycles = more time for the capacitor to settle. Downside: slower maximum sampling rate.

**Fix 2: Use lower-value resistors in voltage dividers**

If your sensor is behind a 100 kΩ + 100 kΩ voltage divider, the source impedance is 50 kΩ. Replace with 1 kΩ + 1 kΩ. Source impedance drops to 500 Ω.

The sampling capacitor fills 100× faster.

Downside: the divider now draws 100× more current continuously.

**Fix 3: Add a small capacitor at the ADC input**

```text
Sensor ──[ high R ]──┬──── ADC pin
                     │
                 [ 10–100 nF ]
                     │
                    GND
```

The external capacitor is a reservoir. Your slow sensor fills it up slowly over time. When the ADC switch closes, the sampling capacitor does not have to wait for the sensor — it instantly pulls charge from the local reservoir. The reservoir is large enough that it barely budges in voltage.

Rule: make the external capacitor at least 10× larger than the internal sampling capacitor.

**Fix 4: Add a buffer op-amp (best accuracy)**

```text
Sensor ──[ voltage divider ]──── [op-amp voltage follower] ──── ADC pin
```

A voltage follower op-amp has:
- Input impedance: gigaohms (draws almost nothing from sensor)
- Output impedance: milliohms (drives the sampling capacitor instantly)

It reads the sensor gently without loading it, then drives the ADC aggressively. The op-amp decouples the two sides completely.

This is the correct solution when the sensor is very weak (e.g., a potentiometer at the high-resistance end) or when you need maximum ADC accuracy.

### 6.6 Quick Diagnostic

```text
Are your ADC readings jumping or wrong?

Yes → Does accuracy improve with slower sampling rate?
  Yes → Source impedance is too high. Use Fix 1 or 2 or 3 or 4.
  No  → Noise is probably entering from another path. Check decoupling and ground.

Does the error change based on which channel you just sampled?
  Yes → Classic source impedance / mux crosstalk. Use Fix 1 or add a cap per Fix 3.
  No  → Could be offset, gain error, or reference noise.
```

---

## SECTION 7 — Bus Capacitance: Why More Devices Slows Everything Down

### 7.1 Where Bus Capacitance Comes From

A communication bus (I2C SDA, I2C SCL, SPI MISO) is a wire. Everything connected to that wire adds capacitance:

| Source | Typical capacitance |
|--------|---------------------|
| PCB trace (per cm, over ground plane) | 1–2 pF/cm |
| Each device input pin | 3–10 pF |
| 30 cm ribbon cable | 50–100 pF |
| 1 m twisted pair cable | 50–100 pF |

Every device you add, every centimeter of trace you route, every cable you use: they all add capacitance to the bus.

### 7.2 Why Capacitance Slows the Rising Edge

The I2C bus is driven LOW by devices (open-drain). It returns HIGH through the pull-up resistor:

```text
3.3V ──[ pull-up R ]──── SDA line ──── device 1, device 2, device 3
                                         (each adds C to the line)
```

When a device releases the line, the pull-up resistor must charge all that bus capacitance up to 3.3 V:

```text
Rising edge time constant: tau = R_pullup × C_total
```

Example with three I2C sensors and a short cable:

```text
C_total = 10pF (trace) + 3 × 5pF (devices) + 20pF (cable) = 45 pF
R = 4.7 kΩ
tau = 4700 × 45e-12 = 211 ns
```

The I2C fast-mode spec (400 kHz) requires rise time under 300 ns. With 211 ns time constant, the 10–90% rise time is approximately:

```text
t_rise = 2.2 × tau = 2.2 × 211 ns = 464 ns
```

That exceeds the 300 ns limit. The bus cannot run reliably at 400 kHz with this configuration.

### 7.3 The Ruined Signal Pattern and Why It Happens

```text
Perfect signal (low capacitance):
3.3V ──┐    ┌── 3.3V
       │    │
0.0V ──┘    └── 0.0V
(sharp, instant transitions)

Ruined signal (high capacitance):
3.3V ──╮    ╭── 3.3V
       │    │
       ╰────╯
0.0V ──╯    ╰── 0.0V
(sloped ramps — the cap must fill before voltage rises)
```

The issue is not just aesthetics. I2C and other protocols have defined thresholds:

```text
Below 0.3 × VCC = definitely LOW
Above 0.7 × VCC = definitely HIGH
Between: undefined region
```

For a 3.3 V bus:
- LOW threshold: below 0.99 V
- HIGH threshold: above 2.31 V

With high bus capacitance, the rising edge is so slow that the signal spends a long time in the undefined region between 0.99 V and 2.31 V. The receiver cannot tell if it is looking at a HIGH or a LOW. Data is corrupted or the device misses the edge entirely.

### 7.4 Fixes for High Bus Capacitance

**Option 1: Lower the pull-up resistor value**

```text
tau = R × C
```

Half the resistance → half the rise time. Downside: more current flows when a device holds the line LOW (more power, more stress on the driving transistor).

**Option 2: Reduce the number of devices**

Each device you remove reduces C_total by 3–10 pF. If the protocol allows it, split a large bus into two smaller buses with their own pull-ups.

**Option 3: Shorten the traces and cables**

Every centimeter removed reduces capacitance. Keep I2C buses short. If you need long cables, investigate I2C buffers.

**Option 4: I2C bus buffer/extender IC**

A dedicated I2C buffer (e.g., PCA9517, LTC4311) repeats the signal with a fresh, low-capacitance drive. The downstream bus capacitance is isolated from the upstream. Each segment can have its own optimal pull-up.

**Option 5: Reduce the communication speed**

Lower the clock frequency. Standard mode I2C at 100 kHz gives you 10 µs per bit. A 300 ns rise time is only 3% of a bit period at 100 kHz, which is fine.

---

## Summary Table: One Line Per Concept

| Concept | Root cause | Quick fix |
|---------|-----------|-----------|
| Floating pin reads random | No defined voltage, stray charge accumulates | Add 10 kΩ pull-up or pull-down |
| Why 10 kΩ specifically | Balance: drains noise without loading signal | Use internal pull-up if chip supports it |
| Signal ringing | Impedance mismatch causes wave reflections | 22–47 Ω series resistor at driver output |
| Works at 1 MHz, fails at 20 MHz | Ringing duration vs bit period: same ringing, less time to settle | Shorten wire, add series resistor, lower SPI speed |
| ADC reads wrong value | Sampling cap not fully charged before window closes | Increase sampling time, lower source impedance |
| ADC depends on previous channel | Leftover charge from previous sample | Increase sampling time, add buffer cap at pin |
| I2C slows down with more devices | Bus capacitance increases, rise time exceeds spec | Lower pull-up value, split bus, use buffer IC |
| Sensor power noise | Digital switching current couples into analog supply | Ferrite bead + 100 nF cap on analog power pin |

---

## What to Read Next

- **If you debug SPI frame drops:** Go to `05-spi-deep-dive.md`. The signal integrity concepts here map directly to the SPI timing diagrams and CS setup/hold analysis.
- **If you debug I2C ACK failures:** Go to `06-i2c-deep-dive.md`. Bus capacitance and pull-up calculations are covered in detail.
- **If your ADC reads noisy sensor data:** Go to `03-opamps-adc-sampling.md`. Op-amp buffers, anti-aliasing filters, and ADC reference design are covered there.
- **If your power rail sags when motors activate:** Review `01-passive-components.md` Section 3.5 (decoupling) and the power rail noise example in `01A` Section 9.4.
