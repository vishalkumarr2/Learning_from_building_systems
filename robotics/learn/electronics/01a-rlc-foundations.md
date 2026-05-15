# 01A — RLC Foundations: Resistance, Reactance, and Impedance

## A slow, practical bridge from "I know a resistor" to "I can reason about real circuits"

**Prerequisite:** You know that a resistor limits current.
**Goal:** Understand how resistors, capacitors, and inductors behave together, why reactance exists, what impedance really means, and how to debug circuits without feeling lost.

---

## Why This Section Exists

If you are still learning resistors and suddenly see words like **impedance**, **reactance**, **high-Z**, **capacitive loading**, **inductive kick**, or **RLC resonance**, it can feel like electronics jumped three floors without telling you where the stairs are.

This lesson is the stairs.

We will build the idea in this order:

1. A resistor limits current immediately.
2. A capacitor resists voltage change.
3. An inductor resists current change.
4. Reactance is frequency-dependent opposition to AC.
5. Impedance is the full opposition a circuit gives to changing signals.
6. RLC circuits are what happen when all three are in the same room.

The senior-engineer version is simple: **every circuit is energy moving through paths, with components either burning energy, storing energy, or returning energy later.** Once you can see that, terms like impedance stop being scary.

---

## Start Here — The Three Parts in Kid Language

Before the long explanation, keep this picture in your head:

| Thing | Simple meaning | If you connect it to DC | If the signal changes fast | Everyday example |
| ----- | -------------- | ----------------------- | -------------------------- | ---------------- |
| Resistance | How hard it is for current to flow | Limits current immediately | Still limits current | LED resistor, pull-up resistor |
| Capacitance | How much charge a part can store for each volt | Charges up, then blocks steady current | Becomes an easy path for fast changes | Decoupling cap, RC delay, touch sensor |
| Inductance | How strongly a coil/wire fights current change | Eventually acts almost like wire | Becomes a hard path for fast changes | Relay coil, motor winding, ferrite bead |

The shortest mental model:

```text
R = current brake
C = voltage cushion
L = current flywheel
```

That is why the three equations look different:

```text
Resistor:  V = I * R       current depends on voltage right now
Capacitor: I = C * dV/dt   current depends on how fast voltage changes
Inductor:  V = L * dI/dt   voltage depends on how fast current changes
```

If the words feel heavy, translate them like this:

- **Resistance** asks: "How much current can flow for this voltage?"
- **Capacitance** asks: "How much charge must move before this voltage changes?"
- **Inductance** asks: "How much voltage is needed to change this current?"

Most circuit understanding comes from noticing which one of those three questions is active.

---

## PART 1 — The Three Questions Every Circuit Asks

When you look at any circuit, ask these three questions before touching formulas:

1. **Where is the voltage pushing from and to?**
2. **Where can current flow in a complete loop?**
3. **Is the signal steady DC, slowly changing, or changing very fast?**

Most beginner confusion comes from answering only the first two. That works for simple resistor circuits, but capacitors and inductors care deeply about the third question: **how fast things are changing**.

Think like this:

| Component | What it hates | What it stores | What it does with energy |
| --------- | ------------- | -------------- | ------------------------- |
| Resistor | Current flow | Nothing useful | Burns energy as heat |
| Capacitor | Voltage changing suddenly | Electric field | Stores and gives back charge |
| Inductor | Current changing suddenly | Magnetic field | Stores and gives back current |

That table is the whole lesson in seed form.

---

## PART 2 — Resistance: The Beginner's Anchor

### 2.1 What Resistance Means

Resistance is opposition to current. A resistor is like a narrow pipe in a water system: more resistance means less flow for the same pressure.

Electrical version:

```text
V = I * R

V = voltage, the push
I = current, the flow
R = resistance, the opposition
```

If you increase resistance, current goes down. If you increase voltage, current goes up.

### 2.2 The Most Important Resistor Example: Protecting an LED

Imagine a 3.3 V GPIO driving an LED. The LED wants about 2.0 V across it and 10 mA through it.

```text
3.3 V GPIO ---- resistor ---- LED ---- GND
```

The resistor must handle the leftover voltage:

```text
Voltage across resistor = 3.3 V - 2.0 V = 1.3 V
Desired current = 10 mA = 0.010 A
R = V / I = 1.3 / 0.010 = 130 ohm
```

Use 150 ohm because it is a standard value and slightly safer.

The lesson: **the resistor turns excess electrical energy into heat so the LED does not die.**

### 2.3 Resistance Is Real Even When You Did Not Add a Resistor

This is where real circuits start becoming interesting.

Everything has some resistance:

- A wire has tiny resistance.
- A MOSFET when ON has `Rds_on`.
- A battery has internal resistance.
- A GPIO output has output resistance.
- A motor winding has copper resistance.

When someone says "this GPIO pin can drive 20 mA," they are not saying it is magic. Inside the chip, transistors and tiny resistances limit what can happen safely.

### 2.4 DC Resistor Thinking

For steady DC circuits, resistor reasoning works beautifully.

Example:

```text
12 V ---- 1 kOhm ---- GND

I = V / R = 12 / 1000 = 0.012 A = 12 mA
P = V * I = 12 * 0.012 = 0.144 W
```

A quarter-watt resistor can survive that. A tiny 0402 SMD resistor may not, depending on its rating.

Practical mentor rule: **always calculate both current and power. Current tells you if the circuit works. Power tells you if it survives.**

---

## PART 3 — Capacitance: The Component That Hates Sudden Voltage Change

### 3.1 What a Capacitor Is

A capacitor is two conductive plates separated by an insulator. It stores energy in an electric field.

The simple picture:

```text
Plate A    Insulator    Plate B
  ||||||||||||||||||||||||
```

When you apply voltage, charge piles up on one plate and leaves the other plate. The capacitor becomes a small energy tank.

### 3.2 The One Rule to Remember

A capacitor resists sudden changes in voltage.

That is the beginner-friendly version of:

```text
I = C * dV/dt
```

Meaning:

- If voltage changes quickly, a lot of current flows.
- If voltage is steady, current eventually stops.
- A capacitor can supply current briefly to keep voltage from dipping.
- A capacitor can absorb current briefly to keep voltage from spiking.

### 3.3 Capacitor With DC

Connect a capacitor to a battery through a resistor:

```text
Battery + ---- resistor ---- capacitor ---- Battery -
```

At the first instant, the capacitor is empty. It behaves almost like a short circuit because it has 0 V across it and is willing to accept charge.

After some time, it charges up to the battery voltage. Once fully charged, it behaves like an open circuit for DC.

So for DC after things settle:

```text
Capacitor = open circuit
```

This is why people say capacitors "block DC."

### 3.4 Capacitor With Changing Signals

Now feed the capacitor a changing signal.

If the voltage keeps rising and falling, the capacitor keeps charging and discharging. Current keeps flowing because `dV/dt` is not zero.

So for AC:

```text
Capacitor passes changing signals more easily as frequency increases.
```

That is not magic. Higher frequency means voltage changes faster. Faster voltage change means more current through the capacitor.

### 3.5 Practical Example: Decoupling a Microcontroller

An STM32 pin switching from LOW to HIGH draws a tiny burst of current from the power rail. If that current must travel from a regulator several centimeters away, the voltage at the chip can dip.

So we place a 100 nF capacitor close to the chip:

```text
3.3 V rail ----+---- MCU VCC pin
               |
              100 nF
               |
GND -----------+---- MCU GND pin
```

When the MCU suddenly needs current, the nearby capacitor supplies it for a few nanoseconds. Then the regulator slowly refills the capacitor.

Beginner translation: **the capacitor is a tiny local battery for fast events.**

Senior translation: **the capacitor provides low impedance at high frequency near the load.**

Same thing. Different language.

### 3.6 Practical Example: RC Delay

Put a resistor before a capacitor:

```text
3.3 V ---- resistor ----+---- output
                        |
                     capacitor
                        |
                       GND
```

The capacitor cannot jump instantly to 3.3 V. It fills gradually through the resistor.

Time constant:

```text
tau = R * C
```

Example:

```text
R = 10 kOhm
C = 100 nF
tau = 10,000 * 0.0000001 = 0.001 s = 1 ms
```

After about 5 time constants, the capacitor is practically full.

```text
5 * tau = 5 ms
```

This same idea controls:

- Button debouncing.
- Reset delays.
- I2C rise time.
- Low-pass filters.
- ADC input settling.

---

## PART 4 — Inductance: The Component That Hates Sudden Current Change

### 4.1 What an Inductor Is

An inductor is a coil of wire. Current through the coil creates a magnetic field. Energy is stored in that magnetic field.

Simple picture:

```text
----((((((((----
```

The bigger the coil and magnetic core, the more energy it can store for the same current.

### 4.2 The One Rule to Remember

An inductor resists sudden changes in current.

That is the beginner-friendly version of:

```text
V = L * dI/dt
```

Meaning:

- If current changes quickly, the inductor creates a large voltage.
- If current is steady, an ideal inductor has almost no voltage across it.
- An inductor tries to keep current flowing the way it was flowing.

### 4.3 Inductor With DC

At steady DC, current is constant.

```text
dI/dt = 0
V = L * 0 = 0
```

So an ideal inductor looks like a short circuit after things settle.

Real inductors are coils of wire, so they still have some copper resistance.

For DC after things settle:

```text
Inductor = almost a wire
```

### 4.4 Inductor With Changing Signals

For a changing signal, the current is trying to increase and decrease. The inductor fights that change.

So for AC:

```text
Inductor blocks changing signals more strongly as frequency increases.
```

This is the opposite of a capacitor.

### 4.5 Practical Example: Relay Coil and Flyback Diode

A relay coil is an inductor. Suppose current is flowing through it and you turn the transistor OFF.

The coil says: "No. Current was flowing. I want it to keep flowing."

To force current through an open switch, it creates a big voltage spike. That spike can destroy the transistor.

The fix is a flyback diode:

```text
12 V ---- relay coil ----+---- transistor ---- GND
                         |
                       diode
                         |
                        12 V
```

The diode gives the coil current a safe loop when the transistor turns off.

Beginner translation: **the diode gives the inductor a gentle way to stop.**

Senior translation: **the diode clamps back-EMF and controls the current decay path.**

### 4.6 Practical Example: Motor Windings

A DC motor winding is also an inductor. When you PWM a motor driver, you are turning voltage ON and OFF quickly, but motor current does not instantly jump up and down.

The winding inductance smooths the current.

That is why motor control talks about current ripple, freewheel paths, and dead-time. The motor is not just a resistor. It is an R-L load with moving mechanical parts attached.

---

## PART 5 — Reactance: Resistance for Changing Signals

### 5.1 Why Resistance Is Not Enough

A resistor behaves mostly the same at DC, 100 Hz, 10 kHz, and 1 MHz.

A capacitor does not.

An inductor does not.

Their opposition depends on frequency, so we need a new word: **reactance**.

Reactance is measured in ohms, just like resistance, but it is not the same thing.

Important difference:

```text
Resistance burns energy as heat.
Reactance stores energy and gives it back later.
```

### 5.2 Capacitive Reactance

Capacitive reactance:

```text
Xc = 1 / (2 * pi * f * C)
```

As frequency goes up, capacitive reactance goes down.

Example: 100 nF capacitor.

| Frequency | Xc approx | What it feels like |
| --------- | --------- | ------------------ |
| DC | infinite | Open circuit |
| 100 Hz | 15.9 kOhm | Almost open |
| 10 kHz | 159 ohm | Noticeable path |
| 1 MHz | 1.6 ohm | Strong high-frequency path |

This is why a 100 nF decoupling capacitor does almost nothing to slow DC power, but helps a lot with fast noise.

### 5.3 Inductive Reactance

Inductive reactance:

```text
Xl = 2 * pi * f * L
```

As frequency goes up, inductive reactance goes up.

Example: 10 uH inductor.

| Frequency | Xl approx | What it feels like |
| --------- | --------- | ------------------ |
| DC | 0 ohm | Almost wire |
| 100 Hz | 0.006 ohm | Wire |
| 10 kHz | 0.63 ohm | Small opposition |
| 1 MHz | 62.8 ohm | Serious opposition |

This is why ferrite beads and inductors are useful for blocking high-frequency noise while allowing DC power through.

### 5.4 The Beginner's Frequency Rule

Memorize this:

```text
Capacitor: low frequency blocked, high frequency passed.
Inductor:  low frequency passed, high frequency blocked.
```

Even simpler:

```text
C likes fast changes.
L hates fast changes.
R does not care much.
```

---

## PART 6 — Impedance: The Full Opposition

### 6.1 What Impedance Means

Impedance is the total opposition a circuit gives to current when signals may be changing.

It includes:

- Resistance from resistors and real wire losses.
- Capacitive reactance from capacitors and stray capacitance.
- Inductive reactance from inductors, coils, wires, and loops.

For DC-only resistor circuits, you can usually say:

```text
impedance = resistance
```

For changing signals, that is no longer enough.

### 6.2 Impedance Is Ohm's Law Growing Up

Beginner Ohm's law:

```text
V = I * R
```

AC version:

```text
V = I * Z
```

`Z` is impedance.

Impedance is measured in ohms, but it also carries phase information. That means it tells you two things:

1. How much current flows.
2. Whether current lines up with voltage or is shifted in time.

### 6.3 Why Phase Exists

With a resistor, voltage and current rise and fall together.

```text
Resistor:
Voltage:  up, down, up, down
Current:  up, down, up, down
```

With a capacitor, current comes first because current must flow to charge the capacitor before voltage appears across it.

```text
Capacitor:
Current leads voltage.
```

With an inductor, voltage comes first because voltage must push against the coil before current ramps up.

```text
Inductor:
Current lags voltage.
```

The classic memory aid:

```text
CIVIL

In a Capacitor: I leads V
In an Inductor: V leads I
```

You do not need to master complex numbers yet. Just remember: capacitors and inductors shift timing. That timing shift is part of impedance.

### 6.4 The Engineer's Formula, Gently

For a series circuit:

```text
Z = R + jX
```

Where:

```text
X = Xl - Xc
```

`j` means the reactive part is sideways from resistance. Think of it like a map:

```text
Resistance: left-right distance
Reactance:  up-down distance
Impedance:  diagonal distance
```

Magnitude:

```text
|Z| = sqrt(R^2 + X^2)
```

If this feels abstract, use it only as a calculator rule for now:

1. Compute `Xc` and `Xl` at the frequency you care about.
2. Subtract them: `X = Xl - Xc`.
3. Combine with `R` using the square root formula.

### 6.5 Example: Same Capacitor, Different Frequencies

Suppose a 100 nF capacitor is connected from a noisy 3.3 V rail to ground.

At 100 Hz:

```text
Xc = 1 / (2 * pi * 100 * 100 nF) = 15.9 kOhm
```

That is high impedance. Low-frequency ripple does not get shunted strongly.

At 1 MHz:

```text
Xc = 1 / (2 * pi * 1,000,000 * 100 nF) = 1.59 ohm
```

That is low impedance. Fast noise gets a short local path to ground.

This is why the same capacitor can be useless for one problem and perfect for another. **Always ask: at what frequency?**

### 6.6 Example: Why "High Impedance" Means "Easy to Disturb"

A GPIO input may have input impedance of megaohms or more. That means it draws almost no current.

That sounds good, but it also means tiny stray currents and electric fields can move the voltage around.

So a floating input behaves randomly:

```text
No pull-up, no pull-down:

nearby noise + tiny input capacitance + high impedance = random HIGH/LOW readings
```

Add a 10 kOhm pull-up:

```text
3.3 V ---- 10 kOhm ---- GPIO input
```

Now the pin has a defined default path. It is no longer floating.

Practical rule: **high impedance is good when you do not want to load a signal, but dangerous when a node is left undefined.**

---

## PART 7 — RLC Circuits: All Three Behaviors Together

### 7.1 What RLC Means

RLC means a circuit containing:

- R: resistance.
- L: inductance.
- C: capacitance.

This can be intentional, like a filter, or accidental, like a long SPI wire.

Intentional RLC:

```text
audio crossover, radio tuner, motor filter, power supply filter
```

Accidental RLC:

```text
PCB trace resistance + wire inductance + input capacitance
```

Many "mysterious" electronics bugs are accidental RLC circuits ringing at a frequency you did not expect.

### 7.2 Series RLC Circuit

```text
Input ---- R ---- L ---- C ---- GND
```

Total series impedance:

```text
Z = R + j(Xl - Xc)
```

Three cases matter.

#### Case 1: Low Frequency

At low frequency:

```text
Xc is large
Xl is small
```

The capacitor dominates. The circuit feels capacitive.

#### Case 2: High Frequency

At high frequency:

```text
Xc is small
Xl is large
```

The inductor dominates. The circuit feels inductive.

#### Case 3: Resonance

At one special frequency:

```text
Xl = Xc
```

The inductor and capacitor cancel each other's reactance. The circuit is left mostly with resistance.

That frequency is resonance:

```text
f0 = 1 / (2 * pi * sqrt(L * C))
```

At resonance, energy sloshes back and forth between the capacitor's electric field and the inductor's magnetic field.

Kid-level analogy: a swing.

- Capacitor is like height at the top of the swing.
- Inductor is like speed at the bottom.
- Energy moves between height and speed.
- Resistance is friction that slowly stops the swing.

### 7.3 Example: Resonance Calculation

Suppose:

```text
L = 10 uH
C = 100 nF
```

Then:

```text
f0 = 1 / (2 * pi * sqrt(10e-6 * 100e-9))
f0 = 1 / (2 * pi * 1e-6)
f0 = 159 kHz approximately
```

At around 159 kHz, that inductor and capacitor naturally want to exchange energy.

If there is little resistance, the circuit rings strongly. If there is more resistance, the ringing is damped.

### 7.4 Damping: Why Resistance Can Be Helpful

Resistance wastes energy as heat. Usually beginners think that sounds bad.

But sometimes resistance is exactly what you need because it stops ringing.

Example: SPI SCLK line ringing.

```text
MCU output ---- long trace/wire ---- device input
```

Hidden components:

- The MCU output has resistance.
- The wire has inductance.
- The input pin and wire have capacitance.

That is an accidental RLC circuit. A sharp clock edge can make it ring.

Add a small series resistor near the driver:

```text
MCU output ---- 22 to 47 ohm ---- SCLK trace ---- device input
```

That resistor slows and damps the edge just enough to reduce ringing.

Beginner translation: **the resistor calms the signal down.**

Senior translation: **series termination damps the transmission-line/reflection behavior and reduces overshoot.**

### 7.5 Quality Factor: How Ringy Is It?

You will see the letter `Q` around RLC circuits.

`Q` means quality factor, but practically it means:

```text
High Q = low damping = rings strongly and selects a narrow frequency band
Low Q  = high damping = broad, gentle, less ringing
```

Examples:

- Radio tuner: wants high Q to select one station.
- Digital clock line: wants low enough Q to avoid ringing.
- Power rail: wants damping so regulator/capacitor networks do not oscillate.

---

## PART 8 — Filters: RLC Thinking in Useful Form

### 8.1 RC Low-Pass Filter

```text
Input ---- R ----+---- Output
                 |
                 C
                 |
                GND
```

Low frequencies pass. High frequencies are sent to ground through the capacitor.

Cutoff:

```text
fc = 1 / (2 * pi * R * C)
```

Example:

```text
R = 3.3 kOhm
C = 100 nF
fc = 482 Hz approximately
```

Use case: remove high-frequency noise before an ADC.

### 8.2 RC High-Pass Filter

```text
Input ---- C ----+---- Output
                 |
                 R
                 |
                GND
```

Low frequencies are blocked. High frequencies pass.

Use case: audio coupling, edge detection, removing DC offset.

### 8.3 LC Low-Pass Filter

```text
Input ---- L ----+---- Output
                 |
                 C
                 |
                GND
```

The inductor resists high-frequency current. The capacitor gives high-frequency noise a path to ground.

Use case: switching power supplies.

### 8.4 Ferrite Bead Plus Capacitor

Common on sensitive power rails:

```text
3.3 V main ---- ferrite bead ----+---- VDDA or sensor power
                                 |
                                100 nF
                                 |
                                GND
```

The ferrite bead blocks high-frequency noise. The capacitor shunts that noise to ground. Together they create a quieter local supply.

---

## PART 9 — Practical Examples That Make the Words Real

### 9.1 I2C Pull-Up and Bus Capacitance

I2C lines are pulled HIGH by resistors. The bus also has capacitance from wires and device pins.

```text
3.3 V ---- pull-up resistor ---- SDA line ---- bus capacitance ---- GND
```

That is an RC charging circuit.

If the pull-up is too weak or the bus capacitance is too large, the signal rises slowly.

Example:

```text
R = 4.7 kOhm
C = 200 pF
tau = R * C = 4700 * 200e-12 = 0.94 us
```

At 400 kHz I2C, bits are only 2.5 us long. A slow rise eats a lot of your timing budget.

Use 2.2 kOhm:

```text
tau = 2200 * 200e-12 = 0.44 us
```

Better. More current when LOW, but faster rise.

This is impedance thinking in daily life: **the pull-up resistor and bus capacitance form a frequency-dependent limitation.**

### 9.2 SPI Clock Ringing

A fast SPI clock edge is not just "HIGH then LOW." It contains high-frequency energy.

A long jumper wire has:

- Resistance.
- Inductance.
- Capacitance to nearby wires and ground.
- A characteristic impedance.

So at 40 MHz, the wire is not just a wire. It is a distributed RLC system.

Symptoms:

- Overshoot above 3.3 V.
- Undershoot below ground.
- Multiple threshold crossings.
- Data works at 1 MHz but fails at 20 MHz.

Fixes:

- Shorten the wire.
- Put ground next to the signal.
- Use a proper ground plane.
- Add 22 to 47 ohm series resistor near the driver.
- Slow the edge if the driver allows it.

### 9.3 ADC Input Source Impedance

An MCU ADC input often has a tiny internal sampling capacitor. When sampling starts, that capacitor must charge to the sensor voltage quickly.

If your sensor has high output impedance, the ADC capacitor charges too slowly.

Symptoms:

- ADC readings depend on previous channel.
- Fast sampling gives wrong values.
- Slower sampling improves accuracy.

Fixes:

- Use lower resistor values in the divider.
- Increase ADC sampling time.
- Add a buffer op-amp.
- Add a small capacitor near the ADC input if appropriate.

This is why "input impedance" matters. The ADC is not only measuring voltage; during sampling it briefly asks the source to provide charge.

### 9.4 Power Rail Noise

A robot board has motors, DC-DC converters, radios, MCUs, and sensors. They all share power somehow.

Motor current changes quickly. DC-DC converters switch quickly. MCU pins switch quickly.

The power rail has hidden impedance:

- Trace resistance.
- Trace inductance.
- Capacitor ESR and ESL.
- Connector resistance.
- Battery internal resistance.

So current bursts become voltage noise:

```text
Voltage noise = current burst * power-path impedance
```

That is why layout, decoupling, ground return paths, and capacitor placement matter.

### 9.5 Speaker Crossover

A speaker crossover is a friendly RLC example.

- Tweeter should receive high frequencies.
- Woofer should receive low frequencies.

Use capacitor for tweeter:

```text
Amplifier ---- capacitor ---- tweeter
```

The capacitor blocks low-frequency current that could damage the tweeter.

Use inductor for woofer:

```text
Amplifier ---- inductor ---- woofer
```

The inductor blocks high-frequency current from the woofer.

Same rules. Different application.

---

## PART 10 — How to Read Any Circuit Like a Calm Engineer

Use this checklist.

### Step 1: Identify DC Paths

For DC steady state:

```text
Capacitor = open circuit
Inductor = short circuit, plus winding resistance
Resistor = resistor
```

Ask:

- What is the default voltage at each node?
- Is anything floating?
- Where does current flow continuously?
- What power is dissipated?

### Step 2: Identify Fast-Change Paths

For fast events:

```text
Capacitor = low impedance path
Inductor = high impedance obstacle
Wire = not ideal anymore
```

Ask:

- Where does switching current return?
- Is the decoupling capacitor close enough?
- Is there a large current loop?
- Is a long wire acting like an inductor or transmission line?

### Step 3: Compute Reactance at the Frequency You Care About

Do not ask "what is the impedance of this capacitor?" without a frequency.

Ask:

```text
At 100 Hz, what is Xc?
At 400 kHz, what is Xc?
At 40 MHz, what is Xc?
```

The answer may change from "nearly open" to "nearly short."

### Step 4: Look for Stored Energy

Capacitors and inductors store energy.

Ask:

- If I turn this off, where does the inductor current go?
- If I plug this in, how fast does the capacitor charge?
- If this rings, what R, L, and C are forming the resonator?

### Step 5: Add Damping When Needed

Ringing means energy is bouncing between L and C.

Resistance damps ringing.

Examples:

- Series resistor on SPI clock.
- Snubber across relay contacts.
- ESR in capacitors.
- Termination resistor on CAN/RS-485.

---

## PART 11 — Tiny Labs You Can Actually Build

### Lab 1: LED Resistor

Build:

```text
3.3 V ---- 330 ohm ---- LED ---- GND
```

Measure:

- Voltage across LED.
- Voltage across resistor.
- Current through resistor.

Expected learning: voltage divides based on component behavior, and current is set by the resistor.

### Lab 2: RC Charging Curve

Build:

```text
3.3 V square wave ---- 10 kOhm ----+---- scope probe
                                   |
                                  100 nF
                                   |
                                  GND
```

Expected learning: the output rises slowly instead of instantly.

Change R to 1 kOhm. The rise becomes faster.

Change C to 1 uF. The rise becomes slower.

### Lab 3: RC Low-Pass Filter

Use the same circuit and feed a square wave.

Watch sharp edges become rounded.

Expected learning: the capacitor gives high-frequency edge energy a path to ground.

### Lab 4: Inductive Kick

Use a relay module, not a bare relay at first.

Compare:

- Relay coil with flyback diode.
- Relay coil without flyback diode only if supervised and using safe equipment.

Expected learning: inductors generate voltage spikes when current is interrupted.

### Lab 5: Long Wire Ringing

Send a fast GPIO square wave down a long jumper wire and observe it on a scope.

Then add a 33 ohm resistor near the GPIO.

Expected learning: the waveform becomes cleaner because the series resistor damps the accidental RLC path.

---

## PART 12 — Common Beginner Confusions

### "Is impedance just resistance?"

For DC resistor circuits, yes, close enough.

For changing signals, no. Impedance includes resistance plus frequency-dependent behavior from capacitance and inductance.

### "Why is impedance measured in ohms if it is not resistance?"

Because it still relates voltage to current:

```text
V = I * Z
```

Ohms are the unit of opposition to current. Resistance is one kind. Reactance is another. Impedance is the full package.

### "Can current pass through a capacitor? There is an insulator inside."

No electrons cross the dielectric in a healthy capacitor. But current flows in the external circuit while one plate charges and the other plate discharges.

For DC after charging, this stops. For AC, it keeps happening back and forth.

### "Why does an inductor make high voltage when switched off?"

Because it tries to keep current flowing. If you suddenly remove the path, it raises voltage until current finds a path through a diode, spark, transistor breakdown, or parasitic capacitance.

### "Why do fast digital circuits need analog thinking?"

Because a digital edge is a fast analog event. A 40 MHz SPI clock is not only 40 MHz; its sharp edges contain much higher-frequency energy. At those frequencies, wires have inductance, capacitance, delay, and impedance.

### "Why does adding a resistor sometimes improve a signal?"

Because it damps energy. A small series resistor can reduce ringing, overshoot, EMI, and false threshold crossings.

---

## PART 13 — Cheat Sheet

### Component Behavior

| Component | DC steady state | High frequency | Stores energy in | Main danger |
| --------- | --------------- | -------------- | ---------------- | ----------- |
| Resistor | Resists current | Resists current | None | Heat |
| Capacitor | Open circuit | Low impedance | Electric field | Inrush current |
| Inductor | Short circuit | High impedance | Magnetic field | Voltage spike |

### Formulas

```text
Ohm's law:          V = I * R
Power:              P = V * I = I^2 * R = V^2 / R
RC time constant:   tau = R * C
Cap reactance:      Xc = 1 / (2 * pi * f * C)
Ind reactance:      Xl = 2 * pi * f * L
Impedance:          Z = R + j(Xl - Xc)
Magnitude:          |Z| = sqrt(R^2 + (Xl - Xc)^2)
Resonance:          f0 = 1 / (2 * pi * sqrt(L * C))
```

### Mental Rules

```text
R burns energy.
C resists voltage change.
L resists current change.
C passes fast changes.
L blocks fast changes.
Impedance depends on frequency.
Fast digital edges are analog events.
Ringing means energy is bouncing between L and C.
Resistance can damp ringing.
```

---

## PART 14 — Practice Questions

Try these without rushing. The point is to build intuition.

1. A 10 kOhm resistor is connected from 5 V to ground. What current flows? What power is dissipated?

2. A 100 nF capacitor is connected from 3.3 V to ground near an MCU. Does it affect DC power much? Why does it help with fast switching noise?

3. A relay coil draws 80 mA. You turn it off with a MOSFET and no flyback diode. What does the inductor try to do?

4. A 4.7 kOhm I2C pull-up is used with 400 pF bus capacitance. What is the RC time constant? Why might this fail at 400 kHz?

5. Calculate the capacitive reactance of 100 nF at 1 kHz and 1 MHz. What changed?

6. Calculate the inductive reactance of 10 uH at 1 kHz and 1 MHz. What changed?

7. An SPI clock line rings badly. Why might adding a 33 ohm resistor near the driver help?

8. For L = 10 uH and C = 100 nF, what is the resonant frequency? What does resonance mean physically?

9. A node is described as "high impedance." Is that always good? When is it bad?

10. In one sentence each, explain resistance, reactance, and impedance.

### Answers

1. `I = 5 / 10000 = 0.5 mA`. `P = 5 * 0.0005 = 2.5 mW`.

2. At DC, the capacitor charges and then draws almost no current. For fast noise, its reactance is low, so it supplies or absorbs quick current pulses locally.

3. It tries to keep the 80 mA flowing. If no safe path exists, it creates a large voltage spike until current finds a path, possibly through MOSFET breakdown.

4. `tau = 4700 * 400e-12 = 1.88 us`. At 400 kHz, the bit period is 2.5 us, so the line may rise too slowly.

5. At 1 kHz: about 1.59 kOhm. At 1 MHz: about 1.59 ohm. Higher frequency makes a capacitor easier to pass through.

6. At 1 kHz: about 0.063 ohm. At 1 MHz: about 62.8 ohm. Higher frequency makes an inductor harder to pass through.

7. The resistor damps the accidental RLC circuit formed by driver resistance, trace/wire inductance, and input capacitance. It reduces overshoot and ringing.

8. About 159 kHz. At resonance, energy moves back and forth between the inductor's magnetic field and the capacitor's electric field.

9. High impedance is good when you do not want to load a signal. It is bad when a node is floating because tiny noise can move the voltage.

10. Resistance is steady opposition that dissipates energy. Reactance is frequency-dependent opposition that stores and returns energy. Impedance is the total opposition, including resistance and reactance.

---

## Final Mentor Summary

If you remember only one thing, remember this:

A resistor decides how much current flows. A capacitor fights voltage changing. An inductor fights current changing. Reactance is how capacitors and inductors oppose changing signals. Impedance is the full answer when a circuit asks, "how hard is it for current to flow at this frequency?"

Once you ask "what frequency?" and "where does the energy go?", you are no longer just reading circuits. You are thinking like an electronics engineer.
