# Electronics & Serial Protocols — Learning Plan
### For: Software engineer (Python/ROS2) building STM32 Zephyr ↔ Jetson Orin SPI bridge at 100Hz
### Goal: Understand the HARDWARE and PROTOCOLS underneath the software APIs

---

## Why This Matters

You already know how to call `spi_transceive()` in Zephyr. But when the 100Hz bridge drops frames at 40MHz SPI, you need to know WHY — and the answer is in the analog world: signal integrity, impedance mismatches, capacitive loading on CS lines, or a slave DMA that arms too late. These notes build you up from electrons to frames.

---

## Dependency Graph

```
                    ┌──────────────────┐
                    │  07-CAN Bus      │  ← differential signaling,
                    │  (standalone)    │     arbitration, error handling
                    └──────────────────┘

┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ 01-Passive  │────→│ 02-Semiconductors│────→│ 03-OpAmps/ADC/   │
│ Components  │     │ Diodes, BJTs,    │     │   Sampling Theory│
│ R, C, L     │     │ MOSFETs          │     │                  │
└─────┬───────┘     └────────┬─────────┘     └──────────────────┘
      │                      │
      │   ┌──────────────────┘
      │   │  (MOSFETs unlock open-drain → unlocks I2C pull-ups)
      │   │  (RC time constants unlock pull-up calculations)
      │   │
      ▼   ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ 04-UART     │     │ 05-SPI           │     │ 06-I2C           │
│ Serial Deep │     │ Deep Dive        │     │ Deep Dive        │
│ Dive        │     │ (YOUR BRIDGE)    │     │ (sensor config)  │
└─────────────┘     └──────────────────┘     └──────────────────┘
```

**Reading order:** 01 → 02 → 03 → then 04/05/06/07 in any order (though 02 before 06 is strongly recommended for open-drain).

---

## Topic Breakdown

| # | Topic | Est. Time | Prerequisites | What It Unlocks |
|---|-------|-----------|---------------|-----------------|
| 01 | Passive Components (R, C, L) | 3-4 hrs | Basic algebra | Pull-ups, decoupling, RC filters, voltage dividers, power dissipation, everything |
| 02 | Semiconductors (Diodes, BJTs, MOSFETs) | 3-4 hrs | 01 (Ohm's law, RC) | Level shifting, open-drain, motor drivers, ESD protection, power switching |
| 03 | Op-Amps, ADC, Sampling Theory | 3-4 hrs | 01 + 02 | Signal conditioning, Nyquist, anti-aliasing, understanding IMU data paths |
| 04 | UART Serial Deep Dive | 2-3 hrs | 01 (basic) | Debug console, GPS modules, RS-485 industrial sensors, DMA patterns |
| 05 | SPI Deep Dive | 3-4 hrs | 01 + 02 (MOSFET) | **Your 100Hz bridge**, IMU reads, flash memory, display drivers |
| 06 | I2C Deep Dive | 3-4 hrs | 01 + 02 (MOSFET, open-drain) | EEPROM, temperature sensors, IO expanders, multi-sensor buses |
| 07 | CAN Bus Deep Dive | 2-3 hrs | 01 + 02 (differential) | Automotive comms, industrial sensors, robot actuator buses |

**Total: ~20-26 hours of focused study**

---

## Checkpoint Questions — "You're Done When You Can Answer..."

### 01 — Passive Components
- [ ] A 3.3V GPIO drives an LED with Vf=2.0V. What resistor value limits current to 10mA? (Answer: 130Ω, use 150Ω standard value)
- [ ] Why does every IC need a 100nF cap within 5mm of its VCC pin? What happens if you skip it?
- [ ] A voltage divider with 10kΩ + 10kΩ divides 5V to 2.5V. You connect a 1kΩ load. What is the actual output voltage now, and why?
- [ ] What is the -3dB frequency of a 1kΩ + 100nF low-pass filter?

### 02 — Semiconductors
- [ ] Why can't you use a regular MOSFET with Vgs(th)=4V as a switch driven by a 3.3V GPIO?
- [ ] Draw the BSS138 level-shifting circuit for 3.3V ↔ 5V I2C. Explain how both directions work.
- [ ] A relay coil draws 50mA at 12V. Design the BJT driver circuit from a 3.3V GPIO (choose transistor, calculate base resistor).
- [ ] What is a flyback diode and why does the relay above need one?

### 03 — Op-Amps, ADC, Sampling
- [ ] A sensor outputs 0-5V but your ADC accepts 0-3.3V. Design the signal chain (hint: voltage divider + buffer).
- [ ] Your 12-bit ADC reads 2048. What voltage is that? (Need to know Vref)
- [ ] You sample a 150Hz signal at 200Hz. What frequency do you actually see? (Aliased to 50Hz)
- [ ] Why does a high-impedance sensor need a buffer before the ADC?

### 04 — UART
- [ ] Draw the waveform for sending 'A' (0x41) at 115200-8N1. How long does it take?
- [ ] Two devices configured at 115200 and 112000 baud. At what byte in a 20-byte frame does corruption start?
- [ ] Why is DMA essential for UART at 1Mbaud?

### 05 — SPI
- [ ] What physically happens during one SPI clock cycle? (Two shift registers exchange one bit)
- [ ] What is the difference between Mode 0 and Mode 3? When does it matter?
- [ ] Your SPI slave reads 0x52 when the master sent 0xA5. What's wrong? (Likely wrong CPOL/CPHA — bit shift)
- [ ] Why is SPI slave mode "hard" compared to master mode?

### 06 — I2C
- [ ] Why does I2C need pull-up resistors and SPI doesn't?
- [ ] Calculate the pull-up resistor for 400kHz I2C with 200pF bus capacitance.
- [ ] An I2C bus is stuck with SDA low. How do you recover? Why does 9 clocks work?
- [ ] Walk through reading register 0x75 from device 0x68: what is every byte on the wire?

### 07 — CAN
- [ ] Why does CAN use differential signaling? What noise rejection advantage does it provide?
- [ ] Two nodes transmit IDs 0x100 and 0x080 simultaneously. Who wins and why?
- [ ] What happens if you forget the 120Ω termination resistors?
- [ ] What is bit-stuffing and why is it needed?

---

## Study Tips

1. **Read with a multimeter and a dev board at hand.** Measure the voltage divider you designed — seeing 2.47V instead of 2.5V teaches you more than any textbook.

2. **Use an oscilloscope or logic analyzer.** A $15 Saleae clone lets you SEE SPI/I2C/UART waveforms. Match what you see to the ASCII diagrams in these notes.

3. **Build the simplest possible circuit for each concept.** An LED + resistor teaches Ohm's law. A 10kΩ + 100nF RC filter with a function generator teaches time constants.

4. **When you read a datasheet, start with page 1.** The first page has: max voltage, max current, pin count, package, and the key parameter (bandwidth, slew rate, Rds_on, etc.). You can ignore 90% of the datasheet until you need it.

5. **Everything is Ohm's law.** V=IR is the most powerful equation in electronics. If something isn't working, measure V, measure I, calculate R — the mismatch tells you what's wrong.

---

## Quick Reference: Units You'll See Everywhere

| Symbol | Unit | What It Measures | "Feel" for the Value |
|--------|------|------------------|---------------------|
| V | Volts | Electrical pressure | 3.3V = MCU logic, 5V = USB, 12V = motors, 120/240V = wall |
| A | Amps | Current flow | 10mA = LED, 500mA = USB, 2A = motor, 15A = circuit breaker |
| Ω | Ohms | Resistance | 100Ω = current limit, 10kΩ = pull-up, 1MΩ = high-Z input |
| F | Farads | Capacitance | 100pF = tiny, 100nF = decoupling, 10µF = bulk, 1000µF = power supply |
| H | Henrys | Inductance | 10µH = DC-DC, 100µH = filter, 1mH = relay coil |
| Hz | Hertz | Frequency | 100Hz = your bridge, 1kHz = audio, 1MHz = I2C fast, 40MHz = SPI |
| W | Watts | Power | 0.25W = resistor limit, 1W = small heatsink, 10W = needs cooling |
