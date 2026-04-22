# Senior Interview Questions — Embedded Control

**Purpose:** Questions that expose the gap between "I know PID" and "I've shipped motor control firmware."
**Level:** Senior embedded/firmware engineer (5+ years)

---

## Question 1: ISR Priority Inversion

> "You have a current loop ISR at 10 kHz and a speed loop ISR at 1 kHz. The speed ISR takes 200 µs. What's the problem?"

**Expected weak answer:** "No problem, they run at different rates."

**Deep answer:**

The speed ISR at 200 µs blocks the current ISR for 200 µs. The current loop period is 100 µs. So **two current loop samples are missed** every time the speed ISR fires.

Result:
- Current loop runs at effectively 5 kHz (every 10th cycle has a 200 µs hole)
- Phase margin drops by ~15° at that frequency
- Possible current oscillation at the current loop crossover frequency

**Fixes:**
1. Speed ISR should be **lower priority** than current ISR (preemptive)
2. Keep speed ISR under 50 µs (half the current loop period)
3. Use DMA for ADC readings so current measurement isn't blocked
4. Or: do speed computation in the current ISR (every 10th call) — same priority, no preemption issue

---

## Question 2: ADC Timing

> "I sample current at the center of the PWM period. Why?"

**Expected weak answer:** "That's when the signal is stable" or "Convention."

**Deep answer:**

PWM switching causes current ripple:

```
PWM:   _____|‾‾‾‾‾|_____|‾‾‾‾‾|_____
                                       
Current: /\/\/\/\/\/\/\/\/\/\/\/\/\/\/
           ↑ center (average value)
```

- At the PWM edge (switching instant), the current has a large transient due to parasitic inductance and reverse recovery of the body diode in the MOSFET
- At the center of the ON or OFF period, the current ripple is at its **midpoint** — sampling here gives the true **average** current without needing a low-pass filter
- If you sample at the wrong time, you get ripple-corrupted readings. This corrupts the current controller and causes **audible noise** (because the current oscillates at sub-harmonics of the PWM frequency)
- Center-aligned PWM makes this easier: the counter counts up then down, and you trigger ADC at the peak/valley

---

## Question 3: Fixed-Point Truncation Drift

> "My fixed-point PID works perfectly in unit tests but drifts 0.1 rad/s over 10 minutes on the actual robot. What's happening?"

**Expected weak answer:** "Maybe the plant model is wrong" or "Sensor drift."

**Deep answer:** **Integral truncation accumulation.**

In Q16.16, the integral update is:
```c
integral += Ki_Ts * error;  // Q16.16 multiplication
```

If `Ki_Ts * error` rounds to zero (because `error` is very small), the integral never accumulates. But the small nonzero analog error still exists. Over millions of iterations, the **missing micro-accumulations** add up.

Example: $K_i T_s = 0.1$ (Q16.16: 6553), error = 0.00001 rad/s (Q16.16: 0 — rounds to zero!)

**Fixes:**
1. **Extended accumulator:** Use a 64-bit accumulator for the integral, only truncate when outputting:
   ```c
   int64_t integral_ext += (int64_t)Ki_Ts * error;  // 64-bit
   int32_t integral = (int32_t)(integral_ext >> 16); // Truncate for output only
   ```
2. **Residual tracking:** Track the truncation remainder and add it back next cycle
3. **Higher Q resolution for integral:** Use Q8.24 for the integral term even if everything else is Q16.16

---

## Question 4: DMA vs ISR for Motor Control

> "Should I use DMA or ISR to read the ADC for current sensing?"

**Expected weak answer:** "DMA is faster" or "ISR is simpler."

**Deep answer:** Both are valid but for different reasons:

**DMA approach:**
- ADC triggers on PWM center → DMA transfers result to memory → DMA complete interrupt fires → ISR reads from memory
- **Advantage:** Zero-jitter sampling. The ADC timing is purely hardware, unaffected by software delays.
- **Advantage:** The ISR only processes data, doesn't wait for conversion
- **Disadvantage:** More complex setup, harder to debug

**ISR approach:**
- PWM center interrupt → ISR starts ADC → ISR waits for conversion (or uses polling) → ISR processes
- **Advantage:** Simpler code, easier to understand
- **Disadvantage:** If the ISR is delayed (by another ISR), the ADC sample point shifts, corrupting the current reading
- **Disadvantage:** ADC conversion time (1-5 µs) is wasted in the ISR

**For OKS motor control:** DMA is preferred. The ADC-trigger-PWM hardware chain guarantees consistent sampling regardless of software load. The ISR computation budget is 100 µs (10 kHz), and wasting 5 µs on ADC wait is acceptable but unnecessary.

---

## Question 5: Watchdog Strategy

> "How would you implement a watchdog for the motor controller?"

**Expected weak answer:** "Use the hardware watchdog timer, pet it in the main loop."

**Deep answer:** Multi-level watchdog strategy:

**Level 1: Hardware WDT** — catches hard lockups (infinite loops, stack overflow)
```c
// Pet in the current loop ISR (10 kHz)
void current_isr(void) {
    wdt_feed();
    // ... control computation
}
// If ISR stops firing → hardware reset in 5 ms
```

**Level 2: Software task watchdog** — catches soft lockups (task starvation, priority inversion)
```c
// Each task increments a counter
volatile uint32_t task_alive[NUM_TASKS];

// Watchdog task checks all counters every 100 ms
void watchdog_task(void) {
    for (int i = 0; i < NUM_TASKS; i++) {
        if (task_alive[i] == last_alive[i]) {
            // Task i is stuck!
            enter_safe_state();
        }
        last_alive[i] = task_alive[i];
    }
}
```

**Level 3: Communication watchdog** — catches Jetson→MCU link failures
```c
if (millis() - last_spi_msg > 10) {  // 10 ms timeout
    // Jetson is silent → decelerate and hold
    enter_reduced_mode();
}
```

**Level 4: Sanity watchdog** — catches control divergence
```c
if (abs(speed_error) > MAX_SPEED_ERROR && 
    abs(current) > 0.9 * CURRENT_LIMIT &&
    duration > 500) {  // 500 ms
    // Something is mechanically wrong
    emergency_stop();
}
```

**Trap:** A single watchdog is not enough. Each level catches a different failure class.

---

## Question 6: Stack Overflow in ISR

> "The motor controller works for 2 hours then crashes randomly. No pattern. What do you suspect?"

**Expected weak answer:** "Memory leak" or "Race condition."

**Deep answer:** Top suspects in order:

1. **Stack overflow:** The ISR uses local variables that overflow into adjacent memory. Works for hours because the corrupted memory location isn't accessed often. Use stack painting to diagnose:
   ```c
   // Fill stack with 0xDEADBEEF at boot
   // Periodically check how many are left
   uint32_t stack_free = check_stack_watermark(current_isr_stack);
   ```

2. **Uninitialized variable read:** A variable on the ISR stack is sometimes read before being written (depends on code path). Works most of the time because the stack happens to contain reasonable values from previous invocations.

3. **Integer overflow accumulation:** A counter or accumulator wraps after exactly $2^{32}$ increments. At 10 kHz: $2^{32} / 10000 / 3600 = 119$ hours. At 100 kHz: 11.9 hours. Close to 2 hours? Check 16-bit counters: $2^{16} / 10000 = 6.5$ seconds — too fast. $2^{16} / 100 = 655$ seconds ≈ 11 minutes. Check all counter sizes.

4. **Heap fragmentation:** If the ISR (or a task) allocates/frees memory. **Never allocate in real-time code.** But library functions sometimes allocate internally (printf, sprintf, malloc in C string operations).

5. **EMI-induced bit flip:** Hardware noise corrupts a RAM cell. More likely in noisy motor control environments. Use ECC RAM if available, or periodic state validation.

---

## Question 7: RTOS Timing Guarantee

> "I need the current loop to run exactly every 100 µs. Can an RTOS guarantee this?"

**Expected weak answer:** "Yes, RTOS gives deterministic timing" or "Use a high-priority task."

**Deep answer:** **No, an RTOS cannot guarantee exact timing for software tasks.** It can guarantee a **worst-case latency bound**, but there's always jitter.

Sources of jitter:
1. **Interrupt latency:** Even the highest-priority ISR has 0.1–5 µs entry latency (context save, pipeline flush)
2. **Critical sections:** Other code that disables interrupts (even briefly) adds jitter
3. **Cache misses:** First execution after a context switch may be slower
4. **Flash wait states:** Code fetched from flash has variable timing if flash cache misses

For guaranteed 100 µs timing:
- **Best approach:** Hardware timer ISR (not RTOS task). Timer triggers ADC → DMA → ISR fires after conversion. Jitter < 1 µs.
- **Good approach:** RTOS task at highest priority with timer interrupt as trigger. Jitter: 1–5 µs.
- **Bad approach:** RTOS periodic task with vTaskDelayUntil(). Jitter: 5–50 µs depending on system load.

**OKS approach:** The current loop runs in a timer ISR, not an RTOS task. The speed loop runs in a lower-priority ISR. Only the communication and monitoring tasks use RTOS scheduling.

---

## Question 8: Motor Identification On-Line

> "We manufacture 1000 robots. Each motor has ±10% parameter variation. How do you handle this?"

**Expected weak answer:** "Use robust gains that work for all motors" or "Tune each robot manually."

**Deep answer:** **On-line identification + adaptive tuning at startup:**

```
Boot sequence:
1. Apply known voltage step (e.g., 2V for 200 ms)
2. Record current and speed transients
3. Fit first-order model: extract τ_m, K, R_a
4. Compute PID gains from identified parameters
5. Validate: run a speed step test, check overshoot < 10%
6. If validation fails: fall back to conservative default gains
7. Store identified parameters in EEPROM for next boot
```

This runs once at every boot (takes ~2 seconds). Benefits:
- Handles motor-to-motor variation
- Handles aging (brush wear increases $R_a$, bearing wear changes $B$)
- Handles motor replacement (new motor, different parameters)
- Provides diagnostic data (if identified $R_a$ is 3× higher than nominal, the motor is failing)

**Trap:** "Robust gains" sounds safe but wastes 30-50% of performance headroom. In a warehouse with tight corridors, that headroom translates to speed and precision.

---

## Question 9: Safety-Critical Braking

> "The Jetson crashes mid-transit. The MCU must stop the robot safely. What's the braking strategy?"

**Expected weak answer:** "Set PWM to zero" or "Emergency stop, cut power."

**Deep answer:** Cutting power is the **worst** option for a moving robot:

1. **No power = no braking.** DC motors coast when unpowered. The robot continues moving with only friction to slow it.
2. **Passive regenerative braking:** Short the motor terminals through a low-resistance path. Back-EMF drives current through the short, creating braking torque. Simple, effective, no control needed.
3. **Active braking:** Continue running the PID with setpoint = 0, decelerating at the maximum safe rate. Best tracking performance, but requires the controller to still be functional.
4. **Mechanical brake:** Engage a physical brake (spring-loaded, power-off-to-engage). Guaranteed stop but harsh — can damage goods on the robot.

**OKS braking cascade:**
```
Level 0: Active deceleration (PID running, normal stop)
Level 1: Passive regen braking (short motor terminals via H-bridge low-side)
Level 2: Mechanical brake engagement (spring-loaded)
Level 3: Power cut (last resort, motor coasts)
```

The MCU firmware implements all four levels. On Jetson communication loss:
- Hold current velocity for 50 ms (tolerate brief jitter)
- Decelerate at 0.5 m/s² (active braking) for up to 2 seconds
- If speed doesn't decrease → engage mechanical brake
- If SPI is still down after 5 seconds → power cut

---

## Question 10: The volatile Keyword

> "Should PID variables shared between the ISR and main loop be marked volatile?"

**Expected weak answer:** "Yes, always use volatile for shared variables."

**Deep answer:** **`volatile` is necessary but not sufficient.**

`volatile` tells the compiler:
- Don't optimize away reads/writes
- Don't reorder accesses
- Re-read from memory every time

But `volatile` does NOT guarantee:
- **Atomicity:** Reading a 32-bit value on an 8/16-bit MCU takes multiple instructions. The ISR can modify it mid-read.
- **Consistency:** Reading `speed` and `position` separately might get speed from cycle N and position from cycle N+1.

**Correct approach:**

```c
// Option 1: Disable interrupts briefly
__disable_irq();
int32_t local_speed = shared_speed;
int32_t local_position = shared_position;
__enable_irq();

// Option 2: Double-buffer with sequence counter
typedef struct {
    volatile uint32_t sequence;
    volatile int32_t speed;
    volatile int32_t position;
} motor_state_t;

motor_state_t state_buf[2];
volatile int active_buf = 0;

// ISR writes to inactive buffer, then flips
// Main loop reads from active buffer
```

**Trap:** Many embedded tutorials say "just use volatile." This causes subtle bugs in multi-variable consistency that only manifest under specific timing conditions.
