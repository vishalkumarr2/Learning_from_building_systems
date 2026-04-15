# Exercises: Zephyr Foundations
## Projects 1–3: Blinky · UART Shell · 100Hz Timer
**Covers:** west, Kconfig, Devicetree, threads, timers, logging

---

## Section A — Conceptual Questions
*Write your answer before revealing. These test understanding, not memorization.*

---

**Q1.** You clone the Zephyr repository directly (`git clone https://github.com/zephyrproject-rtos/zephyr`) and run `west build -b nucleo_h743zi2 myapp/`. What critical pieces are missing compared to using `west init`, and what specific error do you expect?

<details><summary>Answer</summary>

`git clone` only gets the Zephyr kernel source. Zephyr depends on ~50 external modules listed in `west.yml` — `hal_stm32`, `mbedtls`, `cmsis`, `lvgl`, and others. Those repos live at URLs pinned to specific commit SHAs in `west.yml`, but `git clone` doesn't read `west.yml` or fetch them.

The build will fail early with an error like `CMake Error: could not find module FindMbedTLS` or `fatal error: cmsis_core.h: No such file or directory` — depending on which module Zephyr tries to include first.

The correct workflow: `west init ~/zephyrproject` then `west update` — this reads `west.yml` and clones every dependency repo at its pinned version. The `.west/` directory created by `west init` is what tells west "this folder is a workspace root."

</details>

---

**Q2.** Your colleague says "I changed `CONFIG_BLINK_PERIOD_MS` from 500 to 250 in `prj.conf` but the blink rate didn't change after I pressed reset." What's going on, and what must they do?

<details><summary>Answer</summary>

`prj.conf` Kconfig values become `#define` constants in a generated header (`autoconf.h`) at *compile* time. Pressing reset just re-runs the existing firmware binary — the old value `500` is baked into the machine code.

They must **rebuild** the firmware (`west build -b nucleo_h743zi2 myapp/`) and then reflash (`west flash`). Reset alone never applies config changes. If they're unsure whether west will pick up the change incrementally, running with `--pristine` (`west build --pristine ...`) forces a clean rebuild.

</details>

---

**Q3.** Explain in one sentence what the `find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})` line in `CMakeLists.txt` does that a plain `cmake_minimum_required()` by itself does not.

<details><summary>Answer</summary>

It imports Zephyr's entire build infrastructure — the ARM toolchain, all driver CMake targets, the Kconfig pipeline, and the devicetree compiler — so you can simply list your `.c` files and get a complete embedded firmware instead of a desktop executable. Without it, CMake would just set up a generic C project with no knowledge of the Zephyr SDK, and `target_sources(app PRIVATE ...)` would reference an undefined CMake target named `app`.

</details>

---

**Q4.** A prj.conf contains `CONFIG_SHELL=y`. You can see in the Zephyr source that the shell module includes `#include <zephyr/drivers/uart.h>`. Does that mean you need to also add `CONFIG_SERIAL=y` and `CONFIG_UART_CONSOLE=y` to your prj.conf? Why or why not?

<details><summary>Answer</summary>

No — Kconfig has a dependency system. The `Kconfig` file for `CONFIG_SHELL` declares `depends on SERIAL` and/or `select UART_CONSOLE`, meaning enabling `CONFIG_SHELL=y` automatically propagates the required dependencies. The Kconfig solver resolves the full dependency tree before generating `autoconf.h`.

However, knowing this matters: if you see a `CONFIG_SHELL=y` mysteriously bring in `CONFIG_SERIAL`, `CONFIG_UART_INTERRUPT_DRIVEN`, and `CONFIG_UART_CONSOLE` without you asking, they were auto-selected. You can inspect the complete resolved config using `west build -t menuconfig` and searching each symbol to see why it's set.

</details>

---

**Q5.** Your application overlay file is at `myapp/boards/nucleo_h743zi2.overlay`. The Zephyr board file at `zephyr/boards/arm/nucleo_h743zi2/nucleo_h743zi2.dts` also defines a `leds` node. Both define a node named `led_0`. What wins, and what keyword tells you that in the overlay syntax?

<details><summary>Answer</summary>

The **overlay wins**. Devicetree overlays are merged on top of the board DTS in a well-defined order: SoC DTS → board DTS → application overlay. Properties defined in the overlay overwrite properties with the same name in earlier layers.

The syntax that signals "I am modifying an existing node" is the `&` reference operator: `&led_0 { gpios = <&gpiob 0 GPIO_ACTIVE_HIGH>; }`. If you write the path without `&`, you may be creating a new node at that path rather than patching the existing one — a subtle but important distinction.

</details>

---

**Q6.** Explain the difference between `CONFIG_LOG_DEFAULT_LEVEL=3` and `CONFIG_LOG_OVERRIDE_LEVEL=3`. When would you use each?

<details><summary>Answer</summary>

`CONFIG_LOG_DEFAULT_LEVEL=3` sets the *default* maximum level for any module that doesn't explicitly declare its own level in `LOG_MODULE_REGISTER(name, level)`. A module that registers with `LOG_LEVEL_DBG` will still emit debug messages even though the default is 3 (INF).

`CONFIG_LOG_OVERRIDE_LEVEL=3` is a hard ceiling — it forces ALL modules to level 3 regardless of what they declared. Even a module that registered with `LOG_LEVEL_DBG` will have its `LOG_DBG()` calls compiled out.

Use `LOG_DEFAULT_LEVEL` during development when you want verbose modules to stay verbose. Use `LOG_OVERRIDE_LEVEL` before release when you want to globally silence debug noise and guarantee no debug messages reach production builds.

</details>

---

**Q7.** What is the difference between thread priority `0` and thread priority `-1` in Zephyr? Could a thread at priority `0` ever starve a thread at priority `-1`?

<details><summary>Answer</summary>

Priority `0` is the **highest preemptive** priority. The kernel can interrupt it to run a higher-priority thread if one becomes ready, but in practice nothing preempts it in the normal range.

Priority `-1` is **cooperative** — it is in a completely different scheduling class. Cooperative threads are never preempted by the kernel timer tick. They run until they voluntarily call `k_yield()`, `k_sleep()`, `k_sem_take()`, or any blocking primitive.

To answer the second question: **yes**. A cooperative thread at `-1` that never yields will starve ALL preemptive threads including those at priority `0`, because the preemptive scheduler only runs at tick boundaries, and cooperative threads suppress preemption. A misbehaving cooperative thread can lock the system. This is why negative priorities should only be used for very short, ISR-like logic that provably yields quickly.

</details>

---

**Q8.** Your 100Hz thread uses `k_msleep(10)` inside a `while(1)` loop. The work inside the loop takes exactly 0.5ms every iteration. After 1000 iterations, how many milliseconds have elapsed, and how far off is that from the target 10,000ms?

<details><summary>Answer</summary>

`k_msleep(10)` sleeps for 10ms *after* the function is called. Since each iteration does 0.5ms of work then sleeps 10ms, each iteration takes **10.5ms** total.

After 1000 iterations: 1000 × 10.5ms = **10,500ms** elapsed.

That is **500ms too slow** — a 5% drift. More importantly, this error accumulates without bound. By iteration 10,000, you'd be 5 seconds behind. The correct tool is `k_timer` (fire on an absolute schedule) plus `k_sem_take` to block between firings — then 0.5ms of work simply borrows from the next period and the fire times stay anchored to real wall clock.

</details>

---

**Q9.** You call `k_thread_stack_space_get(&my_thread_tid, &unused)` and get `unused=48` out of a `1024` byte stack. Should you be worried? What action do you take?

<details><summary>Answer</summary>

**Yes, be very worried.** Unused = 48 means the high-water mark of stack usage was 1024 − 48 = **976 bytes**, leaving only 48 bytes of safety margin (about 4.7%). Any additional function call, interrupt re-entry, or LOG message that takes more stack space would overflow silently, or crash unpredictably at a later, unrelated point in time.

Immediate actions:
1. Increase the thread's stack to at least 1536 bytes, ideally 2048.
2. Add `CONFIG_STACK_SENTINEL=y` to prj.conf so overflows trigger a kernel panic with a clear message rather than silent corruption.
3. After increasing, re-measure the high-water mark and aim for at least 20% headroom (e.g., with 2048 stack, keep `unused > 410`).

</details>

---

**Q10.** Your overlay file modifies pin PA5 of GPIOA for an LED. You discover the Nucleo-H743ZI2 board DTS already uses PA5 for a different peripheral. What happens at build time, and how do you resolve the conflict without touching the board DTS?

<details><summary>Answer</summary>

Devicetree does not automatically detect pin conflicts — both nodes will happily compile. However, at runtime, whichever driver initializes last will "win" the GPIO pin via the HAL's alternate-function configuration, and the other driver will behave incorrectly (the LED might never blink, or the peripheral might stop working). The conflict is silent.

Resolution without touching board DTS: in your overlay, **disable** the conflicting node that owns PA5 — e.g., `&some_spi { status = "disabled"; };`. Then define your LED on the same pin. Alternatively (better), pick a different unused pin for your LED. Check the Nucleo-H743ZI2 user manual or the board DTS file to identify which pins are free, then use those in your overlay instead.

</details>

---

**Q11.** Why does `LOG_DBG("value=%d", x)` have **zero overhead** at runtime when `CONFIG_LOG_DEFAULT_LEVEL=3`? What makes it different from wrapping it in `if (debug_flag) { printf(...); }`?

<details><summary>Answer</summary>

When the compile-time log level is set to 3 (INF), `LOG_DBG` expands to a macro that resolves to `do { } while(0)` — literally nothing. The compiler sees it as dead code and eliminates it entirely. The string `"value=%d"` doesn't even end up in the flash binary. There is no branch, no function call, no rodata storage.

By contrast, `if (debug_flag) { printf(...); }` still compiles the format string into flash (it's in rodata) and still emits a branch instruction and a potentially expensive printf call sites. Even when `debug_flag` is always 0, the data is in memory. Kconfig-gated macros genuinely produce zero binary size and zero runtime overhead — this is a first-class design goal for embedded firmware.

</details>

---

**Q12.** A teammate says: "I'll just use `printf()` instead of `LOG_INF()` — it's simpler." Name two specific reasons this is wrong in a Zephyr application.

<details><summary>Answer</summary>

1. **No filtering or level gating.** `printf` outputs unconditionally. In a 100Hz thread, `printf("dt=%u\n", dt)` will flood the UART at 100 messages/second. With `LOG_DBG`, you change one Kconfig line to silence all debug output and the call compiles away completely.

2. **No buffering — blocks the calling thread.** Zephyr's log subsystem uses an internal ring buffer and a dedicated log processing thread. `LOG_INF` returns in ~microseconds (writes to buffer, deferred flush). `printf` calls `uart_poll_out` in a tight loop, spinning until every character is transmitted at 115200 baud. A 40-character message takes ~3.5ms of CPU time in the calling thread. At 100Hz that's 35% of your budget, and it disrupts the timing measurement you're trying to make.

(Bonus: `printf` also bypasses `CONFIG_LOG_DEFAULT_LEVEL` filtering and can't be enabled/disabled per-module.)

</details>

---

**Q13.** What does `K_THREAD_DEFINE` do differently from calling `k_thread_create()` at runtime? When would you use `k_thread_create()` instead?

<details><summary>Answer</summary>

`K_THREAD_DEFINE` is a **compile-time macro** that allocates both the stack array and the `k_thread` struct as static variables in BSS. The thread is registered with the kernel before `main()` runs and starts automatically (with an optional `delay` parameter). No heap allocation, no runtime overhead.

`k_thread_create()` is a **runtime function**. You pass it a pre-allocated stack and `k_thread` struct, and it starts the thread dynamically. This is necessary when the number of threads isn't known at compile time (e.g., a thread pool that sizes itself based on a runtime config value), or when threads are created and destroyed during operation (e.g., connection handler threads in a server).

For Projects 1–3, always prefer `K_THREAD_DEFINE` — it's safer (no stack allocation errors at runtime), visible to the kernel debugger by name, and idiomatic Zephyr.

</details>

---

**Q14.** You add `CONFIG_STACK_SENTINEL=y` to prj.conf and see a kernel panic message: `"Stack sentinel for thread worker_tid overflowed"`. What does the sentinel actually check, and what does this message guarantee about **when** the overflow happened?

<details><summary>Answer</summary>

The stack sentinel is a known magic value (a specific byte pattern) written at the **bottom** of the stack allocation at thread creation time. Periodically (on context switches and in some ISR paths), the kernel checks that the magic value is still intact. If it has been overwritten, that means the stack grew past its allocated region and corrupted the sentinel.

The key nuance: the message **does not mean the overflow just happened**. It means the overflow was *detected now*, but the actual overflow happened at some point in the past when the thread's stack reached its limit. By the time the sentinel is checked, the corrupted data may have already caused undefined behavior. The sentinel is a safety net, not a real-time detector. For precise detection, `CONFIG_MPU_STACK_GUARD=y` (if the SoC has an MPU) will trap the overflow at the exact instruction it occurs.

</details>

---

**Q15.** An IRQ fires at hardware priority 3 while a thread at software priority 0 is running. Which preempts which, and what fundamental rule does this demonstrate about IRQ priorities vs thread priorities?

<details><summary>Answer</summary>

The IRQ preempts the thread. An ISR at any hardware priority can preempt *any* running thread — there is no thread priority high enough to block a hardware interrupt. The Cortex-M NVIC handles IRQ preemption entirely in hardware, before the Zephyr scheduler is even consulted.

This demonstrates that **thread priorities and IRQ priorities are completely orthogonal scales with no direct comparison possible**. Thread priority 0 means "highest among runnable threads, give this the CPU first." IRQ priority 3 means "when this hardware event fires, switch to its handler immediately, regardless of what code is running." They live in different worlds (software scheduler vs NVIC hardware), and trying to compare them as if "priority 3 IRQ < priority 0 thread" leads to incorrect reasoning about system behavior.

</details>

---

## Section B — Spot the Bug
*Each snippet has exactly ONE bug. Identify it and explain why it's wrong. Provide the fix.*

---

**Bug 1.** 100Hz timing with k_msleep

```c
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(timer_work, LOG_LEVEL_DBG);

static void worker_thread(void *a, void *b, void *c)
{
    uint32_t last = k_uptime_get_32();

    while (1) {
        k_msleep(10);

        uint32_t now = k_uptime_get_32();
        uint32_t dt = now - last;
        last = now;

        LOG_DBG("dt=%u ms", dt);

        /* Simulate 2ms of processing work */
        k_busy_wait(2000);
    }
}

K_THREAD_DEFINE(worker_tid, 2048, worker_thread, NULL, NULL, NULL, 5, 0, 0);
```

<details><summary>Bug + Fix</summary>

**The bug:** `k_msleep(10)` is called at the *start* of the loop before the work, so the timing reference is "10ms after sleep starts" — but sleep starts *after* the previous iteration's `k_busy_wait(2000)`. Result: each cycle takes 10ms + 2ms = **12ms**. After 1000 cycles, the thread is 2 seconds behind schedule, and drift accumulates forever.

**Why it fails:** `k_msleep` is relative to when it is called. It has no memory of when the previous sleep ended. It cannot compensate for the time consumed by work between sleeps.

**The fix:** Use a `k_timer` + `k_sem` pattern so the timer fires on an absolute 10ms schedule regardless of how long the work takes:

```c
K_SEM_DEFINE(tick_sem, 0, 1);

static void expiry(struct k_timer *t) { k_sem_give(&tick_sem); }
K_TIMER_DEFINE(tick_timer, expiry, NULL);

static void worker_thread(void *a, void *b, void *c)
{
    k_timer_start(&tick_timer, K_MSEC(10), K_MSEC(10));
    uint32_t last = k_uptime_get_32();

    while (1) {
        k_sem_take(&tick_sem, K_FOREVER);  /* blocks until timer fires */

        uint32_t now = k_uptime_get_32();
        uint32_t dt = now - last;
        last = now;

        LOG_DBG("dt=%u ms", dt);
        k_busy_wait(2000);  /* timer fires at exactly t+10ms regardless */
    }
}
```

</details>

---

**Bug 2.** Thread logging without enough stack

```c
static void imu_thread(void *a, void *b, void *c)
{
    LOG_INF("IMU thread started");
    while (1) {
        float ax, ay, az;
        read_imu_i2c(&ax, &ay, &az);
        LOG_INF("ax=%.3f ay=%.3f az=%.3f", ax, ay, az);
        k_msleep(10);
    }
}

K_THREAD_DEFINE(imu_tid, 512, imu_thread, NULL, NULL, NULL, 3, 0, 0);
```

<details><summary>Bug + Fix</summary>

**The bug:** Stack size is 512 bytes — far too small for a thread that calls `LOG_INF` with floating-point format arguments. `LOG_INF` with `%f` formatting requires printf-style formatting internally, which typically uses 300–600 bytes of stack. Combined with the function call chain (`imu_thread` → `read_imu_i2c` → `LOG_INF` → log formatting → uart output), this will overflow the 512-byte stack silently.

**Why it fails:** There is no MMU guard page on embedded. The stack overflow writes into adjacent SRAM — likely another thread's stack or kernel control structures. The thread may appear to work at first, then produce bizarre behavior minutes later, or hard-fault with no obvious connection to this thread.

**The fix:** Increase stack to at least 2048 bytes for any thread that calls LOG functions. Add `CONFIG_STACK_SENTINEL=y` during development to catch overflows explicitly:

```c
K_THREAD_DEFINE(imu_tid, 2048, imu_thread, NULL, NULL, NULL, 3, 0, 0);
```

And in prj.conf:
```ini
CONFIG_STACK_SENTINEL=y
CONFIG_THREAD_STACK_INFO=y
```

</details>

---

**Bug 3.** Custom Kconfig symbol not defined

```
myapp/
├── CMakeLists.txt
├── prj.conf          ← contains CONFIG_SAMPLE_RATE_HZ=100
└── src/
    └── main.c        ← uses CONFIG_SAMPLE_RATE_HZ
```

`prj.conf`:
```ini
CONFIG_GPIO=y
CONFIG_LOG=y
CONFIG_SAMPLE_RATE_HZ=100
```

`src/main.c`:
```c
#include <zephyr/kernel.h>

int main(void)
{
    int period_ms = 1000 / CONFIG_SAMPLE_RATE_HZ;
    k_msleep(period_ms);
    return 0;
}
```

Build produces: `warning: CONFIG_SAMPLE_RATE_HZ' not defined, defaulting to 0` and then a divide-by-zero at runtime.

<details><summary>Bug + Fix</summary>

**The bug:** There is no `Kconfig` file in the project root defining the `SAMPLE_RATE_HZ` symbol. Zephyr's Kconfig system only knows about symbols that are declared in a `Kconfig` file. An undeclared symbol in `prj.conf` is silently set to 0 (or ignored with a warning). The C code then evaluates `1000 / 0`.

**Why it fails:** Kconfig symbols are not freely invented in `prj.conf` — they must be defined in a `Kconfig` file that describes the symbol's type, default, range, and help text. Without the definition, the symbol doesn't exist in the resolved config.

**The fix:** Create `myapp/Kconfig` in the project root:

```kconfig
mainmenu "My Application"

config SAMPLE_RATE_HZ
    int "Sample rate in Hz"
    default 100
    range 1 1000
    help
      The sampling rate for sensor acquisition in Hertz.
      Rebuild required after changing this value.
```

Then the `CONFIG_SAMPLE_RATE_HZ=100` in `prj.conf` correctly overrides the default of 100, and `1000 / CONFIG_SAMPLE_RATE_HZ` evaluates to 10 at compile time.

</details>

---

**Bug 4.** Timer expiry function doing too much

```c
K_SEM_DEFINE(timer_sem, 0, 1);
static float sensor_reading = 0.0f;

static void timer_expiry_fn(struct k_timer *timer_id)
{
    /* Read sensor directly in timer callback */
    sensor_reading = read_i2c_sensor_blocking();  /* takes ~2ms */
    k_sem_give(&timer_sem);
    LOG_DBG("Timer fired, reading=%.3f", sensor_reading);
}

K_TIMER_DEFINE(my_timer, timer_expiry_fn, NULL);
```

<details><summary>Bug + Fix</summary>

**The bug:** `timer_expiry_fn` runs in **ISR context** (or a high-priority system-work-queue context). Calling a blocking I2C function (`read_i2c_sensor_blocking()`) from ISR context is illegal in Zephyr — it will either hang (waiting for a semaphore that can never be given at IRQ level) or corrupt the I2C driver state. Additionally, `LOG_DBG` from an ISR context is unsafe without specific ISR-safe logging configuration.

**Why it fails:** ISRs cannot call any kernel function that blocks. The I2C driver likely uses a semaphore internally to wait for transaction completion. `k_sem_take(K_FOREVER)` from ISR context is undefined behavior — it may deadlock or trigger a kernel assertion.

**The fix:** Keep the expiry function minimal — just give the semaphore. Do all blocking I/O work in the *thread* that waits on the semaphore:

```c
static void timer_expiry_fn(struct k_timer *timer_id)
{
    k_sem_give(&timer_sem);  /* only this — no blocking, no LOG */
}

static void sensor_thread(void *a, void *b, void *c)
{
    k_timer_start(&my_timer, K_MSEC(10), K_MSEC(10));
    while (1) {
        k_sem_take(&timer_sem, K_FOREVER);
        /* Now in thread context — blocking I2C is safe */
        float reading = read_i2c_sensor_blocking();
        LOG_DBG("reading=%.3f", reading);
    }
}
```

</details>

---

**Bug 5.** Overlay modifying a node that doesn't exist

```dts
/* boards/nucleo_h743zi2.overlay */
&my_custom_spi {
    status = "okay";
    cs-gpios = <&gpioa 4 GPIO_ACTIVE_LOW>;
};
```

Build error: `devicetree error: undefined node label 'my_custom_spi'`

<details><summary>Bug + Fix</summary>

**The bug:** The `&label` syntax in a devicetree overlay means "find the existing node with this label and modify it." There is no node labeled `my_custom_spi` in the board DTS or SoC DTS — so the label reference is undefined.

**Why it fails:** You cannot use `&label` to *create* a new node. You can only use it to modify an existing one. To reference the STM32's SPI1 peripheral, you must use its actual label from the SoC DTS (e.g., `&spi1`), not an invented name.

**The fix:** Look up the actual label in `zephyr/dts/arm/st/h7/stm32h743.dtsi` — the SPI peripherals are labeled `spi1`, `spi2`, etc.:

```dts
/* boards/nucleo_h743zi2.overlay */
&spi1 {
    status = "okay";
    cs-gpios = <&gpioa 4 GPIO_ACTIVE_LOW>;
    pinctrl-0 = <&spi1_sck_pa5 &spi1_miso_pa6 &spi1_mosi_pa7>;
    pinctrl-names = "default";
};
```

To discover available labels: `grep -r "spi" zephyr/boards/arm/nucleo_h743zi2/` or use `west build -t guiconfig` and inspect the devicetree viewer.

</details>

---

**Bug 6.** Shell command registered but never found

```c
/* src/commands.c */
#include <zephyr/shell/shell.h>

static int cmd_status(const struct shell *sh, size_t argc, char **argv)
{
    shell_print(sh, "System OK");
    return 0;
}

SHELL_CMD_REGISTER(status, NULL, "Print system status", cmd_status);
```

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(myapp)

target_sources(app PRIVATE src/main.c)
```

At runtime: typing `status` in the shell returns `"Command not found"`.

<details><summary>Bug + Fix</summary>

**The bug:** `src/commands.c` is not listed in `target_sources`. CMake never compiles it. The `SHELL_CMD_REGISTER` macro expands to a linker section snippet that only ends up in the binary if the `.c` file is compiled. Since the file is absent from the build, the command is never registered.

**Why it fails:** `SHELL_CMD_REGISTER` uses Zephyr's iterable sections mechanism — the macro places a small struct in a special ELF linker section. At startup, Zephyr iterates that section to find all registered commands. If the `.c` file isn't compiled, the struct never makes it into the section, and the command is invisible.

**The fix:** Add the file to `CMakeLists.txt`:

```cmake
target_sources(app PRIVATE
    src/main.c
    src/commands.c   # ← was missing
)
```

</details>

---

**Bug 7.** Misusing k_uptime_get_32() for relative timing

```c
static void measure_work_duration(void)
{
    uint32_t start = k_uptime_get_32();

    do_expensive_work();

    /* Report elapsed time */
    uint32_t elapsed = start - k_uptime_get_32();
    LOG_INF("Work took %u ms", elapsed);
}
```

<details><summary>Bug + Fix</summary>

**The bug:** The subtraction is backwards. `start - k_uptime_get_32()` subtracts a *larger* number from a *smaller* one. Because both are `uint32_t`, the result wraps around to a huge positive number (e.g., if start=1000 and now=1005, the result is `1000 - 1005 = 0xFFFFFFF8 = 4294967288`). This will log a wildly incorrect elapsed time.

**Why it fails:** Time moves forward. `start` is an earlier (smaller) value; `k_uptime_get_32()` called after is a later (larger) value. The correct subtraction is `now - start`, not `start - now`. The uint32_t wraparound after ~49 days is handled correctly by `now - start` as long as the interval fits in 32 bits.

**The fix:**

```c
uint32_t start = k_uptime_get_32();
do_expensive_work();
uint32_t elapsed = k_uptime_get_32() - start;  /* now - start, not start - now */
LOG_INF("Work took %u ms", elapsed);
```

</details>

---

**Bug 8.** Two UART backends fighting

```ini
# prj.conf
CONFIG_SHELL=y
CONFIG_LOG=y
CONFIG_LOG_BACKEND_UART=y
CONFIG_UART_CONSOLE=y
CONFIG_SERIAL=y
```

```dts
/* boards/nucleo_h743zi2.overlay — no changes */
```

Terminal shows interleaved garbage: `[00[00:00[INF]:01.23 blin:00:01ky.230 INF] ...]`

<details><summary>Bug + Fix</summary>

**The bug:** Both the shell backend and the log/console backend are configured to use the same UART device (the default `uart0`, wired to the ST-Link VCP). They race each other to write characters to the same UART from different contexts. Shell output and log output interleave at the byte level, producing garbled lines.

**Why it fails:** There is no coordination between the shell backend and the log UART backend at the character level. Both fire `uart_poll_out` from their respective contexts whenever they have data. On a 115200 baud UART, a 40-char log message takes ~3.5ms — plenty of time for the shell to inject characters mid-stream.

**The fix (two options):**

Option A (simpler): Keep everything on one UART but disable the log UART backend, using only the shell's `log` backend so log messages go through the shell output path which IS serialized:
```ini
CONFIG_LOG_BACKEND_UART=n
CONFIG_LOG_BACKEND_SHELL=y
```

Option B (cleanest): Route shell to USART1 (CP2102 on PA9/PA10), keep console/log on USART3 (ST-Link). Add to overlay:
```dts
&usart1 {
    status = "okay";
    current-speed = <115200>;
    pinctrl-0 = <&usart1_tx_pa9 &usart1_rx_pa10>;
    pinctrl-names = "default";
};
```
And add to prj.conf:
```ini
CONFIG_SHELL_BACKEND_SERIAL=y
CONFIG_UART_SHELL_ON_DEV_NAME="USART_1"
```

</details>

---

**Bug 9.** Thread silently stops

```c
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(sensor, LOG_LEVEL_INF);

static void sensor_thread(void *a, void *b, void *c)
{
    LOG_INF("Sensor thread started");
    while (1) {
        int ret = read_sensor_data();
        if (ret < 0) {
            LOG_ERR("Sensor read failed: %d", ret);
            return;  /* exit thread on error */
        }
        process_and_log(ret);
        k_msleep(10);
    }
}

K_THREAD_DEFINE(sensor_tid, 2048, sensor_thread, NULL, NULL, NULL, 5, 0, 0);
```

The sensor occasionally fails, and after a failure the system appears to keep running (main thread works) but sensor data stops being processed. No crash.

<details><summary>Bug + Fix</summary>

**The bug:** `return` from the thread entry function silently terminates the thread. Zephyr does not restart threads automatically when they exit. The thread simply stops, the semaphore/timer keeps firing (if any), and the rest of the system is unaware anything went wrong — no crash, no log message after the `LOG_ERR`.

**Why it fails:** Unlike OS processes which have a process table and watchdog mechanisms, a Zephyr thread that returns just... stops. The kernel marks it as terminated. If it held any mutexes or resources, those are now leaked. The system "works" but silently degrades.

**The fix:** Either retry the failing operation inside the loop (with backoff), or declare the thread essential with `K_ESSENTIAL` (causes kernel panic if it exits — at least you know), or implement a watchdog feeding pattern:

```c
static void sensor_thread(void *a, void *b, void *c)
{
    LOG_INF("Sensor thread started");
    uint32_t consecutive_errors = 0;

    while (1) {
        int ret = read_sensor_data();
        if (ret < 0) {
            consecutive_errors++;
            LOG_WRN("Sensor read failed: %d (error #%u)", ret, consecutive_errors);
            if (consecutive_errors > 10) {
                LOG_ERR("Too many sensor errors — reinitializing");
                init_sensor();  /* attempt recovery */
                consecutive_errors = 0;
            }
            k_msleep(100);  /* back off before retry */
            continue;       /* do NOT return — keep thread alive */
        }
        consecutive_errors = 0;
        process_and_log(ret);
        k_msleep(10);
    }
}
```

</details>

---

**Bug 10.** LOG_MODULE_REGISTER called inside a function

```c
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

int main(void)
{
    LOG_MODULE_REGISTER(myapp, LOG_LEVEL_INF);  /* inside main() */

    LOG_INF("System started");

    while (1) {
        k_msleep(1000);
    }

    return 0;
}
```

Build error: `error: 'log_myapp' undeclared` or linker error about duplicate log module.

<details><summary>Bug + Fix</summary>

**The bug:** `LOG_MODULE_REGISTER` must be called at **file scope** (outside any function), not inside `main()` or any other function. It expands to static variable declarations and linker section annotations, which are only syntactically valid at file scope. Placing it inside a function is a C syntax error or results in undefined behavior.

**Why it fails:** The macro expands to something like `static const struct log_source_const_data __log_myapp = {...}` — a static struct definition. Static struct definitions at block scope (inside a function) are legal C but mean the struct is local to that function scope, which defeats the linker-section registration mechanism entirely.

**The fix:** Move the macro to file scope, before any function definitions:

```c
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(myapp, LOG_LEVEL_INF);  /* file scope — correct */

int main(void)
{
    LOG_INF("System started");
    while (1) {
        k_msleep(1000);
    }
    return 0;
}
```

One `LOG_MODULE_REGISTER` per `.c` file, always at file scope.

</details>

---

## Section C — Fill in the Blank
*Fill in the blanks (marked `_____`) to complete working Zephyr code. Show answers in the details block.*

---

**Exercise C1.** Complete this `prj.conf` for a project that needs: interactive shell on UART, structured logging at INFO level, GPIO, and a 2KB main stack.

```ini
CONFIG_GPIO=_____
CONFIG_SHELL=_____
CONFIG_SHELL_BACKEND_SERIAL=_____
CONFIG_LOG=_____
CONFIG_LOG_DEFAULT_LEVEL=_____
CONFIG_MAIN_STACK_SIZE=_____
CONFIG_UART_CONSOLE=_____
CONFIG_SERIAL=_____
```

<details><summary>Answers</summary>

```ini
CONFIG_GPIO=y
CONFIG_SHELL=y
CONFIG_SHELL_BACKEND_SERIAL=y
CONFIG_LOG=y
CONFIG_LOG_DEFAULT_LEVEL=3          # 3 = INF
CONFIG_MAIN_STACK_SIZE=2048
CONFIG_UART_CONSOLE=y
CONFIG_SERIAL=y
```

Note: `CONFIG_SHELL=y` would auto-select `CONFIG_SERIAL` and `CONFIG_UART_CONSOLE` via Kconfig dependencies, but listing them explicitly makes the intent clear.

</details>

---

**Exercise C2.** Fill in the blanks to define a Kconfig symbol for a configurable sensor threshold:

```kconfig
config SENSOR_THRESHOLD_MG
    _____  "Sensor alarm threshold in milligrams"
    _____ 500
    _____ 10 10000
    help
      Acceleration threshold above which an alarm is triggered.
      Units: milli-g. Rebuild required after changing.
```

<details><summary>Answers</summary>

```kconfig
config SENSOR_THRESHOLD_MG
    int  "Sensor alarm threshold in milligrams"
    default 500
    range 10 10000
    help
      Acceleration threshold above which an alarm is triggered.
      Units: milli-g. Rebuild required after changing.
```

`int` declares the symbol type. `default` sets the value when not overridden. `range` adds a compile-time validity check rejecting values outside the bounds.

</details>

---

**Exercise C3.** Complete this overlay to map the `led0` alias to pin PB14 of GPIOB, active-high:

```dts
/ {
    _____ {
        led0 = &_____custom_led;
    };

    leds {
        compatible = "_____";
        custom_led: led_custom_0 {
            gpios = <&_____ _____ GPIO_ACTIVE_HIGH>;
            label = "Custom LED PB14";
        };
    };
};
```

<details><summary>Answers</summary>

```dts
/ {
    aliases {
        led0 = &custom_led;
    };

    leds {
        compatible = "gpio-leds";
        custom_led: led_custom_0 {
            gpios = <&gpiob 14 GPIO_ACTIVE_HIGH>;
            label = "Custom LED PB14";
        };
    };
};
```

- `aliases` (not `alias`) is the correct DTS node name for the Zephyr alias mechanism
- `gpio-leds` is the required `compatible` string that tells Zephyr's GPIO LED driver what this node represents
- `&gpiob` references the GPIOB controller node defined in the SoC DTS
- `14` is the pin index within port B

</details>

---

**Exercise C4.** Complete this `K_THREAD_DEFINE` call for a thread named `sensor_tid`, stack size 2048, that runs `sensor_fn(NULL, NULL, NULL)` at priority 7, starting 100ms after boot:

```c
K_THREAD_DEFINE(
    _____,    /* thread name */
    _____,    /* stack size */
    _____,    /* entry function */
    _____, _____, _____,   /* arg1, arg2, arg3 */
    _____,    /* priority */
    _____,    /* options */
    _____     /* start delay ms */
);
```

<details><summary>Answers</summary>

```c
K_THREAD_DEFINE(
    sensor_tid,      /* thread name */
    2048,            /* stack size */
    sensor_fn,       /* entry function */
    NULL, NULL, NULL, /* arg1, arg2, arg3 */
    7,               /* priority (preemptive, lower priority than main's 0) */
    0,               /* options (0 = normal thread) */
    100              /* start delay: 100ms */
);
```

</details>

---

**Exercise C5.** Fill in the blanks to implement the correct 100Hz timer setup:

```c
K_SEM_DEFINE(_____, 0, _____);  /* initial=0, max=1 */

static void expiry_fn(struct k_timer *t)
{
    k_sem_give(&_____);
}

K_TIMER_DEFINE(my_timer, _____, NULL);

static void timer_thread(void *a, void *b, void *c)
{
    k_timer_start(&my_timer, K_MSEC(_____), K_MSEC(_____));
    while (1) {
        k_sem_take(&_____, K_FOREVER);
        /* do 100Hz work */
    }
}
```

<details><summary>Answers</summary>

```c
K_SEM_DEFINE(tick_sem, 0, 1);   /* initial=0 (no ticks yet), max=1 (don't accumulate) */

static void expiry_fn(struct k_timer *t)
{
    k_sem_give(&tick_sem);
}

K_TIMER_DEFINE(my_timer, expiry_fn, NULL);

static void timer_thread(void *a, void *b, void *c)
{
    k_timer_start(&my_timer, K_MSEC(10), K_MSEC(10)); /* fire in 10ms, repeat every 10ms */
    while (1) {
        k_sem_take(&tick_sem, K_FOREVER);
        /* do 100Hz work */
    }
}
```

Setting max to 1 is intentional: if the thread misses a tick (overrun), subsequent ticks don't queue up indefinitely. The thread processes one tick, then immediately takes the already-queued next tick — behaviorally correct for rate-controlled work.

</details>

---

**Exercise C6.** Fill in the blanks to measure and log the jitter of a periodic thread. The `dt_ms` variable is already computed:

```c
static uint32_t _____ = 0;
static uint32_t _____ = UINT32_MAX;
static uint32_t sample_n = 0;

/* Inside the 100Hz loop, after computing dt_ms: */
if (dt_ms > _____) { _____ = dt_ms; }
if (dt_ms < _____) { _____ = dt_ms; }
_____++;

if (sample_n == _____) {                     /* every 10 seconds */
    LOG_INF("Jitter 10s: min=%u max=%u spread=%u ms",
            _____, _____, _____ - _____);
    _____ = 0;
    _____ = UINT32_MAX;
    _____ = 0;
}
```

<details><summary>Answers</summary>

```c
static uint32_t max_dt = 0;
static uint32_t min_dt = UINT32_MAX;
static uint32_t sample_n = 0;

if (dt_ms > max_dt) { max_dt = dt_ms; }
if (dt_ms < min_dt) { min_dt = dt_ms; }
sample_n++;

if (sample_n == 1000) {                     /* every 10 seconds at 100Hz */
    LOG_INF("Jitter 10s: min=%u max=%u spread=%u ms",
            min_dt, max_dt, max_dt - min_dt);
    max_dt = 0;
    min_dt = UINT32_MAX;
    sample_n = 0;
}
```

Note: 1000 samples at 100Hz = 10 seconds.

</details>

---

**Exercise C7.** Complete this `CMakeLists.txt` for a project with two source files (`src/main.c` and `src/sensors.c`) that also needs to include a local header directory `src/include/`:

```cmake
cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr _____ HINTS $ENV{ZEPHYR_BASE})
project(sensor_app)

target_sources(app PRIVATE
    _____
    _____
)

target_include_directories(app PRIVATE
    _____
)
```

<details><summary>Answers</summary>

```cmake
cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(sensor_app)

target_sources(app PRIVATE
    src/main.c
    src/sensors.c
)

target_include_directories(app PRIVATE
    src/include/
)
```

`REQUIRED` is not a blank option — it must be present; if Zephyr is not found, the build fails with a clear error rather than silently continuing without it.

</details>

---

**Exercise C8.** Fill in the correct prj.conf settings to enable stack overflow detection during development AND allow inspecting the stack high-water mark at runtime:

```ini
CONFIG_STACK_SENTINEL=_____     # runtime panic on overflow
CONFIG___________=y             # enables k_thread_stack_space_get()
CONFIG_FAULT_DUMP=_____         # emit full register dump on hard fault
```

<details><summary>Answers</summary>

```ini
CONFIG_STACK_SENTINEL=y
CONFIG_THREAD_STACK_INFO=y
CONFIG_FAULT_DUMP=2
```

`CONFIG_FAULT_DUMP=2` emits a detailed register dump including the CFSR (Configurable Fault Status Register) and faulting address on hard fault — much more useful than the default level 1 which just prints a generic message.

</details>

---

## Section D — Lab Tasks
*Hands-on tasks requiring hardware (Nucleo-H743ZI2 + CP2102 USB-UART adapter). Each task has a specific, verifiable success criterion.*

---

**Lab Task D1. Build and flash your first Blinky from scratch**

- **Setup:** West workspace initialized, Zephyr SDK installed, Nucleo connected via USB, ST-Link drivers working.
- **Steps:**
  1. Create the directory structure: `myapp/`, `myapp/src/`, `myapp/boards/`.
  2. Write `CMakeLists.txt` with the minimal Zephyr template (section 2.1 of study notes).
  3. Write `prj.conf` with `CONFIG_GPIO=y` and `CONFIG_BLINK_PERIOD_MS=500`.
  4. Write a `Kconfig` file defining the `BLINK_PERIOD_MS` symbol (int, default 500, range 50–5000).
  5. Write `src/main.c` using `DT_ALIAS(led0)` and `gpio_pin_toggle_dt()`.
  6. Run `west build -b nucleo_h743zi2 myapp/` and verify it compiles.
  7. Run `west flash`.
  8. Change `CONFIG_BLINK_PERIOD_MS=100` in `prj.conf`, rebuild, reflash.

- **Success criterion:** LED blinks at ~1Hz with the 500ms config. After changing to 100ms config and reflashing, LED blinks visibly faster (~5Hz). The rate change persists after pressing the reset button (it doesn't revert to 500ms).

- **If it doesn't work:**
  - No LED blink at all: Add `LOG_INF("main entered")` before the while loop and verify log output. If log appears but no blink, the overlay's `led0` alias is wrong — check it with `west build -t guiconfig` → "Devicetree" → expand `/aliases`.
  - `Board nucleo_h743zi2 not found`: Run `source ~/zephyrproject/zephyr/zephyr-env.sh` or check `ZEPHYR_BASE`.
  - Blink rate doesn't change after rebuild: Confirm you changed the file, saved, and ran `west build` again before `west flash`. Check `build/zephyr/.config` and search for `BLINK_PERIOD` to verify the new value is present.

---

**Lab Task D2. Add UART logging and observe it from minicom**

- **Setup:** CP2102 USB-to-UART adapter connected: CP2102 TX → Nucleo PA10 (USART1 RX), CP2102 RX → Nucleo PA9 (USART1 TX), CP2102 GND → Nucleo GND. (Alternatively: use the ST-Link VCP already wired to USART3.)
- **Steps:**
  1. Add to `prj.conf`: `CONFIG_LOG=y`, `CONFIG_LOG_DEFAULT_LEVEL=4`, `CONFIG_LOG_BACKEND_UART=y`, `CONFIG_UART_CONSOLE=y`.
  2. Add `LOG_MODULE_REGISTER(blinky, LOG_LEVEL_DBG)` at file scope in `main.c`.
  3. Add `LOG_INF("Blinky started, period=%d ms", CONFIG_BLINK_PERIOD_MS)` in `main()` before the loop.
  4. Add `LOG_DBG("Toggle complete, cycle=%d", cycle)` inside the loop (increment a `cycle` counter).
  5. Build, flash, open minicom at 115200 8N1.
  6. Change `CONFIG_LOG_DEFAULT_LEVEL=3` (INF), rebuild, reflash.

- **Success criterion at level 4 (DBG):** minicom shows both INF and DBG messages. You should see `[INF] blinky: Blinky started, period=500 ms` once, then `[DBG] blinky: Toggle complete, cycle=1` etc. at 2 messages per second (toggling every 500ms). After changing to level 3 and reflashing: only the `INF` startup message appears; `DBG` messages are completely gone.

- **If it doesn't work:**
  - Blank terminal: Verify baud rate (115200), hardware flow control OFF in minicom (Ctrl-A → O → Serial port setup → F = N). Check you connected to the right `/dev/ttyUSB*` device.
  - Garbled output: Two UART backends competing. Check that `CONFIG_UART_SHELL_ON_DEV_NAME` (if set) doesn't equal `CONFIG_UART_CONSOLE_ON_DEV_NAME`.
  - `LOG_DBG` messages missing even at level 4: Check that `LOG_MODULE_REGISTER` uses `LOG_LEVEL_DBG` (not `LOG_LEVEL_INF`) as the second argument.

---

**Lab Task D3. Build the UART shell and control blink rate at runtime**

- **Setup:** Same as D2. `CONFIG_SHELL=y` and shell UART working.
- **Steps:**
  1. Add `CONFIG_SHELL=y` and `CONFIG_SHELL_BACKEND_SERIAL=y` to `prj.conf`.
  2. Add `#include <zephyr/shell/shell.h>` and declare a global `volatile uint32_t blink_period_ms = CONFIG_BLINK_PERIOD_MS`.
  3. Implement `cmd_blink_set` and `cmd_blink_get` handlers (section 2.4 of study notes).
  4. Register with `SHELL_CMD_REGISTER`.
  5. Change `main()` to use `blink_period_ms` (the runtime variable) instead of `CONFIG_BLINK_PERIOD_MS` in `k_msleep`.
  6. Build, flash, connect minicom.
  7. Press Enter to get `uart:~$` prompt. Type `blink get`. Then type `blink set 100`. Observe LED rate change.

- **Success criterion:** `blink get` returns the current period. `blink set 100` causes the LED to visibly speed up within one blink cycle — **without reflashing**. `blink set 5001` returns an error message. The shell prompt reappears after each command.

- **If it doesn't work:**
  - No `uart:~$` prompt: Verify `CONFIG_SHELL=y` and `CONFIG_SHELL_BACKEND_SERIAL=y`. Press Enter a few times (shell waits for newline). Check `minicom` is in "no hardware flow" mode.
  - `Command not found` for `blink`: The `.c` file defining `SHELL_CMD_REGISTER` may not be in `target_sources`. Or `CONFIG_SHELL_CMDS=y` may be required — check with `west build -t menuconfig`.
  - Shell appears but LOG messages also appear on the same line: both are writing to same UART. Use `CONFIG_LOG_BACKEND_SHELL=y` instead of `CONFIG_LOG_BACKEND_UART=y` so log output goes through the shell's serialized output path.

---

**Lab Task D4. Implement and verify 100Hz timer with drift measurement**

- **Setup:** minicom connected. `CONFIG_LOG=y` and logging working from D2.
- **Steps:**
  1. Implement the `k_timer` + `k_sem_take` pattern (section 2.7 of study notes) in a separate thread.
  2. Compute `dt_ms = now - last` inside the thread loop.
  3. Log with `LOG_DBG("tick=%u dt_ms=%u", tick_count, dt_ms)` — increment `tick_count` each iteration.
  4. Add `LOG_WRN("drift: dt=%u", dt_ms)` when `dt_ms > 11 || dt_ms < 9`.
  5. Deliberately add `k_busy_wait(500)` (0.5ms of fake work) to the thread and verify it does NOT cause drift.
  6. Then change to `k_busy_wait(15000)` (15ms — longer than the 10ms period). Observe the overrun warning.
  7. Remove the busy_wait.

- **Success criterion:** With 0.5ms of work: `dt_ms` values are consistently 10 (no WRN messages). Over 100 ticks (1 second), the cumulative drift measured as `(actual_elapsed - 1000ms)` is **under 5ms**. With 15ms of work: WRN messages appear immediately, and `dt_ms` shows values of 10 or 20 (skipped ticks), not 25ms (which k_msleep would produce).

- **If it doesn't work:**
  - Thread never starts / no log output: Stack too small. Add `CONFIG_STACK_SENTINEL=y` to get a useful panic. Increase thread stack to 2048.
  - `dt_ms` shows 0 every tick: `last_tick_ms` not being updated. Ensure you do `last_tick_ms = now` inside the loop after computing `dt`.
  - `dt_ms` consistently shows ~10.3ms instead of 10: You're using `k_msleep(10)` not `k_timer`. Double-check the implementation uses `k_sem_take` blocking on the semaphore, not a sleep.

---

**Lab Task D5. Profile stack usage and set appropriate sizes**

- **Setup:** Timer thread from D4 running. Add `CONFIG_THREAD_STACK_INFO=y` to prj.conf.
- **Steps:**
  1. Add a shell command `stack_info` that calls `k_thread_stack_space_get()` on your timer thread and prints the unused bytes.
  2. Run the system for 60 seconds to allow the worst-case stack depth to be reached.
  3. Execute `stack_info` in the shell. Note the `unused` value.
  4. Calculate: high_water = stack_size − unused. Headroom% = (unused / stack_size) × 100.
  5. Experiment: deliberately reduce the timer thread stack to `unused - 50` (just below the actual high-water + 50 margin). Enable `CONFIG_STACK_SENTINEL=y`. Expect a kernel panic. Observe the panic message.
  6. Restore the correct stack size with ≥20% headroom.

- **Success criterion:** Step 4 should show headroom > 20% with the default 2048-byte stack if only LOG_DBG calls are in the thread. Step 5 should produce a visible kernel panic message containing "Stack sentinel" and the thread name within a few seconds of running (not a silent lockup). After step 6, the system runs stably with the panic gone.

- **If it doesn't work:**
  - `k_thread_stack_space_get` returns error: Ensure `CONFIG_THREAD_STACK_INFO=y` is set and you are passing `&timer_tid` (the k_thread struct address), not a raw pointer.
  - No panic even with undersized stack: The sentinel might need a deeper call chain to actually trigger. Add a `LOG_INF` with a long format string and several arguments — this forces more stack usage.

---

## Section E — Think Deeper
*These questions require synthesizing multiple concepts. No single answer is correct — they're thinking prompts for self-reflection or pair discussion.*

---

**E1.** You have three threads: IMU reader (priority 3, 100Hz), UART shell (priority 5, event-driven), and a background data packer (priority 8, continuous). The IMU reader uses `k_timer` + `k_sem`. The UART shell calls `LOG_INF` whenever a command is received.

You observe that every 10th IMU tick has `dt_ms=11` instead of 10. The other 9 are exactly 10. No LOG_WRN about overruns in the shell. The pattern repeats exactly every 10 ticks.

Propose one mechanistic hypothesis that explains this exact pattern, and describe what two measurements you would make to confirm or refute it.

---

**E2.** Consider the statement: "We don't need `CONFIG_STACK_SENTINEL=y` in production because we've profiled the stack and have 30% headroom."

Argue both sides: (a) why this reasoning is sound, and (b) why it could still be wrong. What real-world event could invalidate the 30% headroom measurement after the firmware ships?

---

**E3.** Your blinky app works perfectly. Then you add `CONFIG_SHELL=y` and the LED stops blinking entirely — no crash, no error message. `LOG_INF("main entered")` appears in the terminal, but nothing after it.

Trace through the most likely failure mode: what did enabling `CONFIG_SHELL=y` probably change about UART pin assignments or thread scheduling that could cause this?

---

**E4.** You are debugging a hard fault. The fault handler prints: `CFSR: 0x00000001 MMFAR: 0x00000000`. A CFSR value of `0x00000001` means IACCVIOL — instruction access violation. The MMFAR (MemManage Fault Address Register) is 0x00000000, meaning the CPU tried to fetch an instruction from address 0x0.

Given only this information, construct two different hypotheses about what program error could lead to a function call to address 0x0. Which of `CONFIG_STACK_SENTINEL=y` or `CONFIG_FAULT_DUMP=2` would have given you more useful data, and why?

---

**E5.** West uses a `west.yml` manifest to pin every dependency to a specific commit SHA. A colleague proposes: "We should stop using west and just vendor all the Zephyr files into our repo — then we own them and don't need internet access to build."

Evaluate this proposal. What does it get right? What problems would it create at month 6 when you need to pull a security patch from Zephyr upstream? Suggest a middle ground that preserves the benefits of both approaches.

---

**E6.** In section 1.3 of the study notes, prj.conf is described as controlling "compile-time" behavior. Yet with `CONFIG_LOG_RUNTIME_FILTERING=y` and `log enable dbg my_module` in the shell, you **can** change behavior at runtime without rebuilding. Does this contradict the compile-time model? Explain precisely what is still compile-time vs what becomes runtime when `CONFIG_LOG_RUNTIME_FILTERING` is enabled.

---

*End of exercises for Section 01. Next: Section 02 — I2C IMU integration and DMA.*
