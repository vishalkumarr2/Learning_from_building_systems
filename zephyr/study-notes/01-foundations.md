# Zephyr Foundations — Study Notes
### Projects 1–3: Blinky, UART Shell, 100Hz Timer Thread
**Hardware:** STM32 Nucleo-H743ZI2 · Zephyr RTOS · ST-Link V3 · CP2102 USB-UART

---

## PART 1 — ELI15 Concept Explanations

---

### 1.1 What is a "west workspace" and why Zephyr needs it

**The analogy: Python venv**

When you start a Python project, you don't install libraries globally — you create a `venv` so that project A's `requests==2.28` doesn't break project B's `requests==2.31`. West is exactly this, but for an entire embedded operating system.

Zephyr is not a simple library you `#include`. It's a collection of:
- The kernel source (~1M lines of C)
- Hundreds of driver modules
- Toolchain wrappers (CMake, ninja, gcc-arm)
- Board definitions and devicetree files

All of these need to be at **specific, pinned versions** that match each other. West is the tool that:
1. Clones all these repos into one directory tree (called the workspace)
2. Pins every repo to the exact commit recorded in `west.yml`
3. Knows how to invoke CMake, the ARM compiler, and the flasher — all in the right order

**What the workspace looks like on disk:**
```
~/zephyrproject/          ← this entire folder IS the west workspace
    .west/                ← hidden metadata: where west.yml lives
        config
    zephyr/               ← Zephyr kernel source (don't edit this)
    bootloader/
    modules/
    tools/
    myapp/                ← YOUR application code lives here
```

**The key mental model:** Your application (`myapp/`) is a *guest* inside the Zephyr workspace. It never contains Zephyr itself. West glues them together at build time.

```bash
# One-time setup: initialize workspace
west init ~/zephyrproject
cd ~/zephyrproject
west update                      # clones all repos listed in west.yml

# Every build: west invokes cmake + ninja for you
west build -b nucleo_h743zi2 myapp/
west flash                       # flashes via ST-Link
```

> **Why not just `git clone zephyr` and `make`?** Because Zephyr imports ~50 external modules (hal_stm32, mbedtls, lvgl, …). West is what keeps them all synchronized. Without it, you'd be manually cloning and version-matching 50 repos.

---

### 1.2 CMakeLists.txt vs prj.conf vs .overlay — three completely different jobs

This is the most common point of confusion for beginners. They look similar (all are "config files") but they control three completely different layers.

| File | Layer | When it runs | What it controls |
|------|-------|-------------|-----------------|
| `CMakeLists.txt` | Build system | `west build` invocation | Which source files compile, which Zephyr modules get linked |
| `prj.conf` | Kconfig | Preprocessor / compile time | Feature flags, subsystem enable/disable, numeric tuning constants |
| `boards/nucleo_h743zi2.overlay` | Devicetree | Merged before compile | Hardware topology: which pins, which peripherals, their properties |

**CMakeLists.txt — the recipe card**

Think of a CMakeLists.txt as a recipe that tells the build system: "here are the ingredients (source files), here is the oven (Zephyr SDK), here is what I want to bake (firmware .elf)."

```cmake
# Minimum viable CMakeLists.txt for a Zephyr app
cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(my_blinky)

target_sources(app PRIVATE src/main.c)
```

The `find_package(Zephyr ...)` line is the magic — it pulls in ALL of Zephyr's build infrastructure. After that, you just list your `.c` files.

**prj.conf — the feature switch panel**

Every optional feature in Zephyr is behind a Kconfig symbol. prj.conf is the file where you flip those switches for your project. The syntax is always `CONFIG_SOMETHING=y` (enable) or `CONFIG_SOMETHING=n` (disable) or `CONFIG_SOMETHING=1024` (set a number).

```ini
# prj.conf examples
CONFIG_GPIO=y              # enable GPIO driver subsystem
CONFIG_SERIAL=y            # enable UART drivers
CONFIG_SHELL=y             # enable interactive shell
CONFIG_LOG=y               # enable logging
CONFIG_LOG_DEFAULT_LEVEL=3 # 0=off, 1=ERR, 2=WRN, 3=INF, 4=DBG
CONFIG_MAIN_STACK_SIZE=2048
```

These values become `#define` constants injected into every C file at compile time (via an auto-generated `autoconf.h`). There is no runtime lookup — if you change prj.conf you must rebuild.

**boards/nucleo_h743zi2.overlay — the wiring diagram**

The board's .overlay file describes the *physical hardware* in a structured language called Devicetree Source (DTS). It tells Zephyr: "Pin PA5 is connected to an LED, and it's active-low." Your C code never mentions PA5 directly — it just asks for the node labeled `leds` and uses whatever pin the overlay says.

```dts
/* boards/nucleo_h743zi2.overlay */
/ {
    leds {
        compatible = "gpio-leds";
        my_led: led_0 {
            gpios = <&gpiob 0 GPIO_ACTIVE_HIGH>;
            label = "My LED";
        };
    };
};
```

> **The payoff:** The SAME main.c runs unchanged on a different board — you just swap the overlay. This is why embedded OSes use devicetree.

---

### 1.3 How Kconfig actually works — CONFIG_X=y vs a runtime variable

**The analogy: a light switch wired before the wall is plastered**

A runtime variable is a light switch you can flip any time. Kconfig is a switch that gets wired BEFORE the wall is plastered — you can choose what to build in, but once the firmware is compiled, it cannot change without a rebuild.

Here's the concrete mechanism:

1. You write `CONFIG_SHELL=y` in prj.conf
2. During `west build`, the Kconfig system reads all `Kconfig` files in Zephyr source
3. It resolves dependencies (enabling SHELL auto-enables UART, etc.)
4. It generates a C header: `build/zephyr/include/generated/autoconf.h`
5. That header contains `#define CONFIG_SHELL 1`
6. The shell module's C files have `#ifdef CONFIG_SHELL` guards — so the shell code only compiles when that define exists

**What CONFIG_MAIN_STACK_SIZE=2048 actually does:**

```c
// Somewhere deep in Zephyr kernel:
#define K_KERNEL_STACK_DEFINE(sym, size) ...
K_KERNEL_STACK_DEFINE(_main_stack, CONFIG_MAIN_STACK_SIZE);
```

The number `2048` gets substituted at compile time into a stack array allocation. There is zero runtime overhead — it's the same as writing `char stack[2048]` directly. If you set it too small, the thread overflows silently (or crashes if you enabled `CONFIG_STACK_SENTINEL=y`).

**Why this matters practically:**
- Changing a CONFIG requires a full rebuild (sometimes `west build --pristine`)
- You cannot read CONFIG values at runtime from your app (they're preprocessor defines, not variables)
- Dependencies are automatic: enabling CONFIG_SHELL will auto-enable CONFIG_SERIAL, CONFIG_UART_CONSOLE, etc.

---

### 1.4 What devicetree is and why embedded systems use it

**The analogy: a phone book vs hardcoded phone numbers**

Imagine writing code that calls `gpio_pin_set(GPIOA, 5, 1)` to blink an LED. That works on your exact board — but if you move the LED to pin PB0, every file that references PA5 must be updated. Worse, you need different code for every board variant.

Devicetree solves this with **indirection via labels**. Instead of pin numbers, your C code uses *logical names*:

```c
// C code uses a label, not a pin number
const struct device *led = DEVICE_DT_GET(DT_ALIAS(led0));
```

Somewhere in a DTS file (either in Zephyr's board files or your overlay), that alias is defined:

```dts
aliases {
    led0 = &green_led;   /* green_led is defined with gpios = <&gpiob 0 ...> */
};
```

The mapping from `led0` → PB0 lives in the overlay. Your main.c never changes.

**What devicetree is at a deeper level:**

DTS (Devicetree Source) is a hierarchical description of hardware — think of it as a JSON file for chips. It describes:
- Which buses exist (SPI1, I2C1, USART3…)
- Which peripherals are on which bus
- Which GPIO pins they use
- Their clock sources, interrupts, DMA channels

The DTS files are compiled by `dtc` (devicetree compiler) into a binary blob, then parsed by Zephyr's macros at compile time to generate C code. By the time your application runs, the devicetree data has been compiled into the firmware — there is no parsing at runtime.

**The three layers of devicetree for Zephyr:**
1. **SoC DTS** (in `zephyr/dts/arm/st/h7/`) — describes the STM32H743 chip's peripherals
2. **Board DTS** (in `zephyr/boards/arm/nucleo_h743zi2/`) — describes the Nucleo board layout
3. **Application overlay** (in your app's `boards/` dir) — YOUR customizations. Overlays WIN over everything below them.

---

### 1.5 What k_thread is — the OS concept, scheduling, cooperative vs preemptive

**The analogy: employees sharing one desk**

Imagine three employees who need to use one computer. Without a manager, they'd fight. The OS kernel is the manager — it decides who uses the CPU and for how long.

A **k_thread** is Zephyr's name for one "employee" — a sequence of instructions with its own:
- **Stack**: private scratchpad memory (local variables, function call chain)
- **Priority**: how urgently it needs the CPU
- **State**: running, ready, sleeping, waiting for a semaphore, etc.

**Cooperative vs preemptive scheduling — the key difference:**

| Mode | How it works | Priority range | Danger |
|------|-------------|---------------|--------|
| **Cooperative** | Thread runs until it *voluntarily* yields (calls `k_yield()`, `k_sleep()`, or waits on a sync primitive) | Negative priorities (−16 to −1) | One thread can starve all others |
| **Preemptive** | Kernel can interrupt a thread at any tick boundary if a higher-priority thread is ready | 0 to +14 (lower number = higher priority) | Need to protect shared data with mutexes |

**Priority numbers in Zephyr:**
- Priority **0** = highest preemptive priority
- Priority **14** = lowest preemptive priority  
- Priority **−1** = cooperative (never preempted, only yielded)
- The `main()` thread runs at priority **0** by default

**Rule of thumb for your projects:**
- Use priority **5** for a background worker thread (lower than main, preemptive)
- Use `-1` only if the thread must never be interrupted (ISR-like logic)
- Match stack size to how deep your call chain goes (LOG functions are expensive — add 512 bytes)

```c
// Thread entry function signature
void my_thread_fn(void *arg1, void *arg2, void *arg3)
{
    while (1) {
        /* do work */
        k_msleep(100);  // release CPU, sleep 100ms
    }
}

// Static thread definition (no malloc, stack is a compile-time array)
K_THREAD_DEFINE(
    my_thread,          // thread name (used in kernel debugger)
    1024,               // stack size in bytes
    my_thread_fn,       // entry function
    NULL, NULL, NULL,   // arg1, arg2, arg3
    5,                  // priority (0=highest preemptive, higher=lower)
    0,                  // options (e.g. K_ESSENTIAL)
    0                   // delay before start (ms), 0 = start immediately
);
```

---

### 1.6 k_timer vs k_msleep — why k_msleep is WRONG for 100Hz timing

**The drift math problem:**

Suppose your goal is to fire a callback every 10ms (100Hz).

```
Naive approach with k_msleep(10):
  t=0:    work starts  → consumes 0.3ms
  t=0.3:  k_msleep(10) starts
  t=10.3: k_msleep(10) ends  → NEXT cycle starts at 10.3ms
  t=10.6: work done again
  t=20.6: second k_msleep ends  → NEXT cycle starts at 20.6ms
```

After 100 cycles: expected time = 1000ms, actual time = 1030ms. **3% drift accumulates forever.**

This happens because `k_msleep(N)` means "sleep for N milliseconds *after* the function is called" — it doesn't know or care about when the *previous* iteration ended.

**The correct tool: k_timer**

`k_timer` fires at an absolute wall-clock interval. It does not care how long your work took. When the timer expires, it signals a semaphore (or calls a callback). Your thread wakes up, does work, then *blocks waiting for the next semaphore*. The timer fires again at exactly 10ms from the *previous* fire — not from when the work finished.

```
Correct approach with k_timer + k_sem:
  t=0:    timer fires → k_sem_give()  → thread wakes, work starts (0.3ms)
  t=0.3:  work done → k_sem_take() blocks
  t=10.0: timer fires AGAIN (exactly 10ms from t=0) → thread wakes
  t=10.3: work done
  t=20.0: timer fires at exactly 20ms from t=0
```

The timer fires based on its own absolute schedule. Your 0.3ms work time is irrelevant to when the next fire happens.

**The caveat:** If your work takes *longer than 10ms*, you will miss ticks (the semaphore count will build up). This is intentional — it's an overrun you need to detect and handle. With k_msleep you'd just silently fall further behind with no warning.

---

### 1.7 Stack overflow in embedded — why it's silent by default

**The analogy: writing past the edge of a whiteboard onto the wall**

In desktop programming, the OS gives each process a virtual memory stack and sets up a guard page — write past it, get a segfault immediately. In embedded, there is no MMU (or it's disabled). The stack is just a region of SRAM. If you write past the end of it, you silently overwrite whatever was allocated adjacent to it in memory — often another thread's stack, or global variables, or the kernel control structures.

**What happens in practice:**
1. Thread A is given a 1024-byte stack
2. Thread A calls LOG_INF() which internally uses printf-like formatting — this alone can use 300-600 bytes of stack
3. Thread A starts a deep call chain: `thread_fn → log_msg → log_output_flush → uart_poll_out`
4. Stack usage exceeds 1024 bytes
5. The overflow silently overwrites adjacent memory
6. Thread B starts behaving strangely, or the system hard-faults minutes later with no obvious connection

**Why it's silent by default:** Checking the stack on every function call would require reading/writing a guard word (slow). Zephyr disables this check by default for performance.

**How to enable stack checking:**
```ini
# prj.conf — enable both of these during development
CONFIG_STACK_SENTINEL=y        # writes a magic value at stack boundary, checks it periodically
CONFIG_THREAD_STACK_INFO=y     # enables k_thread_stack_space_get() for profiling
```

With `CONFIG_STACK_SENTINEL=y`, Zephyr will detect the overflow and raise a kernel panic with a useful error instead of silent corruption.

**How to size your stack:**
- Start with 2048 bytes for any thread that calls LOG functions
- Call `k_thread_stack_space_get(my_thread, &unused)` after a few minutes of operation
- The actual high-water mark = stack_size − unused
- Target: at least 20% headroom above your measured high-water mark

---

### 1.8 IRQ priority vs thread priority — completely different number systems

This is a trap that catches nearly everyone. They sound similar but they are orthogonal concepts.

**Thread priority: who the scheduler runs next (0–14, lower = higher)**
- This is a software concept managed by the Zephyr kernel
- Controls which *thread* gets the CPU at the next scheduling point
- Range: 0 (highest) to CONFIG_NUM_PREEMPT_PRIORITIES−1 (lowest, default 14)
- Visible in `k_thread_priority_set()`, `K_THREAD_DEFINE(..., priority, ...)`

**IRQ priority: who the CPU serves next when hardware signals occur (0–15 on ARM, lower = higher)**
- This is a *hardware* concept managed by the CPU's NVIC (Nested Vectored Interrupt Controller)
- Controls which *interrupt handler* runs when two hardware events arrive simultaneously
- Has NOTHING to do with threads — ISRs don't go through the thread scheduler
- Range: on ARM Cortex-M, typically 0–15, where 0 is highest and 15 is lowest
- Zephyr maps these internally; you set them in devicetree (`interrupts = <n priority>`)

**The critical interaction:**
- An ISR can preempt ANY thread — even a thread at priority 0
- An ISR cannot be preempted by a thread under any circumstances
- Zephyr uses `CONFIG_ZERO_LATENCY_IRQ` for ISRs that must not be masked even briefly

**Why this matters for your projects:**
- The UART interrupt that receives shell characters runs at IRQ priority ~5
- Your main thread runs at thread priority 0
- Even though thread priority 0 is "highest thread priority", a UART ISR will interrupt it
- These two priority numbers (0 and 5) are on completely different scales and cannot be compared

**Practical rule:** Don't set thread priorities without understanding IRQ priorities. If your timer callback is running in ISR context, it plays by IRQ rules (no sleeping, no k_sem_take, no LOG). If it runs in thread context (k_timer + k_sem pattern), it plays by thread rules.

---

### 1.9 Zephyr log levels — when to use each

Zephyr's logging macro maps directly to a severity level that can be filtered at compile time or runtime.

| Macro | Level | When to use | Example |
|-------|-------|-------------|---------|
| `LOG_DBG(...)` | 4 — Debug | Verbose state, every-tick data, values you only need during development | `LOG_DBG("dt_ms=%u", dt);` |
| `LOG_INF(...)` | 3 — Info | Normal significant events: startup, mode changes, periodic summaries | `LOG_INF("Shell ready");` |
| `LOG_WRN(...)` | 2 — Warning | Unexpected but recoverable conditions | `LOG_WRN("Timer overrun: dt=%u", dt);` |
| `LOG_ERR(...)` | 1 — Error | Failures that degrade functionality | `LOG_ERR("GPIO init failed: %d", ret);` |
| `LOG_PANIC()` | 0 — Panic | Unrecoverable state, immediately flush log and halt | Rarely called directly |

**Compile-time filtering:** If `CONFIG_LOG_DEFAULT_LEVEL=3` (INF), all `LOG_DBG()` calls compile to nothing — zero overhead. This means you can leave debug logging in your code and just change the Kconfig level.

**Runtime filtering:** With `CONFIG_LOG_RUNTIME_FILTERING=y`, you can change the active level via shell:
```
uart:~$ log enable dbg my_module
```

**Best practice for your projects:**
- Use `LOG_DBG` for the dt_ms measurement — 100 messages/second, you only want this when debugging
- Use `LOG_INF` for the "system started" announcement
- Use `LOG_WRN` for timing overruns
- Never use `printf()` in Zephyr — it bypasses the log subsystem's buffering and filtering

---

## PART 2 — Code Patterns

---

### 2.1 Complete Minimal Blinky

**File structure:**
```
my_blinky/
├── CMakeLists.txt
├── prj.conf
└── src/
    └── main.c
```

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(my_blinky)

target_sources(app PRIVATE src/main.c)
```

**prj.conf:**
```ini
CONFIG_GPIO=y

# Blink interval in milliseconds — controlled by Kconfig
# Change this value and rebuild to change blink rate
CONFIG_BLINK_PERIOD_MS=500
```

**src/main.c:**
```c
#include <zephyr/kernel.h>
#include <zephyr/drivers/gpio.h>

/* DT_ALIAS(led0) looks up the 'led0' alias in devicetree.
 * The macro resolves to the actual GPIO device + pin at compile time.
 * No pin numbers in C code. */
#define LED0_NODE DT_ALIAS(led0)

static const struct gpio_dt_spec led = GPIO_DT_SPEC_GET(LED0_NODE, gpios);

int main(void)
{
    int ret;

    if (!gpio_is_ready_dt(&led)) {
        return -ENODEV;
    }

    ret = gpio_pin_configure_dt(&led, GPIO_OUTPUT_ACTIVE);
    if (ret < 0) {
        return ret;
    }

    while (1) {
        gpio_pin_toggle_dt(&led);
        /* CONFIG_BLINK_PERIOD_MS is a compile-time constant from prj.conf */
        k_msleep(CONFIG_BLINK_PERIOD_MS);
    }

    return 0;
}
```

**Kconfig file** (required to expose your custom CONFIG symbol):
```kconfig
# Kconfig — must exist in project root to define custom symbols
mainmenu "My Blinky"

config BLINK_PERIOD_MS
    int "LED blink period in milliseconds"
    default 500
    range 50 5000
    help
      Controls the on/off duration of the LED blink in milliseconds.
      Rebuild required after changing this value.
```

**Build and flash:**
```bash
cd ~/zephyrproject
west build -b nucleo_h743zi2 my_blinky/
west flash
```

---

### 2.2 Board Overlay — Remapping an LED to a Specific GPIO Pin

Create this file at `my_blinky/boards/nucleo_h743zi2.overlay`:

```dts
/*
 * Override led0 alias to use PB0 (Arduino D26 on Nucleo-H743ZI2)
 * instead of the default board LED.
 *
 * The '&' prefix means we are MODIFYING an existing node, not creating new one.
 * gpios = <&gpiob 0 GPIO_ACTIVE_HIGH> means:
 *   - &gpiob  : reference to the GPIOB controller node
 *   - 0       : pin number within that port (PB0)
 *   - GPIO_ACTIVE_HIGH : pin state HIGH means LED is ON
 */
/ {
    aliases {
        led0 = &my_custom_led;
    };

    leds {
        compatible = "gpio-leds";
        my_custom_led: led_custom_0 {
            gpios = <&gpiob 0 GPIO_ACTIVE_HIGH>;
            label = "Custom LED PB0";
        };
    };
};

/*
 * If you want to keep the existing board LED node AND add a new one,
 * use status = "disabled" on the old one:
 *
 * &green_led_1 {
 *     status = "disabled";
 * };
 */
```

After adding this file, rebuild — no changes to main.c needed. The `led0` alias now points to PB0.

---

### 2.3 LOG_MODULE_REGISTER and Using LOG_INF

```c
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

/*
 * LOG_MODULE_REGISTER(module_name, log_level)
 *
 * - module_name: appears in log output prefix, e.g. "[00:00:01.234 INF] blinky: ..."
 * - log_level:   per-module maximum level. LOG_LEVEL_DBG means all 4 levels active.
 *                Can be overridden at compile time via CONFIG_LOG_OVERRIDE_LEVEL.
 *
 * This macro must appear ONCE per .c file, at file scope (not inside a function).
 */
LOG_MODULE_REGISTER(blinky, LOG_LEVEL_DBG);

int main(void)
{
    LOG_INF("Blinky started. Blink period: %d ms", CONFIG_BLINK_PERIOD_MS);

    int cycle = 0;
    while (1) {
        gpio_pin_toggle_dt(&led);
        k_msleep(CONFIG_BLINK_PERIOD_MS);

        cycle++;
        /* LOG_DBG only emits if compiled log level >= 4 (DBG).
         * At LOG_LEVEL_INF (3), this line compiles to nothing. */
        LOG_DBG("Cycle %d complete", cycle);

        if (cycle % 100 == 0) {
            LOG_INF("100 blink cycles completed");
        }
    }
}
```

**Required prj.conf entries for logging:**
```ini
CONFIG_LOG=y
CONFIG_LOG_DEFAULT_LEVEL=3     # 3=INF; change to 4 for DBG during development
CONFIG_LOG_BACKEND_UART=y      # send log output to UART
CONFIG_UART_CONSOLE=y          # system console on UART
```

---

### 2.4 Registering a Shell Command

```c
#include <zephyr/shell/shell.h>

/*
 * Shell command handler function signature:
 * - shell: opaque shell context (pass to shell_print/shell_error)
 * - argc:  number of arguments (including command name)
 * - argv:  argument strings, argv[0] = command name, argv[1] = first arg
 *
 * Returns 0 on success, negative errno on error.
 */
static int cmd_blink_set(const struct shell *sh, size_t argc, char **argv)
{
    if (argc != 2) {
        shell_error(sh, "Usage: blink set <period_ms>");
        return -EINVAL;
    }

    /* strtol for safe string-to-int conversion */
    char *end;
    long period = strtol(argv[1], &end, 10);

    if (*end != '\0' || period < 50 || period > 5000) {
        shell_error(sh, "Invalid period. Range: 50–5000 ms");
        return -EINVAL;
    }

    blink_period_ms = (uint32_t)period;
    shell_print(sh, "Blink period set to %ld ms", period);
    return 0;
}

/* Show current period */
static int cmd_blink_get(const struct shell *sh, size_t argc, char **argv)
{
    shell_print(sh, "Current blink period: %u ms", blink_period_ms);
    return 0;
}

/*
 * SHELL_STATIC_SUBCMD_SET_CREATE creates a subcommand set.
 * SHELL_CMD_REGISTER registers the top-level command.
 *
 * This creates a two-level command: "blink set 500" and "blink get"
 */
SHELL_STATIC_SUBCMD_SET_CREATE(blink_cmds,
    SHELL_CMD_ARG(set, NULL, "Set blink period: blink set <ms>", cmd_blink_set, 2, 0),
    SHELL_CMD(get, NULL, "Get current blink period", cmd_blink_get),
    SHELL_SUBCMD_SET_END
);

SHELL_CMD_REGISTER(blink, &blink_cmds, "Blink control commands", NULL);
```

**Required prj.conf:**
```ini
CONFIG_SHELL=y
CONFIG_SHELL_BACKEND_SERIAL=y
CONFIG_UART_CONSOLE=y
CONFIG_SERIAL=y

# Shell uses uart0 by default — ensure your overlay points uart0 to the right pins
# See section 2.5 if you need to remap
```

**Connect and test:**
```bash
# Linux: find CP2102 device
ls /dev/ttyUSB*

# Open minicom (115200 8N1, no hardware flow control)
minicom -D /dev/ttyUSB0 -b 115200

# In minicom, press Enter to see the prompt:
uart:~$ blink set 250
Blink period set to 250 ms
uart:~$ blink get
Current blink period: 250 ms
```

---

### 2.5 Overlay: Correct UART Instance for Shell vs Log

The Nucleo-H743ZI2 has multiple UARTs. A common mistake is assigning both the shell and the system console to the same UART — they fight and produce garbage.

**Correct setup: shell on USART3 (Arduino header), log/console on USART3 as well (single UART)**

```dts
/* boards/nucleo_h743zi2.overlay */

/*
 * The Nucleo-H743ZI2 uses USART3 connected to the built-in ST-Link VCP
 * as the default console. This is uart0 in Zephyr's abstraction.
 *
 * If you want to use the CP2102 on separate pins (e.g. PA9/PA10 = USART1),
 * configure that as a SECOND uart and route ONLY the shell there,
 * keeping the console on uart0 for easy debugging.
 */

/* Option A: Single UART for both shell and console (simpler) */
/* No overlay changes needed — default nucleo_h743zi2 board already wires
   USART3 to uart0 and connects it to the ST-Link USB VCP. */

/* Option B: Shell on CP2102 (USART1 on PA9/PA10), console on ST-Link (USART3) */
&usart1 {
    status = "okay";
    current-speed = <115200>;
    pinctrl-0 = <&usart1_tx_pa9 &usart1_rx_pa10>;
    pinctrl-names = "default";
};

/* In prj.conf for Option B, add: */
/* CONFIG_SHELL_BACKEND_SERIAL=y */
/* CONFIG_UART_SHELL_ON_DEV_NAME="USART_1" */
```

**Key rule:** Check `west build -t menuconfig` and search for SHELL — ensure `UART_SHELL_ON_DEV_NAME` and `UART_CONSOLE_ON_DEV_NAME` are NOT set to the same string. If they are, you'll see garbled output because both backends race to write to the same UART from different places.

---

### 2.6 K_THREAD_DEFINE — Stack, Priority, Entry Function

```c
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(worker, LOG_LEVEL_DBG);

/* Thread entry function — must match this signature */
static void worker_thread(void *arg1, void *arg2, void *arg3)
{
    ARG_UNUSED(arg1);
    ARG_UNUSED(arg2);
    ARG_UNUSED(arg3);

    LOG_INF("Worker thread started");

    while (1) {
        /* Do periodic work */
        LOG_DBG("Worker tick");
        k_msleep(1000);
    }
}

/*
 * K_THREAD_DEFINE(name, stack_sz, entry, p1, p2, p3, prio, opts, delay)
 *
 * name:     C identifier for the thread — used by kernel tools
 * stack_sz: in bytes. For threads calling LOG: use minimum 2048.
 * entry:    thread function pointer
 * p1,p2,p3: passed as arg1,arg2,arg3 to entry function
 * prio:     0=highest preemptive; use 5-10 for background threads
 * opts:     usually 0. K_ESSENTIAL means kernel panics if thread dies.
 * delay:    ms before thread starts. 0 = start when scheduler runs.
 *
 * This macro allocates the stack array and the k_thread struct at COMPILE TIME.
 * The thread is automatically started — no k_thread_start() call needed.
 */
K_THREAD_DEFINE(
    worker_tid,     /* thread identifier — you can pass this to k_thread_join() */
    2048,           /* stack size — 2048 for any thread using LOG */
    worker_thread,  /* entry function */
    NULL,           /* arg1 */
    NULL,           /* arg2 */
    NULL,           /* arg3 */
    5,              /* priority (preemptive, lower than main's default 0) */
    0,              /* options */
    0               /* start delay in ms */
);
```

**Checking stack usage at runtime:**
```c
/* After thread has been running a while, check its high-water mark */
size_t unused;
k_thread_stack_space_get(&worker_tid, &unused);
LOG_INF("Worker stack unused: %zu bytes out of %d", unused, 2048);
/* If unused is < 200, your stack is too small */
```

---

### 2.7 k_timer + k_sem Pattern for 100Hz Timing

```c
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(timer_thread, LOG_LEVEL_DBG);

/* Semaphore: initialized to 0 (no tokens), max 1 token.
 * Timer ISR gives one token every 10ms.
 * Thread takes one token and runs. */
K_SEM_DEFINE(timer_sem, 0, 1);

/*
 * Timer expiry function — called from ISR context (or a high-priority system thread).
 * Keep it SHORT. No sleeping, no heavy computation, no LOG macros.
 * Just give the semaphore to wake the worker thread.
 */
static void timer_expiry_fn(struct k_timer *timer_id)
{
    ARG_UNUSED(timer_id);
    k_sem_give(&timer_sem);
}

/* Define the timer (expiry_fn, stop_fn) */
K_TIMER_DEFINE(my_timer, timer_expiry_fn, NULL);

/* Drift measurement state */
static uint32_t last_tick_ms = 0;

static void timer_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    /* Start timer: first fire in 10ms, then every 10ms */
    k_timer_start(&my_timer, K_MSEC(10), K_MSEC(10));

    /* Record start time for first dt calculation */
    last_tick_ms = k_uptime_get_32();

    LOG_INF("100Hz timer thread started");

    while (1) {
        /* Block until timer fires — no busy-waiting, no CPU waste */
        k_sem_take(&timer_sem, K_FOREVER);

        /* Measure ACTUAL elapsed time since last tick */
        uint32_t now_ms = k_uptime_get_32();
        uint32_t dt_ms = now_ms - last_tick_ms;
        last_tick_ms = now_ms;

        /* Log at DBG level — high volume, only useful during profiling */
        LOG_DBG("dt_ms=%u (target=10)", dt_ms);

        /* Warn if we drifted more than 1ms */
        if (dt_ms > 11 || dt_ms < 9) {
            LOG_WRN("Timer drift: dt=%u ms", dt_ms);
        }

        /* === Your 100Hz work goes here === */
    }
}

K_THREAD_DEFINE(
    timer_tid,
    2048,           /* 2048 minimum — LOG inside thread needs headroom */
    timer_thread,
    NULL, NULL, NULL,
    5,              /* priority 5 — background worker */
    0,
    0
);
```

---

### 2.8 Measuring dt_ms with k_uptime_get_32()

```c
#include <zephyr/kernel.h>

/*
 * k_uptime_get_32() returns milliseconds since system boot as uint32_t.
 * Wraps around after ~49.7 days (2^32 ms).
 * Use uint32_t subtraction — subtraction correctly handles wrap-around
 * as long as the interval fits in 32 bits (it does for <49 days).
 *
 * k_uptime_get() returns int64_t — use this if you need >49 day intervals.
 */

/* Pattern 1: Simple one-shot elapsed time measurement */
uint32_t start = k_uptime_get_32();
do_some_work();
uint32_t elapsed = k_uptime_get_32() - start;
LOG_INF("Work took %u ms", elapsed);

/* Pattern 2: Continuous dt measurement in a loop (for drift analysis) */
uint32_t prev = k_uptime_get_32();
while (1) {
    /* ... wait for next tick ... */
    uint32_t now = k_uptime_get_32();
    uint32_t dt = now - prev;   /* safe even if counter wrapped */
    prev = now;

    if (dt != 10) {
        LOG_WRN("Unexpected dt: %u ms (expected 10)", dt);
    }
}

/* Pattern 3: Jitter tracking over N samples */
uint32_t max_dt = 0;
uint32_t min_dt = UINT32_MAX;
uint32_t sample_count = 0;

/* Inside the loop: */
if (dt > max_dt) { max_dt = dt; }
if (dt < min_dt) { min_dt = dt; }
sample_count++;

if (sample_count == 10000) {
    LOG_INF("Jitter over 10000 ticks: min=%u ms, max=%u ms, spread=%u ms",
            min_dt, max_dt, max_dt - min_dt);
    max_dt = 0;
    min_dt = UINT32_MAX;
    sample_count = 0;
}
```

---

## PART 3 — Gotcha Table

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|-------------|-----------------|-----|
| `west build` fails: `Board nucleo_h743zi2 not found` | ZEPHYR_BASE not set or wrong board name | `echo $ZEPHYR_BASE` — should point to zephyr/ dir. Run `west boards | grep nucleo` | `source ~/zephyrproject/zephyr/zephyr-env.sh` before building |
| LED never blinks — no errors | Overlay references wrong GPIOB pin, or `led0` alias not defined | `west build -t guiconfig` → check DT nodes. Or add `LOG_INF` at entry and check UART output | Add overlay file, define `aliases { led0 = &your_led; }` |
| Board flashes fine but no UART output | System console not configured for UART, or wrong baud rate | Check prj.conf for `CONFIG_UART_CONSOLE=y` and `CONFIG_UART_CONSOLE_ON_DEV_NAME` | Add `CONFIG_UART_CONSOLE=y`, connect at 115200 |
| Shell prompt doesn't appear (blank terminal) | Shell not enabled, or UART assignment conflict | Check `CONFIG_SHELL=y` in prj.conf. Check if shell and console share the same uart | Verify `CONFIG_UART_SHELL_ON_DEV_NAME` ≠ `CONFIG_UART_CONSOLE_ON_DEV_NAME` |
| Garbled/garbage output on serial | Shell and log backends racing on same UART | Output looks like interleaved partial strings | Route shell to USART1 (CP2102), keep console on USART3 (ST-Link) |
| Thread silently stops after a few seconds | Stack overflow — LOG calls inside thread exceeded 1KB default | Add `CONFIG_STACK_SENTINEL=y` to prj.conf — will see kernel panic instead | Increase thread stack to 2048+, or check high-water mark with `k_thread_stack_space_get` |
| `LOG_DBG` messages never appear even with `CONFIG_LOG=y` | Default log level is 3 (INF), filtering out DBG | Check `CONFIG_LOG_DEFAULT_LEVEL` in prj.conf | Set `CONFIG_LOG_DEFAULT_LEVEL=4` or use `log enable dbg module_name` in shell |
| 100Hz thread drifts — average interval is 10.3ms not 10ms | Using `k_msleep(10)` instead of `k_timer` | Measure dt_ms over 1000 ticks — drift accumulates vs stays near 10ms | Change to k_timer + k_sem pattern (section 2.7) |
| Timer thread fires correctly but LOG messages missing | LOG buffer overflow — 100 msg/sec flooding a small buffer | Check `CONFIG_LOG_BUFFER_SIZE` | Increase `CONFIG_LOG_BUFFER_SIZE=4096`, or drop to `LOG_INF` every 100 ticks instead of `LOG_DBG` every tick |
| `west flash` fails: `Error: open failed` | ST-Link driver issue or device not found | `lsusb | grep STM`, check dmesg for device connection | Reconnect USB, `west flash --runner=openocd` to try alternate runner |
| Hard fault with no useful message | Stack sentinel not enabled — overflow corrupted kernel structures | Enable `CONFIG_STACK_SENTINEL=y` and `CONFIG_FAULT_DUMP=2` | Both: gives stack trace with function names on hard fault |
| Custom Kconfig symbol `CONFIG_BLINK_PERIOD_MS` not recognized | Missing `Kconfig` file in project root | `west build` will warn about unknown symbols | Create `Kconfig` file in project root defining the symbol (see section 2.1) |
| `gpio_pin_configure_dt` returns -ENODEV | GPIO controller not enabled in prj.conf | `CONFIG_GPIO=y` is missing | Add `CONFIG_GPIO=y` to prj.conf |
| Shell command registered but not found at runtime | Missing `CONFIG_SHELL_CMDS=y` or command file not compiled | Check `SHELL_CMD_REGISTER` is in a compiled `.c` file; check `target_sources` in CMakeLists | Add the .c file to `target_sources(app PRIVATE ...)` |
| k_timer fires but semaphore count keeps growing | Work takes longer than timer period (overrun) | Compare dt_ms against timer period — should be ~10ms | Reduce work in thread, increase timer period, or add overrun counter and skip ticks |

---

## Quick Reference Card

```
Build:          west build -b nucleo_h743zi2 <app_dir>/
Clean build:    west build -b nucleo_h743zi2 <app_dir>/ --pristine
Flash:          west flash
Monitor serial: minicom -D /dev/ttyUSB0 -b 115200
Config GUI:     west build -t menuconfig
Devicetree GUI: west build -t guiconfig

Priority scale (Zephyr threads):  0 (highest) → 14 (lowest preemptive)
Priority scale (Zephyr coop):    -1 → -16 (cooperative, never preempted)
Priority scale (ARM IRQ):         0 (highest) → 15 (lowest) — DIFFERENT SYSTEM

Stack minimum rules:
  Thread with no LOG:        512 bytes
  Thread calling LOG_DBG:   2048 bytes
  Thread calling shell API: 2048 bytes

Timer accuracy:
  k_msleep(10) in loop:    drifts (work time not accounted)
  k_timer + k_sem:         no drift (fires on absolute schedule)
```
