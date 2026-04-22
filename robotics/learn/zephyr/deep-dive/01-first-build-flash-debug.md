# 01 — First Build, Flash, and Debug Session
### STM32 Nucleo-H743ZI2 · Zephyr RTOS · ST-Link V3

**Status:** 🟡 HARDWARE-GATED — fill this in when Nucleo arrives  
**Prerequisite reading:** `zephyr/01_zephyr_intro.md`, `zephyr/00-mastery-plan.md` Sections 1–3  
**Unlocks:** `02-spi-slave-first-frame.md`  
**Time budget:** ~6 hours first session

---

## Why This File Exists

You have 17 Zephyr study files and know the theory cold.
The first hardware session is where 80% of the pain is not from code — it's from:
- Toolchain not finding the board
- west flash failing silently
- Logic analyzer not triggering
- GDB attaching to the wrong core

This file is a prescription for the first 6 hours. Follow it in order.
If something fails at step N, the answer is in the `## Failure Modes` section at the end.

---

## Software Setup (Do This Before Hardware Arrives)

These steps have no hardware dependency. Complete them now so session 1 is pure hardware.

### Step 1: Install west + Zephyr SDK

```bash
# Python venv (don't pollute system Python)
python3 -m venv ~/zephyrproject/.venv
source ~/zephyrproject/.venv/bin/activate

# west — the Zephyr meta-tool
pip install west

# Init the workspace
west init ~/zephyrproject
cd ~/zephyrproject
west update           # ~300MB download, takes 5-10min

# Zephyr SDK (ARM toolchain, OpenOCD, etc.)
wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.17.0/zephyr-sdk-0.17.0_linux-x86_64.tar.xz
tar xvf zephyr-sdk-0.17.0_linux-x86_64.tar.xz
cd zephyr-sdk-0.17.0
./setup.sh           # installs arm-zephyr-eabi + all host tools
```

**Verify now (no hardware needed):**
```bash
source ~/zephyrproject/.venv/bin/activate
cd ~/zephyrproject/zephyr
west build -b nucleo_h743zi2 samples/basic/blinky -- -DCONFIG_SERIAL=y
# Expected: build/zephyr/zephyr.elf and zephyr.bin produced, no errors
```

> If this fails, fix it before the hardware arrives. The error is almost always
> missing `cmake`, missing `dtc`, or wrong Python version. Read the output carefully.

### Step 2: Install udev rules

```bash
# ST-Link udev rule (lets non-root users flash)
sudo cp ~/zephyrproject/zephyr/boards/common/openocd.cfg /etc/udev/rules.d/
# OR the direct way:
sudo tee /etc/udev/rules.d/99-openocd.rules > /dev/null <<'EOF'
ATTRS{idVendor}=="0483", ATTRS{idProduct}=="374b", MODE="0664", GROUP="plugdev", TAG+="uaccess"
ATTRS{idVendor}=="0483", ATTRS{idProduct}=="374e", MODE="0664", GROUP="plugdev", TAG+="uaccess"
EOF
sudo udevadm control --reload-rules
sudo usermod -aG plugdev $USER   # log out and back in
```

### Step 3: Verify GDB works

```bash
arm-zephyr-eabi-gdb --version
# Expected: GNU gdb (Zephyr SDK ...) 14.x.x
```

---

## Session 1 Checklist (Hardware Present)

Work through these in order. Check each box before moving on.

### A. Unbox + Visual Inspection

- [ ] Nucleo board: no bent pins on 144-pin headers
- [ ] Two micro-USB ports present (CN1 = ST-Link, CN13 = USB OTG) — only CN1 is needed initially
- [ ] ST-Link version sticker: **V3** is on H743ZI2. Older boards may need ST-Link firmware update
- [ ] Saleae Logic 8 cable: small USB-C end to Logic 8, larger USB-A end to computer

### B. First Connection — Verify ST-Link Enumerates

```bash
lsusb | grep "STMicroelectronics"
# Expected:
# Bus 001 Device 003: ID 0483:374e STMicroelectronics STLINK-V3

# ST-Link info:
st-info --probe
# Expected:
# Found 1 stlink programmers
# chipid: 0x0450 (STM32H7xx)
# flash: 2097152 (pagesize: 131072)
```

If `st-info` shows `Found 0`, check:
1. `lsusb` — USB enumerated at all?
2. `dmesg | tail -20` — udev/permission error?
3. udev rules installed in step 2 above
4. Did you log out and back in after `usermod`?

### C. Flash Blinky

```bash
cd ~/zephyrproject/zephyr
west flash -b nucleo_h743zi2 -- --openocd /path/to/openocd
# Expected: "** Programming Finished **" and green LED starts blinking at ~500ms
```

LED locations on Nucleo-H743ZI2:
- **LD1 (green)**: PA5 — default Blinky LED
- **LD2 (yellow)**: PE1
- **LD3 (red)**: PB14 — hard fault / error indicator

> **If LD3 (red) blinks instead of green:** HardFault on boot. Most common cause:
> wrong `DTCMRAM` vs `SRAM` placement. Check that you used `nucleo_h743zi2` not `nucleo_h743zi`.
> These are different boards; wrong target = wrong linker script = HardFault.

### D. Open Serial Console

```bash
# Find the Nucleo's serial port (ST-Link also exposes a virtual COM port)
ls /dev/ttyACM*
# Expected: /dev/ttyACM0

# Connect
picocom -b 115200 /dev/ttyACM0
# Or: screen /dev/ttyACM0 115200
```

After reset, you should see:
```
*** Booting Zephyr OS build v3.7.0 ***
```
If nothing: check `CONFIG_SERIAL=y` was set in the build command.

### E. Rebuild with Shell + LOG enabled

```bash
west build -b nucleo_h743zi2 samples/subsys/shell/shell_module -- \
  -DCONFIG_SERIAL=y \
  -DCONFIG_SHELL=y \
  -DCONFIG_SHELL_BACKEND_SERIAL=y \
  -DCONFIG_LOG=y \
  -DCONFIG_LOG_BACKEND_UART=y
west flash
```

Zephyr shell in `picocom`:
```
uart:~$ kernel version
Zephyr version 3.7.0

uart:~$ kernel threads
Threads:
  0x200003b0 (sysworkq):
    options: 0x0, priority: -1 cooperative
    state: pending
  0x200003d0 (main):
    options: 0x0, priority: 0 preemptive
    state: running
```

**Key commands to know:**
```
kernel threads      # show all threads + stack usage
kernel stacks       # show stack high-water marks
log go              # flush any buffered log output
log list            # show all registered log modules
```

### F. Attach GDB

In one terminal:
```bash
# Start OpenOCD server
openocd -f interface/stlink.cfg -f target/stm32h7x.cfg
# Expected: "Listening on port 3333 for gdb connections"
```

In another terminal:
```bash
arm-zephyr-eabi-gdb build/zephyr/zephyr.elf
(gdb) target extended-remote :3333
(gdb) monitor reset halt      # halt the MCU
(gdb) info registers          # see all register values
(gdb) list main               # show source around main()
(gdb) break main              # set breakpoint
(gdb) continue
```

**Key GDB commands for Zephyr debugging:**
```gdb
monitor reset halt            # halt without losing connection
info registers                # all ARM registers including CFSR (crash register)
x/4wx 0x20000000              # inspect memory at SRAM start
bt                            # backtrace (call stack)
p z_current                   # current Zephyr thread pointer
p *z_current                  # inspect current thread struct
```

### G. Decode a HardFault (practice on purpose)

Add this to a test app to trigger a null pointer dereference:
```c
void main(void) {
    volatile int *p = (int *)0xDEADBEEF;
    *p = 42;   // HardFault
}
```

In GDB after the fault:
```gdb
info registers
# Look at:
# PC  = address of faulting instruction
# LR  = return address (where it was called from)
# CFSR = Configurable Fault Status Register
#   bits [7:0] = MemManage faults
#   bits [15:8] = BusFault
#   bits [31:16] = UsageFault

# Decode CFSR:
p/x $xpsr
monitor reg cfsr              # OpenOCD: show CFSR directly
```

---

## Pre-Read for Session 2

Before the next session (`02-spi-slave-first-frame.md`):

1. Re-read `zephyr/05_spi_slave_dma.md` — especially the double-buffering section
2. Re-read `00-mastery-plan.md` Section 3 "SPI Slave DMA Pre-Arming Race"
3. Wire up the Saleae Logic 8 and verify the Saleae Logic 2 app can capture a digital channel
4. Configure SPI decoder in Logic 2: **SPI** · CPOL=0 · CPHA=0 · CS active-low

---

## Failure Modes Reference

| Symptom | Most Likely Cause | Fix |
|---------|------------------|-----|
| `west flash` hangs at "Waiting for target..." | SWD pins not connected; wrong USB port used | Use CN1 (ST-Link USB), not CN13 |
| `st-info --probe` shows 0 stlinks | udev rules not applied | `udevadm control --reload-rules` + re-plug |
| LD3 (red) blinks after flash | HardFault on boot | Wrong board target; check `nucleo_h743zi2` spelling |
| GDB shows PC in `z_fatal_error` | Stack overflow or null dereference | `info registers` → check CFSR; `bt` for call stack |
| Shell prompt never appears | `CONFIG_SERIAL` missing or wrong baud rate | Rebuild with `-DCONFIG_SERIAL=y`; verify `picocom -b 115200` |
| Build fails: `CMake not found` | Zephyr SDK setup incomplete | Re-run `./setup.sh` in SDK dir |
| `arm-zephyr-eabi-gdb: command not found` | PATH not set | `export PATH=~/zephyr-sdk-0.17.0/arm-zephyr-eabi/bin:$PATH` |

---

## What to Write Here After the Session

When you complete this session, add a section:

```markdown
## Session Notes — [DATE]

### What Worked
- ...

### What Didn't Work
- ...

### Surprises
- ...

### Time Spent
- Build + flash: Xmin
- GDB setup: Xmin
- HardFault decode exercise: Xmin

### Questions to Investigate Next Time
- ...
```
