# HardFault Decode Guide
### From register dump to faulting source line — systematic ARM Cortex-M fault analysis

---

## Overview

A HardFault is the ARM Cortex-M catch-all exception for bus faults, memory faults, and
usage faults that can't be handled by their specific handlers. This guide covers:

1. Capturing the register state
2. Reading CFSR/HFSR/MMFAR/BFAR
3. Mapping the fault PC to source line
4. The five most common HardFault causes in Zephyr

---

## 1. The HardFault Register Set

When a fault occurs, the ARM core pushes a stack frame and the fault status registers
are latched. CFSR and HFSR bits are **sticky** — they remain set until explicitly cleared
by writing 1 to the bit (W1C). MMFAR and BFAR validity bits (`MMARVALID` / `BFARVALID`)
are automatically cleared when a new fault of the same type fires.

### 1.1 Configurable Fault Status Register (CFSR)

`CFSR` is at address `0xE000ED28` (32-bit, read/write to clear).

It is split into three sub-registers:

```
Bits [31:16] = UFSR (UsageFault Status Register)
Bits [15:8]  = BFSR (BusFault Status Register)
Bits [7:0]   = MMFSR (MemManage Fault Status Register)
```

#### MMFSR (MemManage) — bits [7:0]:

| Bit | Name | Meaning |
|-----|------|---------|
| 7 | MMARVALID | MMFAR holds a valid fault address |
| 4 | MSTKERR | Stacking for an exception caused the fault |
| 3 | MUNSTKERR | Unstacking after exception caused the fault |
| 1 | DACCVIOL | Data access violation (MMFAR is valid) |
| 0 | IACCVIOL | Instruction fetch from non-executable region |

#### BFSR (BusFault) — bits [15:8]:

| Bit | Name | Meaning |
|-----|------|---------|
| 15 | BFARVALID | BFAR holds a valid fault address |
| 13 | LSPERR | Lazy FP state push caused a BusFault (Cortex-M4/M7/M33 with FPU enabled) |
| 12 | STKERR | Stacking caused the fault |
| 11 | UNSTKERR | Unstacking caused the fault |
| 10 | IMPRECISERR | Imprecise data bus error (async — BFAR may not be valid) |
| 9 | PRECISERR | Precise data bus error (BFAR is valid) |
| 8 | IBUSERR | Instruction fetch bus error |

#### UFSR (UsageFault) — bits [31:16]:

| Bit | Name | Meaning |
|-----|------|---------|
| 25 | DIVBYZERO | Integer divide by zero (if enabled in CCR.DIV_0_TRP) |
| 24 | UNALIGNED | Unaligned access (if enabled in CCR.UNALIGN_TRP) |
| 19 | NOCP | No coprocessor — FPU instruction with FPU disabled |
| 18 | INVPC | Invalid PC on exception return (corrupt LR) |
| 17 | INVSTATE | Invalid state — EPSR.T bit wrong (Thumb mode error) |
| 16 | UNDEFINSTR | Undefined instruction |

---

### 1.2 MMFAR and BFAR — Fault Address Registers

- **MMFAR** (`0xE000ED34`): The address that caused the MemManage fault. Valid only if `MMARVALID=1`.
- **BFAR** (`0xE000ED38`): The address that caused the BusFault. Valid only if `BFARVALID=1`.

These are the addresses that were accessed — not the PC of the faulting instruction.
To find the faulting instruction, use the stacked PC (see Section 2).

### 1.3 HardFault Status Register (HFSR) — `0xE000ED2C`

| Bit | Name | Meaning |
|-----|------|---------|
| 31 | DEBUGEVT | Debug event escalated to HardFault |
| 30 | FORCED | A configurable fault (MemManage/BusFault/UsageFault) was escalated |
| 1 | VECTTBL | Vector table read fault |

`FORCED=1` means the fault escalated from a specific handler — look at CFSR for the
original cause. `VECTTBL=1` means your interrupt vector table address is wrong (check
`VTOR` register at `0xE000ED08`).

---

## 2. Reading the Fault in GDB

### 2.1 The Basic `info registers` Dump

When Zephyr catches a HardFault, it prints a crash log over the serial console. If you
are attached with GDB (OpenOCD or J-Link):

```gdb
(gdb) info registers
r0             0x00000000  0
r1             0x20001234  536874548
...
pc             0x08012abc  0x8012abc <my_function+24>
sp             0x20003f8c  0x20003f8c
lr             0x08012a01  0x8012a01 <my_function+1>
xpsr           0x61000004  1627390980
```

For **precise faults** (MemManage, BusFault `PRECISERR`), `pc` is the exact faulting
instruction. For **imprecise faults** (`BFSR.IMPRECISERR=1`), `pc` may be 1–5 instructions
ahead of the actual faulting write due to the Cortex-M7 write buffer — see below.

**BusFault imprecise**: If `BFSR.IMPRECISERR=1`, `pc` may be 1–5 instructions ahead of
the actual faulting instruction (Cortex-M7 has a deeper write buffer than M4). The fault
was reported asynchronously — the write completed in the buffer before the bus detected
the error. Disable write buffering to force synchronous reporting and get a precise address:

```c
// At startup, for debugging only (reduces performance)
SCnSCB->ACTLR |= SCnSCB_ACTLR_DISDEFWBUF_Msk;
```

### 2.2 Reading Fault Registers from GDB

```gdb
(gdb) x/1xw 0xE000ED28   # CFSR
0xe000ed28:     0x00020000

(gdb) x/1xw 0xE000ED2C   # HFSR
0xe000ed2c:     0x40000000

(gdb) x/1xw 0xE000ED34   # MMFAR
0xe000ed34:     0x00000000

(gdb) x/1xw 0xE000ED38   # BFAR
0xe000ed38:     0x20001234
```

**Decode CFSR = `0x00020000`:**

Bit 17 = UFSR.INVSTATE — Invalid state. The processor tried to execute code with the
Thumb bit (T bit in xPSR) cleared, meaning an ARM instruction executed on a Thumb-only
core (all Cortex-M). Cause: function pointer with bit 0 = 0 (missing `|1` Thumb bit).

### 2.3 Zephyr's Built-In Fault Output

Zephyr prints fault details automatically over the log backend. Example:

```
[00:00:12.345,678] <err> os: ***** HARD FAULT *****
[00:00:12.345,679] <err> os:   Reason: FORCED
[00:00:12.345,680] <err> os:   HFSR: 0x40000000
[00:00:12.345,681] <err> os:   CFSR: 0x00000200  
[00:00:12.345,682] <err> os:   BFAR: 0x00000000
[00:00:12.345,683] <err> os: Faulting instruction: 0x08012abc
```

Use the `Faulting instruction` address to find the source line.

---

## 3. Mapping PC to Source Line

### 3.1 `addr2line` — Quickest Method

```bash
arm-zephyr-eabi-addr2line -e build/zephyr/zephyr.elf -f -p 0x08012abc
```

Output:
```
my_function() at ~/amr/firmware/src/sensor/imu_reader.c:127
```

This is the exact source line. The `-f` flag shows the function name, `-p` formats it
as one line.

### 3.2 `arm-zephyr-eabi-nm` — Find Symbols Near the PC

If `addr2line` returns `??` (missing debug info or wrong ELF), use `nm` to find the
nearest symbol:

```bash
arm-zephyr-eabi-nm -n build/zephyr/zephyr.elf | grep "08012"
```

Output:
```
08012a00 T my_function
08012b80 T other_function
```

The PC `0x08012abc` is between `my_function` (starts at `0x08012a00`) and `other_function`
(starts at `0x08012b80`). The fault is at offset `0x08012abc - 0x08012a00 = 0xbc = 188 bytes`
into `my_function`.

### 3.3 `objdump` — Disassemble Around the PC

```bash
arm-zephyr-eabi-objdump -d build/zephyr/zephyr.elf | \
  awk '/^08012a00/,/^08012b80/' | grep -A5 -B5 "12abc:"
```

This shows the assembly instructions around the fault. Look for:
- `ldr` / `str` to an invalid address → DACCVIOL
- Branch to odd/invalid address → IACCVIOL or INVSTATE
- Division instruction → DIVBYZERO

### 3.4 Linker Map — Verify Section Placement

The `build/zephyr/zephyr.map` file shows where every symbol is placed in memory.

```bash
grep "my_function" build/zephyr/zephyr.map
```

Output:
```
 .text.my_function  0x08012a00  0x180  CMakeFiles/app.dir/src/sensor/imu_reader.c.obj
```

If the faulting address falls outside the `.text` (code) region (e.g., in `.bss` or
`.data`), you have a corrupt function pointer.

---

## 4. Common HardFault Patterns in Zephyr

### 4.1 Stack Overflow → Stacking Fault

**Symptom**: `MMFSR.MSTKERR=1`, PC is at an ISR entry or Zephyr context switch.

**Cause**: Thread stack overflowed. When an exception fires, the CPU tries to push the
exception frame onto the thread stack — but the stack pointer is already below the stack
boundary, causing a MemManage fault.

**Diagnosis**:

```bash
# Check stack sentinel in Zephyr console
(gdb) x/4xw <thread_stack_address>
# Last valid pattern: 0xAABBCCDD (Zephyr's sentinel)
# If it's overwritten, the stack overflow happened here
```

**Fix**: Increase stack size in `prj.conf` or thread definition:

```c
K_THREAD_STACK_DEFINE(my_stack, 4096);  // increase from 2048
```

Enable Zephyr's stack sentinel checking:
```kconfig
CONFIG_STACK_SENTINEL=y
CONFIG_STACK_SENTINEL_DOUBLE_WIDE=y
```

### 4.2 NULL Pointer Dereference

**Symptom**: `MMFSR.DACCVIOL=1`, `MMFAR = 0x00000000` (or small offset like `0x00000004`).

**Cause**: Reading or writing through a NULL pointer. The offset tells you which field:
`0x00000008` with a struct pointer = accessing the third 32-bit field of a NULL struct.

**Find it**: PC + `addr2line` → exact line. Look for any `->` or `[index]` access at
that line.

### 4.3 DMA Writing to Cached Region

**Symptom**: No fault at time of DMA, but HardFault later when reading the data. CFSR
may show UNDEFINSTR if DMA corrupted code. May also show as data corruption without a
fault at all.

**Cause**: STM32H7 D-cache. DMA writes to physical memory but the CPU reads from cache
(which has stale data). DMA cannot write to Flash (where `.text` lives), so code is not
directly corrupted. The more common scenario: DMA writes to a `.data` region containing
**function pointers or vtables**. The CPU reads a stale cache copy, dereferences a garbage
pointer, and the fault appears as `DACCVIOL` (NULL-like address) or `INVSTATE` (pointer
with bit 0 = 0, missing Thumb bit) — not as a code execution fault.

**Fix**: Ensure DMA buffers are placed in non-cacheable SRAM or use cache invalidation
after DMA completion (see session 03).

### 4.4 Unaligned Access with UNALIGN_TRP Enabled

**Symptom**: `UFSR.UNALIGNED=1`.

**Cause**: Accessing a 32-bit or 64-bit value at an address not aligned to its size.
Example: reading a `uint32_t` from address `0x20001003` (not 4-byte aligned).

**Common source**: `memcpy` into a buffer followed by casting the buffer pointer to a
struct pointer and reading multi-byte fields.

**Fix**: Use `memcpy` into a properly aligned local variable:

```c
uint32_t value;
memcpy(&value, &buffer[unaligned_offset], 4);  // safe
// NOT: uint32_t value = *(uint32_t*)&buffer[unaligned_offset];  // UNSAFE
```

### 4.5 FPU Instruction with FPU Disabled

**Symptom**: `UFSR.NOCP=1`.

**Cause**: Floating-point instruction executed but `CPACR.CP10/CP11` bits not set (FPU
not enabled in firmware config).

**Zephyr fix**:

```kconfig
CONFIG_FPU=y
CONFIG_FPU_SHARING=y
```

Or manually in startup code:
```c
SCB->CPACR |= (0xF << 20);  // Enable CP10 and CP11 (FPU)
__DSB();
__ISB();
```

---

## 5. Quick Reference: Decode Workflow

```
HardFault fires
│
├── Read HFSR at 0xE000ED2C
│   ├── FORCED=1 → look at CFSR for actual cause
│   └── VECTTBL=1 → vector table address corrupt (check VTOR)
│
├── Read CFSR at 0xE000ED28
│   ├── MMFSR bits set?
│   │   ├── MMARVALID=1 → read MMFAR for faulting address
│   │   ├── DACCVIOL → NULL ptr or MPU violation
│   │   ├── IACCVIOL → executing non-executable memory
│   │   └── MSTKERR → stack overflow (check thread stack)
│   │
│   ├── BFSR bits set?
│   │   ├── PRECISERR=1 + BFARVALID=1 → read BFAR for address
│   │   └── IMPRECISERR=1 → enable DISDEFWBUF for precise address
│   │
│   └── UFSR bits set?
│       ├── NOCP → FPU not enabled
│       ├── DIVBYZERO → integer division by zero
│       ├── UNALIGNED → unaligned access
│       ├── INVSTATE → corrupt function pointer (bit 0 = 0)
│       └── UNDEFINSTR → executing garbage data as code
│
├── Read stacked PC from GDB: info registers → pc
│
├── arm-zephyr-eabi-addr2line -e zephyr.elf -f -p <PC>
│   → source file + line number
│
└── If addr2line returns "??":
    arm-zephyr-eabi-nm -n zephyr.elf → find nearest symbol
    arm-zephyr-eabi-objdump -d zephyr.elf → disassemble
```

---

## 6. Enabling All Fault Handlers in Zephyr

By default, Zephyr enables MemManage and BusFault handlers separately from HardFault.
Enable all of them explicitly in your early init for maximum information:

```c
/* In main() or board init, before any unsafe code */
SCB->SHCSR |= SCB_SHCSR_MEMFAULTENA_Msk
           |  SCB_SHCSR_BUSFAULTENA_Msk
           |  SCB_SHCSR_USGFAULTENA_Msk;

/* Enable div-by-zero trap */
SCB->CCR |= SCB_CCR_DIV_0_TRP_Msk;

/* ⚠️  UNALIGN_TRP — DEBUGGING USE ONLY:
 * Enabling this traps on ANY unaligned access, including those performed by
 * Zephyr's internal memcpy, newlib string functions, and nanopb's pb_encode.
 * On Cortex-M7, unaligned accesses are hardware-supported by default;
 * enabling this trap will immediately fault on the first Zephyr memcpy call.
 * Uncomment ONLY when hunting a specific unaligned access bug, then remove. */
/* SCB->CCR |= SCB_CCR_UNALIGN_TRP_Msk; */
```

And enable Zephyr's fault reporting:

```kconfig
CONFIG_FAULT_DUMP=2
CONFIG_EXTRA_EXCEPTION_INFO=y
```

`FAULT_DUMP=2` produces a full register dump including all fault status registers.
`EXTRA_EXCEPTION_INFO=y` decodes each bit of CFSR in the log output.
