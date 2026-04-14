# Zephyr Threads — Multitasking on an MCU

## What Is a Thread?

A thread is an **independent execution path** with its own stack and program counter.
Zephyr can run multiple threads on a single-core STM32 — not truly in parallel, but by rapidly switching between them (preemption).

**Without threads**: you'd write one giant loop with `while(1)` — hard to manage, one slow task blocks everything.

**With threads**: each subsystem has its own thread. Zephyr's kernel switches between them automatically.

---

## Thread Basics — The Simple Mental Model

```
             ┌─── K_THREAD_DEFINE ───────────────────────────────┐
             │                                                    │
High         │  Thread: spi_packer     prio=0   stack=2048       │ ← runs first
priority     │  Thread: imu_reader     prio=5   stack=1024       │
             │  Thread: can_reader     prio=5   stack=1024       │
             │  Thread: heartbeat      prio=10  stack=512        │ ← runs last
Low          └───────────────────────────────────────────────────┘
priority
```

Lower number = higher priority.  
If two threads have equal priority, they timeslice (Zephyr can context-switch every `CONFIG_TIMESLICING_SLICE_TICKS`).

---

## Creating Threads

### Static definition (preferred — no heap, no surprises)

```c
#include <zephyr/kernel.h>

/* Define thread function */
void imu_thread_fn(void *arg1, void *arg2, void *arg3)
{
    /* arg1/2/3 are the 3 params from K_THREAD_DEFINE */
    LOG_INF("IMU thread started");

    while (1) {
        /* do work */
        k_msleep(10);   /* yield CPU for 10ms */
    }
    /* Thread functions never return — use infinite loop */
}

/* Declare thread: name, entry, arg1, arg2, arg3, stack_size, priority, start_delay */
K_THREAD_DEFINE(imu_tid,          /* thread ID variable */
                imu_thread_fn,    /* entry function */
                1024,             /* stack size in bytes */
                5,                /* priority (lower = higher) */
                NULL, NULL, NULL, /* 3 optional arguments */
                0);               /* start delay in ms (0 = start immediately) */
```

That's it — `imu_tid` is a `k_tid_t` you can use to suspend/resume/abort.

### Dynamic creation (if you need runtime flexibility)

```c
K_THREAD_STACK_DEFINE(my_stack, 1024);
static struct k_thread my_thread;

k_tid_t tid = k_thread_create(
    &my_thread,           /* struct to store thread state */
    my_stack,             /* stack array */
    K_THREAD_STACK_SIZEOF(my_stack),
    my_thread_fn,         /* entry function */
    NULL, NULL, NULL,     /* 3 optional args */
    5,                    /* priority */
    0,                    /* options (K_FP_REGS if using float on some MCUs) */
    K_NO_WAIT             /* start delay */
);
```

---

## Thread Lifecycle

```
           k_thread_create()
                  │
                  ▼
              ┌────────┐
              │ READY  │ ◄──── k_thread_resume()
              └────────┘
                  │ highest priority ready thread
                  ▼
              ┌──────────┐
              │ RUNNING  │ ← only ONE thread runs at a time
              └──────────┘
                  │ preempted by higher-prio thread, or calls k_msleep()/k_sem_take()
                  ▼
              ┌────────────┐
              │  WAITING   │ (sleeping, blocked on sem/mutex/queue)
              └────────────┘
                  │ timeout expires or sem given
                  ▼
              ( back to READY )
```

```c
/* From another thread: */
k_thread_suspend(imu_tid);       /* pause thread */
k_thread_resume(imu_tid);        /* unpause thread */
k_thread_abort(imu_tid);         /* permanently kill thread */

/* From within the thread itself: */
k_msleep(10);           /* sleep 10ms — yields CPU */
k_sleep(K_USEC(500));   /* sleep 500µs */
k_yield();              /* yield to equal-priority threads (no sleep) */
```

---

## Priority and Preemption

```
Thread A prio=0 (highest) ──────────── ← runs when ready
Thread B prio=5           ─ ─ ─ ─ ─ ─ ← preempted by A, resumes when A sleeps
Thread C prio=10          ─ ─ ─ ─ ─ ─ ← runs only when A and B both sleeping
```

**Preemptive**: if Thread A (prio=0) becomes ready while Thread C (prio=10) is running, Zephyr immediately switches to A.

**Critical for our robot**: the SPI packer thread (prio=0) must run within 1ms of its 10ms deadline. I2C and CAN readers (prio=5) run when SPI packer sleeps. Heartbeat (prio=10) only runs in spare time.

### Cooperative vs Preemptive threads

| | Preemptive (priority < `CONFIG_NUM_COOP_PRIORITIES`) | Cooperative (priority >= ...) |
|---|---|---|
| Can be preempted by higher-prio? | Yes | No (runs until it explicitly yields) |
| Use for | Most threads | ISR-like critical sections |

Most code uses **preemptive threads** — simpler to reason about.

---

## Thread Stacks — How to Size Them

**Too small = stack overflow = random crashes / corruption.**

```c
/* Enable stack sentinel in prj.conf */
/* CONFIG_STACK_SENTINEL=y  (adds guard bytes, panics on overflow) */
/* CONFIG_THREAD_ANALYZER=y (prints stack usage at runtime) */
```

At runtime, check usage:

```
I: thread_analyzer: Name    Priority  Stack    Used
I: thread_analyzer: imu     5         1024     712   (69%)
I: thread_analyzer: packer  0         2048     384   (18%)
```

Rules of thumb:

| Thread type | Minimum stack |
|---|---|
| Simple work loop (no printf) | 256–512 bytes |
| Uses LOG_INF / sprintf | 1024 bytes |
| Calls complex library (nanopb encode) | 1536–2048 bytes |
| Uses C++ exceptions | 4096 bytes |

When in doubt, measure with `CONFIG_THREAD_ANALYZER`.

---

## Complete Thread Architecture for Our Robot

```c
/* All threads in one file — "main" just initializes and blocks */

/* Thread priorities */
#define PRIO_SPI_PACKER    0   /* must meet 10ms deadline */
#define PRIO_IMU_READER    5
#define PRIO_CAN_READER    5
#define PRIO_GPS_READER    7
#define PRIO_HEARTBEAT     10

/* Thread entries */
void imu_thread_fn(void *a, void *b, void *c)
{
    const struct device *i2c = DEVICE_DT_GET(DT_NODELABEL(i2c1));
    /* ... read IMU at 100Hz, publish to IMU ZBus channel ... */
}

void can_thread_fn(void *a, void *b, void *c)
{
    /* ... receives from ZBus queue, can threads use callbacks not polling ... */
}

void spi_packer_fn(void *a, void *b, void *c)
{
    struct imu_data imu;
    struct motor_state motor;

    while (1) {
        /* Wait until next 10ms tick */
        int64_t next = k_uptime_get() + 10;

        /* Collect latest data from channels */
        zbus_chan_read(&imu_chan,   &imu,   K_NO_WAIT);
        zbus_chan_read(&motor_chan, &motor, K_NO_WAIT);

        /* Encode to protobuf + hand to SPI DMA buffer */
        pack_and_swap_spi_buffers(&imu, &motor);

        /* Sleep until next cycle */
        int64_t now = k_uptime_get();
        if (next > now) k_msleep(next - now);
    }
}

void heartbeat_fn(void *a, void *b, void *c)
{
    while (1) {
        gpio_pin_toggle_dt(&led);
        k_msleep(500);
    }
}

/* Static thread definitions */
K_THREAD_DEFINE(imu_tid,    imu_thread_fn,    1024, PRIO_IMU_READER,  NULL, NULL, NULL, 0);
K_THREAD_DEFINE(can_tid,    can_thread_fn,    512,  PRIO_CAN_READER,  NULL, NULL, NULL, 0);
K_THREAD_DEFINE(packer_tid, spi_packer_fn,    2048, PRIO_SPI_PACKER,  NULL, NULL, NULL, 0);
K_THREAD_DEFINE(hb_tid,     heartbeat_fn,     256,  PRIO_HEARTBEAT,   NULL, NULL, NULL, 0);

void main(void)
{
    LOG_INF("System started — all threads running");

    /* main() can just block forever; threads are already running */
    k_sleep(K_FOREVER);
}
```

---

## Common Thread Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Stack too small | Random crashes, HardFault | Enable `CONFIG_STACK_SENTINEL`, analyze with `THREAD_ANALYZER` |
| Blocking in ISR | System hangs | ISR must never call `k_msleep`, only `k_sem_give/k_fifo_put` |
| No `k_msleep` in thread loop | 100% CPU, other threads starve | Always call `k_msleep` or block on a sync primitive |
| Shared global without mutex | Corrupted data | Use `k_mutex_lock/unlock` (see next document) |
| Equal priority threads spinning | One thread monopolizes CPU | Use timeslicing or `k_yield()` |
| Thread never returns | Fine — this is correct | Threads use infinite loops, never return |
