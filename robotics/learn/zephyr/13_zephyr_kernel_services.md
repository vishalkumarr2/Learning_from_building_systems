# Zephyr Kernel Services — Synchronization Primitives

## Why You Need Synchronization

Multiple threads share resources: global variables, hardware peripherals, ring buffers.
Without protection, two threads can corrupt shared data:

```
Thread A: reads x (x=5)         Thread B: reads x (x=5)
Thread A: computes x+1=6        Thread B: computes x+1=6
Thread A: writes x=6            Thread B: writes x=6   ← x should be 7!
```

Zephyr provides several primitives. Each solves a different problem:

| Primitive | Problem it solves |
|---|---|
| `k_sem` (semaphore) | Signal between threads ("event happened") |
| `k_mutex` | Protect shared data from simultaneous access |
| `k_msgq` | Pass data + signal between threads (queue) |
| `k_fifo` | Linked-list queue, zero-copy, variable-size items |
| `k_timer` | Periodic callbacks / timeouts |
| `k_event` | Multibit flags, wait for any/all |

---

## k_sem — Semaphore

A semaphore is a **counter** that threads use to coordinate:
- `k_sem_give()` increments the counter
- `k_sem_take()` decrements the counter — **blocks if counter is 0**

### Binary semaphore (signal: "event occurred")

```c
#include <zephyr/kernel.h>

K_SEM_DEFINE(data_ready_sem, 0, 1);
/* initial_count=0 (nothing ready), limit=1 (binary semaphore) */

/* ISR or producer thread — signals that data is ready */
void drdy_isr(const struct device *dev, struct gpio_callback *cb, uint32_t pins)
{
    k_sem_give(&data_ready_sem);   /* Non-blocking, safe from ISR */
}

/* Consumer thread — waits for signal */
void imu_reader_thread(void *a, void *b, void *c)
{
    while (1) {
        k_sem_take(&data_ready_sem, K_FOREVER);   /* Block until ISR signals */
        imu_burst_read(ACCEL_REG, raw, 12);        /* Data guaranteed ready */
        publish_to_zbus(raw);
    }
}
```

### Counting semaphore (track available resources)

```c
K_SEM_DEFINE(free_buffers_sem, 4, 4);
/* 4 buffers available initially, max=4 */

void producer(void)
{
    k_sem_take(&free_buffers_sem, K_FOREVER); /* Wait for free buffer */
    write_to_buffer();
}

void consumer(void)
{
    process_buffer();
    k_sem_give(&free_buffers_sem);            /* Return buffer to pool */
}
```

### ISR-safe version
`k_sem_give()` is always ISR-safe. `k_sem_take()` is NOT ISR-safe (cannot block in ISR).

---

## k_mutex — Mutual Exclusion

A mutex ensures only **one thread** accesses a resource at a time.
Unlike semaphores, mutexes have **ownership** — only the thread that locked it can unlock it.

Zephyr mutexes support **priority inheritance**: if thread A (prio=0) is waiting for a mutex held by thread B (prio=10), Zephyr temporarily raises B's priority to 0, so B finishes faster and A doesn't starve.

```c
K_MUTEX_DEFINE(spi_buf_mutex);

uint8_t shared_buffer[256];

/* Thread A — writing to buffer */
void writer_thread(void *a, void *b, void *c)
{
    while (1) {
        k_mutex_lock(&spi_buf_mutex, K_FOREVER);
        memcpy(shared_buffer, new_data, sizeof(new_data));
        k_mutex_unlock(&spi_buf_mutex);
        k_msleep(10);
    }
}

/* Thread B — reading from buffer */
void reader_thread(void *a, void *b, void *c)
{
    uint8_t local_copy[256];
    while (1) {
        k_mutex_lock(&spi_buf_mutex, K_FOREVER);
        memcpy(local_copy, shared_buffer, sizeof(local_copy));
        k_mutex_unlock(&spi_buf_mutex);

        process(local_copy);  /* Process outside lock — minimize hold time */
    }
}
```

### Rules for mutexes
1. **Never lock a mutex from an ISR** (ISRs cannot sleep/block)
2. **Hold the mutex for the shortest time possible** — other threads spin-wait
3. **Don't call blocking functions while holding a mutex** (can cause deadlock)
4. **Always unlock** — if thread returns/crashes with mutex locked, other threads block forever

### Deadlock example to avoid

```c
/* Thread A */
k_mutex_lock(&mutex_a, K_FOREVER);
k_mutex_lock(&mutex_b, K_FOREVER);  /* waits for B */

/* Thread B (simultaneously) */
k_mutex_lock(&mutex_b, K_FOREVER);
k_mutex_lock(&mutex_a, K_FOREVER);  /* waits for A */

/* Both threads block forever — deadlock! */
/* Fix: always lock in the same order (mutex_a first, then mutex_b) */
```

---

## k_msgq — Message Queue

A message queue passes **fixed-size messages** between threads — combines data transfer and signaling.

```c
/* Define queue: element size = sizeof(struct imu_data), max 4 items */
K_MSGQ_DEFINE(imu_msgq, sizeof(struct imu_data), 4, 4);

/* Producer: push message */
void imu_reader(void *a, void *b, void *c)
{
    struct imu_data msg;
    while (1) {
        read_imu(&msg);
        if (k_msgq_put(&imu_msgq, &msg, K_NO_WAIT) != 0) {
            LOG_WRN("Queue full — dropping IMU sample");
        }
        k_msleep(10);
    }
}

/* Consumer: pop messages */
void packer_thread(void *a, void *b, void *c)
{
    struct imu_data msg;
    while (1) {
        k_msgq_get(&imu_msgq, &msg, K_FOREVER);  /* block until message arrives */
        encode_and_send(&msg);
    }
}
```

**When to use k_msgq vs ZBus**:
- **k_msgq**: one-to-one pipe. Consumer gets events in order.
- **ZBus**: one-to-many pub/sub. Multiple subscribers can see the same message. Subscribers only see "latest" unless they use queued subscription.

---

## k_timer — Periodic Callbacks

A timer calls a function after a timeout, or periodically.

```c
void timer_callback(struct k_timer *timer)
{
    /* This runs in ISR context (or system workqueue) — keep it short! */
    k_sem_give(&tick_sem);
}

K_TIMER_DEFINE(tick_timer, timer_callback, NULL);

void main_init(void)
{
    /* Start periodic timer: fire after 10ms, then every 10ms */
    k_timer_start(&tick_timer, K_MSEC(10), K_MSEC(10));
}

void packer_thread(void *a, void *b, void *c)
{
    while (1) {
        k_sem_take(&tick_sem, K_FOREVER);  /* exactly every 10ms */
        pack_and_send_frame();
    }
}
```

**Timer callback rules**:
- Runs in ISR context (if using `K_TIMER_DEFINE`) — keep it tiny
- Never call `k_msleep`, `k_mutex_lock`, or other blocking functions inside
- Use `k_sem_give` to wake a thread that does the real work

---

## k_event — Multiple Flags At Once

When you need to wait for **any of several** events, or **all of several** events simultaneously:

```c
K_EVENT_DEFINE(system_events);

#define EVENT_IMU_READY   BIT(0)
#define EVENT_CAN_READY   BIT(1)
#define EVENT_GPS_READY   BIT(2)
#define EVENT_SHUTDOWN    BIT(15)

/* Post an event bit */
void imu_isr(void)
{
    k_event_post(&system_events, EVENT_IMU_READY);
}

/* Wait for any sensor data */
void fusion_thread(void *a, void *b, void *c)
{
    while (1) {
        /* Wait until IMU or CAN data ready */
        uint32_t events = k_event_wait(&system_events,
                                        EVENT_IMU_READY | EVENT_CAN_READY,
                                        false,  /* false = any; true = all */
                                        K_FOREVER);

        k_event_clear(&system_events, events);  /* clear what we consumed */

        if (events & EVENT_IMU_READY) handle_imu();
        if (events & EVENT_CAN_READY) handle_can();
    }
}
```

---

## k_work — System Workqueue

When you need to **defer work** from an ISR to a thread context:

```c
#include <zephyr/kernel.h>

static void process_data_work_fn(struct k_work *work)
{
    /* Runs in system workqueue thread — can call blocking functions */
    LOG_INF("Processing data");
}

static K_WORK_DEFINE(process_work, process_data_work_fn);

/* ISR: submit work item instead of doing heavy work in IRQ context */
void uart_isr(const struct device *dev, void *user_data)
{
    /* Quick: read bytes into ring buffer */
    // ...
    /* Defer processing to thread context */
    k_work_submit(&process_work);   /* safe from ISR */
}
```

---

## Choosing the Right Primitive

```
                Signal only?     ──Yes──► k_sem
                │
                No
                │
           Transfer data?        ──Yes──► k_msgq (fixed size) or k_fifo (variable)
                │
                No
                │
          Protect shared data?   ──Yes──► k_mutex
                │
                No
                │
          Multiple event flags?  ──Yes──► k_event
                │
                No
                │
          Periodic operation?    ──Yes──► k_timer
```

---

## Common Synchronization Bugs

| Bug | Symptom | Fix |
|---|---|---|
| No mutex on shared buffer | Intermittent data corruption | Add `k_mutex_lock/unlock` |
| `k_mutex_lock` in ISR | HardFault / panic | Use `k_sem_give` in ISR |
| Holding mutex across `k_msleep` | Other threads starve | Release before sleeping |
| `k_msgq` full, dropping | LOG_WRN messages | Increase queue size or consume faster |
| Sem count > max | Ignored (capped at max) | Set max high enough |
| k_timer callback blocking | System timer miss | Never block in timer callback |
