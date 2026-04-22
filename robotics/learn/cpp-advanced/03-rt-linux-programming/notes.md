# Week 3 — Real-Time Linux Programming

## What "Real-Time" Actually Means

Real-time does **not** mean fast. It means **deterministic** — the system guarantees a
response within a known time bound. A response that arrives 1μs late can be as
catastrophic as one that never arrives.

### Hard Real-Time

**Definition:** Missing a deadline is a system failure.

**Examples:**
- **Airbag controller** — must fire within 15ms of sensor trigger. 16ms = passenger dies.
- **Anti-lock braking (ABS)** — must modulate brakes every 5ms. Miss one = wheel locks.
- **Pacemaker** — electrical pulse must fire within the cardiac cycle. Late = fibrillation.
- **Industrial robot arm** — position update must arrive every 1ms. Late = arm overshoots into worker.

**Guarantee:** Worst-Case Execution Time (WCET) is analyzed and proven at design time.

### Soft Real-Time

**Definition:** Occasional deadline misses are tolerable; quality degrades gracefully.

**Examples:**
- **Video playback** — drop a frame, you see a flicker. Nobody dies.
- **VoIP call** — lose a packet, you hear a click. Conversation continues.
- **Web server** — 99th percentile latency of 200ms is fine; 500ms occasionally is ugly but OK.

**Guarantee:** Statistical — p99/p99.9 within bounds.

### Firm Real-Time

**Definition:** A late result has zero value but doesn't cause catastrophe.

**Examples:**
- **High-frequency trading** — a stale quote is worthless. Late order = missed profit, not death.
- **Radar tracking** — a late return is discarded. The target has moved.
- **warehouse AMR control loop** — a late cmd_vel is useless; the robot is now 10mm further along.

**Key insight for our work:** The warehouse robot estimator + controller runs at ~100Hz (10ms).
That makes it firm-RT. The line-sensor must deliver fresh data every 10ms. When it delivers
duplicated data (stale), the estimator diverges — exactly the failure mode we've been
investigating in Cases G, Case-H, Case-I.

---

## The RT-Forbidden Operations

In a real-time code path (the "hot loop"), these operations are **forbidden**.
Not "discouraged" — **forbidden**. Each one can cause unbounded latency spikes.

### 1. `malloc` / `operator new`

**Why forbidden:** `malloc` uses a freelist with locks. It can:
- Acquire a global mutex (blocks on contention)
- Call `mmap`/`brk` (system call → context switch → page fault)
- Trigger the OOM killer (process gets killed)

**How bad:** Typical: 0.5-5μs. Worst case: 10-500μs. With OOM pressure: milliseconds.

```
Typical malloc timeline:
╭─────────────────────────────────────────────────────╮
│ Check freelist → found → return ptr         ~50ns   │
│ Check freelist → empty → call mmap()        ~5μs    │
│ Check freelist → empty → mmap → page fault  ~50μs   │
│ Check freelist → lock contention → spin     ~100μs  │
╰─────────────────────────────────────────────────────╯
```

**Fix:** Pre-allocate everything before entering the RT loop. Use pool allocators (see ex04).

### 2. `printf` / `std::cout` / any I/O to file or console

**Why forbidden:** These call `write()`, which:
- Acquires a stdio lock (contention with other threads)
- May flush the buffer (triggering a `write` syscall)
- `write` to a terminal can block if the terminal buffer is full
- `write` to a file can trigger filesystem journaling (ext4 journal commit = 5ms)

**How bad:** Typical: 1-10μs. Worst case: 100μs-5ms (filesystem flush).

**Fix:** Write to a pre-allocated ring buffer. A separate low-priority thread drains
the buffer to disk/console. This is the telemetry pattern in ex05.

### 3. Mutex (without Priority Inheritance)

**Why forbidden:** Causes **priority inversion** — the Mars Pathfinder bug:

```
Priority Inversion Timeline:
─────────────────────────────────────────────────────
Time →
LOW  ████████████████░░░░░░░████████  (holds mutex, runs)
MED  ░░░░░░░░░░░░░░░██████████░░░░░░  (ready, preempts LOW)
HIGH ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  (BLOCKED on mutex)
                    ↑ HIGH blocked here because MED
                      preempts LOW, which holds lock

With Priority Inheritance:
LOW  ████████████████░░░████░░░░░░░░  (boosted to HIGH prio)
MED  ░░░░░░░░░░░░░░░░░░░░░░███████░  (can't preempt boosted LOW)
HIGH ░░░░░░░░░░░░░░░░░░░███░░░░░░░░  (gets mutex quickly)
─────────────────────────────────────────────────────
```

**Fix:** Use `PTHREAD_PRIO_INHERIT` mutexes. Or better: use lock-free data structures
(SPSC queues from Week 2).

### 4. Disk I/O (read/write/open/fsync)

**Why forbidden:** Filesystem operations can:
- Wait for disk seek (HDD: 5-10ms)
- Wait for journal commit (ext4: every 5s, or at fsync)
- Trigger page cache eviction (kernel does I/O under your allocation)
- Wait for NFS/CIFS network timeout (seconds)

**Fix:** All logging goes through a lock-free queue to a non-RT thread.

### 5. System calls that can page fault

**Why forbidden:** `mmap`, `brk`, and even `read` can trigger page faults.
A **major page fault** requires reading from disk — 100μs to 10ms.

**Fix:** Call `mlockall(MCL_CURRENT | MCL_FUTURE)` at startup.
Touch all stack pages ("prefault the stack") before entering the RT loop.

### Summary Table

```
╔════════════════════╦═════════╦═══════════╦═══════════════════════╗
║ Operation          ║ Typical ║ Worst     ║ Fix                   ║
╠════════════════════╬═════════╬═══════════╬═══════════════════════╣
║ malloc/new         ║ 0.5μs   ║ 500μs     ║ Pool allocator / PMR  ║
║ printf/cout        ║ 5μs     ║ 5ms       ║ Lock-free log queue   ║
║ mutex (no PI)      ║ 0.1μs   ║ unbounded ║ PI mutex / lock-free  ║
║ disk I/O           ║ 50μs    ║ 10ms      ║ Separate I/O thread   ║
║ page fault         ║ 1μs     ║ 10ms      ║ mlockall + prefault   ║
╚════════════════════╩═════════╩═══════════╩═══════════════════════╝
```

---

## PREEMPT_RT Patch

Standard Linux is a **general-purpose OS**, not a real-time OS. The PREEMPT_RT patch
(mainlined into Linux 6.12+) converts it to a deterministic system. Here's what it changes:

### What PREEMPT_RT Does

1. **Makes spinlocks preemptible** — In vanilla Linux, `spin_lock()` disables preemption.
   Under PREEMPT_RT, most spinlocks become sleeping mutexes with priority inheritance.
   This means a high-priority RT thread can preempt kernel code that holds a "spinlock".

2. **Threaded interrupt handlers** — In vanilla Linux, hardware interrupts run in a
   special context that can't be preempted. PREEMPT_RT moves interrupt handlers into
   kernel threads with configurable priorities. Your RT thread can have higher priority
   than a network interrupt.

3. **High-resolution timers** — `clock_nanosleep` becomes accurate to ~1μs instead of
   the default timer tick (typically 1ms or 4ms on vanilla kernels).

4. **Sleeping spinlocks become PI-aware** — Since spinlocks become mutexes, they inherit
   priority inheritance automatically, preventing priority inversion in kernel code.

### How to check if you have RT

```bash
uname -a    # look for "PREEMPT_RT" in the output
cat /sys/kernel/realtime   # "1" if RT kernel (if file exists)
```

### Typical worst-case latencies

```
╔═══════════════════╦═══════════════╦═══════════════╗
║                   ║ Vanilla Linux ║ PREEMPT_RT    ║
╠═══════════════════╬═══════════════╬═══════════════╣
║ Timer jitter      ║ 50-500μs      ║ 1-15μs        ║
║ IRQ latency       ║ 5-100μs       ║ 3-10μs        ║
║ Scheduling jitter ║ 10-1000μs     ║ 2-20μs        ║
╚═══════════════════╩═══════════════╩═══════════════╝
```

---

## Thread Affinity and CPU Isolation

### Why Core 0 is Never for RT

Core 0 handles:
- Timer interrupts (the scheduler)
- RCU callbacks (kernel memory reclamation)
- Workqueues (deferred kernel work)
- Most IRQs by default

Pinning an RT thread to core 0 means it fights the kernel for CPU time.

### CPU Isolation: `isolcpus`

Boot parameter that tells the scheduler "don't put anything on these cores":

```bash
# In /etc/default/grub:
GRUB_CMDLINE_LINUX="isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3"
```

- `isolcpus=2,3` — scheduler won't place tasks on cores 2,3
- `nohz_full=2,3` — disable timer tick on these cores when possible
- `rcu_nocbs=2,3` — move RCU callbacks off these cores

```
CPU Core Layout for RT System:
╔════════╦════════════════════════════════════════════╗
║ Core 0 ║ OS kernel, IRQs, housekeeping             ║
║ Core 1 ║ Non-RT threads (telemetry, logging)        ║
║ Core 2 ║ ★ RT control loop (isolated)               ║
║ Core 3 ║ ★ RT sensor processing (isolated)          ║
╚════════╩════════════════════════════════════════════╝
```

### Setting Thread Affinity Programmatically

```cpp
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(2, &cpuset);  // pin to core 2
int rc = sched_setaffinity(0, sizeof(cpuset), &cpuset);
if (rc != 0) {
    perror("sched_setaffinity failed");
}
```

### Setting RT Scheduling Policy

```cpp
struct sched_param param;
param.sched_priority = 80;  // 1-99, higher = more urgent
int rc = sched_setscheduler(0, SCHED_FIFO, &param);
if (rc != 0) {
    // Requires CAP_SYS_NICE or root
    perror("sched_setscheduler failed (need root or CAP_SYS_NICE)");
}
```

**SCHED_FIFO vs SCHED_RR:**
- `SCHED_FIFO` — runs until it voluntarily yields or a higher-priority thread arrives.
  No timeslicing. Best for RT — predictable.
- `SCHED_RR` — like FIFO but with timeslicing among same-priority threads.
  Adds scheduling jitter. Avoid for hard RT.

---

## The Cyclic Executive Pattern

The fundamental RT control loop pattern. Every robot controller, flight controller,
and industrial PLC uses a variation of this:

```
Cyclic Executive Timing (1kHz = 1ms period):
                                                          
│◄──────── 1ms ─────────►│◄──────── 1ms ─────────►│
│                         │                         │
│  Sensor  │ Compute │Act│  Sensor  │ Compute │Act│
│  200μs   │  400μs  │50│  200μs   │  400μs  │50│
│          │         │μs│          │         │μs│
├──────────┴─────────┴───┤──────────┴─────────┴───┤
│     slack: 350μs       │     slack: 350μs       │
│◄ sleep ──────────────► │◄ sleep ──────────────► │
                                                          
Overrun Detection:
│◄──────── 1ms ─────────►│
│                         │
│  Sensor  │ Compute (overrun!) │
│  200μs   │  900μs              │
│                                │← missed deadline!
│  LOG OVERRUN, skip to next     │
```

### Key Implementation Details

**Absolute time, not relative:**
```cpp
// WRONG — drifts by processing time each cycle:
while (running) {
    do_work();
    clock_nanosleep(CLOCK_MONOTONIC, 0, &period, nullptr);
}

// CORRECT — always wakes at exact multiples of period:
clock_gettime(CLOCK_MONOTONIC, &next_wake);
while (running) {
    do_work();
    next_wake.tv_nsec += period_ns;
    normalize_timespec(next_wake);  // handle nsec overflow
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_wake, nullptr);
}
```

**Overrun detection:**
```cpp
struct timespec now;
clock_gettime(CLOCK_MONOTONIC, &now);
if (timespec_diff_us(now, next_wake) > 0) {
    overrun_count++;
    // Skip to the next aligned period
    next_wake = now;
    align_to_period(next_wake, period_ns);
}
```

---

## Pool Allocators and PMR

### The Problem

`std::vector::push_back` can call `malloc`. In a 1kHz loop, even one malloc
per cycle means 1000 mallocs/second, any of which can stall.

### Solution 1: Fixed Pool Allocator

Pre-allocate a fixed number of objects. Alloc and free are O(1):

```
FixedPool<Sensor, 128> Layout:
╔══════════════════════════════════════════════════╗
║ storage[0] │ storage[1] │ ... │ storage[127]     ║
║  (Sensor)  │  (Sensor)  │     │  (Sensor)        ║
╠══════════════════════════════════════════════════╣
║ free_list → [4] → [7] → [12] → ... → [null]     ║
║ (singly-linked list through unused slots)        ║
╚══════════════════════════════════════════════════╝

alloc():  pop head of free_list     O(1)
free(p):  push p onto free_list     O(1)
```

### Solution 2: PMR (Polymorphic Memory Resource)

C++17 added `<memory_resource>`. The idea: containers take an allocator
parameter that can be swapped without changing the container type.

```cpp
// Stack buffer — no heap allocation ever
alignas(16) std::byte buf[4096];
std::pmr::monotonic_buffer_resource pool{buf, sizeof(buf)};

// Vector that allocates from our stack buffer
std::pmr::vector<int> data{&pool};
data.reserve(100);  // allocates from buf, not from heap
```

```
PMR monotonic_buffer_resource Layout:
╔════════════════════════════════════════════════╗
║ Stack buffer (4096 bytes)                      ║
║ ┌──────┬──────┬──────┬─────────────────────┐  ║
║ │ used │ used │ used │  free space          │  ║
║ │ (vec)│(str1)│(str2)│← next allocation here│  ║
║ └──────┴──────┴──────┴─────────────────────┘  ║
║                       ↑ current pointer        ║
╚════════════════════════════════════════════════╝
Note: monotonic = never frees individual chunks.
      Reset deallocates everything at once.
```

**Key PMR types:**
- `monotonic_buffer_resource` — bumps a pointer, never frees. Fastest. Reset all at once.
- `unsynchronized_pool_resource` — pool of fixed-size blocks. Good for mixed sizes.
- `synchronized_pool_resource` — thread-safe version (adds locking — avoid in RT).

---

## The Prefault Stack Trick

When a function is called, the stack grows. If the stack page hasn't been
accessed before, the first write triggers a **page fault** — the kernel must
allocate a physical page. That's 5-50μs of latency in your RT loop.

### The Fix

```cpp
void prefault_stack() {
    // Lock all current and future pages in RAM
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed (need CAP_IPC_LOCK or root)");
    }

    // Touch every page of a large stack allocation
    volatile char stack_touch[1024 * 1024];  // 1MB
    for (size_t i = 0; i < sizeof(stack_touch); i += 4096) {
        stack_touch[i] = 0;  // write touches the page
    }
}
```

**Why `volatile`?** Without it, the compiler optimizes away the dead writes.

**Why every 4096 bytes?** That's one page. We need to touch at least one byte
per page to force the kernel to map a physical frame.

**Why `mlockall`?** Without it, the kernel can swap out pages we've faulted in.
`mlockall(MCL_FUTURE)` prevents future pages from being swappable too.

```
Stack Prefault Sequence:
                                      
Before:   [mapped] [mapped] [guard] [unmapped] [unmapped] ...
                              ↑ stack grows down
                              
After mlockall + touch:
          [locked] [locked] [locked] [locked] [locked] ...
          All pages physically present and pinned in RAM
```

---

## Putting It All Together: RT Initialization Checklist

Before entering the RT loop, every real-time Linux application must:

```
RT Application Startup Sequence:
╔═══╦══════════════════════════════════════════════════╗
║ 1 ║ mlockall(MCL_CURRENT | MCL_FUTURE)               ║
║ 2 ║ Prefault stack (volatile touch 1MB)               ║
║ 3 ║ Pre-allocate all buffers (pool/PMR)               ║
║ 4 ║ Set thread affinity (pin to isolated core)        ║
║ 5 ║ Set SCHED_FIFO with appropriate priority          ║
║ 6 ║ Start non-RT threads (logging, telemetry)         ║
║ 7 ║ ═══ ENTER RT LOOP ═══                            ║
║   ║ No malloc, no printf, no mutex, no I/O            ║
╚═══╩══════════════════════════════════════════════════╝
```

---

## Connection to Our Work

The warehouse robot runs ROS1 on Linux. The navigation estimator and controller
run as ROS nodes at ~100Hz. They are **not** configured for hard RT:
- No `SCHED_FIFO`
- No `mlockall`
- No isolated CPUs
- `rospy` and `roscpp` both call `malloc` freely

This explains why the system is vulnerable to latency spikes: when the kernel
is busy (disk flush, network burst, RCU grace period), the estimator loop
gets delayed, stale line-sensor data accumulates, and the robot diverges.

Understanding RT programming helps us:
1. **Diagnose** why certain failures only happen under load
2. **Design** better firmware (STM32 Zephyr runs actual RTOS)
3. **Build** deterministic test tools (like `nav_tile_test.py` with timing)
4. **Read** controller code critically — spot non-RT-safe patterns

---

## Exercises This Week

| Exercise | What You Build | Key Concept |
|----------|---------------|-------------|
| ex01 | RT-forbidden demo | Measure malloc/printf latency tails |
| ex02 | Thread affinity | `sched_setaffinity`, SCHED_FIFO, cache effects |
| ex03 | Cyclic executive | 1kHz loop, jitter measurement, overrun detection |
| ex04 | Pool allocator | `FixedPool<T,N>`, PMR, heap detection |
| ex05 | RT + telemetry | Full mini-project: RT loop + SPSC + CSV telemetry |

| Puzzle | What You Explore | Key Insight |
|--------|-----------------|-------------|
| puzzle01 | Relative vs absolute sleep | Drift accumulation |
| puzzle02 | Priority inversion | Mars Pathfinder, `PTHREAD_PRIO_INHERIT` |
| puzzle03 | Page fault latency | `mlockall` + prefault eliminates spikes |

---

## Further Reading

- "The Design of the PREEMPT_RT Patch" — LWN.net series
- "Real-Time Linux Wiki" — rt.wiki.kernel.org
- "Is Parallel Programming Hard?" — Paul McKenney, Ch. 7-8 on RT
- "Linux Device Drivers, 3rd Ed." — Ch. 7 (Time, Delays, Deferred Work)
- cyclictest tool — the standard RT latency benchmark
- robot firmware: `navigation_estimator` runs a cyclic pattern at 100Hz
