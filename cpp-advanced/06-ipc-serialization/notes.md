# Week 6 — IPC, Serialization & Software-Defined Patterns

## 1. Inter-Process Communication with POSIX Shared Memory

IPC lets separate processes exchange data without going through the kernel on every
message.  Shared memory is the fastest IPC primitive because, after the initial
`mmap`, reads and writes touch the same physical pages — zero syscalls on the hot
path.

### 1.1 The shm_open + mmap Dance

```
Process A                          Process B
────────                          ────────
shm_open("/my_shm", O_CREAT|O_RDWR)
ftruncate(fd, size)
ptr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0)
                                   fd = shm_open("/my_shm", O_RDWR)
                                   ptr = mmap(..., MAP_SHARED, fd, 0)
*ptr = 42;                         printf("%d\n", *ptr);  // 42
munmap(ptr, size)                  munmap(ptr, size)
shm_unlink("/my_shm")
```

Key API summary:

| Function        | Purpose                                    |
|-----------------|--------------------------------------------|
| `shm_open`      | Create/open a named shared memory object   |
| `ftruncate`     | Set the size of the shared region          |
| `mmap`          | Map the object into process address space  |
| `munmap`        | Unmap the region                           |
| `shm_unlink`    | Remove the named object from `/dev/shm`    |

Link with `-lrt` on Linux (the POSIX realtime library).

### 1.2 Shared Memory Layout

When placing structs in shared memory, alignment matters.  The standard approach is
to define a header struct and use `placement new` to construct it at the mapped
address.

```
Shared Memory Region (4096 bytes mapped)
┌─────────────────────────────────────────────────────────┐
│ offset 0: SharedHeader                                  │
│   ┌──────────────────────────────────────────────────┐  │
│   │ magic       : uint32_t  (0xDEADBEEF)            │  │
│   │ version     : uint32_t  (1)                      │  │
│   │ write_seq   : atomic<uint64_t>                   │  │
│   │ read_seq    : atomic<uint64_t>                   │  │
│   │ payload_off : uint32_t  (offset to data region)  │  │
│   └──────────────────────────────────────────────────┘  │
│ offset 64: Data region                                  │
│   ┌──────────────────────────────────────────────────┐  │
│   │ Payload[0] ... Payload[N-1]                      │  │
│   │ (ring buffer of fixed-size messages)             │  │
│   └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 1.3 Seqlock Pattern for Lock-Free Reads

A seqlock lets one writer publish data and many readers consume it without
mutexes.  The writer increments a sequence counter before and after writing.
The reader checks the counter: if it changed or is odd, the read was torn —
retry.

```
Writer:                           Reader:
  seq.store(seq+1, release)         do {
  <write payload>                     s1 = seq.load(acquire)
  seq.store(seq+1, release)           <read payload>
                                      s2 = seq.load(acquire)
                                    } while (s1 != s2 || s1 & 1)
```

This gives sub-microsecond IPC latency because the reader never blocks.

### 1.4 Producer-Consumer with Ring Buffer

```
        Producer                     Consumer
        ────────                     ────────
           │                            │
           ▼                            ▼
    ┌──write_idx──┐             ┌──read_idx───┐
    │             │             │             │
    ▼             │             ▼             │
┌───┬───┬───┬───┬───┬───┬───┬───┐           │
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │  (slots)  │
└───┴───┴───┴───┴───┴───┴───┴───┘           │
    ▲                       ▲                │
    └───── data region ─────┘                │
                                             │
    Full when: (write_idx + 1) % N == read_idx
    Empty when: write_idx == read_idx
```

The producer writes to `slots[write_idx % N]` and then advances `write_idx`.
The consumer reads from `slots[read_idx % N]` and then advances `read_idx`.
Both indices are `std::atomic<uint64_t>` with monotonically increasing values.

---

## 2. Serialization: From memcpy to Schema-Driven

Serialization is the bridge between in-memory layout and wire/disk format.
The choice profoundly affects IPC throughput.

### 2.1 Method Comparison

| Method              | Encode ns/msg | Decode ns/msg | Zero-copy | Schema | Human-readable |
|---------------------|---------------|---------------|-----------|--------|----------------|
| Raw memcpy          | ~2–5          | ~2–5          | Yes*      | No     | No             |
| Hand-rolled binary  | ~5–15         | ~5–15         | No        | No     | No             |
| FlatBuffers-style   | ~10–30        | ~1–5          | Yes       | Yes    | No             |
| JSON (sprintf)      | ~200–800      | ~300–1000     | No        | No     | Yes            |
| Protobuf            | ~50–150       | ~50–150       | No        | Yes    | Semi           |

*memcpy is "zero-copy" only when the struct is POD and layout matches on both sides.

### 2.2 Raw memcpy — The Fastest and Most Fragile

```cpp
struct Msg { uint64_t ts; double x, y, z; };
char buf[sizeof(Msg)];
std::memcpy(buf, &msg, sizeof(Msg));   // serialize
std::memcpy(&msg2, buf, sizeof(Msg));  // deserialize
```

Pros: Fastest possible.  Cons: Tied to one platform, one compiler, one ABI.
Adding a field breaks all readers.

### 2.3 FlatBuffer-like Zero-Copy

The key insight: write fields at known byte offsets into a pre-allocated buffer.
Readers cast directly from the buffer without copying.

```cpp
// Write:  [ts:8][x:8][y:8][z:8]  = 32 bytes
void encode(char* buf, uint64_t ts, double x, double y, double z) {
    std::memcpy(buf + 0,  &ts, 8);
    std::memcpy(buf + 8,  &x,  8);
    std::memcpy(buf + 16, &y,  8);
    std::memcpy(buf + 24, &z,  8);
}
```

Readers access fields by offset without deserializing the whole message.
This is essentially what FlatBuffers and Cap'n Proto do at scale.

### 2.4 JSON — The Universal Tax

JSON trades 10–100× performance for human-readability and language
interoperability.  Fine for config files and REST APIs, painful for
high-frequency IPC.  `sprintf`/`sscanf` is the simplest JSON-ish approach
without pulling in a library.

---

## 3. Type Erasure: Concept/Model Pattern

Type erasure lets you store objects of any type behind a uniform interface
without inheritance at the user's call site.

### 3.1 The Classic Concept/Model

```cpp
class AnyDrawable {
    struct Concept {                     // Abstract interface
        virtual void draw() const = 0;
        virtual ~Concept() = default;
    };
    template<typename T>
    struct Model : Concept {             // Wraps any T with draw()
        T obj;
        Model(T o) : obj(std::move(o)) {}
        void draw() const override { obj.draw(); }
    };
    std::unique_ptr<Concept> pimpl_;
public:
    template<typename T>
    AnyDrawable(T obj) : pimpl_(std::make_unique<Model<T>>(std::move(obj))) {}
    void draw() const { pimpl_->draw(); }
};
```

This is how `std::function`, `std::any`, and `std::move_only_function` work
internally.  The user never writes a base class — the Model wrapper
generates the vtable automatically.

### 3.2 Small Buffer Optimization (SBO)

Heap allocation on every type-erased construction is expensive.  SBO stores
small objects inline in a fixed buffer:

```
AnyCallable layout with SBO:
┌────────────────────────────────────────┐
│ vtable_ptr (8 bytes)                   │
│ ┌────────────────────────────────────┐ │
│ │ inline buffer (32 bytes)           │ │  ← small objects live here
│ │ [                                ] │ │
│ └────────────────────────────────────┘ │
│ is_local flag (1 byte)                 │
└────────────────────────────────────────┘

If sizeof(T) <= 32 && alignof(T) <= alignof(max_align_t):
    placement-new into inline buffer
Else:
    heap-allocate via new Model<T>(...)
```

`std::function` typically uses SBO with a 16–32 byte buffer.  Lambdas that
capture only a pointer or two fit inline; larger captures spill to the heap.

### 3.3 Performance Hierarchy

| Mechanism          | Indirection  | Allocation  | Typical ns/call |
|--------------------|-------------|-------------|-----------------|
| Direct call        | None        | None        | ~1              |
| `std::function`    | vtable + SBO| Possible    | ~3–8            |
| Type-erased SBO    | vtable      | None (SBO)  | ~3–5            |
| Virtual dispatch   | vtable      | Heap        | ~2–4            |
| `std::function` (heap) | vtable  | Always      | ~8–15           |

The SBO path is competitive with raw virtual dispatch when the object fits
in the inline buffer.

---

## 4. Plugin Architecture with dlopen

Dynamic loading lets a program discover and load code at runtime without
recompilation.

### 4.1 The dlopen API

```cpp
void* handle = dlopen("./libplugin.so", RTLD_LAZY);
auto create = (Widget*(*)()) dlsym(handle, "create_widget");
Widget* w = create();
w->run();
dlclose(handle);
```

Link with `-ldl`.

### 4.2 Plugin Contract Pattern

Define a C-linkage factory function that plugins must export:

```cpp
// plugin_api.h — shared between host and plugins
struct Plugin {
    virtual void execute() = 0;
    virtual ~Plugin() = default;
};
extern "C" Plugin* create_plugin();  // each .so implements this
```

The host `dlopen`s each `.so`, calls `dlsym("create_plugin")`, and gets back
a polymorphic object.  This is how audio plugins (VST), game engines, and
database extensions work.

### 4.3 Safety Considerations

- Always check `dlerror()` after `dlopen`/`dlsym`.
- Use `RTLD_NOW` during development to catch missing symbols early.
- Never `dlclose` while objects from the plugin are still alive.
- Plugin ABI must match the host — same compiler, same stdlib, same flags.

---

## 5. IPC Latency Comparison

Practical measurements on a typical Linux x86-64 system:

| IPC Method             | Latency (one-way) | Throughput (msgs/s) | Kernel crossings |
|------------------------|--------------------|---------------------|------------------|
| Shared memory + atomic | 50–200 ns          | 5–20 M              | 0                |
| Unix domain socket     | 1–5 µs             | 200k–1M             | 2                |
| TCP loopback           | 5–20 µs            | 50k–200k            | 2+               |
| Pipe                   | 2–8 µs             | 100k–500k           | 2                |
| POSIX message queue    | 1–5 µs             | 200k–1M             | 2                |
| D-Bus                  | 50–200 µs          | 5k–20k              | 4+               |
| gRPC (loopback)        | 100–500 µs         | 2k–10k              | 2+ (+ proto)     |

Shared memory wins by 10–1000× because there are zero kernel crossings after
the initial mmap.  This is why robotics middleware (ROS 2 Fast-DDS with SHM
transport, Iceoryx) and financial trading systems prefer shared memory IPC.

---

## 6. Putting It Together: Software-Defined Architecture

Modern C++ systems combine these patterns:

1. **Shared memory ring buffer** for zero-copy IPC between processes
2. **FlatBuffer-style serialization** for schema evolution without copying
3. **Type erasure** for plugin interfaces without forcing inheritance on users
4. **dlopen** for runtime extensibility

Example: a sensor fusion pipeline where each stage is a plugin loaded at
startup, communicating via shared memory ring buffers with FlatBuffer messages.
The host knows nothing about the concrete sensor types — type erasure hides
the details behind a uniform `SensorReader` interface.

---

## Exercises This Week

| Exercise | Topic                          | Key Concepts                            |
|----------|--------------------------------|-----------------------------------------|
| ex01     | Shared memory IPC              | shm_open, mmap, seqlock, placement new  |
| ex02     | Serialization benchmark        | memcpy, binary, zero-copy, JSON         |
| ex03     | Type erasure                   | Concept/Model, SBO, move-only erasure   |
| ex04     | Plugin loader (future)         | dlopen, dlsym, factory pattern          |
| ex05     | Ring buffer IPC (future)       | SPSC lock-free, cache-line padding      |
| puzzle01 | ABI mismatch trap              | struct padding across compilers         |
| puzzle02 | False sharing in shared memory | cache-line contention measurement       |

---

## References

- *C++ Software Design* — Klaus Iglberger (Type Erasure chapters)
- *The Linux Programming Interface* — Michael Kerrisk (Ch. 49–54: Shared Memory, Semaphores)
- FlatBuffers internals: https://flatbuffers.dev/flatbuffers_internals.html
- Iceoryx (zero-copy IPC for robotics): https://iceoryx.io
- Sean Parent's "Inheritance Is The Base Class of Evil" talk (type erasure motivation)
