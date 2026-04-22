# Exercise01 — Publisher, Subscriber, Timer
### Companion exercises for `01-nodes-topics-actions.md`

**Estimated time:** 60 minutes  
**Prerequisite:** [01-nodes-topics-actions.md](../01-nodes-topics-actions.md)

**Self-assessment guide:** Write your answer on paper before expanding each details block.
If you can answer Section A without peeking, you understand the executor model well enough
to diagnose silent subscriber failures in field logs.

---

## Overview

A large share of "node running but not publishing" incidents come from exactly three root
causes: the executor is blocked by a long callback, a QoS incompatibility prevents
message delivery, or the publisher/subscriber was created but never has `spin()` called.
These exercises drill each scenario from first principles, then ask you to build complete
nodes from scratch.

---

## Section A — Conceptual Questions

**A1.** What does `rclpy.spin(node)` block on?  
Describe exactly what the thread is doing while it waits for the next event, and what
happens to the spin call when your callback takes 5 seconds to execute.

<details><summary>Answer</summary>

`rclpy.spin(node)` calls into the `SingleThreadedExecutor.spin()` loop, which repeatedly
calls `executor.spin_once()`. Inside `spin_once()`, the thread blocks on
`rcl_wait()` — a system call that sleeps until at least one waitable handle (subscription,
timer, service, action) fires or the timeout expires.

**While it waits:** The thread is parked in `epoll_wait` (on Linux). CPU usage is
effectively zero. No callbacks run.

**What happens when your callback takes 5 seconds:**
The executor is single-threaded. It cannot call any other callback until your 5-second
callback returns. This means:
- All timers that fire during those 5 seconds queue up but are NOT missed — they are
  delivered as a burst when your callback finally returns.
- All incoming subscription messages that arrive during the 5 seconds accumulate in the
  DDS receive queue (up to `depth` messages if using `KEEP_LAST`).
- The calling code does not see any output for 5 seconds.

In practice, if a timer at 10Hz fires while a 5-second callback is running, you get:
50 pending timer events → burst of 50 callbacks delivered back-to-back as fast as the
executor can run them. This is the "callback pile-up" failure mode that causes apparent
jitter even when the underlying signal is smooth.

**Fix:** Keep all callbacks short (< 1 ms for high-rate topics). Move slow work to a
separate thread or use `MultiThreadedExecutor`.

</details>

- [ ] Done

---

**A2.** A publisher uses `KEEP_LAST(1)` on a 100 Hz topic. Describe three distinct
scenarios where this QoS choice is correct and one where it would be the wrong choice.

<details><summary>Answer</summary>

**Correct choices for `KEEP_LAST(1)`:**

1. **Current pose / odometry:** You always want the most recent position estimate.
   The consumer (controller) only cares where the robot is *right now*. Replaying stale
   positions would corrupt the control loop. `KEEP_LAST(1)` ensures the subscriber always
   gets the freshest sample.

2. **Sensor status / heartbeat:** A watchdog subscriber only wants to know the last
   reported health state. Queuing 100 stale "all-good" messages wastes memory and delays
   detection of a newly-arrived "fault" message.

3. **Display / visualization topics:** An RViz subscriber rendering at 10Hz on a 100Hz
   topic. No value in rendering 90 intermediate frames — only the latest matters.

**Wrong choice:**
- **Commanded velocity log / audit trail:** If a safety monitor must receive and record
  *every* velocity command ever issued (e.g., for post-incident replay), `KEEP_LAST(1)` would
  silently drop messages when the executor is busy. Use `KEEP_ALL` or a large
  `KEEP_LAST(N)` with a reliable transport.

</details>

- [ ] Done

---

**A3.** What happens if a subscriber callback takes longer than the publish rate of its
topic? Walk through what happens to message delivery over 3 publish periods.

<details><summary>Answer</summary>

Assume publish rate = 100 Hz (period = 10 ms) and callback takes 25 ms.

**Period 1 (t=0 ms):** Message M1 arrives. Executor picks it up and calls the callback.
The callback runs for 25 ms.

**Period 2 (t=10 ms):** Publisher emits M2 while the callback from M1 is still running.
M2 enters the DDS receive queue. The executor cannot call the callback for M2 because the
single thread is occupied with M1.

**Period 3 (t=20 ms):** Publisher emits M3. With `KEEP_LAST(1)`, M3 *overwrites* M2 in
the queue. M2 is silently dropped.

**At t=25 ms:** M1 callback returns. Executor calls the callback for M3 (M2 was dropped).

**Result:** Effective callback rate = 1/25ms = 40 Hz (not 100 Hz). 60% of messages are
dropped. This is not an error — it is by design when using `KEEP_LAST(1)`. The callback
always gets the latest message, never a stale one.

With `KEEP_LAST(10)`, messages queue up until memory is exhausted or the executor catches
up. If the callback is always slower than the rate, the queue fills up, and when the queue
is full, the oldest message is dropped (oldest-first eviction).

**Key insight:** A callback that is slower than the publish rate will never process all
messages in real-time with a single-threaded executor. This is normal for display/logging
callbacks but a bug for control-loop callbacks.

</details>

- [ ] Done

---

**A4.** What is the difference between `rclpy.Node` and `rclpy_lifecycle.LifecycleNode`?
Name at least three states of a lifecycle node and explain what operation is typically
performed in each.

<details><summary>Answer</summary>

**`rclpy.Node`:** A standard ROS2 node. Publishers, subscribers, and timers become active
immediately when created in `__init__`. There is no concept of "not ready yet" — as soon
as `__init__` returns, all communication primitives are live.

**`rclpy_lifecycle.LifecycleNode`:** A node that implements the Managed Node interface
(REP-2001). Communication primitives are created but remain inactive until the node
transitions through its state machine.

**Three key states and their operations:**

| State | Description | Typical operation |
|---|---|---|
| `UNCONFIGURED` | Node created, `__init__` done | Nothing active. No publishers, no timers. |
| `INACTIVE` (after `on_configure`) | Resources allocated | Create publishers/subscribers/timers, load params, connect to hardware — but do NOT start sending/receiving data. |
| `ACTIVE` (after `on_activate`) | Node is fully operational | Activate publishers (allow `publish()` to transmit), start timers, begin reading from hardware. |
| `FINALIZED` (after `on_cleanup` / `on_shutdown`) | Resources released | Destroy publishers, disconnect from hardware, write logs. |

**Why lifecycle nodes matter for navigation stacks:** Nav2 uses lifecycle nodes so the
entire stack can be configured (parameters loaded, TF initialized) and then activated in a
coordinated sequence. Without this, a node might start publishing data before the TF tree
is ready, causing a burst of lookup failures on startup.

</details>

- [ ] Done

---

**A5.** Name the three things `create_publisher` requires. Then describe what happens at
the DDS layer when two nodes publish to the same topic but with incompatible QoS.

<details><summary>Answer</summary>

**`create_publisher` requires:**
1. **Message type** — the Python/C++ class for the message (e.g., `std_msgs.msg.Float64`)
2. **Topic name** — a string (e.g., `'/robot/speed'`)
3. **QoS profile** — a `rclpy.qos.QoSProfile` object or integer depth (default: `KEEP_LAST(10)`)

**What happens at the DDS layer with incompatible QoS:**

DDS negotiates QoS via *offered vs requested* incompatibility. The rule is:
- The publisher *offers* a QoS profile.
- The subscriber *requests* a QoS profile.
- The subscription is only established if the publisher's offer is *at least as good* as the
  subscriber's request.

Example: publisher offers `BEST_EFFORT`, subscriber requests `RELIABLE`.
`BEST_EFFORT < RELIABLE` → the offered QoS is weaker than requested → **incompatible**.

**Concrete outcome:**
- No message is ever delivered. The subscriber's callback is never called.
- Both nodes log an `IncompatibleQosEvent` warning (though it is often missed).
- `ros2 topic info /topic_name --verbose` shows the QoS mismatch.

**This is completely silent** — the subscriber node is ACTIVE, the topic appears to have
data (`ros2 topic hz`), but the callback receives nothing. This is one of the most common
"why isn't my node getting data" bugs.

</details>

- [ ] Done

---

## Section B — Spot the Bug

Each snippet below has exactly one bug. Read the code, identify the bug, and write the fix
before expanding the answer.

---

**B1.** This node is supposed to publish `/status` at 1Hz. It never publishes anything.

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class StatusNode(Node):
    def __init__(self):
        super().__init__('status_node')
        self.pub = self.create_publisher(String, '/status', 10)
        self.timer = self.create_timer(1.0, self.timer_cb)

    def timer_cb(self):
        msg = String()
        msg.data = 'ok'
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = StatusNode()
    # nothing else
```

<details><summary>Answer</summary>

**Bug:** `main()` creates the node but never calls `rclpy.spin(node)`. The executor never
runs, so the timer callback is never invoked.

**Fix:**
```python
def main():
    rclpy.init()
    node = StatusNode()
    rclpy.spin(node)          # <-- this line is missing
    node.destroy_node()
    rclpy.shutdown()
```

Without `spin()`, the process creates all ROS2 primitives (publisher, timer) and then
immediately returns from `main()`. The timer is registered but never fired because the
executor never has a chance to poll the timer handle.

</details>

- [ ] Done

---

**B2.** This subscriber is intended to cache the latest message. Under concurrent access
the cached value is sometimes corrupted.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from rclpy.executors import MultiThreadedExecutor
import threading

class ImuCache(Node):
    def __init__(self):
        super().__init__('imu_cache')
        self.latest_imu = None
        self.sub = self.create_subscription(Imu, '/imu', self.cb, 10)

    def cb(self, msg):
        self.latest_imu = msg   # shared write, no lock

    def get_latest(self):
        return self.latest_imu  # shared read, no lock

def main():
    rclpy.init()
    node = ImuCache()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.spin()
```

<details><summary>Answer</summary>

**Bug:** `self.latest_imu` is written in `cb()` (called by executor threads) and read in
`get_latest()` (potentially called from any thread). With a `MultiThreadedExecutor`, multiple
callbacks can run in parallel. The read/write is not atomic in CPython for complex objects
(the GIL protects simple attribute assignments, but the read-modify idiom `if node.get_latest()
is not None` can still race). On non-CPython runtimes (PyPy, future no-GIL Python) this is a
data race.

**Fix:** Protect the shared variable with a lock.

```python
import threading

class ImuCache(Node):
    def __init__(self):
        super().__init__('imu_cache')
        self.latest_imu = None
        self._lock = threading.Lock()          # <-- add lock
        self.sub = self.create_subscription(Imu, '/imu', self.cb, 10)

    def cb(self, msg):
        with self._lock:                        # <-- acquire before write
            self.latest_imu = msg

    def get_latest(self):
        with self._lock:                        # <-- acquire before read
            return self.latest_imu
```

**Note:** With `SingleThreadedExecutor`, the lock is not strictly required because
callbacks are serialised. The bug only manifests with `MultiThreadedExecutor`. This is a
common source of hard-to-reproduce races in sensor fusion nodes.

</details>

- [ ] Done

---

**B3.** This QoS config is intended to receive `/map` from `map_server`, which publishes
with `TRANSIENT_LOCAL` durability. The subscriber never receives the map on startup.

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

map_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,   # <-- look here
    depth=1,
)

self.map_sub = self.create_subscription(
    OccupancyGrid, '/map', self.map_cb, map_qos
)
```

<details><summary>Answer</summary>

**Bug:** The subscriber uses `VOLATILE` durability. `map_server` publishes `/map` with
`TRANSIENT_LOCAL`, meaning it *stores* the last published map and delivers it to any new
subscriber that connects after publication.

The `VOLATILE` subscriber says "I do not want stored messages — only live ones." Since the
map was published before this node started, the subscriber misses it entirely and waits
forever.

**Fix:** Match the publisher's durability.

```python
map_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,   # <-- must match publisher
    depth=1,
)
```

**Rule:** For `TRANSIENT_LOCAL` to work, *both* publisher and subscriber must use
`TRANSIENT_LOCAL`. If either uses `VOLATILE`, the late-join delivery is disabled and
new subscribers miss any data published before they connected.

This is the most common cause of "costmap never loads" or "static obstacles not visible"
on nav stack startup — the costmap's static layer subscribes to `/map` with the wrong
durability.

</details>

- [ ] Done

---

## Section C — Build Exercises

### C1. Minimal Publisher

Write a **complete**, runnable Python ROS2 node that publishes a `Float64` to `/robot/speed`
at 10 Hz. Include all imports, the `Node` subclass, `main()`, and the `spin()` call.

The published value should be `1.0` (a constant placeholder).

```python
# TODO: fill in here
import rclpy
# TODO: remaining imports

class SpeedPublisher(Node):
    def __init__(self):
        super().__init__('speed_publisher')
        # TODO: create publisher on /robot/speed, Float64, depth 10
        # TODO: create timer at 10Hz calling self.timer_cb

    def timer_cb(self):
        # TODO: create Float64 message, set data=1.0, publish it
        pass

def main(args=None):
    # TODO: init, create node, spin, destroy, shutdown
    pass

if __name__ == '__main__':
    main()
```

<details><summary>Answer</summary>

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class SpeedPublisher(Node):
    def __init__(self):
        super().__init__('speed_publisher')
        self.pub = self.create_publisher(Float64, '/robot/speed', 10)
        self.timer = self.create_timer(0.1, self.timer_cb)   # 1/10Hz = 0.1s

    def timer_cb(self):
        msg = Float64()
        msg.data = 1.0
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SpeedPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

**Key points:**
- `create_timer(0.1, ...)` → period in seconds; 0.1s = 10 Hz.
- `depth=10` is the QoS history depth (KEEP_LAST(10)).
- `destroy_node()` and `shutdown()` ensure clean teardown when Ctrl-C interrupts spin.

</details>

- [ ] Done

---

### C2. Subscriber with State and Fusion Timer

Write a node that:
1. Subscribes to `/odom` (`nav_msgs/Odometry`) and stores the latest linear x-speed.
2. Subscribes to `/imu` (`sensor_msgs/Imu`) and stores the latest linear x-acceleration.
3. On a **1 Hz** timer, publishes a `Float64` to `/fusion/speed_comparison` equal to the
   absolute difference between the latest odom speed and the latest IMU linear x-acceleration.
4. If no message has been received yet on either topic, skip the publish and log a warning.

```python
import rclpy
from rclpy.node import Node
# TODO: remaining imports

class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')
        self._latest_odom_speed = None   # float or None
        self._latest_imu_accel  = None   # float or None

        # TODO: subscriber to /odom
        # TODO: subscriber to /imu
        # TODO: publisher on /fusion/speed_comparison
        # TODO: 1Hz timer

    def odom_cb(self, msg):
        # TODO: store msg.twist.twist.linear.x
        pass

    def imu_cb(self, msg):
        # TODO: store msg.linear_acceleration.x
        pass

    def timer_cb(self):
        # TODO: guard against None, compute abs diff, publish
        pass

def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

<details><summary>Answer</summary>

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64


class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')
        self._latest_odom_speed = None
        self._latest_imu_accel  = None

        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.odom_cb, 10)
        self.sub_imu  = self.create_subscription(
            Imu, '/imu', self.imu_cb, 10)
        self.pub = self.create_publisher(Float64, '/fusion/speed_comparison', 10)
        self.timer = self.create_timer(1.0, self.timer_cb)

    def odom_cb(self, msg: Odometry):
        self._latest_odom_speed = msg.twist.twist.linear.x

    def imu_cb(self, msg: Imu):
        self._latest_imu_accel = msg.linear_acceleration.x

    def timer_cb(self):
        if self._latest_odom_speed is None or self._latest_imu_accel is None:
            self.get_logger().warn(
                'Waiting for sensor data: '
                f'odom={self._latest_odom_speed is not None}, '
                f'imu={self._latest_imu_accel is not None}'
            )
            return

        diff = Float64()
        diff.data = abs(self._latest_odom_speed - self._latest_imu_accel)
        self.pub.publish(diff)


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

**Design notes:**
- Storing `None` as sentinel allows distinguishing "never received" from "received zero".
- `get_logger().warn(...)` uses the node's logger so the message includes the node name,
  timestamp, and severity — better than `print()` for diagnosing startup hangs.
- `msg.twist.twist.linear.x` — note the double `twist`; `Odometry.twist` is a
  `TwistWithCovariance`, and its inner `twist` field is the actual `Twist`.

</details>

- [ ] Done

---

### C3. Lifecycle Node Publisher

Convert the publisher from C1 into a **Lifecycle Node**. The publisher should only
be created in `on_configure()` and should only be actively used in the `ACTIVE` state.
Show `on_activate()` and `on_deactivate()` callbacks.

```python
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State
# TODO: remaining imports

class LifecycleSpeedPublisher(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_speed_publisher')
        self._pub = None
        self._timer = None

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """Called when transitioning UNCONFIGURED → INACTIVE."""
        # TODO: create publisher (but do NOT start timer yet)
        self.get_logger().info('Configured.')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Called when transitioning INACTIVE → ACTIVE."""
        # TODO: create the 10Hz timer
        self.get_logger().info('Activated.')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Called when transitioning ACTIVE → INACTIVE."""
        # TODO: destroy/cancel the timer (publisher stays allocated)
        self.get_logger().info('Deactivated.')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        """Called when transitioning INACTIVE → UNCONFIGURED."""
        # TODO: destroy publisher
        self.get_logger().info('Cleaned up.')
        return TransitionCallbackReturn.SUCCESS

    def timer_cb(self):
        # TODO: publish Float64(data=1.0) to /robot/speed
        pass

def main(args=None):
    rclpy.init(args=args)
    node = LifecycleSpeedPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

<details><summary>Answer</summary>

```python
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State
from std_msgs.msg import Float64


class LifecycleSpeedPublisher(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_speed_publisher')
        self._pub = None
        self._timer = None

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self._pub = self.create_lifecycle_publisher(Float64, '/robot/speed', 10)
        self.get_logger().info('Configured: publisher created (inactive).')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        super().on_activate(state)              # activates the managed publisher
        self._timer = self.create_timer(0.1, self.timer_cb)
        self.get_logger().info('Activated: timer started.')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        if self._timer is not None:
            self._timer.cancel()
            self.destroy_timer(self._timer)
            self._timer = None
        super().on_deactivate(state)            # deactivates the managed publisher
        self.get_logger().info('Deactivated: timer stopped.')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        if self._pub is not None:
            self.destroy_publisher(self._pub)
            self._pub = None
        self.get_logger().info('Cleaned up: publisher destroyed.')
        return TransitionCallbackReturn.SUCCESS

    def timer_cb(self):
        if self._pub is not None and self._pub.is_activated:
            msg = Float64()
            msg.data = 1.0
            self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LifecycleSpeedPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

**Key points:**
- `create_lifecycle_publisher()` (not `create_publisher()`) returns a managed publisher
  that knows about the node's lifecycle state. Its `is_activated` property returns `False`
  in `INACTIVE` state even if `publish()` is called.
- `super().on_activate(state)` must be called to trigger the base class mechanism that
  sets `is_activated = True` on all managed publishers. Missing this is a common bug.
- The timer is created in `on_activate` (not `on_configure`) so it only fires when the
  node is in the `ACTIVE` state.
- Destroying the timer in `on_deactivate` is important — if you only `cancel()` without
  `destroy_timer()`, the timer handle remains registered with the executor and will be
  checked on every `spin_once()`, wasting a small amount of time.

**To trigger transitions manually:**
```bash
ros2 lifecycle set /lifecycle_speed_publisher configure
ros2 lifecycle set /lifecycle_speed_publisher activate
ros2 lifecycle set /lifecycle_speed_publisher deactivate
```

</details>

- [ ] Done
