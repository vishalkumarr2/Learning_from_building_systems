# Debug Session 3 — cmd_vel Gap: "The Robot Drifts Between Waypoints"

**Skills tested:** Two-layer architecture timing, ROS2 real-time issues, Jetson↔MCU coordination
**Difficulty:** Advanced
**Time estimate:** 60–90 minutes

---

## The Scenario

**Field report from warehouse (automated site, no operator present):**

> Multiple robots intermittently drift 5–15 cm off the planned path during corridor transit. The drift happens mid-straight-line, not at corners. It corrects within 1–2 seconds but causes near-misses with rack posts. AMCL localization is accurate (checked). Path planner output is correct (checked). The issue happens more under heavy CPU load (many robots active, cameras streaming).

**What you have:**
- Nav2 controller (Regulated Pure Pursuit) runs at 20 Hz
- Motor bridge (SPI) runs at 1 kHz
- Velocity smoother at 50 Hz
- Jetson Xavier NX, 6 cores, running ROS2 + AMCL + 2 cameras + DDS

---

## Step 1 — Gather Timing Data

### 1.1 cmd_vel Publication Intervals

Log the time between consecutive cmd_vel publications:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class CmdVelMonitor(Node):
    def __init__(self):
        super().__init__('cmdvel_monitor')
        self.last_time = None
        self.intervals = []
        self.create_subscription(Twist, '/cmd_vel', self.cb, 10)
    
    def cb(self, msg):
        now = self.get_clock().now()
        if self.last_time:
            dt = (now - self.last_time).nanoseconds / 1e6  # ms
            self.intervals.append(dt)
            if dt > 80:  # Flag gaps > 80 ms (4× nominal)
                self.get_logger().warn(f'cmd_vel gap: {dt:.1f} ms')
        self.last_time = now
```

### 1.2 Results

```
cmd_vel interval statistics (5 minute window):
  Mean:   50.2 ms  (expected: 50 ms = 20 Hz ✓)
  Median: 49.8 ms
  P95:    62.3 ms
  P99:    98.7 ms   ← Almost 2× period
  Max:    187.4 ms  ← Nearly 4× period!
  
  Gaps > 80 ms:  23 occurrences in 5 minutes
  Gaps > 100 ms: 8 occurrences
  Gaps > 150 ms: 2 occurrences
```

**Finding:** The Nav2 controller is supposed to publish cmd_vel every 50 ms, but under load, it occasionally takes 100–187 ms. During these gaps, the motor bridge sends **the last cmd_vel** repeatedly.

---

## Step 2 — Why Gaps Cause Drift

### 2.1 The Straight-Line Case

The robot is driving straight at $v = 0.5$ m/s, $\omega = 0$. A small perturbation makes the robot rotate slightly ($\omega_{actual} = 0.01$ rad/s). Normally, the Nav2 controller corrects this in 50 ms:

```
t=0:    cmd_vel: v=0.5, ω=0.0     Robot heading: 0.000 rad
t=50ms: Nav2 sees slight drift     Heading drifted to: 0.0005 rad
        cmd_vel: v=0.5, ω=-0.01   (corrective angular velocity)
t=100ms: Corrected back            Heading: 0.0001 rad
```

**With a 150 ms gap:**

```
t=0:    cmd_vel: v=0.5, ω=0.0     Robot heading: 0.000 rad
t=50ms: Nav2 SHOULD correct...     Heading: 0.0005 rad
        But Nav2 is delayed!       Motor bridge sends ω=0.0 (stale)
t=100ms: Still delayed!            Heading: 0.0015 rad (drift growing)
t=150ms: Nav2 finally updates      Heading: 0.0025 rad = 0.14°
        cmd_vel: v=0.5, ω=-0.05   (larger correction needed)
t=200ms: Heading: 0.001 rad       Still correcting...
```

**Cross-track error from 150 ms gap:**

$$e_{cross} = v \cdot \sin(\Delta\theta) \cdot t_{gap} \approx 0.5 \times 0.0025 \times 0.15 = 0.19 \text{ mm}$$

Wait — that's tiny. So where does the 5–15 cm drift come from?

### 2.2 The Real Mechanism: Compounding

One gap doesn't cause 15 cm drift. But **multiple gaps in sequence** do. And the real problem is more subtle:

**The estimator is also affected.** When the Nav2 controller's timer fires late:
1. The TF tree (`odom` → `base_link`) is stale
2. The controller computes the error using an old robot pose
3. The correction is **wrong** — it corrects for where the robot WAS, not where it IS
4. This wrong correction causes a temporary overshoot in the other direction
5. The next cycle (also potentially late) compounds the error

### 2.3 DDS Queue Saturation

Under heavy CPU load, the ROS2 DDS middleware also suffers:

```
Camera 1 topics: 30 Hz × 2 MB = 60 MB/s
Camera 2 topics: 30 Hz × 2 MB = 60 MB/s  
LiDAR:           10 Hz × 100 KB = 1 MB/s
AMCL particles:  5 Hz × 50 KB = 0.25 MB/s
cmd_vel:         20 Hz × 48 bytes = ~1 KB/s  ← Tiny!

Total DDS load: ~120 MB/s
```

The cmd_vel message is tiny but competes with camera topics for DDS thread time. When the DDS executor is busy serializing a 2 MB image, cmd_vel publication can be delayed by **up to one image serialization time** (~5–15 ms on Xavier NX).

---

## Step 3 — Diagnose Root Causes

### 3.1 CPU Profiling

```bash
# On the Jetson
sudo perf top -p $(pgrep nav2_controller)
```

```
  35.2%  librcl.so          [.] rmw_take
  22.1%  nav2_controller    [.] RegulatedPurePursuit::computeVelocityCommands
   8.3%  libfastrtps.so     [.] eprosima::fastrtps::rtps::MessageReceiver
   7.1%  libtf2_ros.so      [.] tf2_ros::BufferCore::lookupTransform
```

**Finding:** 35% of the controller's time is spent in `rmw_take` (receiving DDS messages). When other nodes flood DDS, the controller spends more time waiting for the executor.

### 3.2 Timer Jitter Analysis

```python
# Measure actual timer callback intervals
import statistics

intervals = [50.1, 49.8, 50.3, 51.2, 48.9, 50.0, 50.5, 
             89.3,  # ← gap
             10.7,  # ← catch-up (timer fires immediately after gap)
             50.2, 50.1, 50.0, 50.3,
             102.5, # ← larger gap
             47.5,  # ← partial catch-up
             50.0]

print(f"Jitter std: {statistics.stdev(intervals):.1f} ms")
# Result: 15.3 ms standard deviation (should be < 2 ms for 20 Hz)
```

---

## Step 4 — Fix It (Multiple Layers)

### 4.1 Fix 1: Real-Time Priority for Nav2 Controller

```bash
# Set SCHED_FIFO priority for the controller process
sudo chrt -f 50 ros2 run nav2_controller controller_server
```

Or in the ROS2 launch file:

```python
# Use ROS2 real-time executor
from rclpy.executors import MultiThreadedExecutor

executor = MultiThreadedExecutor(num_threads=2)
executor.add_node(controller_node)
# Pin to isolated CPU cores
os.sched_setaffinity(0, {4, 5})  # Cores 4-5 for controller
```

### 4.2 Fix 2: Separate DDS Domain for Control

Isolate control topics from camera topics:

```yaml
# nav2_controller params
ros__parameters:
  use_sim_time: false
  # Use intra-process communication for cmd_vel
  # (bypasses DDS serialization entirely)
  use_intra_process_comms: true
```

### 4.3 Fix 3: cmd_vel Staleness Detection on Motor Bridge

The motor bridge should detect and handle stale commands:

```python
class MotorBridge(Node):
    def __init__(self):
        super().__init__('motor_bridge')
        self.cmd_vel_age_threshold = 0.08  # 80 ms = 1.5× period
        self.decel_rate = 0.5  # m/s² when cmd_vel is stale
        
    def spi_callback(self):
        age = (self.get_clock().now() - self.cmd_vel_stamp).nanoseconds / 1e9
        
        if age > self.cmd_vel_age_threshold:
            # STALE: decelerate smoothly instead of holding
            self.get_logger().warn(f'cmd_vel stale: {age*1000:.0f} ms')
            
            # Reduce speed toward zero
            v = self.current_v
            if abs(v) > 0.01:
                sign = 1.0 if v > 0 else -1.0
                v -= sign * self.decel_rate * 0.001  # 1 ms step
                v = max(0, v) if sign > 0 else min(0, v)
            
            # Hold angular velocity at zero (don't turn while stale)
            omega = 0.0
        else:
            v, omega = self.last_cmd_vel
        
        self.send_to_mcu(v, omega)
```

### 4.4 Fix 4: Velocity Smoother with Gap Awareness

```python
class GapAwareVelocitySmoother:
    def __init__(self, max_accel, max_decel, nominal_dt):
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.nominal_dt = nominal_dt
        self.current_v = 0.0
        self.current_omega = 0.0
        self.last_update_time = None
    
    def smooth(self, target_v, target_omega, current_time):
        if self.last_update_time is not None:
            actual_dt = current_time - self.last_update_time
            
            # If actual_dt > 2× nominal, we had a gap
            if actual_dt > 2.0 * self.nominal_dt:
                # Use nominal_dt for smoothing (don't allow huge jumps)
                dt = self.nominal_dt
                # Log the gap
                print(f"Gap detected: {actual_dt*1000:.0f} ms")
            else:
                dt = actual_dt
        else:
            dt = self.nominal_dt
        
        self.last_update_time = current_time
        
        # Standard acceleration limiting
        dv = target_v - self.current_v
        max_dv = (self.max_accel if dv > 0 else self.max_decel) * dt
        dv = max(min(dv, max_dv), -max_dv)
        self.current_v += dv
        
        dw = target_omega - self.current_omega
        max_dw = self.max_accel * dt
        dw = max(min(dw, max_dw), -max_dw)
        self.current_omega += dw
        
        return self.current_v, self.current_omega
```

---

## Step 5 — Verify

### 5.1 Before Fixes

```
cmd_vel gaps > 80 ms:  23 per 5 minutes
Max path deviation:     15.2 cm
Near-misses per hour:   3.1
```

### 5.2 After Fixes (All Four Layers)

```
cmd_vel gaps > 80 ms:  2 per 5 minutes (RT priority)
Max path deviation:     2.1 cm (gap detection + decel)
Near-misses per hour:   0.0
```

---

## Step 6 — The Systemic Lesson

### 6.1 The Real-Time Budget

On a shared Jetson, you have a **timing budget** just like a financial budget:

```
Jetson Xavier NX: 6 cores, 8 GB RAM

Core allocation:
  Cores 0-1: Linux kernel, DDS, cameras, misc
  Cores 2-3: AMCL, costmap, planner (can be slow)
  Core  4:   Nav2 controller (20 Hz, SCHED_FIFO)
  Core  5:   Motor bridge (1 kHz, SCHED_FIFO)

CPU budget at 20 Hz (50 ms per cycle):
  TF lookup:              2 ms
  Costmap query:          5 ms
  RPP computation:        3 ms
  DDS publish:            1 ms
  ─────────────────────────────
  Total:                 11 ms (22% utilization, safe)
  Headroom:              39 ms (78% margin for jitter)
```

### 6.2 Why This Bug Is Hard to Find

1. **Intermittent:** Only happens under load. Works perfectly in lab testing.
2. **Small individually:** Each gap causes < 1 mm error. Only compounding makes it visible.
3. **Correct at every layer:** Nav2's algorithm is correct. Motor bridge is correct. SPI is correct. The bug is in the **timing between layers**.
4. **Not a "bug" in code:** The code is correct. The deployment configuration (CPU affinity, priorities, DDS tuning) is the issue.

---

## Checkpoint Questions

1. The Nav2 controller runs at 20 Hz. A cmd_vel gap of 150 ms is how many missed cycles?
2. Why does DDS queue saturation affect cmd_vel (48 bytes) when camera topics (2 MB) are the bottleneck?
3. The motor bridge detects stale cmd_vel and decelerates. Why decelerate instead of holding the last velocity?
4. `SCHED_FIFO` with priority 50 is used for the controller. What happens to other ROS2 nodes when the controller needs CPU?
5. Why is this bug almost impossible to reproduce in simulation?
6. The velocity smoother uses `nominal_dt` during gaps instead of `actual_dt`. Why? What would happen with `actual_dt`?
7. Core pinning (CPU affinity) is applied. What happens if the Jetson has only 4 cores and you try to pin 3 real-time processes to dedicated cores?

---

## Key Takeaways

- **cmd_vel gaps** are the most common source of path tracking drift in our robot
- **Root cause:** Linux scheduler jitter under CPU load — not a control algorithm bug
- **Defense in depth:** RT priority + CPU pinning + DDS isolation + staleness detection + graceful deceleration
- **Timing between layers** matters as much as the algorithms in each layer
- **Lab testing rarely catches this** because lab conditions have no CPU contention
- **Monitor cmd_vel intervals in production** — it's the canary for system health
