# Exercise02 — TF2 Broadcaster and Listener
### Companion exercises for `02-tf2-time-qos.md`

**Estimated time:** 60 minutes  
**Prerequisite:** [02-tf2-time-qos.md](../02-tf2-time-qos.md)

**Self-assessment guide:** Work Section A from memory. If you can correctly answer A2 and A3
without hints, you can diagnose `ExtrapolationException` errors in field bags without reading
source code.

---

## Overview

TF2 lookup failures are responsible for a disproportionate share of navigation stack failures
that are hard to diagnose because the error message (`ExtrapolationException`) is often printed
at WARN level and scrolls past unnoticed. This exercise set builds the muscle to reason about
the TF tree topology, understand time interpolation, and write broadcasters and listeners from
scratch.

---

## Section A — Conceptual Questions

**A1.** What is the difference between a `StaticTransformBroadcaster` and a
`TransformBroadcaster`? When should you use each?

<details><summary>Answer</summary>

| Property | `StaticTransformBroadcaster` | `TransformBroadcaster` |
|---|---|---|
| Publication rate | Published once on startup; re-broadcast on new subscriber connection | Published every time you call `sendTransform()` |
| Topic | `/tf_static` | `/tf` |
| QoS | `TRANSIENT_LOCAL` (stored, delivered to late joiners) | `BEST_EFFORT` or `RELIABLE` (not stored) |
| Use case | Fixed sensor mounts: laser bolted to chassis, camera mounted on arm | Moving joints: wheel odometry → base_link, map → odom from localization |

**Key rule:** If the transform *never changes at runtime* (physical mounting), use
`StaticTransformBroadcaster`. If the transform changes because the robot moves or a joint
moves, use `TransformBroadcaster`.

**Why it matters:** A static broadcaster publishes with `TRANSIENT_LOCAL` QoS, so any node
that joins the ROS2 graph after the broadcaster started will still receive the transform.
A dynamic broadcaster publishes on `/tf` which has no history — if a consumer starts after
a dynamic broadcaster, it misses all previous transforms and must wait for the next one.

</details>

- [ ] Done

---

**A2.** A `lookupTransform` call throws `tf2_ros.ExtrapolationException`. What does this
exception mean physically? Describe two different conditions that trigger it.

<details><summary>Answer</summary>

An `ExtrapolationException` means the TF2 buffer was asked for a transform at a time that
is **outside the range of available data** — the buffer cannot interpolate or extrapolate
reliably to answer the query.

**Condition 1 — Query timestamp is too new (into the future):**
The caller asks for `target_frame` at time `T`, but the TF buffer's most recent entry for
the relevant edge is at time `T - Δ` where `Δ` is larger than the TF buffer's tolerance.
This happens when there is a lag between the broadcaster and the listener — e.g., a 50ms
processing delay or a clock offset between machines in a distributed system.

*Fix:* Use `rclpy.time.Time()` (time=0) to request "the latest available transform", or
call `lookupTransform` with a timeout so TF2 waits for the data to arrive.

**Condition 2 — Query timestamp is too old:**
The caller asks for `target_frame` at a timestamp from a message that was received long
ago. TF2 buffers have a finite capacity (default 10 seconds). Transforms older than the
buffer window have been evicted. Using an old header stamp from a queued message can trigger
this condition.

*Fix:* Process messages promptly (don't queue indefinitely), or increase the TF buffer
duration with `tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))`.

</details>

- [ ] Done

---

**A3.** Why does `time=rclpy.time.Time()` (time = zero / "latest available") work
correctly for static frames but can give wrong results for dynamic frames?

<details><summary>Answer</summary>

`rclpy.time.Time()` (time = zero, also written as `Time(seconds=0)`) is a special sentinel
that tells TF2: *"give me the most recently available transform, regardless of timestamp."*

**For static frames (e.g., `base_link → laser_link`):**
The transform is constant — the laser is physically bolted at a fixed offset. The transform
at time T=0 (latest available) is the same as the transform at any other time. Returning
"latest" is always correct.

**For dynamic frames (e.g., `odom → base_link`):**
The transform changes continuously as the robot moves. When you call
`lookupTransform('odom', 'laser_link', Time())`, you get:
- `base_link → laser_link` at its (constant) latest time, AND
- `odom → base_link` at *its* latest time

But in a sensor fusion pipeline, you typically want to transform a laser scan that was taken
at a specific hardware timestamp `T_scan`. Using `Time()` returns the robot's *current*
position, not where it was when the scan was captured. For a robot moving at 0.5 m/s and a
100ms scan-to-processing delay, this error is 5 cm per scan — enough to smear obstacles in
the costmap.

**Correct approach for dynamic frames:**
```python
# Use the scan's own timestamp, not Time()
lookupTransform('odom', 'laser_link', scan_msg.header.stamp, timeout=Duration(seconds=0.1))
```

</details>

- [ ] Done

---

**A4.** What causes a `ConnectivityException`?

<details><summary>Answer</summary>

A `ConnectivityException` means there is **no path** in the TF tree between the requested
source frame and target frame. The frames exist in the TF buffer, but they are not connected
to each other.

**Common causes:**

1. **Missing edge in the tree:** The robot has both `map → odom` (published by localization)
   and `camera_link → camera_optical_link` (from the URDF), but `odom → base_link` is not
   being published because the odometry node crashed. The two subtrees are disconnected.

2. **Typo in frame names:** One node broadcasts `base_link` and another listens for
   `base_footprint`. These are two separate nodes in the TF graph with no edge between them.

3. **Node not started yet:** At startup, the URDF robot state publisher may not have started
   yet, so `base_link → laser_link` does not exist in the buffer.

**Diagnosis:**
```bash
ros2 run tf2_tools view_frames   # generates frames.pdf showing the tree topology
ros2 topic echo /tf --once       # see what transforms are being published right now
```

</details>

- [ ] Done

---

**A5.** How do you check which frames exist in the TF tree from the command line?
Give two commands and explain what each shows.

<details><summary>Answer</summary>

**Command 1:**
```bash
ros2 run tf2_tools view_frames
```
Listens to `/tf` and `/tf_static` for 5 seconds, then generates a `frames.pdf` showing
the complete transform tree as a graph: nodes are frame names, edges are transforms labelled
with the broadcaster, update rate, and buffer age. Useful for seeing the overall topology at
a glance and identifying disconnected subtrees.

**Command 2:**
```bash
ros2 topic echo /tf_static --once
ros2 topic echo /tf --once
```
Shows the raw `TransformStamped` messages being published. Useful for checking exact frame
names (case, underscores), verifying a specific broadcaster is publishing, and reading the
exact translation/rotation values.

**Bonus — interactive query:**
```bash
ros2 run tf2_ros tf2_echo map base_link
```
Continuously prints the `map → base_link` transform, showing both the value and whether
lookups succeed. If the lookup throws, the error message is printed inline.

</details>

- [ ] Done

---

## Section B — Trace the Frame Chain

The robot below has four nodes publishing into the TF tree:

```
Static TF broadcaster (URDF):   base_link → laser_link
                                 (offset: x=0.3m, y=0, z=0.1m, no rotation)

Odometry node:                   odom → base_link
                                 (updates at 50Hz as robot moves)

AMCL localization:               map → odom
                                 (updates at ~5Hz from laser scan matching)
```

Answer the four questions below **without running any code** — reason from the tree topology.

---

**B1.** A costmap node needs to insert a laser scan point into the map frame. The point
is at `(1.0, 0.0)` in `laser_link`. What is the full chain of transforms applied?
In what order are they composed?

<details><summary>Answer</summary>

The chain from `laser_link` to `map` traverses up the tree:

```
laser_link → base_link → odom → map
```

Composed as:
```
P_map = T(map→odom) × T(odom→base_link) × T(base_link→laser_link) × P_laser
```

TF2 handles this automatically — you call
`lookupTransform('map', 'laser_link', stamp)` and it composes the three matrix multiplications
internally.

**Timestamps:** For the costmap use case, all three transforms should be evaluated at (or
interpolated to) `stamp` — the hardware timestamp of the laser scan point. If any one of the
three transforms is not yet available at `stamp`, TF2 will throw `ExtrapolationException`.

</details>

---

**B2.** AMCL crashes and stops publishing `/tf`. After 11 seconds, what exceptions will
`lookupTransform('map', 'base_link', Time())` throw and why?

<details><summary>Answer</summary>

After 11 seconds without AMCL publishing, the `map → odom` transform has been evicted from
the TF buffer (default buffer duration is 10 seconds).

**Exception thrown:** `tf2_ros.ExtrapolationException`

**Reason:** The TF buffer still knows the `map` and `odom` frames exist (they were recently
seen), and there is technically a path `map → odom → base_link`. But the stored data for the
`map → odom` edge is older than the buffer window, so TF2 cannot extrapolate to "latest" and
throws `ExtrapolationException`.

**If AMCL had never started** (from robot power-on), the exception would be
`ConnectivityException` instead — `map → odom` would have no data at all, making the graph
disconnected.

**Practical consequence:** The costmap stops updating, the planner cannot plan in the map
frame, and the robot will stop navigating. In logs you will see repeated
`[costmap_2d] ExtrapolationException` or `[bt_navigator] TF error` messages.

</details>

---

**B3.** You want to know the robot's laser position relative to the `odom` frame (not
`map`). AMCL is not running. Is this lookup possible? What is the risk of using it for
autonomous navigation?

<details><summary>Answer</summary>

**Yes, the lookup is possible.** The chain `odom → base_link → laser_link` does not require
AMCL. The odometry node broadcasts `odom → base_link` independently.

```python
t = tf_buffer.lookup_transform('odom', 'laser_link', Time())
```

**Risk for autonomous navigation:**
The `odom` frame drifts over time due to wheel slip, encoder errors, and IMU integration
drift. Without the `map → odom` correction from a localization system (AMCL), the robot's
position in the `odom` frame will increasingly diverge from its true position in the
physical world. Over a long mission:

- A robot navigating only in `odom` may be commanded to a pose that is correct in `odom`
  coordinates but 50 cm off in physical space.
- Obstacles in the map frame (walls, shelves) will not be correctly projected into the
  costmap in `odom` space.

**Safe use case:** Short-term dead-reckoning (< 5m travel) without map-based obstacles.
For full autonomous navigation, `map → odom` (localization) is required.

</details>

---

**B4.** A node requests `lookupTransform('odom', 'map', stamp)` instead of
`lookupTransform('map', 'odom', stamp)`. What is the difference in the result?

<details><summary>Answer</summary>

`lookupTransform(target_frame, source_frame, stamp)` returns the transform that takes a
point expressed in `source_frame` and produces it in `target_frame`.

- `lookupTransform('map', 'odom', stamp)` → matrix that maps a point from `odom` to `map`.
  This is what the costmap uses: "take this point in odom coordinates and give me its
  position in the global map."

- `lookupTransform('odom', 'map', stamp)` → matrix that maps a point from `map` to `odom`.
  This is the *inverse* transform.

**They are valid** — TF2 automatically inverts edges to answer queries in either direction.
The returned `TransformStamped.transform` will have the inverted translation/rotation.

**The danger** is confusing which frame a point is expressed in. If you apply a `map→odom`
transform to a point that is already in `odom` coordinates, the result is meaningless.
Always be explicit: read the frame_id of your data and match it to the `source_frame`.

</details>

- [ ] Done

---

## Section C — Build Exercises

### C1. Static Transform Broadcaster

Write a **complete** Python ROS2 node that broadcasts a static transform:

- Parent frame: `base_link`
- Child frame: `camera_link`
- Translation: x=0.1m, y=0.0m, z=0.5m
- Rotation: identity (no rotation — all quaternion zeros except w=1.0)

```python
import rclpy
from rclpy.node import Node
# TODO: remaining imports

class StaticCameraPublisher(Node):
    def __init__(self):
        super().__init__('static_camera_tf')
        # TODO: create StaticTransformBroadcaster
        # TODO: build TransformStamped with correct values
        # TODO: broadcast it

def main(args=None):
    rclpy.init(args=args)
    node = StaticCameraPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

<details><summary>Answer</summary>

```python
import rclpy
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math


class StaticCameraPublisher(Node):
    def __init__(self):
        super().__init__('static_camera_tf')

        self._broadcaster = StaticTransformBroadcaster(self)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id  = 'camera_link'

        # Translation
        t.transform.translation.x = 0.1
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.5

        # Identity rotation (no rotation)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self._broadcaster.sendTransform(t)
        self.get_logger().info('Static transform base_link → camera_link published.')


def main(args=None):
    rclpy.init(args=args)
    node = StaticCameraPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

**Key points:**
- `StaticTransformBroadcaster.sendTransform()` publishes to `/tf_static` with
  `TRANSIENT_LOCAL` QoS — any node joining later will receive this transform automatically.
- Setting `w=1.0` with all other quaternion components zero is the identity rotation.
- The static broadcaster only needs to call `sendTransform()` once. Calling it repeatedly
  is harmless but wastes bandwidth.
- Verify with: `ros2 topic echo /tf_static`

</details>

- [ ] Done

---

### C2. Dynamic Circular-Motion Broadcaster

Write a node that simulates a robot driving in a circle. Every 50 ms, compute and broadcast
the `odom → base_link` transform.

**Robot parameters:**
- Speed: 0.5 m/s along the circumference
- Circle radius: 2.0 m
- Starting position: (2.0, 0.0) — on the positive X axis relative to the circle centre
- Heading: initially pointing in the +Y direction (tangent to the circle)

The angular velocity is `ω = v / r`. Update `θ` each tick by `ω × dt`.

```python
import rclpy
from rclpy.node import Node
import math
# TODO: remaining imports

class CircleBroadcaster(Node):
    def __init__(self):
        super().__init__('circle_broadcaster')
        self._angle = 0.0          # current angle around the circle (radians)
        self._radius = 2.0
        self._speed  = 0.5         # m/s along circumference
        self._dt     = 0.05        # 50ms

        # TODO: create TransformBroadcaster
        # TODO: create 20Hz timer (dt=0.05s)

    def timer_cb(self):
        # TODO: advance self._angle by omega * dt
        # TODO: compute x = radius * cos(angle), y = radius * sin(angle)
        # TODO: robot heading = angle + pi/2 (tangent to circle)
        # TODO: build TransformStamped, fill translation and rotation (yaw only)
        # TODO: broadcast
        pass

def main(args=None):
    rclpy.init(args=args)
    node = CircleBroadcaster()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

<details><summary>Answer</summary>

```python
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math


class CircleBroadcaster(Node):
    def __init__(self):
        super().__init__('circle_broadcaster')
        self._angle  = 0.0
        self._radius = 2.0
        self._speed  = 0.5   # m/s
        self._dt     = 0.05  # seconds

        self._broadcaster = TransformBroadcaster(self)
        self._timer = self.create_timer(self._dt, self.timer_cb)

    def timer_cb(self):
        omega = self._speed / self._radius       # rad/s
        self._angle += omega * self._dt

        x = self._radius * math.cos(self._angle)
        y = self._radius * math.sin(self._angle)
        yaw = self._angle + math.pi / 2          # tangent heading

        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id  = 'base_link'

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0

        # Convert yaw to quaternion (rotation around Z only)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(yaw / 2.0)
        t.transform.rotation.w = math.cos(yaw / 2.0)

        self._broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = CircleBroadcaster()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

**Key points:**
- `TransformBroadcaster` (not `StaticTransformBroadcaster`) — this transform changes every tick.
- Yaw-to-quaternion for planar rotation: `qz = sin(yaw/2)`, `qw = cos(yaw/2)`, `qx = qy = 0`.
- `header.stamp` must be set to the current time on every broadcast so TF2 knows when each
  pose was recorded. Using `Time()` (zero) here would break all `lookupTransform` calls that
  query by specific timestamp.
- Verify with: `ros2 run tf2_ros tf2_echo odom base_link`

</details>

- [ ] Done

---

### C3. TF Listener with Exception Handling

Write a node that every second looks up the transform from `map` to `base_link` and prints
the robot's `(x, y, yaw_degrees)` to the console. Handle all three TF2 exception types with
informative messages.

```python
import rclpy
from rclpy.node import Node
import math
# TODO: remaining imports — Buffer, TransformListener, and all three exception types

class PoseReporter(Node):
    def __init__(self):
        super().__init__('pose_reporter')
        # TODO: create tf2_ros.Buffer and tf2_ros.TransformListener
        # TODO: create 1Hz timer

    def timer_cb(self):
        try:
            # TODO: lookupTransform('map', 'base_link', rclpy.time.Time())
            # TODO: extract x, y, yaw from the result
            # TODO: log the pose
            pass
        except Exception as e:
            # TODO: catch LookupException, ConnectivityException, ExtrapolationException
            # separately with informative messages
            pass

def main(args=None):
    rclpy.init(args=args)
    node = PoseReporter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

<details><summary>Answer</summary>

```python
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import tf2_ros
import math


class PoseReporter(Node):
    def __init__(self):
        super().__init__('pose_reporter')
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._timer = self.create_timer(1.0, self.timer_cb)

    def timer_cb(self):
        try:
            t = self._tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),           # latest available
            )
            x   = t.transform.translation.x
            y   = t.transform.translation.y
            qz  = t.transform.rotation.z
            qw  = t.transform.rotation.w
            yaw = math.degrees(2.0 * math.atan2(qz, qw))
            self.get_logger().info(
                f'Robot pose: x={x:.3f}m  y={y:.3f}m  yaw={yaw:.1f}°'
            )

        except tf2_ros.LookupException as e:
            self.get_logger().warn(
                f'LookupException: frame "map" or "base_link" has never been published. '
                f'Details: {e}'
            )
        except tf2_ros.ConnectivityException as e:
            self.get_logger().warn(
                f'ConnectivityException: no connected path from "base_link" to "map" '
                f'in the TF tree. Is localization running? Details: {e}'
            )
        except tf2_ros.ExtrapolationException as e:
            self.get_logger().warn(
                f'ExtrapolationException: requested time is outside the TF buffer window. '
                f'Localization may have stalled. Details: {e}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = PoseReporter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

**Key points:**
- `TransformListener(buffer, node)` automatically subscribes to `/tf` and `/tf_static` and
  populates `buffer`. The listener must be kept alive as long as you want fresh data.
- `math.atan2(qz, qw) * 2` extracts yaw from a quaternion that has only Z rotation. For a
  full 3D rotation, use `transforms3d` or `scipy.spatial.transform.Rotation`.
- The three exception classes are in `tf2_ros` directly — no sub-module needed.
- Each error message says *what the symptom means* (not just the exception name), which
  makes the log useful when reading it an hour later.

</details>

- [ ] Done
