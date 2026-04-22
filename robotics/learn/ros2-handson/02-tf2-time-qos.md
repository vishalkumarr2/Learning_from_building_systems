# 02 — TF2, Time Synchronization, and QoS
### Why your subscriber receives nothing and your transform lookup throws exceptions
**Prerequisite:** 01-nodes-topics-actions.md (publisher/subscriber, DDS discovery)
**Unlocks:** Diagnosing silent subscription failures, fixing TF lookup exceptions, debugging bag replay timing issues, understanding why nodes on different machines can't see each other

---

## Why Should I Care? (Context)

TF lookup failures and QoS mismatches together account for a large share of "navigation stack broke but all nodes are running" incidents. Both are invisible at first glance:

- A **TF failure** looks like the robot believing it is at the wrong position, or the costmap not updating. The error (`ExtrapolationException`) appears in logs but is easy to miss if you're watching the wrong node.
- A **QoS mismatch** is completely silent. The subscriber node exists, it is ACTIVE, it has a callback registered — but it receives zero messages. `ros2 topic echo` shows data flowing, but your node sees nothing.

Understanding how the transform tree is structured, and exactly which QoS dimensions must match, turns these from hour-long debugging sessions into two-minute fixes.

---

# PART 1 — TF2: THE TRANSFORM TREE

---

## 1.1 What TF2 Solves

In a mobile robot, sensor data arrives in different coordinate frames:
- Laser scan points are in `laser_link` frame (relative to the laser)
- The costmap needs obstacle positions in `map` frame (global fixed frame)
- The controller needs the goal in `base_link` frame (robot body)

Without TF2, every node would need to know the entire kinematic chain and recompute transforms manually. With TF2, any node can ask: *"Where is `laser_link` relative to `map` at time T?"* — and TF2 handles the chain traversal.

**Key insight:** TF2 stores a time-indexed tree of transforms. It can interpolate between recorded transforms to answer queries at arbitrary timestamps. This is what makes bag-based replay and sensor fusion possible.

---

## 1.2 The Standard Mobile Robot Frame Tree

```
map
 │
 └── odom
       │
       └── base_link
                │
                ├── laser_link
                │
                ├── camera_link
                │     └── camera_optical_link
                │
                └── imu_link

Each edge is a transform (translation + rotation quaternion + timestamp).
```

**Frame semantics:**

| Frame | Published by | Properties |
|---|---|---|
| `map` | AMCL / localization | Global fixed frame. Can have discrete jumps during relocalization. |
| `odom` | Odometry node / EKF | Continuous, no jumps. Drifts over time. Represents wheel motion. |
| `base_link` | TF broadcaster (identity: base_link = odom + offset) | Robot body center. |
| `laser_link` | Static TF (robot URDF) | Fixed relative to `base_link`. Never changes at runtime. |

**`map → odom`** transform: published by the localization system (AMCL). Represents the correction between where odometry thinks the robot is and where the map-matched pose is.

**`odom → base_link`** transform: published by the odometry/EKF node. Represents the robot's motion since startup.

---

## 1.3 Static vs Dynamic Transforms

**Static transforms** — for fixed sensor mounts (laser bolted to the robot body):
```python
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

broadcaster = StaticTransformBroadcaster(node)
t = TransformStamped()
t.header.stamp = node.get_clock().now().to_msg()
t.header.frame_id = 'base_link'
t.child_frame_id = 'laser_link'
t.transform.translation.x = 0.15    # laser is 15cm forward of base center
t.transform.translation.z = 0.30    # laser is 30cm above ground
t.transform.rotation.w = 1.0        # no rotation (quaternion identity)
broadcaster.sendTransform(t)
```

Static transforms are published once to `/tf_static` with TRANSIENT_LOCAL durability — late-joining nodes receive them immediately.

**Dynamic transforms** — for odometry (robot moves every control cycle):
```python
from tf2_ros import TransformBroadcaster

broadcaster = TransformBroadcaster(node)

# Inside your 50Hz odom callback:
t = TransformStamped()
t.header.stamp = odom_msg.header.stamp   # ← use the sensor timestamp, not now()!
t.header.frame_id = 'odom'
t.child_frame_id = 'base_link'
t.transform.translation.x = odom_msg.pose.pose.position.x
# ... fill rotation from quaternion
broadcaster.sendTransform(t)
```

**Critical rule:** Use the original sensor timestamp, not `node.get_clock().now()`. Processing latency will cause TF extrapolation failures if you use wall clock.

---

## 1.4 The TF Buffer

The TF buffer stores transform history in a sliding window (default: 10 seconds).

```
TF Buffer timeline (10-second window):
                                          now
◄────────────────────────────────────────┤
│  t-10s  t-9s  t-8s ... t-1s  t-0.1s  now │
│   tf    tf    tf  ... tf     tf      tf   │
└──────────────────────────────────────────►

Query at t-5s: ✓ within buffer, can interpolate
Query at t-15s: ✗ outside buffer → ExtrapolationException (too old)
Query at t+0.1s: ✗ in the future → ExtrapolationException (too new, usually >40ms)
```

Increase buffer duration for bag replay or slow systems:
```python
tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=30.0))
```

---

# PART 2 — LOOKING UP TRANSFORMS

---

## 2.1 The lookup_transform API

```python
import tf2_ros

tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer, node)

try:
    # Get the transform: where is 'laser_link' in 'map' frame?
    transform = tf_buffer.lookup_transform(
        target_frame='map',            # frame we want result in
        source_frame='laser_link',     # frame we're transforming FROM
        time=rclpy.time.Time(),        # Time() = "most recent available"
        timeout=rclpy.duration.Duration(seconds=1.0)
    )
except tf2_ros.TransformException as e:
    node.get_logger().error(f'TF lookup failed: {e}')
```

**`time=rclpy.time.Time()`** — zero time means "give me the latest available transform." Safe, but the transform may be up to one control cycle old.

**`time=msg.header.stamp`** — gives the transform at exactly the message timestamp. Fails if the timestamp is outside the buffer window. Use this for precise sensor data alignment.

---

## 2.2 The Three Exceptions

| Exception | Meaning | Common Cause |
|---|---|---|
| `ExtrapolationException` | Requested time is outside buffer window | Future time (>40ms ahead) or buffer too short |
| `ConnectivityException` | No chain exists between the two frames | A node in the chain hasn't started publishing yet |
| `LookupException` | The frame name doesn't exist at all | Typo in frame name, or node not started |

**Debugging the frame tree:**
```python
# Print all frames currently in the buffer
print(tf_buffer.all_frames_as_string())
```

```bash
# From command line:
ros2 run tf2_tools view_frames    # generates frames.pdf diagram
ros2 topic echo /tf --once        # see what's being published
ros2 topic echo /tf_static --once # see static transforms
```

---

## 2.3 Worked Example: Transform Laser Points to Map Frame

**Problem:** Your laser subscriber receives scan data in `laser_link` frame. You want to log the first laser point's position in the global `map` frame.

```python
import tf2_ros
import tf2_geometry_msgs          # registers PointStamped transform support
from geometry_msgs.msg import PointStamped

class LaserMapProjector(Node):
    def __init__(self):
        super().__init__('laser_map_projector')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

    def scan_callback(self, scan_msg):
        # Take the first range reading, compute its Cartesian position
        r = scan_msg.ranges[0]
        angle = scan_msg.angle_min

        point_in_laser = PointStamped()
        point_in_laser.header = scan_msg.header     # frame='laser_link', stamp=scan time
        point_in_laser.point.x = r * math.cos(angle)
        point_in_laser.point.y = r * math.sin(angle)

        try:
            # Transform to map frame AT the scan timestamp
            point_in_map = self.tf_buffer.transform(
                point_in_laser,
                'map',
                timeout=rclpy.duration.Duration(seconds=0.1))
            self.get_logger().info(
                f'Point in map: ({point_in_map.point.x:.2f}, {point_in_map.point.y:.2f})')
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f'Transform failed: {e}')
```

**Step by step:**
1. Extract range and angle from the scan message
2. Compute Cartesian XY in the laser frame (polar to Cartesian)
3. Wrap in `PointStamped` with the scan's original header (preserves timestamp)
4. Call `tf_buffer.transform()` — TF2 looks up `laser_link → map` at the scan's stamp
5. Result: point position in global `map` frame

---

# PART 3 — TIME SYNC AND use_sim_time

---

## 3.1 use_sim_time: Clock Source Switching

In normal operation, ROS2 nodes use the system wall clock. In simulation or bag replay, time is published on `/clock` by the simulator or `rosbag2 play --clock`.

```yaml
# In your launch file or node parameters:
use_sim_time: true
```

When `use_sim_time: true`, the node's `get_clock().now()` reads from `/clock` instead of the OS. This means:
- Time advances when the bag/simulator advances it
- Time can be paused (useful for debugging)
- Time can run faster or slower than real time

---

## 3.2 Wall Time vs Sim Time

```
Wall time  ──────────────────────────────────────────────────────► (always advances)
            0s     1s     2s     3s     4s     5s     6s

Sim time   ──────────────────────────────────────────────────────► (controlled by /clock)
            0s     1s     2s    [PAUSE]  2s    3s     4s
                              ↑                      ↑
                         bag paused               bag resumed at half speed
```

**Common bug:** One node has `use_sim_time: true`, another has `use_sim_time: false` (default). The wall-clock node publishes transforms with wall timestamps, but the sim-time TF consumer tries to look them up at sim timestamps. The times never match → constant `ExtrapolationException`.

**Fix:** In bag replay, set `use_sim_time: true` for ALL nodes, including the TF broadcaster.

---

## 3.3 Synchronizing Two Topics by Timestamp

Sometimes you need data from two topics at the same timestamp (e.g., camera image + IMU for visual-inertial odometry).

**Exact synchronizer** — both messages must have identical stamps:
```python
from message_filters import TimeSynchronizer, Subscriber

image_sub = Subscriber(node, Image, '/camera/image_raw')
imu_sub = Subscriber(node, Imu, '/imu/data')

sync = TimeSynchronizer([image_sub, imu_sub], queue_size=10)
sync.registerCallback(my_callback)

def my_callback(image_msg, imu_msg):
    # Both messages have the same header.stamp
    pass
```

**Approximate synchronizer** — matches messages within a time tolerance:
```python
from message_filters import ApproximateTimeSynchronizer

sync = ApproximateTimeSynchronizer(
    [image_sub, imu_sub],
    queue_size=10,
    slop=0.1)        # ← 100ms tolerance between stamps
sync.registerCallback(my_callback)
```

**When to use approximate:** When two sensors have different publish rates (camera at 30Hz, lidar at 10Hz), exact sync will rarely find matches. Use approximate with a slop slightly larger than one period of the slower sensor.

---

# PART 4 — QoS PROFILES

---

## 4.1 The Three Key Dimensions

QoS (Quality of Service) is a set of policies agreed between publisher and subscriber at discovery time. Mismatch = silent failure.

### Dimension 1: Reliability

```
RELIABLE:    Publisher retransmits until subscriber acknowledges.
             Overhead: ACK traffic, retransmit buffers.
             Use for: commands, goals, critical state.

BEST_EFFORT: Publisher fires and forgets. No retransmit.
             Overhead: none.
             Use for: sensor data (50Hz+) where stale data > delayed data.
```

### Dimension 2: Durability

```
VOLATILE:        New subscriber gets NOTHING from before it subscribed.
                 Use for: real-time data (odom, scan, cmd_vel).

TRANSIENT_LOCAL: Publisher keeps last N messages. New subscriber gets them immediately.
                 Use for: latched data (map, initial pose, static configuration).
```

### Dimension 3: History + Depth

```
KEEP_LAST(N):  Buffer at most N messages. Evict oldest on overflow.
               Use for: all real-time topics. N=10 is typical.

KEEP_ALL:      Buffer everything. Memory can grow without bound.
               Use for: never in production.
```

---

## 4.2 The Mismatch Rule

QoS mismatch is **silent** — no warning at runtime unless you explicitly check `ros2 topic info --verbose`.

```
Compatibility table:
                         Subscriber
                    RELIABLE    BEST_EFFORT
Publisher  RELIABLE     ✓            ✓       ← subscriber relaxes, it's ok
          BEST_EFFORT   ✗            ✓       ← publisher can't guarantee what sub demands
```

```
                         Subscriber
                    TRANSIENT    VOLATILE
Publisher TRANSIENT     ✓           ✓       ← subscriber relaxes
          VOLATILE       ✗           ✓       ← publisher can't serve what sub demands
```

**Memory device:** The subscriber can be *less strict* than the publisher, but not *more strict*. If the publisher is BEST_EFFORT, the subscriber cannot demand RELIABLE.

---

## 4.3 Worked Example: Matching QoS for Odometry

```python
# The odometry publisher (sensor driver) uses:
pub_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,   # fast, no ack overhead
    durability=DurabilityPolicy.VOLATILE,         # real-time only
    history=HistoryPolicy.KEEP_LAST,
    depth=10)

# CORRECT subscriber - matches or is looser:
sub_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,   # matches publisher
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10)

# WRONG subscriber - more strict than publisher → SILENT FAILURE:
sub_qos_wrong = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,      # ← demands ACKs from BEST_EFFORT pub
    ...
)
```

**Diagnosing a suspected QoS mismatch:**
```bash
# Shows publisher and subscriber QoS side-by-side
ros2 topic info /odom --verbose
```

Look for: `Requested incompatible QoS` in the output or node logs.

---

## 4.4 Common Predefined QoS Profiles

| Profile | Reliability | Durability | Depth | Use for |
|---|---|---|---|---|
| `rclcpp::QoS(10)` (default) | RELIABLE | VOLATILE | 10 | General purpose |
| `rclcpp::SensorDataQoS()` | BEST_EFFORT | VOLATILE | 5 | `/scan`, `/imu`, `/odom` |
| `rclcpp::ServicesQoS()` | RELIABLE | VOLATILE | 10 | Service calls |
| `rclcpp::ParametersQoS()` | RELIABLE | VOLATILE | 1000 | `/parameter_events` |
| `rclcpp::SystemDefaultsQoS()` | SYSTEM_DEFAULT | SYSTEM_DEFAULT | — | Rare; DDS decides |

**Practical rule:** Use `SensorDataQoS()` for any topic published at >10Hz. Use default `QoS(10)` for commands and events.

---

## 4.5 Durability for Static Data (Maps)

```python
# Map server publishes with TRANSIENT_LOCAL:
map_pub_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,   # ← key
    history=HistoryPolicy.KEEP_LAST,
    depth=1)

# Costmap subscriber MUST also use TRANSIENT_LOCAL, otherwise:
# → costmap starts before map server → subscribes → gets nothing
# → costmap never has a map → nav2 won't plan paths

map_sub_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,   # ← must match
    depth=1)
```

---

# PART 5 — DDS INTERNALS (Enough to Debug)

---

## 5.1 Discovery Without a Broker

```
Node A (machine 1)                    Node B (machine 2)
     │                                      │
     ├── multicast UDP 239.255.0.1:7400 ───►│ (SPDP: Simple Participant Discovery)
     │◄── multicast UDP 239.255.0.1:7400 ───│
     │                                      │
     ├── unicast SEDP endpoint exchange ────►│ (SEDP: Simple Endpoint Discovery)
     │◄── unicast SEDP endpoint exchange ────│
     │                                      │
     └──── QoS negotiation complete ─────────┘
                data flows directly
```

**Key insight:** If the multicast packet can't reach Node B (e.g., it's blocked by a managed switch with IGMP snooping), discovery never completes. Both nodes appear to be running, but they cannot see each other's topics.

---

## 5.2 ROS_DOMAIN_ID: Fleet Isolation

```bash
# Robot 1's operator console:
export ROS_DOMAIN_ID=1
ros2 launch nav2_bringup ...

# Robot 2's operator console:
export ROS_DOMAIN_ID=2
ros2 launch nav2_bringup ...

# These two robots are completely invisible to each other.
# /scan from robot 1 is NOT delivered to robot 2's costmap.
```

DDS domains are network-level isolation. Nodes in different domains do not discover each other even on the same LAN. Use this when deploying a multi-robot fleet on a shared network.

---

## 5.3 Fixing Discovery on Managed Networks

```bash
# If multicast is blocked, use static peer list:
export ROS_STATIC_PEERS="192.168.1.10;192.168.1.11;192.168.1.12"

# Or configure in a FastDDS XML profile:
# <participants>
#   <participant>
#     <rtps>
#       <builtin>
#         <initialPeersList>
#           <locator>
#             <udpv4><address>192.168.1.10</address></udpv4>
#           </locator>
#         </initialPeersList>
```

**Fastcheck for multi-machine issues:**
```bash
# On machine A:
ros2 daemon stop && ros2 daemon start
ros2 node list    # should show nodes from machine B if same domain

# If not: check ROS_DOMAIN_ID, check ping, check firewall, check multicast routing
```

---

## 5.4 Quick QoS Mismatch Check

```bash
# Full QoS profile for all pubs and subs on a topic:
ros2 topic info /scan --verbose

# Output includes:
# Publisher count: 1
#   Node name: lidar_driver
#   QoS profile:
#     Reliability: Best effort      ← note
#     Durability: Volatile
#
# Subscription count: 2
#   Node name: costmap_node
#   QoS profile:
#     Reliability: Reliable         ← MISMATCH: sub wants Reliable, pub is Best effort
```

Any mismatch in `Reliability` or `Durability` (where sub is more strict than pub) = silent no-data.

---

## Summary — What to Remember

| Concept | Rule |
|---|---|
| TF frame tree | `map → odom → base_link → sensor_link`. Each edge is a timestamped transform. |
| Static TF | Published once to `/tf_static`. For fixed sensor mounts. Late joiners get it immediately (TRANSIENT_LOCAL). |
| Dynamic TF | Published every control cycle to `/tf`. Use the sensor timestamp, not wall clock. |
| TF buffer | Default 10-second history window. Increase to 30s for bag replay. |
| `ExtrapolationException` | Timestamp is too far in the future (>~40ms) or past (beyond buffer). |
| `ConnectivityException` | No TF chain between the two frames yet. Usually a node hasn't started. |
| `LookupException` | Frame name doesn't exist. Check for typos with `all_frames_as_string()`. |
| `use_sim_time` | ALL nodes must use same clock source. Mix = guaranteed TF failures in bag replay. |
| QoS Reliability | BEST_EFFORT pub + RELIABLE sub = silent failure. Sub cannot be more strict than pub. |
| QoS Durability | VOLATILE pub + TRANSIENT_LOCAL sub = silent failure (map never delivered to late-joining costmap). |
| `SensorDataQoS()` | The right default for any topic >10Hz: BEST_EFFORT + VOLATILE + depth 5. |
| `ros2 topic info --verbose` | The first command to run when a subscriber receives nothing. |
| `ROS_DOMAIN_ID` | Isolates robot fleets on shared LAN. Must match between communicating nodes. |
| `ROS_STATIC_PEERS` | Fix discovery when managed switches block multicast. |
