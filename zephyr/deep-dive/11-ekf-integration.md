# 11 — EKF Integration + Pipeline Validation
### Jetson Orin NX · robot_localization · ROS 2 Humble/Jazzy · TF2

**Status:** 🟡 HARDWARE-GATED  
**Prerequisite:** Session `10` milestone complete; `06-ros2-publisher.md` publishing `/imu/raw` at 100Hz  
**Hardware required:** Full pipeline active: STM32 → SPI → Jetson → ROS 2  
**Unlocks:** Full end-to-end pipeline complete  
**Time budget:** ~12 hours  
**Mastery plan:** Project 11

---

## Goal of This Session

Feed IMU + Odometry data into `robot_localization`'s EKF node to produce fused `odometry/filtered` at 100Hz. Visualize in rviz2. Verify the complete `map → odom → base_link` TF tree.

**Milestone**: `ros2 topic hz /odometry/filtered` shows ~100Hz. In rviz2 with `Fixed Frame = base_link`, the `odometry/filtered` arrow updates smoothly when you physically move the STM32 board. `ros2 run tf2_tools view_frames` shows the complete frame tree without disconnected nodes.

---

## Architecture: What robot_localization Does

```
/imu/raw   (sensor_msgs/Imu, 100Hz)  ─┐
/odom/raw  (nav_msgs/Odometry, 100Hz) ─┤─► ekf_node ──► /odometry/filtered
/gps/fix   (sensor_msgs/NavSatFix, 1Hz) ┘    (robot_localization)
                                              └──► TF: odom → base_link (100Hz)
```

The EKF fuses all sensors into a single consistent state estimate. It handles:
- Sensor delays (using message timestamps, not receive time)
- Different update rates (IMU at 100Hz, GPS at 1Hz)
- Sensor noise (via covariance matrices in each message)

---

## Step 1: Install robot_localization

```bash
# ROS 2 Humble
sudo apt-get install -y ros-humble-robot-localization

# ROS 2 Jazzy
sudo apt-get install -y ros-jazzy-robot-localization

# Verify
ros2 pkg list | grep robot_localization
```

---

## Step 2: Publish Odometry Topic

If you have CAN encoder data from session 07, add an `OdomPublisherNode` to your bridge:

```python
# Add to imu_bridge_node.py from session 06, or create a new file

from nav_msgs.msg import Odometry

class OdomPublisherNode(Node):
    def __init__(self):
        super().__init__('odom_publisher')
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.pub = self.create_publisher(Odometry, '/odom/raw', qos)

    def publish(self, left_vel_mms: float, right_vel_mms: float,
                timestamp_ns: int) -> None:
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'

        # Convert mm/s to m/s; differential drive forward velocity
        fwd_vel = (left_vel_mms + right_vel_mms) / 2.0 / 1000.0

        msg.twist.twist.linear.x = fwd_vel
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.angular.z = 0.0   # add yaw rate if you have it

        # Covariance: diagonal elements
        # For a simple encoder: velocity uncertainty ≈ 5mm/s (0.005 m/s)
        # Index [0] = linear.x variance, [35] = angular.z variance
        msg.twist.covariance[0]  = 0.005 ** 2   # linear.x
        msg.twist.covariance[7]  = 0.005 ** 2   # linear.y
        msg.twist.covariance[35] = 0.01  ** 2   # angular.z

        self.pub.publish(msg)
```

If you don't have CAN data yet, publish a static zero-velocity odometry for testing:

```bash
ros2 topic pub /odom/raw nav_msgs/Odometry \
  '{header: {frame_id: odom}, child_frame_id: base_link,
    twist: {covariance: [0.0025,0,0,0,0,0, 0,0.0025,0,0,0,0, 0,0,999999,0,0,0,
                         0,0,0,999999,0,0, 0,0,0,0,999999,0, 0,0,0,0,0,0.0001]}}' \
  -r 100
```

---

## Step 3: TF2 Static Transforms

robot_localization needs to know where the IMU is relative to the robot base frame.

```bash
# Create a launch file to publish static TF transforms
# File: launch/static_tf.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # imu_link is at the origin of base_link (same position, no rotation)
        # Adjust x,y,z,yaw,pitch,roll if the IMU is offset on your robot
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='imu_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'imu_link'],
        ),
        # base_link is at the origin of odom (EKF will override this dynamically)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='odom_base_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link'],
        ),
    ])
```

---

## Step 4: robot_localization EKF Config

```yaml
# config/ekf.yaml
ekf_filter_node:
  ros__parameters:
    frequency: 100.0           # output rate, Hz
    sensor_timeout: 0.1        # seconds before sensor considered stale
    two_d_mode: true           # constrain to 2D (no roll/pitch estimation)
    publish_tf: true           # publish odom → base_link TF
    map_frame: map
    odom_frame: odom
    base_link_frame: base_link
    world_frame: odom          # use odom as the world frame (no GPS fusion for now)

    # IMU: topic, config vector, relative, queue_size, nodelay, threshold
    # Config vector: [roll, pitch, yaw, vroll, vpitch, vyaw, ax, ay, az]
    #                14 booleans — which fields to USE from this sensor
    imu0: /imu/raw
    imu0_config: [false, false, false,   # x,  y,  z  position  — IMU doesn't provide absolute position
                  false, false, true,    # roll, pitch, yaw orientation — use yaw from integrated gyro
                  false, false, false,   # vx, vy, vz velocity  — don't use IMU velocity
                  true,  true,  true,    # vroll, vpitch, vyaw angular velocity — USE
                  true,  true,  false]   # ax, ay, az linear accel — USE x and y (not z — gravity)
    imu0_remove_gravitational_acceleration: true
    imu0_queue_size: 5
    imu0_nodelay: true

    odom0: /odom/raw
    odom0_config: [false, false, false,   # position — odometry doesn't give absolute position
                   false, false, false,   # orientation
                   true,  false, false,   # vx — USE forward velocity
                   false, false, true,    # vyaw — USE yaw rate if available
                   false, false, false]
    odom0_queue_size: 5
    odom0_nodelay: true
```

**Launch file:**

```python
# launch/ekf.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('your_package'),
        'config', 'ekf.yaml'
    )
    return LaunchDescription([
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[config],
        ),
    ])
```

---

## Step 5: Verify the TF Tree

```bash
# In a new terminal:
ros2 run tf2_tools view_frames

# Opens frames.pdf in current directory.
# You need to see:
#   map (if using GPS) or odom as root
#   └── base_link (published by ekf_node at 100Hz)
#       └── imu_link (static TF from launch file)

# If any frame is missing: check static_transform_publisher is running
ros2 run tf2_ros tf2_echo odom base_link
# Should print transform at 100Hz. If silent: EKF not running or no input data.
```

---

## The Three Failure Points (from Mastery Plan, Project 11)

### Failure Point 1: Zero covariance diagonal → NaN propagation

EKF ignores sensors with `0.0` on the covariance diagonal — it interprets "variance=0" as "infinitely precise" and triggers NaN propagation in the Kalman gain matrix.

**Symptom**: `/odometry/filtered` publishes NaN in position fields. `ros2 topic echo /odometry/filtered | grep nan` shows results.

**Fix**: every diagonal element of every covariance matrix must be small but non-zero:
```python
# In ImuBridgeNode (session 06):
# Check these specifically:
msg.linear_acceleration_covariance[0] > 0   # must not be 0
msg.linear_acceleration_covariance[4] > 0
msg.linear_acceleration_covariance[8] > 0
msg.angular_velocity_covariance[0] > 0
# etc.
```

### Failure Point 2: Timestamp lag → EKF velocity drift

`get_clock().now()` called *after* `spi.xfer2()` completes introduces 1–3ms lag per sample. At 100Hz with angular velocity from the gyro, 3ms lag accumulates to detectable yaw drift.

**Fix**: capture the timestamp immediately after ioctl, before any Python processing:

```python
# WRONG:
rx = spi.xfer2([0]*130)
# ... 2ms of Python processing ...
msg.header.stamp = self.get_clock().now().to_msg()  # 2ms late

# CORRECT:
import ctypes, ctypes.util

CLOCK_MONOTONIC_RAW = 4
class timespec(ctypes.Structure):
    _fields_ = [('tv_sec', ctypes.c_long), ('tv_nsec', ctypes.c_long)]

librt = ctypes.CDLL(ctypes.util.find_library('rt'))

def get_monotonic_raw_ns() -> int:
    ts = timespec()
    librt.clock_gettime(CLOCK_MONOTONIC_RAW, ctypes.byref(ts))
    return ts.tv_sec * 1_000_000_000 + ts.tv_nsec

rx = spi.xfer2([0]*130)
stamp_ns = get_monotonic_raw_ns()   # immediately after ioctl
# Convert to ROS 2 time:
msg.header.stamp.sec     = stamp_ns // 1_000_000_000
msg.header.stamp.nanosec = stamp_ns %  1_000_000_000
```

### Failure Point 3: rviz2 shows nothing

**Symptom**: rviz2 opens, add display `Odometry` for `/odometry/filtered`, nothing appears.

**Cause**: `Fixed Frame` set to `map` but no `map → base_link` TF exists (EKF uses `odom` as world frame if GPS is disabled).

**Fix**: set `Fixed Frame = odom` in rviz2 Global Options. Only use `Fixed Frame = map` after adding GPS fusion with a separate `navsat_transform_node`.

---

## Validation Sequence

```bash
# Terminal 1: Start STM32 bridge (session 06)
ros2 run your_package imu_bridge_node

# Terminal 2: Start static TF
ros2 launch your_package static_tf.launch.py

# Terminal 3: Start EKF
ros2 launch your_package ekf.launch.py

# Terminal 4: Verify topics
ros2 topic hz /imu/raw              # target: ~100Hz
ros2 topic hz /odom/raw             # target: ~100Hz
ros2 topic hz /odometry/filtered    # target: ~100Hz

# Terminal 5: Check EKF output
ros2 topic echo /odometry/filtered --once

# Terminal 6: Check TF
ros2 run tf2_ros tf2_echo odom base_link

# Terminal 7: rqt_graph for pipeline view
ros2 run rqt_graph rqt_graph
```

---

## rviz2 Setup for IMU Visualization

```bash
# Open rviz2
rviz2

# In rviz2:
# 1. Global Options → Fixed Frame: set to "odom"
# 2. Add → By topic → /imu/raw → Imu display
#    Set "Topic" to /imu/raw
#    Set "Fixed Frame" to imu_link  (if IMU display shows nothing)
# 3. Add → By topic → /odometry/filtered → Odometry display
#    Uncheck "Show Covariance" initially (clutters view)
# 4. Move/tilt the STM32 board → rviz2 arrow should update

# If IMU arrow doesn't move:
#   ros2 topic echo /imu/raw | grep angular_velocity
#   Confirm non-zero values when moving the board
```

---

## Pipeline Validation Checklist

```bash
# Full pipeline validation script
#!/bin/bash

echo "=== Pipeline Validation ==="

echo "1. IMU topic rate:"
timeout 5 ros2 topic hz /imu/raw 2>&1 | tail -3

echo "2. EKF output rate:"
timeout 5 ros2 topic hz /odometry/filtered 2>&1 | tail -3

echo "3. TF chain:"
timeout 3 ros2 run tf2_ros tf2_echo odom base_link 2>&1 | head -10

echo "4. NaN check in EKF output:"
timeout 3 ros2 topic echo /odometry/filtered 2>&1 | grep -i "nan" || echo "  No NaN found"

echo "5. IMU covariance diagonals:"
ros2 topic echo /imu/raw --once 2>&1 | grep -A5 "linear_acceleration_covariance"
```

---

## Milestone Checklist

- [ ] `ros2 topic hz /odometry/filtered` shows ~100Hz
- [ ] `view_frames` shows `odom → base_link → imu_link` connected
- [ ] No NaN in `/odometry/filtered` during 5-minute run
- [ ] rviz2 `Odometry` display updates when STM32 board is tilted
- [ ] Timestamp lag fix applied: stamp captured immediately after ioctl
- [ ] IMU covariance diagonals all non-zero (verified with `ros2 topic echo`)
- [ ] `rqt_graph` shows complete pipeline: bridge → EKF → rviz2

---

## Session Notes Template

```markdown
## Session Notes — [DATE]

### Setup
- robot_localization version: ___
- ROS 2 distro: humble / jazzy

### EKF
- First NaN-free run: yes/no (if no: which covariance was zero?)
- First rviz2 display: working at yes/no (if no: Fixed Frame issue?)
- EKF output rate (hz): ___Hz

### TF Tree
- view_frames shows complete tree: yes/no
- Missing frames (if any): ___

### Timestamp
- Lag fix applied: yes/no
- Measured lag without fix: ___ms
- Measured lag with fix: ___µs

### Issues
- ...
```
