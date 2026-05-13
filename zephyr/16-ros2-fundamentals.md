# ROS2 Fundamentals — Robot Operating System 2

## What Is ROS2?

ROS2 (Robot Operating System 2) is a **middleware and toolset** for robotics software.
It is **not** an OS — it runs on Linux (or Windows/macOS). It provides:

- Structured **communication** between software modules (publish/subscribe, services, actions)
- **Build system** (colcon + ament_cmake)
- **CLI tools** for debugging (`ros2 topic echo`, `ros2 node list`)
- Large ecosystem of **ready-made packages**: navigation, visualization, sensor drivers, simulation

In our pipeline: Jetson Orin runs ROS2. It reads SPI data from STM32, decodes protobuf, and publishes `sensor_msgs/Imu` messages at 100Hz for the navigation stack to consume.

---

## Core Concepts — The 5 Building Blocks

### 1. Nodes

A **node** is a process that does one thing. The fundamental unit.

Good: one node per sensor, one node for navigation, one node for visualization.
Bad: one giant node that does everything.

```
stm32_bridge_node     → reads SPI, publishes /imu/raw at 100Hz
imu_filter_node       → subscribes /imu/raw, publishes /imu/filtered
nav_stack_node        → subscribes /imu/filtered, /odom, publishes /cmd_vel
motor_driver_node     → subscribes /cmd_vel, writes to CAN
```

### 2. Topics

A **topic** is a named channel for streaming data. **Publish/subscribe** model.

```
Publisher                                    Subscriber(s)
─────────                                   ──────────────
stm32_bridge_node ──/imu/raw──────────────► imu_filter_node
                                           ► imu_logger_node
                                           ► rviz2 (visualizer)
```

Key properties:
- **Decoupled**: publisher doesn't know who subscribes
- **Asynchronous**: publisher doesn't wait for subscriber
- **N-to-M**: many publishers, many subscribers on same topic

```python
# Publisher
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

class ImuPublisher(Node):
    def __init__(self):
        super().__init__('stm32_bridge')
        self.pub = self.create_publisher(Imu, '/imu/raw', 10)
        # 10 = queue depth (how many messages to buffer)

    def publish_imu(self, data):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'
        msg.linear_acceleration.x = data.accel_x
        # ...
        self.pub.publish(msg)
```

```python
# Subscriber
class ImuFilter(Node):
    def __init__(self):
        super().__init__('imu_filter')
        self.sub = self.create_subscription(Imu, '/imu/raw', self.callback, 10)

    def callback(self, msg):
        # Called in executor thread when message arrives
        filtered = self.apply_filter(msg)
        # process...
```

### 3. Services

A **service** is a request-reply pattern. One node asks, another answers — synchronous.

```
Client:         "set_motor_speed(1.5)"   ──►  Service server:  processes, returns OK
                "get_battery_status()"  ──►  Service server:  returns 87%
```

Use for: configuration changes, one-time commands, status queries.
Do NOT use for: high-rate sensor data (use topics instead — services are slow).

```python
from std_srvs.srv import SetBool

class MotorNode(Node):
    def __init__(self):
        super().__init__('motor_driver')
        self.srv = self.create_service(SetBool, 'enable_motors', self.handle_enable)

    def handle_enable(self, request, response):
        enabled = request.data
        self.set_motor_power(enabled)
        response.success = True
        response.message = f"Motors {'on' if enabled else 'off'}"
        return response
```

### 4. Actions

An **action** is like a service, but for **long-running tasks** with progress feedback.

```
Client:    "navigate_to(x=5, y=3)"  ──►  Server starts moving robot
           ← "feedback: 30% done"
           ← "feedback: 60% done"
           ← "result: arrived" (or "result: failed")
Client can cancel at any time.
```

Use for: navigation goals, arm movements, multi-second operations.

### 5. Parameters

Nodes can have **parameters** — configuration values you can change at runtime without restarting.

```python
self.declare_parameter('imu_frame', 'imu_link')
self.declare_parameter('publish_rate_hz', 100.0)

frame = self.get_parameter('imu_frame').get_parameter_value().string_value
rate = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
```

```bash
ros2 param set /stm32_bridge imu_frame base_imu_link
ros2 param get /stm32_bridge publish_rate_hz
```

---

## QoS — Quality of Service

ROS2 lets you configure reliability, durability, and deadline of message delivery.

### Key QoS profiles

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# BEST_EFFORT: don't retry dropped messages — use for sensors (lidar, IMU, cameras)
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

# RELIABLE: retry until delivered — use for commands, events
reliable_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    ...
)
```

**Rule for our IMU**: use `BEST_EFFORT`. At 100Hz, a dropped message is 10ms old — it's better to skip and show the next fresh one than to retry stale data.

```python
self.pub = self.create_publisher(Imu, '/imu/raw', sensor_qos)
```

---

## Complete STM32 Bridge Node

This is the real node that reads from SPI and publishes to ROS2:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
import spidev
import struct
from robot_msgs.pb2 import RobotFrame  # generated from .proto

class Stm32BridgeNode(Node):
    def __init__(self):
        super().__init__('stm32_bridge')

        # SPI setup
        self.spi = spidev.SpiDev()
        self.spi.open(1, 0)              # /dev/spidev1.0
        self.spi.max_speed_hz = 10_000_000
        self.spi.mode = 0

        # Publisher with BEST_EFFORT QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10,
        )
        self.imu_pub = self.create_publisher(Imu, '/imu/raw', sensor_qos)

        # Timer at 100Hz
        self.timer = self.create_timer(0.01, self.read_and_publish)

        # Stats
        self.frame_count = 0
        self.error_count = 0

        self.get_logger().info('STM32 bridge node started')

    def read_and_publish(self):
        """Called every 10ms by timer"""
        try:
            # Read 150-byte SPI frame
            raw = bytes(self.spi.readbytes(150))

            # Validate header
            if raw[0] != 0xAA or raw[1] != 0x55:
                self.error_count += 1
                return

            # Length field
            payload_len = struct.unpack_from('>H', raw, 2)[0]
            payload = raw[4:4 + payload_len]

            # Decode protobuf
            frame = RobotFrame()
            frame.ParseFromString(payload)

            # Build Imu message
            msg = Imu()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'imu_link'

            msg.linear_acceleration.x = frame.accel_x
            msg.linear_acceleration.y = frame.accel_y
            msg.linear_acceleration.z = frame.accel_z

            msg.angular_velocity.x = frame.gyro_x
            msg.angular_velocity.y = frame.gyro_y
            msg.angular_velocity.z = frame.gyro_z

            # Covariances unknown (-1 diagonal = unknown)
            msg.linear_acceleration_covariance[0] = -1.0

            self.imu_pub.publish(msg)
            self.frame_count += 1

            if self.frame_count % 1000 == 0:
                self.get_logger().info(
                    f'Published {self.frame_count} frames, errors={self.error_count}'
                )

        except Exception as e:
            self.get_logger().error(f'Bridge error: {e}')
            self.error_count += 1


def main(args=None):
    rclpy.init(args=args)
    node = Stm32BridgeNode()
    rclpy.spin(node)     # runs the executor, calls callbacks
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

---

## Package Structure

```
my_robot/
├── CMakeLists.txt         ← ament_cmake build
├── package.xml            ← metadata, dependencies
├── my_robot/
│   ├── __init__.py
│   ├── stm32_bridge_node.py
│   └── imu_filter_node.py
├── launch/
│   └── robot.launch.py
├── config/
│   └── params.yaml
└── msg/
    └── WheelSpeed.msg     ← custom message type
```

**package.xml** (declare dependencies):

```xml
<?xml version="1.0"?>
<package format="3">
  <name>my_robot</name>
  <version>0.1.0</version>
  <description>My robot</description>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

---

## Launch Files

A launch file starts multiple nodes at once with configured parameters:

```python
# launch/robot.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot',
            executable='stm32_bridge',
            name='stm32_bridge',
            parameters=[{'publish_rate_hz': 100.0, 'imu_frame': 'imu_link'}],
        ),
        Node(
            package='my_robot',
            executable='imu_filter',
            name='imu_filter',
        ),
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            parameters=['/path/to/amcl_params.yaml'],
        ),
    ])
```

Run: `ros2 launch my_robot robot.launch.py`

---

## Build and Run

```bash
# Build
cd ~/ros2_ws
colcon build --packages-select my_robot
source install/setup.bash

# Run single node
ros2 run my_robot stm32_bridge

# Run launch file
ros2 launch my_robot robot.launch.py

# Debug
ros2 node list                     # show running nodes
ros2 topic list                    # show all topics
ros2 topic echo /imu/raw           # print messages live
ros2 topic hz /imu/raw             # measure publish frequency
ros2 topic bw /imu/raw             # measure bandwidth
ros2 param list /stm32_bridge      # list parameters
```

---

## Common ROS2 Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Missing `source install/setup.bash` | Package not found | Run after every `colcon build` |
| QoS mismatch publisher/subscriber | No messages received | Both must use compatible QoS (BEST_EFFORT matches BEST_EFFORT, RELIABLE matches RELIABLE) |
| Heavy work in callback | 100Hz drops to 10Hz | Move heavy work to separate thread, use timer callback only for receiving |
| Not calling `rclpy.spin()` | Callbacks never fire | `spin()` runs the event loop |
| Wrong topic name (typo) | Subscriber never receives | `ros2 topic list` to verify |
| Using Python GIL for parallelism | Callbacks block each other | Use `MultiThreadedExecutor` for concurrent callbacks |
