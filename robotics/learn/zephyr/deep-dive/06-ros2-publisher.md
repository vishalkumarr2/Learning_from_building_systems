# 06 — ROS 2 IMU Publisher
### Jetson Orin NX · rclpy · sensor_msgs/Imu · 100Hz

**Status:** 🟡 HARDWARE-GATED — fill this in after `05` milestone  
**Prerequisite:** Session `05` milestone: valid SPI frames arriving at Jetson  
**Hardware required:** Same as `05` + Jetson running ROS 2 Humble  
**Unlocks:** `07-estimator-replay.md`  
**Time budget:** ~3 hours

---

## Goal of This Session

Turn the raw SPI frames from `05` into proper `sensor_msgs/Imu` messages published
to `/imu/raw` at 100Hz. Verify with `ros2 topic hz` and `ros2 topic echo`.

**Milestone**: `ros2 topic hz /imu/raw` prints `average rate: 100.000`, `rviz2` shows
IMU arrow updating visually when you tilt the board.

---

## theory: sensor_msgs/Imu Fields

```
sensor_msgs/Imu:
    Header header               # frame_id, timestamp
    geometry_msgs/Quaternion orientation       # leave ZERO if unknown
    float64[9] orientation_covariance          # [0,0,0,...] = unknown
    geometry_msgs/Vector3 angular_velocity     # rad/s (gyro)
    float64[9] angular_velocity_covariance     # from datasheet noise density
    geometry_msgs/Vector3 linear_acceleration  # m/s² (accel, WITHOUT gravity removal)
    float64[9] linear_acceleration_covariance  # from datasheet noise density
```

**Important**: `linear_acceleration` includes gravity. Do NOT subtract gravity here — let
the estimator (robot_localization, etc.) handle that. Subtracted gravity in the raw topic
causes double-subtraction bugs downstream.

**Covariance values** from ICM-42688-P datasheet:
- Accel noise density: 70 µg/√Hz → in ±2g mode at 200Hz: σ² ≈ (70e-6 × 9.81)² × 200
- Gyro noise density:  0.0028 °/s/√Hz → σ² ≈ (0.0028 × π/180)² × 200

```python
# Covariance diagonal values (simplified; tune after Allan variance analysis)
ACCEL_NOISE_DENSITY = 70e-6 * 9.81    # m/s²/√Hz
GYRO_NOISE_DENSITY  = 0.0028 * 3.14159 / 180  # rad/s/√Hz
ODR = 200.0  # Hz

ACCEL_COV = ACCEL_NOISE_DENSITY**2 * ODR
GYRO_COV  = GYRO_NOISE_DENSITY**2 * ODR

ACCEL_COVARIANCE = [ACCEL_COV, 0, 0,  0, ACCEL_COV, 0,  0, 0, ACCEL_COV]
GYRO_COVARIANCE  = [GYRO_COV,  0, 0,  0, GYRO_COV,  0,  0, 0, GYRO_COV ]
```

---

## ROS 2 Node: imu_bridge_node.py

```python
#!/usr/bin/env python3
"""
imu_bridge_node.py — reads SPI frames from STM32 and publishes sensor_msgs/Imu

Run with:
  ros2 run my_imu_pkg imu_bridge_node
or directly:
  python3 imu_bridge_node.py
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
import spidev
import struct
import math

FRAME_FMT    = '<BBHhhhhhhIB5xH'
FRAME_FIELDS = ('magic', 'version', 'seq', 'ax', 'ay', 'az',
                'gx', 'gy', 'gz', 'timestamp_us', 'status', 'crc')
SPI_FRAME_SIZE = 32
MAGIC    = 0xA5
VERSION  = 0x01

ACCEL_SCALE = 9.81 / 16384.0
GYRO_SCALE  = 1.0 / (0.0028 * math.pi / 180.0 * (1 / (200.0 / 65.5)))
# Simpler: ICM-42688 ±500dps → 65.5 LSB/(°/s)
GYRO_LSB_PER_DPS = 65.5
GYRO_SCALE = (math.pi / 180.0) / GYRO_LSB_PER_DPS   # rad/s per LSB

ACCEL_COV = (70e-6 * 9.81)**2 * 200
GYRO_COV  = (0.0028 * math.pi / 180)**2 * 200


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = (crc << 1) ^ 0x1021 if crc & 0x8000 else crc << 1
        crc &= 0xFFFF
    return crc


class ImuBridgeNode(Node):
    def __init__(self):
        super().__init__('imu_bridge')

        # Parameters
        self.declare_parameter('frame_id', 'imu_link')
        self.declare_parameter('spi_bus', 0)
        self.declare_parameter('spi_device', 0)
        self.declare_parameter('spi_speed_hz', 4_000_000)
        self.declare_parameter('rate_hz', 100.0)

        frame_id      = self.get_parameter('frame_id').value
        spi_bus       = self.get_parameter('spi_bus').value
        spi_device    = self.get_parameter('spi_device').value
        spi_speed_hz  = self.get_parameter('spi_speed_hz').value
        rate_hz       = self.get_parameter('rate_hz').value

        # SPI
        self.spi = spidev.SpiDev()
        self.spi.open(spi_bus, spi_device)
        self.spi.max_speed_hz = spi_speed_hz
        self.spi.mode = 0
        self.dummy_tx = [0x00] * SPI_FRAME_SIZE
        self.get_logger().info(f"SPI {spi_bus}.{spi_device} @ {spi_speed_hz/1e6:.0f}MHz")

        # Publisher — SENSOR_DATA QoS: best-effort, keep-last 10
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.pub = self.create_publisher(Imu, 'imu/raw', qos)
        self.frame_id = frame_id

        # State
        self.last_seq = None
        self.crc_errors = 0
        self.dropped_frames = 0
        self.total_frames = 0

        # 100Hz timer
        period = 1.0 / rate_hz
        self.timer = self.create_timer(period, self.timer_cb)
        self.get_logger().info(f"IMU bridge started at {rate_hz:.0f}Hz")

    def timer_cb(self):
        raw = bytes(self.spi.xfer2(self.dummy_tx))
        self.total_frames += 1

        frame = dict(zip(FRAME_FIELDS, struct.unpack(FRAME_FMT, raw)))

        # Validate magic/version
        if frame['magic'] != MAGIC or frame['version'] != VERSION:
            self.get_logger().warn(
                f"Bad frame header: magic=0x{frame['magic']:02x} ver={frame['version']}",
                throttle_duration_sec=1.0)
            return

        # Validate CRC
        expected_crc = crc16_ccitt(raw[:-2])
        if frame['crc'] != expected_crc:
            self.crc_errors += 1
            self.get_logger().warn(
                f"CRC error #{self.crc_errors}: got 0x{frame['crc']:04x} "
                f"expected 0x{expected_crc:04x}",
                throttle_duration_sec=1.0)
            return

        # Check for dropped frames
        if self.last_seq is not None:
            expected_seq = (self.last_seq + 1) & 0xFFFF
            if frame['seq'] != expected_seq:
                dropped = (frame['seq'] - self.last_seq - 1) & 0xFFFF
                self.dropped_frames += dropped
                self.get_logger().warn(
                    f"Dropped {dropped} frames (seq {self.last_seq}→{frame['seq']})",
                    throttle_duration_sec=1.0)
        self.last_seq = frame['seq']

        # Build Imu message
        msg = Imu()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.linear_acceleration.x = frame['ax'] * ACCEL_SCALE
        msg.linear_acceleration.y = frame['ay'] * ACCEL_SCALE
        msg.linear_acceleration.z = frame['az'] * ACCEL_SCALE
        msg.linear_acceleration_covariance = [
            ACCEL_COV, 0, 0,
            0, ACCEL_COV, 0,
            0, 0, ACCEL_COV,
        ]

        msg.angular_velocity.x = frame['gx'] * GYRO_SCALE
        msg.angular_velocity.y = frame['gy'] * GYRO_SCALE
        msg.angular_velocity.z = frame['gz'] * GYRO_SCALE
        msg.angular_velocity_covariance = [
            GYRO_COV, 0, 0,
            0, GYRO_COV, 0,
            0, 0, GYRO_COV,
        ]

        # Orientation unknown — set covariance[0]=-1 to signal this
        msg.orientation_covariance[0] = -1.0

        self.pub.publish(msg)

    def destroy_node(self):
        self.get_logger().info(
            f"Shutting down. Frames: {self.total_frames}, "
            f"CRC errors: {self.crc_errors}, Dropped: {self.dropped_frames}")
        self.spi.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImuBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

---

## Verification Commands

```bash
# In terminal 1: run the node
python3 imu_bridge_node.py

# In terminal 2: verify rate
ros2 topic hz /imu/raw
# Expected: average rate: 100.000

# In terminal 3: echo a few messages
ros2 topic echo /imu/raw --once
# Check: linear_acceleration.z ≈ 9.81 when flat

# Check topic info (verify BEST_EFFORT QoS)
ros2 topic info /imu/raw --verbose

# Record a 10-second bag for offline analysis
ros2 bag record /imu/raw -d 10 -o imu_test_$(date +%Y%m%d_%H%M%S)

# Play back and check
ros2 bag info <bag_dir>
ros2 bag play <bag_dir>
ros2 topic echo /imu/raw
```

---

## RViz2 IMU Visualization

```bash
# Launch RViz2
rviz2

# In RViz2:
# 1. Fixed Frame: "imu_link" (or "map" if you have a TF tree)
# 2. Add → Imu (from sensor_msgs)
# 3. Topic: /imu/raw
# 4. Arrow Scale: 0.5
```

You should see the arrow pointing "up" (in the direction of gravity) and rotating
when you physically rotate the board.

---

## Milestone Checklist

- [ ] `ros2 topic hz /imu/raw` shows `100.000 ± 2.0`
- [ ] `ros2 topic echo /imu/raw --once` shows `linear_acceleration.z ≈ 9.81`
- [ ] `orientation_covariance[0] = -1.0` (orientation unknown)
- [ ] RViz2 IMU arrow responds to board rotation
- [ ] CRC and dropped frame counters at 0 after 60-second run
- [ ] `ros2 bag record` bag file playable and parseable

---

## Pre-Read for Session 7

Before `07-estimator-replay.md`:
1. Review the STUDY-PLAN notes on AMR estimator architecture
2. Read `knowledge/systems/amr-estimator.md` (if it exists)
3. Look at `scripts/analysis/` — what estimator analysis scripts exist?
4. Review any past RCA in `docs/rca/` mentioning `ESTIMATOR_STATE_INVALID`

---

## Session Notes Template

```markdown
## Session Notes — [DATE]

### ROS 2 Setup
- ROS 2 version: ___
- Python spidev version: ___

### Rate Measurement
- `ros2 topic hz` result: ___Hz
- Variance: ±___Hz

### Message Validation
- Stationary accel_z: ___ m/s²
- Gyro bias (stationary): x=___ y=___ z=___ rad/s

### Issues
- ...

### Bag File
- Filename: ...
- Duration: ___s
- Frames: ___
```
