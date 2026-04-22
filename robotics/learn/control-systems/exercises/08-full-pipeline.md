# Exercise 8 — Full Pipeline: Motor to Waypoint

**Covers:** All lessons (01–10)
**Difficulty:** Capstone / Senior-Level

---

## The Challenge

Build an end-to-end motor control pipeline in simulation. You start with a DC motor model and end with a differential-drive robot tracking waypoints. This is the full stack — exactly what OKS does, simplified.

**Time estimate:** 4–8 hours

---

## Stage 1: Motor Model (Lesson 02)

Create a Python DC motor simulator:

```python
class DCMotor:
    def __init__(self):
        self.Ra = 1.0       # Ohms
        self.La = 0.5e-3    # Henries (can neglect for speed loop)
        self.Kt = 0.05      # N·m/A
        self.Ke = 0.05      # V·s/rad
        self.J = 0.001      # kg·m²
        self.B = 0.005      # N·m·s/rad
        self.gear_ratio = 10.0
        self.efficiency = 0.85
        
        # State
        self.current = 0.0
        self.speed = 0.0     # Motor shaft
        self.position = 0.0  # Motor shaft
    
    def step(self, voltage, dt, load_torque=0.0):
        """Advance motor state by dt seconds."""
        # Your implementation here
        # Return: (current, speed, output_position)
        pass
```

**Deliverable:** Step response plot showing current, speed, and position for a 12V step input.

---

## Stage 2: PID Controller (Lessons 03–04)

Implement a discrete PID with anti-windup:

```python
class DiscretePID:
    def __init__(self, Kp, Ki, Kd, dt, output_min, output_max):
        # Your implementation here
        pass
    
    def update(self, setpoint, measurement):
        """Compute control output."""
        # Use velocity form
        # Include derivative-on-measurement
        # Include back-calculation anti-windup
        pass
    
    def reset(self):
        """Reset integrator and state."""
        pass
```

**Deliverable:** Speed step response: command 10 rad/s, show output and control effort. Tune for < 10% overshoot.

---

## Stage 3: Cascade Controller (Lesson 07)

Build current → speed → position cascade:

```python
class CascadeController:
    def __init__(self):
        self.current_pid = DiscretePID(...)  # 10 kHz
        self.speed_pid = DiscretePID(...)    # 1 kHz  
        self.position_pid = DiscretePID(...) # 100 Hz
    
    def update(self, position_setpoint, measured_position, 
               measured_speed, measured_current):
        # Outer to inner
        speed_cmd = self.position_pid.update(position_setpoint, measured_position)
        current_cmd = self.speed_pid.update(speed_cmd, measured_speed)
        voltage_cmd = self.current_pid.update(current_cmd, measured_current)
        return voltage_cmd
```

**Deliverable:** Position step response (1 revolution). Show all three loop outputs: voltage, current command, speed command, position.

---

## Stage 4: Differential Drive (Lesson 08)

Create a differential drive robot:

```python
class DiffDriveRobot:
    def __init__(self):
        self.wheel_sep = 0.4      # meters
        self.wheel_radius = 0.075 # meters
        
        self.left_motor = DCMotor()
        self.right_motor = DCMotor()
        self.left_cascade = CascadeController()
        self.right_cascade = CascadeController()
        
        # Robot pose
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
    
    def set_velocity(self, v, omega):
        """Convert (v, omega) to wheel speed setpoints."""
        pass
    
    def step(self, dt):
        """Advance simulation by dt."""
        pass
```

**Deliverable:** Drive in a 2m radius circle. Plot the robot's x-y trajectory.

---

## Stage 5: Trajectory Tracking (Lesson 08)

Implement Pure Pursuit to track a series of waypoints:

```python
waypoints = [
    (0.0, 0.0),
    (3.0, 0.0),
    (3.0, 3.0),
    (0.0, 3.0),
    (0.0, 0.0),  # Back to start
]
```

**Deliverable:**
1. Plot desired path (dashed) and actual path (solid)
2. Plot cross-track error over time
3. Plot wheel speeds over time
4. Maximum cross-track error should be < 5 cm at $v = 0.3$ m/s

---

## Stage 6: Two-Layer Split (Lesson 09)

Separate your simulation into two threads (or two update loops with different rates):

**MCU layer** (1 kHz):
- Receives speed setpoint
- Runs speed and current PID
- Commands motor voltage (PWM)
- Sends actual speed and position back

**Jetson layer** (20 Hz):
- Receives position feedback
- Runs Pure Pursuit
- Computes $(v, \omega)$ → wheel speed setpoints
- Sends to MCU layer

Add communication delay: 5 ms average, 15 ms worst case (random).

**Deliverable:** Same waypoint test as Stage 5, but now with communication delay. Compare cross-track error.

---

## Stage 7: Stress Test

Add realistic disturbances:

```python
# Random load torque (hitting a bump)
load_torque = random.gauss(0, 0.01)

# Wheel slip (one wheel on smooth floor)
if random.random() < 0.01:  # 1% chance per step
    left_motor.friction *= 0.5  # Slip for 100 ms

# cmd_vel gap (Jetson timer jitter under load)
if random.random() < 0.005:
    jetson_delay = random.uniform(0.05, 0.15)  # 50-150 ms extra
```

**Deliverable:**
1. Run 1000 laps of the square path
2. Report: max cross-track error, number of near-misses (> 10 cm), recovery time
3. Add your cmd_vel staleness detection from Debug Session 3
4. Compare statistics with vs without staleness detection

---

## Grading Rubric

| Stage | Points | Criteria |
|-------|--------|----------|
| 1 | 10 | Motor model matches expected dynamics |
| 2 | 15 | PID works, anti-windup verified, < 10% overshoot |
| 3 | 15 | Cascade runs at correct rates, inside-out tuned |
| 4 | 10 | Differential drive kinematics correct |
| 5 | 15 | Waypoint tracking < 5 cm error |
| 6 | 20 | Two-layer split works with delay |
| 7 | 15 | Staleness detection improves robustness |
| **Total** | **100** | |

---

## Bonus Challenges (Advanced)

1. **Fixed-point version:** Convert Stage 3's PID to Q16.16 C code. Run in a C subprocess called from Python.

2. **MPC comparison:** Replace Pure Pursuit with a simple MPC (Lesson 10). Does it track tighter corners?

3. **Gain scheduling:** Add speed-dependent gains (Lesson 10). Does it improve tracking at different speeds?

4. **Realistic encoder:** Add quantization noise (2048 CPR encoder). How does it affect speed estimation?

5. **Fault injection:** Simulate a motor fault (one wheel stuck) mid-path. Does the robot recover or crash?
