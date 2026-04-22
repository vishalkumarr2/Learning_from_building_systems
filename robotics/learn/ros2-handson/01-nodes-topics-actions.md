# 01 — Nodes, Topics, Services, Actions, Lifecycle
### The communication primitives behind every AMR navigation stack
**Prerequisite:** Basic Python or C++, familiarity with the concept of processes and threads
**Unlocks:** Reading node graphs, debugging silent subscribers, understanding why a robot stops responding during CPU spikes, interpreting Nav2 startup sequences

---

## Why Should I Care? (Context)

A typical AMR navigation stack runs 15–30 ROS2 nodes simultaneously: localization, costmap servers, planners, controllers, recovery behaviors, sensor drivers, fleet connectors. When a robot stops mid-route or refuses a new goal, the first question is always: *which node stopped responding, and why?*

Understanding the execution model answers 80% of "why is node X not processing messages" questions:
- **Executor + spin** explains why a node goes silent during a CPU spike
- **Lifecycle nodes** explain "configured but not active" startup failures
- **Action vs service** explains why some goals never return a result
- **Topic QoS** (covered in 02) explains silent subscription failures

This chapter gives you the mental model. After reading it, you can trace a message from publisher to subscriber in your head and identify exactly where it breaks.

---

# PART 1 — THE ROS2 NODE MODEL

---

## 1.1 What a Node IS Physically

A ROS2 node is not just a software concept — it maps to concrete OS and network primitives.

```
OS Process / Thread
    │
    ├── rclcpp::Node (or rclpy.Node)
    │       ├── DDS Participant (one per Context, usually one per process)
    │       │       ├── DataWriter (for each publisher)
    │       │       └── DataReader (for each subscriber)
    │       ├── Timers
    │       ├── Services
    │       └── Action servers/clients
    │
    └── Executor (runs the event loop)
```

**DDS Participant** is the network identity. When a node starts, it announces its presence via multicast. Other participants learn about it and negotiate QoS. There is no central broker — this is peer-to-peer discovery.

**Key insight:** One ROS2 process can host multiple nodes (a `NodeContainer` / component container does this). All nodes in the same process share the same executor unless you explicitly create multiple.

---

## 1.2 The Executor: The Heart of Every Node

The executor is the event loop. It polls for:
- Incoming messages on subscriptions
- Timer firings
- Service requests
- Action goal/cancel/result requests

When an event arrives, the executor calls your callback.

```
                  Event Queue
                 ┌───────────┐
  Subscription   │ msg_cb    │
  callback ─────►│ timer_cb  │──► Executor ──► calls callbacks
  Timer          │ srv_cb    │     (spin)       one at a time
  Service ───────│ ...       │                  (single-threaded)
                 └───────────┘
```

### SingleThreadedExecutor vs MultiThreadedExecutor

| Executor | Behaviour | Use case |
|---|---|---|
| `SingleThreadedExecutor` | One callback at a time, in order | Default. Fine for most nodes. |
| `MultiThreadedExecutor` | Multiple callbacks concurrently | High-frequency subscriptions + slow services |
| `StaticSingleThreadedExecutor` | Pre-allocates, lower latency | Real-time critical nodes |

**`spin()` vs `spin_once()`:**
- `spin(node)` — blocks forever, processes callbacks as they arrive. The normal mode.
- `spin_once(node, timeout)` — processes at most one pending callback, then returns. Useful when you need to interleave ROS processing with other work.

**The critical trap:**

```python
# DANGER: blocking callback in single-threaded executor
def my_callback(msg):
    time.sleep(2.0)          # ← blocks for 2 seconds
    # During this sleep, ALL other callbacks (timers, other subscriptions)
    # are queued and NOT processed. The node appears frozen.
```

**Fix:** Use a MultiThreadedExecutor + `ReentrantCallbackGroup`, or offload slow work to a thread.

---

## 1.3 rclcpp::Node vs rclcpp_lifecycle::LifecycleNode

```
rclcpp::Node
    │
    └── rclcpp_lifecycle::LifecycleNode
            │
            ├── on_configure()    ← called on configure transition
            ├── on_activate()     ← called on activate transition
            ├── on_deactivate()   ← called on deactivate transition
            └── on_cleanup()      ← called on cleanup transition
```

A plain `Node` starts and immediately begins processing. A `LifecycleNode` starts in `UNCONFIGURED` state and requires explicit transitions before it processes anything. Nav2 uses lifecycle nodes for every major component.

---

## 1.4 Useful Node Options

```cpp
rclcpp::NodeOptions options;
options.use_intra_process_comms(true);   // zero-copy for same-process nodes
options.arguments({"--ros-args", "-r", "__node:=my_node_name"});

auto node = std::make_shared<MyNode>(options);
```

- `use_intra_process_comms`: bypasses serialization when pub and sub are in the same process. Critical for high-frequency image topics.
- Namespace: `ros2 run my_pkg my_node --ros-args -r __ns:=/robot1` — prefixes all topics
- Remapping: `ros2 run my_pkg my_node --ros-args -r /scan:=/robot1/scan`

---

# PART 2 — TOPICS: PUBLISH/SUBSCRIBE

---

## 2.1 DDS Discovery: No Broker Required

Unlike MQTT or ROS1 (which needs `rosmaster`), ROS2 DDS discovery is fully decentralized.

```
Node A starts               Node B starts
    │                           │
    │── multicast announce ─────►│
    │                           │── checks: do I want this topic?
    │◄─ multicast announce ─────│
    │── checks: do I want that? │
    │                           │
    └── both agree QoS ─────────┘
            │
         Data flows directly peer-to-peer
         (no broker in the middle)
```

Discovery uses DDS Simple Discovery Protocol (SDP). By default, multicast on `239.255.0.1`. This can be blocked by managed network switches — a common fleet deployment problem.

---

## 2.2 Message Flow: What Happens Inside

```
Publisher side                          Subscriber side
──────────────                          ───────────────
Your data struct                        Your callback
    │                                       ▲
    ▼                                       │
Serialize (CDR binary)              Deserialize (CDR binary)
    │                                       │
    ▼                                       │
DDS DataWriter                      DDS DataReader
    │                                       │
    └──────────── network/IPC ──────────────┘
```

**CDR** (Common Data Representation) is the binary format. For intra-process comms with `use_intra_process_comms(true)`, serialization is skipped — the pointer is passed directly.

---

## 2.3 Creating a Publisher and Subscriber

**Publisher (C++):**
```cpp
// In node constructor:
auto pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
    "/cmd_vel",          // topic name
    rclcpp::QoS(10)      // QoS: depth=10, default RELIABLE+VOLATILE
);

// To publish:
auto msg = geometry_msgs::msg::Twist();
msg.linear.x = 0.5;
pub_->publish(msg);
```

**Subscriber (C++):**
```cpp
auto sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    "/odom",
    rclcpp::QoS(10),
    std::bind(&MyNode::odom_callback, this, std::placeholders::_1)
);

void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    // process msg
}
```

**The depth (queue size):** If your callback is slow and messages arrive fast, the queue fills. With `KEEP_LAST(10)`, the 11th message evicts the 1st. If you use `KEEP_ALL`, memory grows unboundedly.

---

## 2.4 Worked Example: Odometry Velocity Magnitude Subscriber

**Problem:** Subscribe to `/odom` and print the robot's speed (magnitude of linear velocity).

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math

class SpeedMonitor(Node):
    def __init__(self):
        super().__init__('speed_monitor')
        self.sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_cb,
            10)                         # queue depth

    def odom_cb(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        speed = math.sqrt(vx**2 + vy**2)   # m/s
        self.get_logger().info(f'Speed: {speed:.3f} m/s')

def main():
    rclpy.init()
    rclpy.spin(SpeedMonitor())

# Note: msg.twist.twist.linear is the velocity in the robot body frame.
# msg.pose.pose.position is the global position.
```

**Step by step:**
1. `Odometry.twist.twist.linear.x` — forward velocity in body frame
2. `Odometry.twist.twist.linear.y` — lateral velocity (zero for diff-drive)
3. Speed = √(vx² + vy²) — scalar magnitude, direction-agnostic

---

## 2.5 Timers: Driving a Publish Loop

```cpp
// Publish at 50 Hz
timer_ = this->create_wall_timer(
    std::chrono::milliseconds(20),          // 1000ms / 50 = 20ms period
    std::bind(&MyNode::timer_callback, this)
);

void timer_callback() {
    auto msg = std_msgs::msg::Float64();
    msg.data = compute_something();
    pub_->publish(msg);
}
```

`create_wall_timer` uses wall clock (real time). For simulation/bag replay with `use_sim_time: true`, use `create_timer` with the node clock — it respects `/clock`.

---

## 2.6 Topic Graph: ASCII Diagram

```
  ┌─────────────────────┐          ┌─────────────────────┐
  │   lidar_driver      │          │   slam_node         │
  │  (sensor driver)    │          │  (localization)      │
  │                     │          │                     │
  │  [pub] /scan ───────┼──────────┼──► [sub] /scan      │
  │                     │          │                     │
  │  [pub] /diagnostics │          │  [pub] /map ────────┼──► ...
  └─────────────────────┘          └─────────────────────┘

  ┌─────────────────────┐          ┌─────────────────────┐
  │   odom_publisher    │          │   nav2_controller   │
  │                     │          │                     │
  │  [pub] /odom ───────┼──────────┼──► [sub] /odom      │
  └─────────────────────┘          └─────────────────────┘
```

Inspect the live graph: `ros2 topic list`, `ros2 topic echo /topic`, `ros2 node info /node_name`

---

# PART 3 — SERVICES: SYNCHRONOUS REQUEST-RESPONSE

---

## 3.1 When to Use Services vs Topics

| Criteria | Use Topic | Use Service |
|---|---|---|
| Data rate | Continuous / high-frequency | Occasional / on-demand |
| Direction | One-way (pub → sub) | Bidirectional (request → response) |
| Examples | `/odom`, `/scan`, `/cmd_vel` | `clear_costmap`, `set_map`, `get_plan` |
| Persistence | Subscriber gets only future messages | Client waits for response |

**Rule of thumb:** If you need a reply and the call is infrequent, use a service. If you're streaming data, use a topic.

---

## 3.2 Service Server and Client

**Server (Python):**
```python
from nav2_msgs.srv import ClearEntireCostmap
import rclpy

class CostmapClearer(Node):
    def __init__(self):
        super().__init__('costmap_clearer')
        self.srv = self.create_service(
            ClearEntireCostmap,
            'clear_entirely_global_costmap/clear',
            self.clear_callback)

    def clear_callback(self, request, response):
        # do the work
        self.get_logger().info('Clearing costmap...')
        response.return_code = 0
        return response
```

**Client (Python) — async pattern:**
```python
client = node.create_client(ClearEntireCostmap,
                            'clear_entirely_global_costmap/clear')

# Wait for server to be available (non-blocking with timeout)
if not client.wait_for_service(timeout_sec=2.0):
    node.get_logger().error('Service not available!')
else:
    future = client.call_async(ClearEntireCostmap.Request())
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    if future.done():
        result = future.result()
```

---

## 3.3 The Deadlock Trap

```
   Callback thread (executor)
        │
        ▼
   service_callback() running
        │
        ▼
   client.call()  ← BLOCKS waiting for a response
        │
        ▼
   ... but the executor is blocked inside service_callback()
       so it cannot process the response that arrives!
       DEADLOCK.
```

**Fix:** Never call a service synchronously from within a callback. Use `call_async()` and a future, or use a separate thread.

---

## 3.4 Service Timeline

```
Client                                    Server
  │                                         │
  │─── Request (goal params) ──────────────►│
  │                                         │ (processes...)
  │◄─── Response (result) ──────────────────│
  │                                         │
  │ (client unblocks, future is ready)      │

  Total time: one round-trip.
  No intermediate feedback.
  If server crashes: client times out.
```

---

# PART 4 — ACTIONS: ASYNC WITH FEEDBACK

---

## 4.1 What Actions Add Over Services

Actions are for **long-running goals** that need:
1. An async goal request (don't block the caller)
2. Periodic feedback while executing
3. A final result when done
4. The ability to cancel mid-execution

Nav2's `NavigateToPose` is an action. The robot might take 30 seconds to navigate — you don't want a blocking service call for that.

---

## 4.2 The 5 Message Types

```
Client ──── GoalRequest ────────────────────────────────────► Server
Client ◄─── GoalResponse (accepted/rejected) ───────────────  Server
Client ◄─── FeedbackMessage (current_pose, ETA...) ──────────  Server (repeated)
Client ──── CancelGoalRequest ──────────────────────────────► Server (optional)
Client ◄─── GetResultResponse (final result / error code) ───  Server (once)
```

The action interface (`.action` file) defines three sections:
```
# Goal
geometry_msgs/PoseStamped pose
---
# Result
std_msgs/Empty result
---
# Feedback
geometry_msgs/PoseStamped current_pose
float32 navigation_time
```

---

## 4.3 Preempting Goals (Cancellation)

```python
# Cancel an in-progress goal:
cancel_future = goal_handle.cancel_goal_async()
rclpy.spin_until_future_complete(node, cancel_future)

# On the server side, implement cancel handling:
def cancel_callback(self, goal_handle):
    self.get_logger().info('Received cancel request')
    return CancelResponse.ACCEPT       # or REJECT
```

**Key insight:** Not all action servers honour cancellation. If `cancel_callback` returns `REJECT`, the goal continues. Nav2's `NavigateToPose` server accepts cancellations and stops the robot.

---

## 4.4 Worked Example: Counter Action Server (Python Pseudocode)

```python
# Action: count from 0 to N, publishing feedback each second

class CounterActionServer(Node):
    def __init__(self):
        super().__init__('counter_server')
        self._server = ActionServer(
            self,
            CountTo,                       # action type
            'count_to',                    # action name
            self.execute_callback)

    async def execute_callback(self, goal_handle):
        n = goal_handle.request.target     # e.g. 10
        feedback = CountTo.Feedback()

        for i in range(n):
            feedback.current_count = i
            goal_handle.publish_feedback(feedback)
            await asyncio.sleep(1.0)       # 1 second per step

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return CountTo.Result(final_count=i)

        goal_handle.succeed()
        result = CountTo.Result()
        result.final_count = n
        return result
```

**Step by step:**
1. Server accepts goal → extracts `request.target`
2. Loop: publish feedback every iteration
3. Check for cancellation every loop iteration
4. On loop completion: call `succeed()`, return `Result`

---

## 4.5 Action Timeline

```
Client                                         Server
  │                                               │
  │─── GoalRequest {target: 10} ────────────────►│
  │◄── GoalResponse {accepted: true} ────────────│
  │                                               │ (executing loop)
  │◄── Feedback {current_count: 0} ──────────────│
  │◄── Feedback {current_count: 1} ──────────────│
  │     ...                                       │
  │◄── Feedback {current_count: 9} ──────────────│
  │◄── Result {final_count: 10, status: SUCCEEDED}│
  │                                               │
  (Client's future is now ready)
```

---

# PART 5 — LIFECYCLE NODES

---

## 5.1 The 4 Primary States

```
                    configure()
  UNCONFIGURED ─────────────────► INACTIVE
       ▲                              │  ▲
       │            cleanup()         │  │ deactivate()
       │◄───────────────────────      │  │
       │                         activate()
       │                              │
       │                              ▼
  FINALIZED ◄────────────────── ACTIVE
              shutdown()
```

- **UNCONFIGURED** — node just started, no resources allocated. This is the starting state.
- **INACTIVE** — node is configured (parameters loaded, publishers/subscribers created), but not processing data.
- **ACTIVE** — node is fully operational. Timers fire, callbacks run, data is processed.
- **FINALIZED** — node has been shut down cleanly. Cannot transition out.
- **ERROR** — any transition callback can throw to enter this state. Requires `cleanup()` to recover.

---

## 5.2 The Transition Callbacks

```cpp
class MyLifecycleNode : public rclcpp_lifecycle::LifecycleNode {
public:
    // Called when: UNCONFIGURED → INACTIVE
    // Use for: loading parameters, creating pubs/subs
    CallbackReturn on_configure(const State &) override {
        pub_ = this->create_publisher<...>("topic", 10);
        return CallbackReturn::SUCCESS;   // or FAILURE to stay UNCONFIGURED
    }

    // Called when: INACTIVE → ACTIVE
    // Use for: starting timers, enabling processing
    CallbackReturn on_activate(const State &) override {
        timer_ = this->create_wall_timer(...);
        return CallbackReturn::SUCCESS;
    }

    // Called when: ACTIVE → INACTIVE
    // Use for: stopping timers, pausing processing
    CallbackReturn on_deactivate(const State &) override {
        timer_.reset();
        return CallbackReturn::SUCCESS;
    }

    // Called when: INACTIVE → UNCONFIGURED
    // Use for: releasing resources, resetting state
    CallbackReturn on_cleanup(const State &) override {
        pub_.reset();
        return CallbackReturn::SUCCESS;
    }
};
```

**Key insight:** Publishers and subscribers created in `on_configure()` still exist in INACTIVE state — they just don't do anything because timers and data processing start in `on_activate()`. This is why a node in INACTIVE state can receive messages but silently drop them.

---

## 5.3 The Lifecycle Manager

Nav2 ships a `lifecycle_manager` node that automates the state transitions for all Nav2 nodes in the correct order:

```
lifecycle_manager calls:
    1. configure(amcl)
    2. configure(map_server)
    3. configure(costmap_2d_global)
    4. configure(costmap_2d_local)
    5. configure(nav2_planner)
    6. configure(nav2_controller)
    7. configure(bt_navigator)
    ... (same for activate)
```

If any node fails its `on_configure()`, the lifecycle manager halts and logs the failure. This is the most common cause of "Nav2 won't start" — one node's configuration callback returning `FAILURE`.

---

## 5.4 Debugging Lifecycle State

```bash
# Check current state of all managed nodes
ros2 lifecycle list /amcl
ros2 lifecycle list /bt_navigator

# Manually trigger a transition (useful for testing)
ros2 lifecycle set /amcl configure
ros2 lifecycle set /amcl activate

# Get the full list of available transitions from current state
ros2 lifecycle get /amcl
```

**"Node is in INACTIVE state"** — `on_configure()` succeeded, but `on_activate()` has not been called yet. Either the lifecycle manager hasn't reached that step, or `on_activate()` returned FAILURE on a previous attempt.

---

## Summary — What to Remember

| Concept | Rule |
|---|---|
| Node | A DDS participant + executor event loop. One process can host many nodes. |
| Executor | Runs callbacks one at a time (single-threaded). Blocking callback = all callbacks stall. |
| `spin()` | Never returns. Processes events forever. The normal launch pattern. |
| `spin_once()` | Processes at most one event, then returns. For manual event loop control. |
| Publisher | `create_publisher<T>(topic, QoS)`. Depth = how many messages queue before eviction. |
| Subscriber | `create_subscription<T>(topic, QoS, callback)`. Callback runs inside executor. |
| Topic | Unidirectional, continuous stream. No reply. No delivery guarantee by default. |
| Service | Request-response. Bidirectional. Avoid blocking calls inside callbacks (deadlock). |
| Action | Async goal + feedback stream + result. Use for long-running tasks (navigation). |
| Lifecycle INACTIVE | Configured but not active. Pubs/subs exist; timers do not. |
| Lifecycle ACTIVE | Fully operational. Timers fire. Callbacks run. |
| Lifecycle manager | Coordinates ordered startup/shutdown of all Nav2 nodes. |
