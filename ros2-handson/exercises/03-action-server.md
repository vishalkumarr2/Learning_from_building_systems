# Exercise03 — Action Server and Client
### Companion exercises for `01-nodes-topics-actions.md` (Actions section)

**Estimated time:** 75 minutes  
**Prerequisite:** [01-nodes-topics-actions.md](../01-nodes-topics-actions.md)

**Self-assessment guide:** Actions are the hardest ROS2 communication primitive to debug
because their state machine has 7 states. If you can draw the state machine from memory
before expanding A1, you are ready for Section C.

---

## Overview

Actions power every long-running navigation primitive: `NavigateToPose`, `FollowWaypoints`,
`ComputePathToPose`. Debugging a stuck robot almost always involves reading action server
state transitions in logs. Understanding *why* a goal was rejected vs. aborted vs. cancelled
is the difference between a 2-minute fix and a 2-hour debugging session.

These exercises build a complete `CountDown` action from the `.action` file through server
implementation, client, and preemption — the same pattern used by all Nav2 action servers.

---

## Section A — Conceptual Questions

**A1.** Name all 5 message types involved in a ROS2 action and describe the flow of data
in each direction.

<details><summary>Answer</summary>

A ROS2 action is compiled from a `.action` file into three auto-generated message types,
but internally it uses five distinct message structures:

| Message | Direction | Purpose |
|---|---|---|
| **Goal** | Client → Server | The desired outcome (e.g., target pose, countdown start value) |
| **Goal Response** | Server → Client | `accepted` or `rejected` — the server's decision on whether to execute |
| **Feedback** | Server → Client (streaming) | Intermediate progress updates while the action runs |
| **Cancel Request** | Client → Server | Request to stop an in-progress goal |
| **Result** | Server → Client | Final outcome, delivered once the goal completes, is cancelled, or aborts |

**Data flow timeline:**
```
Client                               Server
  │── GoalRequest ──────────────────►│  (is goal feasible?)
  │◄─ GoalResponse (accept/reject) ──│
  │                                  │  (start executing)
  │◄─ Feedback ──────────────────────│  (periodic updates)
  │◄─ Feedback ──────────────────────│
  │── CancelRequest ────────────────►│  (optional: client requests cancel)
  │◄─ CancelResponse ────────────────│
  │◄─ Result (cancelled/done/abort) ─│  (terminal state)
```

**Key distinction:** The `GoalResponse` only says "I will try" or "I refuse to try". It
does NOT contain the result. The result comes later, asynchronously, after the server
finishes (or fails, or is cancelled).

</details>

- [ ] Done

---

**A2.** How does a client cancel an action that is in progress? Describe the sequence from
the client's perspective and what the server must do to honour the cancel.

<details><summary>Answer</summary>

**Client side:**
```python
# Obtain the goal handle (returned when goal was accepted)
cancel_future = goal_handle.cancel_goal_async()
cancel_response = await cancel_future     # CancelGoal.Response
if cancel_response.return_code == CancelGoal.Response.ERROR_NONE:
    self.get_logger().info('Cancel request accepted by server')
```

The client calls `goal_handle.cancel_goal_async()`, which sends a `CancelGoal` service
request to the action server. This is a *request* — the server is free to ignore it or
accept it.

**Server side:**
The action server receives the cancel request and must decide whether to honour it. In the
execute callback:
```python
def execute(self, goal_handle):
    while not done:
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()               # mark as cancelled
            result = MyAction.Result()
            result.outcome = 'cancelled'
            return result
        # ... do work ...
```

The server **must** poll `goal_handle.is_cancel_requested` in its execute loop and call
`goal_handle.canceled()` when it decides to honour the cancel. Simply not calling it means
the cancel is silently ignored.

**Important:** `canceled()` is not automatic. A server that ignores `is_cancel_requested`
will continue executing until it finishes naturally, even if the client "cancelled" it.

</details>

- [ ] Done

---

**A3.** Why does Nav2 use actions instead of services for `NavigateToPose`?
Give two concrete reasons based on the technical differences between the two primitives.

<details><summary>Answer</summary>

**Reason 1 — Actions support streaming feedback; services do not.**

A robot navigating to a goal takes tens of seconds. With a service, the client is blocked
waiting for a response with no intermediate information. There is no way to stream "current
pose", "distance remaining", "current BT state" back to the caller during execution.

An action sends `Feedback` messages throughout execution. The navigation monitor (fleet
manager, UI, watchdog) can observe progress in real time and detect if the robot is stuck
*before* the action terminates.

**Reason 2 — Actions support cancellation; service calls cannot be cancelled mid-flight.**

Navigation can be interrupted at any time: an emergency stop, a higher-priority task, a
new destination. With a service, once the call is sent, the client can only wait or
disconnect. The server has no cancel signal.

With an action, the client calls `cancel_goal_async()`. The server receives a cancel
request, can cleanly stop the robot (bring velocity to zero, update costmap), and return
a result indicating cancellation. This is the safe way to preempt navigation in a
production robot.

**Additional reason (bonus):** Services use a synchronous request-reply model that
ties up a thread on the server for the duration of the call. For a 60-second navigation
task, this would block the service server thread. Actions execute their callback in a
separate thread (or coroutine) and release the server thread immediately after the goal
is accepted.

</details>

- [ ] Done

---

**A4.** What is the difference between a goal being "rejected" and a goal being "aborted"?
At what point in the goal lifecycle does each occur?

<details><summary>Answer</summary>

| | Rejected | Aborted |
|---|---|---|
| **When** | Before execution starts | During execution |
| **Who decides** | The `goal_callback` (called synchronously when goal arrives) | The `execute_callback` (called asynchronously after acceptance) |
| **Client receives** | `GoalResponse` with `accepted=False` | `Result` with status `ABORTED` |
| **Cause** | Goal is semantically invalid or preconditions are not met | An unexpected error occurred during execution |

**Rejected — in goal_callback:**
```python
def goal_callback(self, goal_request):
    if goal_request.count_from < 0:
        return GoalResponse.REJECT    # immediately refuse
    return GoalResponse.ACCEPT
```
The client's `send_goal_async` future resolves with `accepted=False`. No execution thread
is started.

**Aborted — in execute_callback:**
```python
def execute(self, goal_handle):
    # ... running for 5 seconds ...
    if hardware_fault_detected():
        goal_handle.abort()
        result = CountDown.Result()
        result.outcome = 'hardware fault'
        return result
```
The goal was accepted and execution began, but something went wrong that prevents
completion. The client's result callback receives `GoalStatus.STATUS_ABORTED`.

**Practical distinction for navigation:** "Rejected" = robot refused to try (probably the
goal was in a wall). "Aborted" = robot started moving but could not complete (TF error,
controller timeout, obstacle directly in path with no recovery).

</details>

- [ ] Done

---

## Section B — Spot the Bug: Action Client State Machine

This action client has three deliberate bugs. Read the code and identify each bug before
expanding the answer.

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from example_interfaces.action import Fibonacci


class BuggyClient(Node):
    def __init__(self):
        super().__init__('buggy_client')
        self._client = ActionClient(self, Fibonacci, 'fibonacci')
        # Bug 1: where is it?
        goal = Fibonacci.Goal()
        goal.order = 10
        future = self._client.send_goal_async(goal)   # Bug 2: missing callback
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        # Bug 3: incorrect result check
        if result == Fibonacci.Result():
            self.get_logger().info(f'Got result: {result.sequence}')
        else:
            self.get_logger().error('Action failed')
```

<details><summary>Answer</summary>

**Bug 1 — Server availability check missing.**

The client calls `send_goal_async()` immediately after creating `ActionClient`, without
waiting for the action server to be available. If the server is not running, the goal is
sent to nobody and the future may never resolve (or resolves with an error that is not
checked).

**Fix:**
```python
while not self._client.wait_for_server(timeout_sec=1.0):
    self.get_logger().warn('Waiting for action server...')
```

---

**Bug 2 — No feedback callback registered.**

`send_goal_async()` accepts an optional `feedback_callback` parameter. Without it, feedback
messages are silently discarded. For monitoring long-running actions, you almost always want
to register one.

**Fix:**
```python
future = self._client.send_goal_async(
    goal,
    feedback_callback=self.feedback_cb   # register handler
)

def feedback_cb(self, feedback_msg):
    self.get_logger().info(f'Feedback: {feedback_msg.feedback.partial_sequence}')
```

---

**Bug 3 — Incorrect result check.**

```python
if result == Fibonacci.Result():
```
This compares the entire `WrappedResult` object to a freshly constructed (empty)
`Fibonacci.Result()`. This comparison is almost always `False` because:
- `result` here is a `ClientGoalHandle.WrappedResult`, not a `Fibonacci.Result` directly.
- You need `result.result` to get the inner message.
- Comparing to an empty result object is not a valid success check — the action server
  returns a populated result even on success.

**Fix:** Check the goal status instead.
```python
from rclpy.action.client import GoalStatus

if result.status == GoalStatus.STATUS_SUCCEEDED:
    self.get_logger().info(f'Got result: {result.result.sequence}')
else:
    self.get_logger().error(f'Action failed with status: {result.status}')
```

</details>

- [ ] Done

---

## Section C — Build Exercises

### C1. CountDown Action Server

First, write the `.action` definition file. Then implement the action server.

**Action definition** (`CountDown.action`):
```
# Goal
int32 count_from
---
# Result
string outcome
---
# Feedback
int32 remaining
```

Implement the server that:
1. Accepts any goal with `count_from >= 0`. Rejects negative values.
2. Counts down from `count_from` to 0, publishing feedback every second.
3. If a cancel request arrives, stops immediately and returns `outcome = "cancelled"`.
4. On successful completion, returns `outcome = "done"`.

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
# TODO: remaining imports — CountDown action, time

class CountDownServer(Node):
    def __init__(self):
        super().__init__('countdown_server')
        # TODO: create ActionServer with goal_callback, cancel_callback, execute_callback

    def goal_callback(self, goal_request):
        # TODO: reject if count_from < 0
        pass

    def cancel_callback(self, goal_handle):
        # TODO: always accept cancel requests
        pass

    def execute(self, goal_handle):
        # TODO: countdown loop with feedback and cancel check
        pass

def main(args=None):
    rclpy.init(args=args)
    node = CountDownServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

<details><summary>Answer</summary>

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
import time

# Assumes the CountDown action package is built and sourced.
# from my_actions.action import CountDown
# For this exercise we inline a compatible structure:
from example_interfaces.action import CountDown   # replace with your package


class CountDownServer(Node):
    def __init__(self):
        super().__init__('countdown_server')
        self._action_server = ActionServer(
            self,
            CountDown,
            'count_down',
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            execute_callback=self.execute,
        )
        self.get_logger().info('CountDown action server ready.')

    def goal_callback(self, goal_request):
        if goal_request.count_from < 0:
            self.get_logger().warn(
                f'Rejected goal: count_from={goal_request.count_from} (must be >= 0)'
            )
            return GoalResponse.REJECT
        self.get_logger().info(
            f'Accepted goal: count_from={goal_request.count_from}'
        )
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Cancel request received — will honour it.')
        return CancelResponse.ACCEPT

    def execute(self, goal_handle):
        self.get_logger().info('Executing countdown...')
        feedback_msg = CountDown.Feedback()
        result = CountDown.Result()

        remaining = goal_handle.request.count_from

        while remaining >= 0:
            # Check for cancel before each step
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.outcome = 'cancelled'
                self.get_logger().info(f'Goal cancelled at remaining={remaining}')
                return result

            feedback_msg.remaining = remaining
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Feedback: remaining={remaining}')

            time.sleep(1.0)
            remaining -= 1

        goal_handle.succeed()
        result.outcome = 'done'
        self.get_logger().info('Countdown complete.')
        return result


def main(args=None):
    rclpy.init(args=args)
    node = CountDownServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

**Key points:**
- `goal_callback` is called synchronously on the executor thread that received the goal.
  It must return quickly (no blocking).
- `execute` runs in a separate thread spawned by the action server.
- `time.sleep(1.0)` is fine in `execute` because it runs in its own thread.
- `goal_handle.succeed()` must be called *before* returning the result when the goal
  completes normally. Forgetting this leaves the goal in `EXECUTING` state.
- The cancel check `is_cancel_requested` is polled, not pushed. The server must call it
  in its loop; there is no automatic interruption.

</details>

- [ ] Done

---

### C2. CountDown Action Client

Write the client that:
1. Sends a goal with `count_from = 5`.
2. Registers a feedback callback that prints each `remaining` value.
3. Waits for the result and prints `outcome`.

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
# TODO: remaining imports

class CountDownClient(Node):
    def __init__(self):
        super().__init__('countdown_client')
        # TODO: create ActionClient for 'count_down' topic
        # TODO: send goal count_from=5 with feedback callback
        # TODO: wait for result and log it

    def feedback_cb(self, feedback_msg):
        # TODO: log remaining
        pass

def main(args=None):
    rclpy.init(args=args)
    node = CountDownClient()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

<details><summary>Answer</summary>

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import GoalStatus
from example_interfaces.action import CountDown   # replace with your package


class CountDownClient(Node):
    def __init__(self):
        super().__init__('countdown_client')
        self._client = ActionClient(self, CountDown, 'count_down')

        self.get_logger().info('Waiting for action server...')
        self._client.wait_for_server()

        goal = CountDown.Goal()
        goal.count_from = 5

        self.get_logger().info(f'Sending goal: count_from={goal.count_from}')
        self._send_future = self._client.send_goal_async(
            goal,
            feedback_callback=self.feedback_cb,
        )
        self._send_future.add_done_callback(self.goal_response_cb)

    def feedback_cb(self, feedback_msg):
        remaining = feedback_msg.feedback.remaining
        self.get_logger().info(f'[feedback] remaining = {remaining}')

    def goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal was rejected by server.')
            return
        self.get_logger().info('Goal accepted. Waiting for result...')
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self.result_cb)

    def result_cb(self, future):
        result = future.result()
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Result: outcome = "{result.result.outcome}"')
        elif result.status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warn(f'Goal was cancelled. outcome = "{result.result.outcome}"')
        else:
            self.get_logger().error(f'Goal aborted. Status = {result.status}')


def main(args=None):
    rclpy.init(args=args)
    node = CountDownClient()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

**Key points:**
- The client is fully asynchronous — `send_goal_async` returns a future and `spin` drives
  the callbacks. No blocking `spin_until_future_complete` is needed when using done callbacks.
- `goal_response_cb` → `result_cb` is the standard two-step callback chain for actions.
- `feedback_msg.feedback.remaining` — note the extra `.feedback` field wrapping the
  actual `Feedback` message inside `FeedbackMessage`.

</details>

- [ ] Done

---

### C3. Preemption — Goal Rejection and Mid-Goal Cancellation

Modify the server from C1 to:
1. **Reject** any goal where `count_from >= 10`.
2. After the server is started, write a test client that:
   - Sends `count_from = 8` (accepted).
   - After receiving **3 feedback messages**, sends a cancel request.
   - Verifies that `outcome = "cancelled"` is returned.

```python
# Server modification — fill in the rejection logic:
def goal_callback(self, goal_request):
    # TODO: reject if count_from < 0 OR count_from >= 10
    pass

# Test client with cancel after 3 feedbacks:
class PreemptTestClient(Node):
    def __init__(self):
        super().__init__('preempt_test_client')
        self._client = ActionClient(self, CountDown, 'count_down')
        self._feedback_count = 0
        self._goal_handle = None

        self._client.wait_for_server()
        goal = CountDown.Goal()
        goal.count_from = 8
        self.get_logger().info('Sending goal count_from=8...')
        future = self._client.send_goal_async(
            goal,
            feedback_callback=self.feedback_cb
        )
        future.add_done_callback(self.goal_response_cb)

    def goal_response_cb(self, future):
        # TODO: store goal_handle, set up result callback
        pass

    def feedback_cb(self, feedback_msg):
        # TODO: increment count, cancel after 3
        pass

    def result_cb(self, future):
        # TODO: assert outcome == 'cancelled'
        pass
```

<details><summary>Answer</summary>

**Server modification:**
```python
def goal_callback(self, goal_request):
    if goal_request.count_from < 0 or goal_request.count_from >= 10:
        self.get_logger().warn(
            f'Rejected: count_from={goal_request.count_from} '
            f'(must be in range [0, 9])'
        )
        return GoalResponse.REJECT
    return GoalResponse.ACCEPT
```

**Test client with cancel:**
```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import GoalStatus
from example_interfaces.action import CountDown


class PreemptTestClient(Node):
    def __init__(self):
        super().__init__('preempt_test_client')
        self._client = ActionClient(self, CountDown, 'count_down')
        self._feedback_count = 0
        self._goal_handle = None

        self._client.wait_for_server()
        goal = CountDown.Goal()
        goal.count_from = 8
        self.get_logger().info('Sending goal count_from=8...')
        future = self._client.send_goal_async(
            goal, feedback_callback=self.feedback_cb
        )
        future.add_done_callback(self.goal_response_cb)

    def goal_response_cb(self, future):
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.get_logger().error('Goal rejected! (count_from may be >= 10)')
            return
        self.get_logger().info('Goal accepted.')
        result_future = self._goal_handle.get_result_async()
        result_future.add_done_callback(self.result_cb)

    def feedback_cb(self, feedback_msg):
        self._feedback_count += 1
        remaining = feedback_msg.feedback.remaining
        self.get_logger().info(
            f'Feedback #{self._feedback_count}: remaining={remaining}'
        )
        if self._feedback_count == 3 and self._goal_handle is not None:
            self.get_logger().info('Received 3 feedbacks → sending cancel...')
            cancel_future = self._goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_response_cb)

    def cancel_response_cb(self, future):
        response = future.result()
        if response.return_code == 0:   # ERROR_NONE
            self.get_logger().info('Cancel request accepted by server.')
        else:
            self.get_logger().warn(f'Cancel rejected, code={response.return_code}')

    def result_cb(self, future):
        result = future.result()
        outcome = result.result.outcome
        if result.status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info(f'[PASS] Goal cancelled. outcome="{outcome}"')
            assert outcome == 'cancelled', f'Expected "cancelled", got "{outcome}"'
        else:
            self.get_logger().error(
                f'[FAIL] Expected CANCELED, got status={result.status}, outcome="{outcome}"'
            )


def main(args=None):
    rclpy.init(args=args)
    node = PreemptTestClient()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

**Test for count_from=10 rejection:**
```python
# Quick rejection test — add to __init__ before the count_from=8 goal:
reject_goal = CountDown.Goal()
reject_goal.count_from = 10
future = self._client.send_goal_async(reject_goal)
rclpy.spin_until_future_complete(self, future)
handle = future.result()
assert not handle.accepted, 'Expected rejection for count_from=10'
self.get_logger().info('[PASS] count_from=10 correctly rejected.')
```

**Timing note:** The feedback callback fires in the executor thread. The cancel is sent
asynchronously, so the server will receive it while still running. With `time.sleep(1.0)`
per step and a cancel after 3 feedbacks, the server will cancel at approximately the 4th
iteration (remaining ≈ 5 for count_from=8).

</details>

- [ ] Done
