# Day 41: LLM for Robotics

> **Phase III — LLMs: Training & Alignment | Week 6 | 2.5 hours**
> *"Language is the interface between human intent and robot action." — Google Brain*

## Navigation
- **Previous:** [Day 40: RAG & Tool Use](day-40-rag-tool-use.md)
- **Next:** [Day 42: Phase III Capstone Day 1](day-42-phase3-capstone-1.md)
- **Week:** [Week 6 Overview](README.md)
- **Phase:** [Phase III: LLM Engineering](../../study-notes/07-llm-engineering.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 41.1 The Language-Action Gap

LLMs understand language. Robots execute actions. The challenge: bridge the gap.

```
Human intent: "Pick up the red cup on the table"
    │
    ▼ (language understanding)
Semantic understanding: object=red_cup, location=table, action=grasp
    │
    ▼ (grounding)
Physical grounding: position=(0.3, 0.5, 0.1), grasp_type=top_pinch
    │
    ▼ (planning)
Action sequence: approach → pre-grasp → close_gripper → lift → verify
    │
    ▼ (control)
Motor commands: joint velocities, torques, trajectories
```

Three paradigms have emerged:

### 41.2 SayCan: Language as Task Selector

**SayCan (Google, 2022):** LLM scores *what makes sense* + robot scores *what's possible*.

$$
\text{action}^* = \arg\max_a \underbrace{P_{\text{LLM}}(a \mid \text{instruction})}_{\text{language score}} \cdot \underbrace{P_{\text{affordance}}(a \mid \text{state})}_{\text{feasibility score}}
$$

```
Instruction: "I spilled my drink, can you help?"

LLM scoring (what makes sense):
  "Pick up sponge"     → 0.8
  "Go to kitchen"      → 0.6
  "Pick up apple"      → 0.1
  "Dance"              → 0.0

Affordance scoring (what's possible):
  "Pick up sponge"     → 0.9  (sponge is reachable)
  "Go to kitchen"      → 0.7  (path exists)
  "Pick up apple"      → 0.3  (apple not nearby)
  "Dance"              → 0.1  (no dance skill)

Combined:
  "Pick up sponge"     → 0.8 × 0.9 = 0.72  ← selected!
  "Go to kitchen"      → 0.6 × 0.7 = 0.42
```

### 41.3 Code as Policies: LLM Writes Robot Code

**Code as Policies (Google, 2023):** Instead of selecting from predefined actions, the LLM *generates executable code*.

```python
# User: "Stack the blocks in a pyramid"
# LLM generates:

def stack_pyramid():
    blocks = detect_objects("block")
    sorted_blocks = sorted(blocks, key=lambda b: b.size, reverse=True)

    # Bottom row
    base_y = 0.3
    for i, block in enumerate(sorted_blocks[:3]):
        pick(block)
        place(x=0.2 + i * 0.1, y=base_y, z=0.05)

    # Middle row
    for i, block in enumerate(sorted_blocks[3:5]):
        pick(block)
        place(x=0.25 + i * 0.1, y=base_y, z=0.1)

    # Top
    pick(sorted_blocks[5])
    place(x=0.3, y=base_y, z=0.15)
```

**Advantages over SayCan:**
- Not limited to predefined action primitives
- Can express complex logic (loops, conditionals)
- More generalizable — new tasks don't need new training

### 41.4 LLM for Log Analysis & Diagnostics

Directly applicable to OKS robot fleet management:

```
LLM-Powered Robot Diagnostics:

Input: Robot error logs + telemetry
  "[14:23:01] WARN: sensorbar SPI timeout (3 consecutive)
   [14:23:05] ERROR: NAV_ESTIMATED_STATE_NOT_FINITE
   [14:23:05] INFO: Guardian triggering safety stop
   [14:23:06] ERROR: Robot stopped — manual recovery needed"

LLM Analysis:
  "Root cause: SPI communication failure between sensorbar and
   main controller. The 3 consecutive timeouts caused the navigation
   estimator to receive stale data, resulting in NaN propagation.
   
   Recommended actions:
   1. Check SPI wiring and connectors
   2. Verify sensorbar firmware version (expect v1.2.4+)
   3. Inspect for debris on sensor surface
   4. Review IMU calibration date"
```

### 41.5 Task Planning for Fleets

LLMs can decompose high-level warehouse goals into multi-robot task plans:

```
Goal: "Process all incoming shipments in Dock B"

LLM task decomposition:
  1. Robot-1: Navigate to Dock B, scan incoming pallets
  2. Robot-2: Transport pallet #1 to Sorting Zone A
  3. Robot-3: Transport pallet #2 to Sorting Zone B
  4. Robot-1: Verify Dock B is clear, report to WMS
  5. Robot-2: Return to staging area
  6. Robot-3: Return to staging area

Constraints extracted:
  - Robot-1 must complete step 1 before steps 2-3 can begin
  - Steps 2 and 3 can execute in parallel
  - Step 4 depends on steps 2-3 completion
```

### 41.6 Limitations & Safety

```
⚠️ LLM hallucination → dangerous in robotics
  "Place the hot coffee on the baby"
  → LLM might generate valid code for this harmful action!

Safety layers required:
  1. Action validation against physics constraints
  2. Safety zone checking before execution
  3. Human confirmation for irreversible actions
  4. Affordance checking (can the robot actually do this?)
  5. Semantic safety filter (is this action safe?)

Reality: LLMs are planners, not controllers.
  ✅ Use LLMs for: task decomposition, error diagnosis, NL interface
  ❌ Don't use for: real-time control, safety-critical decisions
```

---

## Implementation (60 min)

### Build an LLM Robot Command Translator

```python
"""
Day 41 Implementation: LLM-powered robot command translator.
Translates natural language to structured robot commands.
"""
import json
import re
from dataclasses import dataclass, field
from enum import Enum


class CommandType(Enum):
    NAVIGATE = "navigate"
    PICK = "pick"
    PLACE = "place"
    STOP = "stop"
    STATUS = "status"
    SCAN = "scan"
    CHARGE = "charge"


@dataclass
class RobotCommand:
    command_type: CommandType
    parameters: dict = field(default_factory=dict)
    safety_check: bool = True
    confidence: float = 0.0

    def to_ros_message(self) -> dict:
        """Convert to ROS-style message format."""
        if self.command_type == CommandType.NAVIGATE:
            return {
                "topic": "/move_base/goal",
                "msg_type": "MoveBaseGoal",
                "data": {
                    "target_pose": {
                        "position": self.parameters.get("position", {}),
                        "orientation": self.parameters.get("orientation", {}),
                    }
                },
            }
        elif self.command_type == CommandType.STOP:
            return {
                "topic": "/cmd_vel",
                "msg_type": "Twist",
                "data": {"linear": {"x": 0}, "angular": {"z": 0}},
            }
        elif self.command_type == CommandType.PICK:
            return {
                "topic": "/pick_action/goal",
                "msg_type": "PickGoal",
                "data": {
                    "object_id": self.parameters.get("object_id", ""),
                    "grasp_type": self.parameters.get("grasp_type", "auto"),
                },
            }
        return {"topic": "unknown", "data": self.parameters}


class CommandParser:
    """Rule-based fallback parser for robot commands."""

    PATTERNS = {
        CommandType.NAVIGATE: [
            r"(?:go|move|navigate|drive)\s+to\s+(.+)",
            r"(?:head|travel)\s+(?:to|towards)\s+(.+)",
        ],
        CommandType.PICK: [
            r"(?:pick|grab|grasp|get)\s+(?:up\s+)?(.+)",
            r"(?:take|collect)\s+(.+)",
        ],
        CommandType.PLACE: [
            r"(?:place|put|drop|set)\s+(.+?)(?:\s+(?:on|at|in)\s+(.+))?$",
        ],
        CommandType.STOP: [
            r"(?:stop|halt|freeze|emergency)",
            r"e-?stop",
        ],
        CommandType.STATUS: [
            r"(?:status|battery|health|diagnostics)",
            r"(?:what|how)\s+(?:is|are)\s+(?:your|the)\s+(.+)",
        ],
        CommandType.CHARGE: [
            r"(?:charge|recharge|go\s+charge)",
            r"(?:battery\s+low|need\s+charging)",
        ],
    }

    def parse(self, text: str) -> RobotCommand | None:
        text_lower = text.lower().strip()

        for cmd_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    params = {}
                    if cmd_type == CommandType.NAVIGATE:
                        params["destination"] = match.group(1).strip()
                    elif cmd_type == CommandType.PICK:
                        params["object"] = match.group(1).strip()
                    elif cmd_type == CommandType.PLACE:
                        params["object"] = match.group(1).strip()
                        if match.lastindex and match.lastindex >= 2:
                            params["location"] = match.group(2).strip()

                    return RobotCommand(
                        command_type=cmd_type,
                        parameters=params,
                        confidence=0.8,
                    )
        return None


class LLMCommandTranslator:
    """Translates NL instructions to robot commands using LLM prompting."""

    SYSTEM_PROMPT = """You are a warehouse robot command interpreter.
Convert natural language instructions into structured JSON commands.

Available command types:
- navigate: Move to a location {destination, speed}
- pick: Pick up an object {object_id, grasp_type}
- place: Place an object {object_id, location}
- stop: Emergency stop {}
- status: Report status {subsystem}
- scan: Scan area {area, scan_type}
- charge: Go to charging station {}

Respond ONLY with valid JSON:
{"command": "type", "params": {...}, "safety_notes": "..."}"""

    def create_prompt(self, instruction: str) -> str:
        return (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Instruction: {instruction}\n"
            f"Command JSON:"
        )

    def parse_llm_output(self, output: str) -> RobotCommand | None:
        """Parse LLM JSON output into a RobotCommand."""
        try:
            # Extract JSON from output
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if not json_match:
                return None
            data = json.loads(json_match.group())

            cmd_type = CommandType(data.get("command", "status"))
            return RobotCommand(
                command_type=cmd_type,
                parameters=data.get("params", {}),
                confidence=0.9,
            )
        except (json.JSONDecodeError, ValueError):
            return None


class SafetyValidator:
    """Validate robot commands before execution."""

    RESTRICTED_ZONES = {"zone_x", "maintenance_area", "human_zone"}
    MAX_SPEED = 1.5  # m/s

    def validate(self, command: RobotCommand) -> tuple[bool, str]:
        """Return (is_safe, reason)."""
        # Check restricted zones
        dest = command.parameters.get("destination", "")
        if dest.lower().replace(" ", "_") in self.RESTRICTED_ZONES:
            return False, f"Destination '{dest}' is a restricted zone"

        # Check speed limits
        speed = command.parameters.get("speed", 0.5)
        if speed > self.MAX_SPEED:
            return False, f"Speed {speed} exceeds maximum {self.MAX_SPEED} m/s"

        # Emergency stop always allowed
        if command.command_type == CommandType.STOP:
            return True, "Emergency stop — always allowed"

        return True, "Command validated"


# --- SayCan-style scoring ---
def saycan_score(
    language_scores: dict[str, float],
    affordance_scores: dict[str, float],
) -> list[tuple[str, float]]:
    """Combine language and affordance scores (SayCan style)."""
    combined = {}
    for action in language_scores:
        lang = language_scores.get(action, 0.0)
        aff = affordance_scores.get(action, 0.0)
        combined[action] = lang * aff

    return sorted(combined.items(), key=lambda x: -x[1])


# --- Demo ---
if __name__ == "__main__":
    parser = CommandParser()
    translator = LLMCommandTranslator()
    validator = SafetyValidator()

    test_commands = [
        "Go to the charging station",
        "Pick up the package from shelf B3",
        "Stop immediately!",
        "Navigate to zone_x",  # restricted zone
        "What's your battery level?",
        "Place the box on conveyor belt 2",
    ]

    print("=" * 60)
    print("Robot Command Translation")
    print("=" * 60)

    for cmd_text in test_commands:
        print(f"\nInput: \"{cmd_text}\"")

        # Rule-based parsing
        cmd = parser.parse(cmd_text)
        if cmd:
            is_safe, reason = validator.validate(cmd)
            print(f"  Parsed: {cmd.command_type.value} | {cmd.parameters}")
            print(f"  Safety: {'✅' if is_safe else '❌'} {reason}")
            if is_safe:
                ros_msg = cmd.to_ros_message()
                print(f"  ROS topic: {ros_msg['topic']}")
        else:
            print("  ⚠️ Could not parse — would fall back to LLM")
            prompt = translator.create_prompt(cmd_text)
            print(f"  LLM prompt ready ({len(prompt)} chars)")

    # SayCan demo
    print("\n" + "=" * 60)
    print("SayCan Scoring Demo")
    print("=" * 60)

    lang_scores = {
        "Pick up sponge": 0.8,
        "Navigate to kitchen": 0.6,
        "Pick up apple": 0.1,
        "Dance": 0.0,
    }
    aff_scores = {
        "Pick up sponge": 0.9,
        "Navigate to kitchen": 0.7,
        "Pick up apple": 0.3,
        "Dance": 0.1,
    }

    ranked = saycan_score(lang_scores, aff_scores)
    print("Instruction: 'I spilled my drink, can you help?'")
    for action, score in ranked:
        print(f"  {score:.2f} | {action}")
```

---

## Exercise (45 min)

### E41.1 — Multi-Step Task Planner (25 min)
Build a simple task planner that decomposes complex instructions:
1. Input: "Sort all packages by destination and deliver to zones A, B, C"
2. Generate a dependency graph of sub-tasks
3. Identify which steps can run in parallel across multiple robots
4. Output a Gantt-chart-style execution plan

### E41.2 — Error Diagnosis Agent (20 min)
Create an error diagnosis pipeline:
1. Input: a sequence of robot log entries (5-10 lines)
2. Use rule-based pattern matching to identify known error codes
3. Generate a structured diagnosis: root cause, confidence, recommended actions
4. Compare: how does adding context (past incidents) improve diagnosis quality?

---

## Key Takeaways

1. **SayCan** combines LLM reasoning with robot affordances — say what's useful, can what's possible
2. **Code as Policies** lets LLMs generate executable robot code — more flexible than action selection
3. **LLMs are planners, not controllers** — use for high-level reasoning, not real-time motor commands
4. **Safety validation** is mandatory between LLM output and robot execution
5. **Log analysis + RAG** is the most immediately useful LLM application for existing robot fleets

---

## Connection to the Thread

This day directly connects our LLM knowledge to the robotics domain. SayCan and Code as Policies are precursors to VLAs (Vision-Language-Action models) — Phase VII of our curriculum. The difference: VLAs skip the code/action-selection intermediate step and directly output motor commands from vision + language. But understanding the language→action pipeline here is essential for understanding what VLAs replace.

---

## Further Reading

- [SayCan: Do As I Can, Not As I Say](https://arxiv.org/abs/2204.01691) — Google, 2022
- [Code as Policies: Language Model Programs for Embodied Control](https://arxiv.org/abs/2209.07753) — Liang et al., 2023
- [ProgPrompt: Generating Situated Robot Task Plans](https://arxiv.org/abs/2209.11302) — Singh et al., 2023
- [Inner Monologue: Embodied Reasoning through Planning with LLMs](https://arxiv.org/abs/2207.05608) — Huang et al., 2022
- [Voyager: An Open-Ended Embodied Agent with LLMs](https://arxiv.org/abs/2305.16291) — Wang et al., 2023
