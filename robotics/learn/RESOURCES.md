# External Resources

Curated external references for the robot learning curriculum.

**Tag axes** (one per entry):
- **Domain:** `embedded` `zephyr` `ros2` `navigation` `llm` `electronics` `linux-rt` `python`
- **Format:** `course` `paper` `repo` `video` `reference-manual` `blog-post` `book`
- **Depth:** `quickref` `tutorial` `comprehensive` `deep-dive`
- **When:** `before-hardware` `when-debugging` `theory-only` `hands-on`
- **Level:** `beginner` `intermediate` `expert-only`

**Linking convention:** study notes reference entries by anchor, e.g. `[RESOURCES.md#zephyr-docs](../RESOURCES.md#zephyr-docs)`

---

## AI / LLM Engineering

### [mlabonne/llm-course](https://github.com/mlabonne/llm-course) {#llm-course}
`llm` `transformers` `course` `jupyter` `comprehensive` `hands-on` `intermediate`
> End-to-end LLM engineering: fundamentals → fine-tuning → quantisation → RAG → inference → deployment. 50+ notebooks. Read Part 1 (The LLM Scientist) before building any agent pipeline; Part 2 (The LLM Engineer) before deploying.

### [microsoft/generative-ai-for-beginners](https://github.com/microsoft/generative-ai-for-beginners) {#genai-beginners}
`llm` `course` `comprehensive` `hands-on` `beginner`
> 21-lesson Microsoft course (110 k ★) covering GenAI fundamentals, prompt engineering, RAG, agents, and fine-tuning. Each lesson has a video + Python/TypeScript code samples. Read before building any LLM-powered tool or working on the AI curriculum track.

---

## Embedded / Zephyr

### [Zephyr Project Documentation](https://docs.zephyrproject.org) {#zephyr-docs}
`embedded` `zephyr` `reference-manual` `quickref` `before-hardware`
> Authoritative. Focus on §Devicetree, §Subsystems (SPI, CAN, ZBus), and §Kernel. Not for reading cover-to-cover — use as a reference when the study notes send you here.

### [ARM Cortex-M4/M7 Technical Reference Manual](https://developer.arm.com/documentation/ddi0489/latest) {#arm-trm}
`embedded` `cortex-m` `reference-manual` `deep-dive` `when-debugging` `expert-only`
> Canonical source for fault registers (CFSR, HFSR, MMFAR, BFAR), CCR trap bits, and pipeline behaviour. Use when `12-hardfault-decode.md` doesn't go deep enough.

### [Understanding the Cortex-M fault exceptions — Joseph Yiu](https://www.embedded.com/programming-embedded-systems-in-c-cpp/) {#cortex-m-faults}
`embedded` `cortex-m` `blog-post` `tutorial` `when-debugging` `intermediate`
> Practical walkthrough of HardFault, BusFault, MemManage. Good supplement to the deep-dive notes before real hardware debugging.

### [m3y54m/Embedded-Engineering-Roadmap](https://github.com/m3y54m/Embedded-Engineering-Roadmap) {#embedded-roadmap}
`embedded` `roadmap` `comprehensive` `before-hardware` `all-levels`
> Visual roadmap + curated resource list for embedded engineers covering hardware, firmware, RTOS, and protocols. Use as an orientation map to see where each Zephyr/ARM topic fits in the wider embedded world before starting lab work.

### [gurugio/lowlevelprogramming-university](https://github.com/gurugio/lowlevelprogramming-university) {#low-level-university}
`embedded` `linux-rt` `course` `comprehensive` `theory-only` `intermediate`
> Self-study curriculum covering C, assembly, Linux kernel internals, and OS design. Builds the low-level mental model needed to understand Zephyr scheduler behaviour and HAL; useful background before the kernel-internals labs.

### [RT-Thread/rt-thread](https://github.com/RT-Thread/rt-thread) {#rt-thread}
`embedded` `repo` `comprehensive` `before-hardware` `intermediate`
> Mature IoT RTOS (1.2KB RAM minimum) supporting ARM Cortex-M/A, RISC-V and 200+ boards; useful for studying modular RTOS architecture and comparing scheduling/IPC patterns against Zephyr. 11.9k ★

---

## ROS 2 / Navigation

### [ROS 2 Documentation (Jazzy)](https://docs.ros.org/en/jazzy/) {#ros2-docs}
`ros2` `reference-manual` `quickref` `before-hardware`
> Official ROS 2 API and concept docs. Start with §Concepts → §Tutorials → §How-to guides. Most useful once you have a working workspace.

### [nav2 documentation](https://navigation.ros.org/) {#nav2-docs}
`ros2` `navigation` `reference-manual` `comprehensive` `before-hardware`
> Nav2 stack: costmaps, planners, controllers, BT navigator. Read §Concepts before touching any AMR navigation code.

### [AtsushiSakai/PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) {#python-robotics}
`robotics` `navigation` `python` `repo` `comprehensive` `hands-on` `intermediate`
> Python code + textbook for robotics algorithms: EKF, Particle Filter, FastSLAM, A*, RRT, MPC, and more (29 k ★). Run the simulations to build intuition for the same algorithms used in the robot navigation stack before reading estimator source code.

### [MichaelGrupp/evo](https://github.com/MichaelGrupp/evo) {#evo}
`ros2` `navigation` `python` `repo` `deep-dive` `when-debugging` `intermediate`
> CLI + Python library for computing APE/RPE trajectory error metrics from odometry or SLAM bag files. Use when comparing estimator branches against rosbag ground truth or validating sensor-fusion changes.

### [henki-robotics/robotics_essentials_ros2](https://github.com/henki-robotics/robotics_essentials_ros2) {#robotics-essentials-ros2}
`ros2` `course` `comprehensive` `hands-on` `beginner`
> Hands-on ROS2 + Gazebo course covering SLAM, navigation, and odometry with lab exercises. Good companion or prerequisite for the robot navigation-estimator track.

### [fkromer/awesome-ros2](https://github.com/fkromer/awesome-ros2) {#awesome-ros2}
`ros2` `repo` `quickref` `before-hardware` `all-levels`
> Curated index of ROS2 packages, tools, tutorials, and conference talks. First stop when searching for existing packages or examples before writing custom nodes.

### [Kiloreux/awesome-robotics](https://github.com/Kiloreux/awesome-robotics) {#awesome-robotics}
`robotics` `repo` `quickref` `before-hardware` `all-levels`
> Aggregated list of robotics courses, textbooks, simulators, competitions, and software libraries (6.4 k ★). Use to map the wider robotics landscape or find learning resources beyond the robot curriculum.

### [mithi/robotics-coursework](https://github.com/mithi/robotics-coursework) {#robotics-coursework}
`robotics` `course` `comprehensive` `before-hardware` `intermediate`
> Curated list of robotics course series (Coursera, EdX, university lecture playlists) covering kinematics, dynamics, SLAM, motion planning, and computer vision. Use to find a structured study path before tackling AMR navigation theory. 4.5k ★

### [ai-winter/ros_motion_planning](https://github.com/ai-winter/ros_motion_planning) {#ros-motion-planning}
`navigation` `cpp` `repo` `hands-on` `intermediate`
> ROS implementations of 20+ path-planning algorithms (A*, JPS, D*, RRT, RRT*, DWA, MPC, TEB, Pure Pursuit) with animated demos. Use to build hands-on intuition for the planners and local controllers before reading AMR navigation source. (ROS1/Noetic) 3.5k ★

### [cartographer-project/cartographer](https://github.com/cartographer-project/cartographer) {#cartographer}
`ros2` `navigation` `repo` `deep-dive` `theory-only` `expert-only`
> Google's 2D/3D SLAM via pose-graph optimization and scan matching — the theoretical reference for understanding how slam_toolbox and Nav2's SLAM integrations work. Note: archived, no active dev. 7.8k ★

---

## Electronics / Signals

### [The Art of Electronics — Horowitz & Hill](https://artofelectronics.net/) {#aoe}
`electronics` `book` `comprehensive` `theory-only` `intermediate`
> Gold standard reference. Most useful: Chapter 1 (foundations), Chapter 12 (logic families), Chapter 14 (ADC/DAC). Don't read linearly — treat as a reference.

---

## Linux / Real-Time

### [PREEMPT_RT wiki](https://wiki.linuxfoundation.org/reallinux/start) {#preempt-rt}
`linux-rt` `reference-manual` `deep-dive` `when-debugging`
> RT patch internals, latency tuning, cyclictest interpretation. Use when `07-nanopb-rt-ekf.md` cyclictest results look wrong.
