# Learning_from_building_systems

Notes and self-study curricula from building real systems: robotics, embedded
software, optimization, and machine learning.

**Read it here:** https://vishalkumarr2.github.io/Learning_from_building_systems/

## Where to start

| Curriculum | What it covers | Size |
|---|---|---|
| [Robotics](https://vishalkumarr2.github.io/Learning_from_building_systems/robotics/) | Embedded C++, Zephyr RTOS, electronics, control, navigation estimation, ROS 2, Python, logic | 8 tracks |
| [LLM to VLA](https://vishalkumarr2.github.io/Learning_from_building_systems/llm-to-vla/) | Deep learning foundations to Vision-Language-Action models | 112 daily lessons |
| [ML Compilers](https://vishalkumarr2.github.io/Learning_from_building_systems/ml-compilers/) | CUDA, TVM, Triton, torch.compile, distributed training | 70 daily lessons |
| [Optimization](https://vishalkumarr2.github.io/Learning_from_building_systems/optimization/) | Linear algebra to Lie groups and game theory | 70 daily lessons (weeks 5–10 not written yet) |

Every lesson page has a **Mark as done** button. Progress and the "continue where
you left off" link are stored in your browser only.

## Editing the content

Lessons are written in Markdown. The `.html` file next to each `.md` file is
generated, so edit the Markdown and rebuild:

```bash
pip install markdown                 # once
python3 tools/build_site.py          # render all curriculum markdown to HTML
python3 tools/check_site.py          # broken links, orphan pages, unrendered markdown
python3 -m unittest discover -s testing -p 'test_*.py'
```

`ml-compilers/` and the top-level area pages are hand-written HTML; edit those directly.
`check_site.py` also scans for internal names listed in a local, untracked
`.internal-terms` file (one regex per line) so they never reach the public site.

## Principles I keep returning to

-  Setup way to debug before you build something
  -  like loggers in software or electronics( this will improve your development time exponentially)
  -  How do you debug mechanical
  -  Setup testing bench
    -  Testing in software
    -  Testing in electronics
    -  Testing in Mechanical
    -  System Testing
-  System should not be able to kill itself, other entity is required to kill it and report its status.
-  Have mocking system, which help you test your flows.
-  Think about what state your system should be in when it start.
-  How your system life will be at end of life( full or partial).
-  All system if are not ideal, then you have to set threshold for operation where you are going to assume them ideal. like 2 rpm for 0 speed, 4.5000000000001 for 4.5
-  Enable debouncing if you think that is required
-  Have initialisation/starting procudures to have a defined procedures

-  This one is for software:
  -  Verify the input first
  -  Type of the data
  -  Constaints satisfaction
  -  value of data
  -  sources of the data

For Simulation:
-   Reproducability
-   Logging
-   Able to Fast Forwared time for quick simulations.
-   Have models
