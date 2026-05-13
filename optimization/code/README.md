# Optimization Curriculum — Code

Companion code for the 10-week optimization curriculum.

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# For C++ (Week 6): install Ceres Solver
# Ubuntu/Debian:
sudo apt install libceres-dev

# Or build from source: http://ceres-solver.org/installation.html
```

## Structure

```
code/
├── requirements.txt          # Python dependencies
├── week01/                   # Phase I: Math Foundations (Week 1)
├── week02/                   # Phase I: Advanced Foundations (Week 2)
├── week03/                   # Phase II: Unconstrained (Week 3)
├── week04/                   # Phase II: Advanced Unconstrained (Week 4)
│   └── miniopt/              # Mini optimizer library (capstone)
├── week05/                   # Phase III: NLS (Week 5)
├── week06_cpp/               # Phase III: Ceres Solver — C++ (Week 6)
│   └── CMakeLists.txt
├── week07/                   # Phase IV: Constrained (Week 7)
├── week08/                   # Phase IV: Convex + MPC (Week 8)
├── week09/                   # Phase V: SLAM (Week 9)
└── week10/                   # Phase V: Applications (Week 10)
    └── final_capstone/       # Final capstone project
```

## Running Code

Each `.py` file is self-contained and runnable:

```bash
cd code/week01
python gram_schmidt.py
```

C++ files (Week 6) use CMake:

```bash
cd code/week06_cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
./hello_ceres
```

## Testing

```bash
# Run all tests
pytest code/ -v

# Run specific week
pytest code/week04/ -v

# With coverage
pytest code/ --cov=code/ --cov-report=term-missing
```
