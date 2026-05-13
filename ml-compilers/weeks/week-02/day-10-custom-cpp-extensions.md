# Day 10: Custom C++ Extensions & pybind11

> **Phase I · Week 2 · Day 10 of 70 · 2.5 hours**

*"If the operation you need doesn't exist in ATen, you don't file a feature request — you write a kernel."*

| Previous | Next | Week | Phase | Curriculum |
|----------|------|------|-------|------------|
| [Day 9: Memory Management](day-09-memory-management-pytorch.md) | [Day 11: torch.profiler & Trace Analysis](day-11-torch-profiler-trace.md) | [Week 2: PyTorch Internals](README.md) | Phase I: Foundations | [Curriculum Home](../../STUDY-PLAN.md) |

---

## Why This Matters

Fusing operations into a single CUDA kernel eliminates memory round-trips to global
memory. A naive `bias + GELU(x)` requires two kernel launches and one intermediate
tensor write/read. A fused kernel does both in registers — saving ~2× bandwidth and
1 kernel launch overhead. Custom C++ extensions are how production systems (FlashAttention,
xFormers, Triton-compiled ops) deliver their speedups. Knowing this pathway is essential
for any ML compiler engineer.

---

## 1. Extension Architecture

PyTorch offers two paths for custom C++/CUDA code:

```
┌─────────────────────────────────────────────────┐
│                 Python User Code                │
│         my_op = my_extension.fused_bias_gelu    │
├─────────────┬───────────────────────────────────┤
│ Ahead-of-   │  JIT Compilation                  │
│ Time Build  │  torch.utils.cpp_extension.load() │
│ setup.py /  │  ← compiles on first call,        │
│ setuptools  │    caches in ~/.cache/torch_ext/   │
├─────────────┴───────────────────────────────────┤
│            torch.utils.cpp_extension             │
│  CppExtension / CUDAExtension                   │
├─────────────────────────────────────────────────┤
│  pybind11 bindings (auto-included by PyTorch)    │
├─────────────────────────────────────────────────┤
│  Your C++ / CUDA source files                    │
│  (.cpp for CPU, .cu for CUDA)                    │
└─────────────────────────────────────────────────┘
```

---

## 2. JIT Compilation with `load()`

The fastest way to iterate on a custom op:

```python
from torch.utils.cpp_extension import load

# This compiles on first call, caches for subsequent runs
my_extension = load(
    name='my_extension',
    sources=['my_op.cpp', 'my_op_cuda.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=True,
)

# Use it like any Python function
y = my_extension.fused_bias_gelu(x, bias)
```

Behind the scenes, `load()`:
1. Invokes `nvcc` for `.cu` files and `g++` for `.cpp` files
2. Links against `libtorch` and `libc10`
3. Creates a shared object (`.so`) via pybind11
4. Caches in `~/.cache/torch_extensions/`

---

## 3. Full Example: Fused Bias + GELU

### 3.1 The Math

GELU (Gaussian Error Linear Unit):

$$\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5 \, x \left(1 + \tanh\!\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715\, x^3\right)\right]\right)$$

We want to compute $\text{GELU}(x + \text{bias})$ in a single kernel.

### 3.2 CUDA Kernel (`fused_bias_gelu_cuda.cu`)

```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function: GELU approximation (tanh variant)
__device__ __forceinline__ float gelu_fwd(float x) {
    const float kSqrt2OverPi = 0.7978845608f;  // sqrt(2/pi)
    const float kCoeff = 0.044715f;
    float cube = x * x * x;
    float inner = kSqrt2OverPi * (x + kCoeff * cube);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Device function: GELU gradient
__device__ __forceinline__ float gelu_bwd(float x) {
    const float kSqrt2OverPi = 0.7978845608f;
    const float kCoeff = 0.044715f;
    float cube = x * x * x;
    float inner = kSqrt2OverPi * (x + kCoeff * cube);
    float tanh_inner = tanhf(inner);
    float sech2 = 1.0f - tanh_inner * tanh_inner;
    float inner_deriv = kSqrt2OverPi * (1.0f + 3.0f * kCoeff * x * x);
    return 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * inner_deriv;
}

// Forward kernel: y = GELU(x + bias)
__global__ void fused_bias_gelu_fwd_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        float val = input[idx] + bias[col];
        output[idx] = gelu_fwd(val);
    }
}

// Backward kernel
__global__ void fused_bias_gelu_bwd_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ grad_input,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        float val = input[idx] + bias[col];
        grad_input[idx] = grad_out[idx] * gelu_bwd(val);
    }
}

// C++ wrapper functions (called from Python)
torch::Tensor fused_bias_gelu_fwd_cuda(
    torch::Tensor input,
    torch::Tensor bias
) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be on CUDA");
    TORCH_CHECK(input.size(-1) == bias.size(0), "bias size mismatch");

    auto output = torch::empty_like(input);
    int rows = input.numel() / input.size(-1);
    int cols = input.size(-1);
    int total = rows * cols;

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    fused_bias_gelu_fwd_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, cols
    );

    return output;
}

torch::Tensor fused_bias_gelu_bwd_cuda(
    torch::Tensor grad_out,
    torch::Tensor input,
    torch::Tensor bias
) {
    auto grad_input = torch::empty_like(input);
    int rows = input.numel() / input.size(-1);
    int cols = input.size(-1);
    int total = rows * cols;

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    fused_bias_gelu_bwd_kernel<<<blocks, threads>>>(
        grad_out.data_ptr<float>(),
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        rows, cols
    );

    return grad_input;
}
```

### 3.3 C++ Bindings (`fused_bias_gelu.cpp`)

```cpp
#include <torch/extension.h>

// Forward declarations of CUDA functions
torch::Tensor fused_bias_gelu_fwd_cuda(torch::Tensor input,
                                        torch::Tensor bias);
torch::Tensor fused_bias_gelu_bwd_cuda(torch::Tensor grad_out,
                                        torch::Tensor input,
                                        torch::Tensor bias);

// Input validation + dispatch
torch::Tensor fused_bias_gelu_fwd(torch::Tensor input,
                                   torch::Tensor bias) {
    TORCH_CHECK(input.device().is_cuda(),
                "fused_bias_gelu only supports CUDA tensors");
    return fused_bias_gelu_fwd_cuda(input, bias);
}

torch::Tensor fused_bias_gelu_bwd(torch::Tensor grad_out,
                                   torch::Tensor input,
                                   torch::Tensor bias) {
    return fused_bias_gelu_bwd_cuda(grad_out, input, bias);
}

// pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &fused_bias_gelu_fwd,
          "Fused bias + GELU forward (CUDA)");
    m.def("backward", &fused_bias_gelu_bwd,
          "Fused bias + GELU backward (CUDA)");
}
```

---

## 4. Autograd Integration

To make the custom op work with `loss.backward()`, wrap it in a
`torch.autograd.Function`:

```python
import torch
from torch.utils.cpp_extension import load

# JIT compile
fused_op = load(
    name='fused_bias_gelu',
    sources=['fused_bias_gelu.cpp', 'fused_bias_gelu_cuda.cu'],
    extra_cuda_cflags=['-O3'],
)

class FusedBiasGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias):
        # Save for backward
        ctx.save_for_backward(input, bias)
        return fused_op.forward(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        grad_input = fused_op.backward(grad_output.contiguous(),
                                        input, bias)
        # grad_bias = sum of grad_input along batch dims
        grad_bias = grad_input.sum(dim=tuple(range(grad_input.dim()-1)))
        return grad_input, grad_bias

# Convenience wrapper
def fused_bias_gelu(input, bias):
    return FusedBiasGELU.apply(input, bias)
```

### 4.1 Gradient Check

Always validate your backward implementation:

```python
from torch.autograd import gradcheck

input = torch.randn(4, 8, dtype=torch.float64,
                     device='cuda', requires_grad=True)
bias = torch.randn(8, dtype=torch.float64,
                    device='cuda', requires_grad=True)

# gradcheck uses finite differences to verify analytical gradients
assert gradcheck(FusedBiasGELU.apply, (input, bias), eps=1e-6)
print("Gradient check PASSED")
```

---

## 5. Registering with the Dispatcher

For proper integration with `torch.compile` and other PyTorch subsystems,
register your op with the dispatcher using `torch.library`:

```python
import torch
from torch import Tensor

# Define the operator schema
torch.library.define(
    "myops::fused_bias_gelu",
    "(Tensor input, Tensor bias) -> Tensor"
)

# Register CUDA implementation
@torch.library.impl("myops::fused_bias_gelu", "cuda")
def fused_bias_gelu_cuda(input: Tensor, bias: Tensor) -> Tensor:
    return FusedBiasGELU.apply(input, bias)

# Register a "fake" implementation for torch.compile tracing
@torch.library.impl_abstract("myops::fused_bias_gelu")
def fused_bias_gelu_abstract(input: Tensor, bias: Tensor) -> Tensor:
    # Return a tensor with the correct shape/dtype but no data
    return torch.empty_like(input)

# Now usable as:
y = torch.ops.myops.fused_bias_gelu(x, bias)
```

---

## 6. Ahead-of-Time Build with setuptools

For distribution, use `setup.py`:

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fused_bias_gelu',
    ext_modules=[
        CUDAExtension(
            name='fused_bias_gelu',
            sources=[
                'fused_bias_gelu.cpp',
                'fused_bias_gelu_cuda.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math',
                         '-gencode=arch=compute_80,code=sm_80',  # A100
                         '-gencode=arch=compute_89,code=sm_89'], # RTX 4090
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
```

Build and install:

```bash
pip install -e .
# OR for development iteration:
python setup.py build_ext --inplace
```

---

## Hands-On Exercises

### Exercise 1: Benchmark Fused vs Unfused (30 min)

```python
import torch
import torch.nn.functional as F
import time

def unfused_bias_gelu(x, bias):
    return F.gelu(x + bias, approximate='tanh')

# Compare against unfused PyTorch implementation
x = torch.randn(512, 4096, device='cuda')
bias = torch.randn(4096, device='cuda')

# Warmup
for _ in range(10):
    unfused_bias_gelu(x, bias)
    # fused_bias_gelu(x, bias)  # your custom op

torch.cuda.synchronize()

# Benchmark unfused
t0 = time.time()
for _ in range(1000):
    y_ref = unfused_bias_gelu(x, bias)
torch.cuda.synchronize()
t_unfused = time.time() - t0

print(f"Unfused: {t_unfused*1000:.1f} ms (1000 iters)")
# Compare with your fused version — expect 1.3-2x speedup

# TASK: Also measure peak memory for both versions
```

### Exercise 2: Add Half-Precision Support (30 min)

```cpp
// Extend the CUDA kernel to support fp16 using __half types.
// Hints:
//   - Use AT_DISPATCH_FLOATING_TYPES_AND_HALF macro
//   - For __half arithmetic, use __hadd(), __hmul()
//   - Or cast to float for computation, cast back to __half for storage

// Template the kernel:
template <typename scalar_t>
__global__ void fused_bias_gelu_fwd_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        float val = static_cast<float>(input[idx])
                  + static_cast<float>(bias[col]);
        output[idx] = static_cast<scalar_t>(gelu_fwd(val));
    }
}

// Dispatch:
// AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fused_gelu", [&] {
//     fused_bias_gelu_fwd_kernel<scalar_t><<<blocks, threads>>>(...);
// });
```

### Exercise 3: Vectorized Memory Access (20 min)

```cpp
// The basic kernel loads one float per thread. Optimize by loading
// 4 floats at once using float4:

__global__ void fused_bias_gelu_fwd_vec4(
    const float4* __restrict__ input,
    const float* __restrict__ bias,
    float4* __restrict__ output,
    int rows, int cols_div4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols_div4) {
        int col4 = (idx % cols_div4) * 4;
        float4 in = input[idx];
        float4 out;
        out.x = gelu_fwd(in.x + bias[col4 + 0]);
        out.y = gelu_fwd(in.y + bias[col4 + 1]);
        out.z = gelu_fwd(in.z + bias[col4 + 2]);
        out.w = gelu_fwd(in.w + bias[col4 + 3]);
        output[idx] = out;
    }
}

// TASK: Integrate this into the extension and benchmark.
// Expected speedup: 1.5-2x over scalar version due to memory coalescing.
```

---

## Key Takeaways

1. **`torch.utils.cpp_extension.load()`** provides zero-setup JIT compilation for
   rapid C++/CUDA kernel development
2. **Kernel fusion** eliminates intermediate memory writes — the primary speedup
   mechanism for bandwidth-bound operations
3. **`torch.autograd.Function`** bridges custom CUDA kernels with PyTorch's autograd
   engine; always verify with `gradcheck`
4. **`torch.library`** registers ops with the dispatcher for compatibility with
   `torch.compile`, `vmap`, and other transforms
5. **`AT_DISPATCH_FLOATING_TYPES_AND_HALF`** handles dtype dispatch across fp32/fp64/fp16
   without duplicating kernel code

---

## Further Reading

- [PyTorch Custom C++ and CUDA Extensions tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [torch.library documentation](https://pytorch.org/docs/stable/library.html)
- [pybind11 documentation](https://pybind11.readthedocs.io/)
- [CUDA C++ Programming Guide: Memory Access Patterns](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)

---

## Tomorrow's Preview

**Day 11: torch.profiler & Trace Analysis** — We'll learn to profile PyTorch programs
with `torch.profiler`, visualize execution traces in Chrome/Perfetto, identify GPU
idle time, and measure CPU-GPU overlap to find the real bottleneck.
