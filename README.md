# Gaussian Elimination on GPU

This is an OpenCL implementation of Gaussian Elimination method for solving systems of linear equations (Ax = b).

## Overview

This project compares strategies for GPU-accelerated Gaussian Elimination, from naive implementations to two optimized versions. The focus is on performance analysis, and to understand GPU performance characteristics through benchmarking results. 

The project includes the necessary code and library, and the report itself contains the detailed analysis.

## Problem Description

**Computational Complexity**: O(n³/3) floating-point operations

## Project Structure

```
.
├── gaussianElimination.cpp          # Naive GPU implementation (baseline)
├── optimizedGaussianElimination.cpp # Optimized implementations (2 strategies)
├── GaussianEliminationCI.cpp        # Compute-intensive variant for analysis
├── basicKernels.ocl                 # Basic OpenCL kernels
├── optimizedKernels.ocl             # Optimized OpenCL kernels
└── README.md                        # This file
```

## Implementations

### 1. Sequential CPU (Reference)
- Standard Gaussian Elimination with partial pivoting
- Used for correctness verification
- Performance baseline for speedup calculations
- **File**: All `.cpp` files contain CPU reference implementation

### 2. Naive GPU Implementation
- Basic parallelization: one work item per row
- Sequential back substitution
- No optimization
- **Purpose**: Baseline GPU performance
- **File**: `gaussianElimination.cpp`
- **Kernels**: `forwardElimination`, `backSubstitution` in `basicKernels.ocl`

### 3. Optimized GPU Implementations

#### Strategy A: Blocked with Shared Memory
- Caches pivot row in local/shared memory
- Reduces global memory traffic
- Dynamic shared memory allocation
- Cooperative data loading
- **File**: `optimizedGaussianElimination.cpp`
- **Kernel**: `forwardEliminationBlocked` in `optimizedKernels.ocl`

#### Strategy B: Coalesced Memory Access
- One thread per matrix element
- Optimized memory access patterns
- Maximizes memory bandwidth utilization
- Minimal thread divergence
- **File**: `optimizedGaussianElimination.cpp`
- **Kernel**: `forwardEliminationCoalesced` in `optimizedKernels.ocl`

### 4. Compute-Intensive Variant
- Controllable computation intensity
- Studies memory vs compute-bound behavior
- Analyzes operational intensity impact
- **File**: `GaussianEliminationCI.cpp`
- **Kernel**: `forwardEliminationCoalescedIntensive` in `optimizedKernels.ocl`

## Running the Project

### Prerequisites
- OpenCL SDK (AMD, NVIDIA, or Intel)
- C++ compiler with C++11 support
- JC utility library (for OpenCL helpers)

### Compilation
- CMake would be ideal to establish the project structure
- Use Visual Studio Code to compile and run locally

### Basic Usage
```bash
# Naive implementation
./gaussianElimination -p 0 -d 0 -n 512

# Optimized with blocked strategy
./optimizedGaussian -p 0 -d 0 -n 512 -s blocked

# Optimized with coalesced strategy
./optimizedGaussian -p 0 -d 0 -n 512 -s coalesced

# Compute-intensive variant
./gaussianCI -p 0 -d 0 -n 512 -c 100
```

### Command Line Arguments
- `-p <id>`: OpenCL platform ID (default: 0)
- `-d <id>`: OpenCL device ID (default: 0)
- `-n <size>`: Matrix dimension (default: 512)
- `-s <strategy>`: Optimization strategy - "blocked" or "coalesced" (optimized version only)
- `-c <count>`: Computation loop count (compute-intensive version only)
- `-h`: Show help message

### Example Output
```
Matrix size: 512x512
Running sequential Gaussian elimination...
Sequential time: 125000 us
Sequential solution verified successfully!

Running optimized GPU solver...
Optimized GPU time: 8500 us

=== Results ===
Verification: PASSED
Speedup: 14.7x
CPU Performance: 0.89 GFLOPS
GPU Performance: 13.1 GFLOPS
GPU Memory Bandwidth: 45.2 GB/s
```

## Performance Metrics

All implementations measure and give:

1. **Execution Time**: Microseconds for computation only
2. **Speedup**: Ratio of CPU time to GPU time
3. **GFLOPS**: Computational performance (billion floating-point operations per second)
4. **Memory Bandwidth**: GB/s of memory throughput
5. **Solution Accuracy**: Maximum error compared to CPU reference
6. **Operational Intensity**: FLOPS per byte (compute-intensive variant)
