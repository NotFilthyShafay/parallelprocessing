# Parallel Computing Project

This project implements and compares various parallel computing approaches for two different computational problems:
1. Matrix Multiplication
2. Laplace Equation Solver

Each implementation provides single-threaded, OpenMP, CUDA, and (optionally) OpenCL versions for performance comparison.

## Project Structure

```
/home/shafay/Parallel/
├── matmul/                         # Matrix multiplication implementations
│   ├── cpu/
│   │   ├── matrix.h                # Matrix class definition
│   │   ├── matrix_multiplication_omp.cpp
│   │   └── matrix_multiplication_single_threaded.cpp
│   └── cuda/
│       └── matmulcuda.cu
├── laplace_solver/                 # Laplace equation solver implementations
│   ├── cpu/
│   │   ├── grid.h                  # Grid class definition
│   │   ├── laplace_single_thread.cpp
│   │   └── laplace_omp.cpp
│   ├── cuda/
│   │   └── laplace.cu
│   ├── openCL/
│   │   ├── laplace_kernel.cl       # OpenCL kernels
│   │   └── laplace_main.cpp        # OpenCL host code
│   └── plotter.py                  # Performance visualization script
├── bin/                            # Compiled binaries
├── Makefile                        # Build system
└── README.md                       # This file
```

## Prerequisites

- C++ compiler with C++11 support
- OpenMP library
- NVIDIA CUDA Toolkit
- OpenCL SDK (optional, for OpenCL version)
- Python with pandas and matplotlib (for plotting results)

## Building the Project

The project includes a Makefile to simplify building all implementations:

```bash
# Build all implementations
make all

# Build specific groups
make matmul      # Build all matrix multiplication implementations
make laplace     # Build all Laplace solver implementations

# Build specific implementations
make bin/matmul_st        # Matrix multiplication - single-threaded
make bin/matmul_omp       # Matrix multiplication - OpenMP
make bin/matmul_cuda      # Matrix multiplication - CUDA
make bin/laplace_st       # Laplace solver - single-threaded
make bin/laplace_omp      # Laplace solver - OpenMP
make bin/laplace_cuda     # Laplace solver - CUDA
```

## Usage

### Matrix Multiplication

#### Single-threaded Version
```bash
./bin/matmul_st [matrix_size]
```

#### OpenMP Version
```bash
./bin/matmul_omp [matrix_size] [num_threads]
```

#### CUDA Version
```bash
./bin/matmul_cuda [matrix_size] [kernel_type]
```
Parameters:
- `kernel_type`: Type of CUDA kernel (0: naive, 1: shared memory)

### Laplace Equation Solver

#### Single-threaded Version
```bash
./bin/laplace_st [grid_size] [tolerance] [max_iterations]
```

#### OpenMP Version
```bash
./bin/laplace_omp [grid_size] [num_threads] [tolerance] [max_iterations]
```

#### CUDA Version
```bash
./bin/laplace_cuda [grid_size] [block_size] [tolerance] [max_iterations]
```

#### OpenCL Version (Build Separately)
```bash
# Build with your preferred OpenCL compiler
g++ -std=c++11 -O3 laplace_solver/openCL/laplace_main.cpp -lOpenCL -o bin/laplace_opencl

# Run with grid size, workgroup size, and kernel type
./bin/laplace_opencl [grid_size] [workgroup_index] [use_optimized] [tolerance] [max_iterations]
```

## Running Automated Tests

The Makefile includes targets to run benchmarks across different problem sizes:

```bash
# Run all tests
make run

# Run matrix multiplication tests
make run_matmul
make run_matmul_st
make run_matmul_omp
make run_matmul_cuda

# Run Laplace solver tests
make run_laplace
make run_laplace_st
make run_laplace_omp
make run_laplace_cuda
```

## Visualization

To generate performance comparison plots:

```bash
python laplace_solver/plotter.py
```

This will create plots showing:
- Execution time comparison between implementations
- Speedup relative to single-threaded CPU implementation
- OpenMP scaling with thread count
- CUDA block size performance impact
- OpenCL workgroup size performance impact (if data available)

## Performance Reports

Generate performance reports from the CSV data:

```bash
# Generate all reports
make report

# Generate specific reports
make matmul_report
make laplace_report
```

## Output Files

The programs generate several output files:

### Matrix Multiplication
- Performance data: `single_threaded_performance.csv`, `omp_performance.csv`, `cuda_performance.csv` 
- Checksums for verification: `single_threaded_checksum.txt`, `omp_checksum.txt`, `cuda_checksum.txt`
- Summary report: `performance_report.txt`

### Laplace Solver
- Performance data: `laplace_cpu_performance.csv`, `laplace_omp_performance.csv`, `laplace_performance.csv`
- Solution CSV files: `laplace_solution_cpu.csv`, `laplace_solution_omp.csv`, `laplace_solution_cuda.csv`
- Statistics: `laplace_cpu_stats.txt`, `laplace_omp_stats.txt`, `laplace_cuda_stats.txt`
- Summary report: `laplace_performance_report.txt`

## Analysis

The Laplace solver implementations demonstrate different optimization techniques:
- OpenMP: Uses collapse(2) for efficient parallelization of nested loops and reduction for finding maximum differences
- CUDA: Uses shared memory for block-wise reduction and efficient boundary condition application
- OpenCL: Provides both basic and optimized kernels using local memory

Workgroup and block size analysis:
- For the CUDA implementation, 16×16 typically provides the best performance balance
- The reason is optimal occupancy, efficient memory access patterns, and balanced register pressure
