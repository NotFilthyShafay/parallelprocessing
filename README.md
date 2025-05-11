# Parallel Matrix Multiplication

This project implements and compares matrix multiplication using various parallel computing approaches:
- Single-threaded CPU implementation
- OpenMP multi-threaded CPU implementation
- CUDA GPU implementation (with both naive and shared memory kernels)

## Project Structure

```
/home/shafay/Parallel/
├── cpu/
│   ├── matrix.h                     # Matrix class definition
│   ├── matrix_multiplication_omp.cpp       # OpenMP implementation
│   └── matrix_multiplication_single_threaded.cpp  # Single-threaded implementation
├── cuda/
│   └── matmulcuda.cu                # CUDA implementation
├── bin/                             # Compiled binaries
├── Makefile                         # Build system
└── README.md                        # This file
```

## Prerequisites

- C++ compiler with C++11 support
- OpenMP library
- NVIDIA CUDA Toolkit
- Matrix class implementation (`matrix.h`)

## Building the Project

The project includes a Makefile to simplify building all implementations:

```bash
# Build all implementations
make all

# Build specific implementations
make bin/matmul_st    # Single-threaded
make bin/matmul_omp   # OpenMP
make bin/matmul_cuda  # CUDA
```

You can also compile manually:

```bash
# Single-threaded version
g++ -O3 -std=c++11 -o bin/matmul_st cpu/matrix_multiplication_single_threaded.cpp

# OpenMP version
g++ -fopenmp -O3 -std=c++11 -o bin/matmul_omp cpu/matrix_multiplication_omp.cpp

# CUDA version
nvcc -O3 -arch=compute_89 -code=sm_89 -o bin/matmul_cuda cuda/matmulcuda.cu
```

## Usage

### Single-threaded Matrix Multiplication

```bash
./bin/matmul_st [matrix_size]
```

Parameters:
- `matrix_size`: Size of the square matrices (default: 1000)

### OpenMP Matrix Multiplication

```bash
./bin/matmul_omp [matrix_size] [num_threads]
```

Parameters:
- `matrix_size`: Size of the square matrices (default: 1000)
- `num_threads`: Number of OpenMP threads to use (default: max available threads)

### CUDA Matrix Multiplication

```bash
./bin/matmul_cuda [matrix_size] [kernel_type]
```

Parameters:
- `matrix_size`: Size of the square matrices (default: 1000)
- `kernel_type`: Type of CUDA kernel to use (0: naive, 1: shared memory, default: 1)

## Running Automated Tests

The Makefile includes targets to run benchmarks across different matrix sizes:

```bash
# Run all implementations with predefined matrix sizes
make run

# Run specific implementations
make run_st    # Single-threaded
make run_omp   # OpenMP with 2, 4, and 8 threads
make run_cuda  # CUDA with both kernel types

# Generate a performance comparison report
make report
```

## Output Files

The programs generate the following output files:

1. Performance CSV files:
   - `single_threaded_performance.csv`: Single-threaded performance metrics
   - `omp_performance.csv`: OpenMP performance metrics
   - `cuda_performance.csv`: CUDA performance metrics

2. Checksum files for verification:
   - `single_threaded_checksum.txt`
   - `omp_checksum.txt`
   - `cuda_checksum.txt`

3. Performance report:
   - `performance_report.txt`: Comparison of all implementations (generated with `make report`)

## Example

```bash
# Run with 2000x2000 matrices
./bin/matmul_st 2000          # Single-threaded
./bin/matmul_omp 2000 4       # OpenMP with 4 threads
./bin/matmul_cuda 2000 1      # CUDA with shared memory

# Run all benchmarks and generate report
make report
```

Expected output (OpenMP example):
```
Running OpenMP matrix multiplication with size: 2000x2000 using 4 threads
Matrix multiplication completed in XXXX.XX ms
```

## Performance Analysis

You can use the generated CSV files to analyze performance across different:
- Matrix sizes
- Number of threads (for OpenMP)
- Kernel implementations (for CUDA)

This is useful for studying:
- Scalability
- Parallel efficiency
- CPU vs GPU performance
- Effect of shared memory optimization in GPU computing
