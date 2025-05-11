Matrix Multiplication Performance Comparison
This project compares the performance of matrix multiplication implemented using different parallel computing techniques: single-threaded CPU, OpenMP CPU parallelization, and CUDA GPU acceleration.
Project Overview
Matrix multiplication is a fundamental operation in many computational applications, from scientific simulations to machine learning. This project provides implementations in:

Single-threaded CPU code
OpenMP parallel CPU code (with configurable thread count)
CUDA GPU acceleration (with both naive and shared memory optimization)

Performance is measured across different matrix sizes to demonstrate how each approach scales.
Requirements
To build and run this project, you'll need:

A C++ compiler with C++11 support (g++ recommended)
OpenMP support (included in most modern C++ compilers)
NVIDIA CUDA Toolkit (12.x or newer recommended)
An NVIDIA GPU with compute capability 8.9 or compatible with your CUDA version
Linux environment (WSL supported with proper GPU passthrough)

Project Structure
.
├── bin/                 # Compiled executables
├── cpu/                 # CPU implementations
│   ├── matrix_multiplication_single_threaded.cpp
│   └── matrix_multiplication_omp.cpp
├── cuda/                # CUDA implementations
│   └── matmulcuda.cu
├── opencl/             # OpenCL implementation (not used in current Makefile)
├── Makefile            # Build and run automation
└── README.md           # This file
Building the Project
To build all implementations:
bashmake
This will compile the single-threaded, OpenMP, and CUDA versions of the matrix multiplication code.
Running the Tests
Running All Implementations
To run all implementations with various matrix sizes:
bashmake run
Running Specific Implementations
Run only the single-threaded version:
bashmake run_st
Run only the OpenMP version (with 2, 4, and 8 threads):
bashmake run_omp
Run only the CUDA version (with both naive and shared memory kernels):
bashmake run_cuda
Generating a Performance Report
To run all tests and generate a performance comparison report:
bashmake report
This will create a performance_report.txt file containing performance metrics for all implementations.
Performance Metrics
The performance is measured in:

Execution time in milliseconds
GFLOPS (Giga Floating Point Operations Per Second)

The report compares how performance scales with:

Matrix size (1000x1000 to 4000x4000)
Number of threads (for OpenMP)
Kernel optimization technique (for CUDA)

Customizing the Tests
You can modify the SIZES variable in the Makefile to test with different matrix dimensions:
makeSIZES = 1000 2000 3000 4000
Cleaning Up
To remove all compiled executables:
bashmake clean
Notes

For the OpenMP version, specify the number of threads as a command-line argument
For the CUDA version, specify the kernel type (0 for naive, 1 for shared memory) as a command-line argument
Performance may vary significantly depending on your hardware configuration