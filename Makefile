# Compiler settings
CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -std=c++11
OMPFLAGS = -fopenmp
CUDAFLAGS = -O3 -arch=compute_89 -code=sm_89
LDFLAGS = 

# Directories
CPU_DIR = cpu
CUDA_DIR = cuda
OPENCL_DIR = opencl
BIN_DIR = bin

# Source files
ST_SRC = $(CPU_DIR)/matrix_multiplication_single_threaded.cpp
OMP_SRC = $(CPU_DIR)/matrix_multiplication_omp.cpp
CUDA_SRC = $(CUDA_DIR)/matmulcuda.cu

# Executable names
ST_EXE = $(BIN_DIR)/matmul_st
OMP_EXE = $(BIN_DIR)/matmul_omp
CUDA_EXE = $(BIN_DIR)/matmul_cuda

# Matrix sizes to test with
SIZES = 1000 2000 3000 4000

# Default target
all: directories $(ST_EXE) $(OMP_EXE) $(CUDA_EXE)

# Create directories
directories:
	@mkdir -p $(BIN_DIR)

# Single-threaded version
$(ST_EXE): $(ST_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# OpenMP version
$(OMP_EXE): $(OMP_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS)

# CUDA version
$(CUDA_EXE): $(CUDA_SRC)
	$(NVCC) $(CUDAFLAGS) $< -o $@ $(LDFLAGS)

# Clean build files
clean:
	rm -rf $(BIN_DIR)

# Run single-threaded version with different sizes
run_st: $(ST_EXE)
	@echo "Running single-threaded matrix multiplication..."
	@for size in $(SIZES); do \
		echo "Size: $$size x $$size"; \
		$(ST_EXE) $$size; \
		echo ""; \
	done

# Run OpenMP version with different sizes
run_omp: $(OMP_EXE)
	@echo "Running OpenMP matrix multiplication..."
	@for size in $(SIZES); do \
		for threads in 2 4 8; do \
			echo "Size: $$size x $$size, Threads: $$threads"; \
			$(OMP_EXE) $$size $$threads; \
			echo ""; \
		done \
	done

# Run CUDA version with different sizes
run_cuda: $(CUDA_EXE)
	@echo "Running CUDA matrix multiplication..."
	@for size in $(SIZES); do \
		for kernel in 0 1; do \
			kernelType=$$([ $$kernel -eq 0 ] && echo "naive" || echo "shared memory"); \
			echo "Size: $$size x $$size, Kernel: $$kernelType"; \
			$(CUDA_EXE) $$size $$kernel; \
			echo ""; \
		done \
	done

# Run all implementations
run: run_st run_omp run_cuda

# Generate comparison report
report: run
	@echo "Generating performance comparison report..."
	@echo "Matrix Multiplication Performance Comparison" > performance_report.txt
	@echo "==========================================" >> performance_report.txt
	@echo "" >> performance_report.txt
	@echo "Single-threaded performance:" >> performance_report.txt
	@cat single_threaded_performance.csv >> performance_report.txt
	@echo "" >> performance_report.txt
	@echo "OpenMP performance:" >> performance_report.txt
	@cat omp_performance.csv >> performance_report.txt
	@echo "" >> performance_report.txt
	@echo "CUDA performance:" >> performance_report.txt
	@cat cuda_performance.csv >> performance_report.txt
	@echo "" >> performance_report.txt
	@echo "Report saved to performance_report.txt"

.PHONY: all directories clean run_st run_omp run_cuda run report
