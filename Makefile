# Compiler settings
CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -std=c++11
OMPFLAGS = -fopenmp
CUDAFLAGS = -O3 -arch=compute_89 -code=sm_89
LDFLAGS = 

# Directories
MATMUL_DIR = matmul
BIN_DIR = bin
LAPLACE_DIR = laplace_solver

# Matrix multiplication source files
ST_SRC = $(MATMUL_DIR)/cpu/matrix_multiplication_single_threaded.cpp
OMP_SRC = $(MATMUL_DIR)/cpu/matrix_multiplication_omp.cpp
CUDA_SRC = $(MATMUL_DIR)/cuda/matmulcuda.cu

# Laplace solver source files
LAPLACE_ST_SRC = $(LAPLACE_DIR)/cpu/laplace_single_thread.cpp
LAPLACE_OMP_SRC = $(LAPLACE_DIR)/cpu/laplace_omp.cpp
LAPLACE_CUDA_SRC = $(LAPLACE_DIR)/cuda/laplace.cu

# Matrix multiplication executable names
ST_EXE = $(BIN_DIR)/matmul_st
OMP_EXE = $(BIN_DIR)/matmul_omp
CUDA_EXE = $(BIN_DIR)/matmul_cuda

# Laplace solver executable names
LAPLACE_ST_EXE = $(BIN_DIR)/laplace_st
LAPLACE_OMP_EXE = $(BIN_DIR)/laplace_omp
LAPLACE_CUDA_EXE = $(BIN_DIR)/laplace_cuda

# Matrix sizes to test with
SIZES = 1000 2000 3000 4000

# Laplace grid sizes to test with
LAPLACE_SIZES = 256 512 1024

# Default target
all: directories matmul laplace

# Group targets
matmul: $(ST_EXE) $(OMP_EXE) $(CUDA_EXE)
laplace: $(LAPLACE_ST_EXE) $(LAPLACE_OMP_EXE) $(LAPLACE_CUDA_EXE)

# Create directories
directories:
	@mkdir -p $(BIN_DIR)

# Matrix multiplication - single-threaded version
$(ST_EXE): $(ST_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -I$(MATMUL_DIR)/cpu

# Matrix multiplication - OpenMP version
$(OMP_EXE): $(OMP_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS) -I$(MATMUL_DIR)/cpu

# Matrix multiplication - CUDA version
$(CUDA_EXE): $(CUDA_SRC)
	$(NVCC) $(CUDAFLAGS) $< -o $@ $(LDFLAGS) -I$(MATMUL_DIR)/cpu

# Laplace solver - single-threaded version
$(LAPLACE_ST_EXE): $(LAPLACE_ST_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -I$(LAPLACE_DIR)/cpu

# Laplace solver - OpenMP version
$(LAPLACE_OMP_EXE): $(LAPLACE_OMP_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS) -I$(LAPLACE_DIR)/cpu

# Laplace solver - CUDA version
$(LAPLACE_CUDA_EXE): $(LAPLACE_CUDA_SRC)
	$(NVCC) $(CUDAFLAGS) $< -o $@ $(LDFLAGS) -I$(LAPLACE_DIR)/cpu

# Clean build files
clean:
	rm -rf $(BIN_DIR)

# Run matrix multiplication - single-threaded version
run_matmul_st: $(ST_EXE)
	@echo "Running single-threaded matrix multiplication..."
	@for size in $(SIZES); do \
		echo "Size: $$size x $$size"; \
		$(ST_EXE) $$size; \
		echo ""; \
	done

# Run matrix multiplication - OpenMP version
run_matmul_omp: $(OMP_EXE)
	@echo "Running OpenMP matrix multiplication..."
	@for size in $(SIZES); do \
		for threads in 2 4 8; do \
			echo "Size: $$size x $$size, Threads: $$threads"; \
			$(OMP_EXE) $$size $$threads; \
			echo ""; \
		done \
	done

# Run matrix multiplication - CUDA version
run_matmul_cuda: $(CUDA_EXE)
	@echo "Running CUDA matrix multiplication..."
	@for size in $(SIZES); do \
		for kernel in 0 1; do \
			kernelType=$$([ $$kernel -eq 0 ] && echo "naive" || echo "shared memory"); \
			echo "Size: $$size x $$size, Kernel: $$kernelType"; \
			$(CUDA_EXE) $$size $$kernel; \
			echo ""; \
		done \
	done

# Run all matrix multiplication implementations
run_matmul: run_matmul_st run_matmul_omp run_matmul_cuda

# Run Laplace solver - single-threaded version
run_laplace_st: $(LAPLACE_ST_EXE)
	@echo "Running single-threaded Laplace solver..."
	@for size in $(LAPLACE_SIZES); do \
		echo "Grid Size: $$size x $$size"; \
		$(LAPLACE_ST_EXE) $$size 1e-4 10000; \
		echo ""; \
	done

# Run Laplace solver - OpenMP version
run_laplace_omp: $(LAPLACE_OMP_EXE)
	@echo "Running OpenMP Laplace solver..."
	@for size in $(LAPLACE_SIZES); do \
		for threads in 2 4 8; do \
			echo "Grid Size: $$size x $$size, Threads: $$threads"; \
			$(LAPLACE_OMP_EXE) $$size $$threads 1e-4 10000; \
			echo ""; \
		done \
	done

# Run Laplace solver - CUDA version
run_laplace_cuda: $(LAPLACE_CUDA_EXE)
	@echo "Running CUDA Laplace solver..."
	@for size in $(LAPLACE_SIZES); do \
		for block_size in 8 16 32; do \
			echo "Grid Size: $$size x $$size, Block Size: $$block_size"; \
			$(LAPLACE_CUDA_EXE) $$size $$block_size 1e-4 10000; \
			echo ""; \
		done \
	done

# Run all Laplace solver implementations
run_laplace: run_laplace_st run_laplace_omp run_laplace_cuda

# Run all implementations
run: run_matmul run_laplace

# Generate comparison report for matrix multiplication
matmul_report: run_matmul
	@echo "Generating matrix multiplication performance comparison report..."
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

# Generate comparison report for Laplace solver
laplace_report: run_laplace
	@echo "Generating Laplace solver performance comparison report..."
	@echo "Laplace Equation Solver Performance Comparison" > laplace_performance_report.txt
	@echo "==========================================" >> laplace_performance_report.txt
	@echo "" >> laplace_performance_report.txt
	@echo "Single-threaded performance:" >> laplace_performance_report.txt
	@cat laplace_cpu_performance.csv >> laplace_performance_report.txt
	@echo "" >> laplace_performance_report.txt
	@echo "OpenMP performance:" >> laplace_performance_report.txt
	@cat laplace_omp_performance.csv >> laplace_performance_report.txt
	@echo "" >> laplace_performance_report.txt
	@echo "CUDA performance:" >> laplace_performance_report.txt
	@cat laplace_performance.csv >> laplace_performance_report.txt
	@echo "" >> laplace_performance_report.txt
	@echo "Report saved to laplace_performance_report.txt"

# Generate all reports
report: matmul_report laplace_report

.PHONY: all directories clean matmul laplace \
        run_matmul_st run_matmul_omp run_matmul_cuda run_matmul \
        run_laplace_st run_laplace_omp run_laplace_cuda run_laplace \
        run matmul_report laplace_report report
