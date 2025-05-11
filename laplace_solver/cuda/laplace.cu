#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include "../cpu/grid.h"

using namespace std;

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                 << cudaGetErrorString(error) << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CUDA kernel for Laplace equation using Jacobi iteration
__global__ void laplaceJacobiKernel(const float* oldGrid, float* newGrid, int width, int height) {
    // Calculate thread indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Only update interior points (skip boundaries)
    if (row > 0 && row < height - 1 && col > 0 && col < width - 1) {
        // Calculate linear index
        int idx = row * width + col;
        
        // Apply Jacobi iteration (average of neighbors)
        newGrid[idx] = 0.25f * (
            oldGrid[idx - width] +  // top neighbor
            oldGrid[idx + width] +  // bottom neighbor
            oldGrid[idx - 1] +      // left neighbor
            oldGrid[idx + 1]        // right neighbor
        );
    }
}

// CUDA kernel for setting boundary conditions
__global__ void setBoundaryConditionsKernel(float* grid, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Top boundary (5V)
    if (idx < width) {
        grid[idx] = 5.0f;
    }
    
    // Bottom boundary (-5V)
    if (idx < width) {
        grid[(height - 1) * width + idx] = -5.0f;
    }
    
    // Left and right boundaries (0V)
    if (idx > 0 && idx < height - 1) {
        grid[idx * width] = 0.0f;            // Left
        grid[idx * width + (width - 1)] = 0.0f;  // Right
    }
}

// CUDA kernel to calculate the maximum difference between two grids
__global__ void calcMaxDiffKernel(const float* oldGrid, const float* newGrid, 
                               float* maxDiffs, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Calculate linear index for global memory access
    int idx = row * width + col;
    
    // Initialize shared memory for block-wise reduction
    __shared__ float blockMaxDiff[256]; // Assuming block size <= 256
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Initialize with 0 or the difference if in bounds
    float diff = 0.0f;
    if (row > 0 && row < height - 1 && col > 0 && col < width - 1) {
        diff = fabs(newGrid[idx] - oldGrid[idx]);
    }
    
    blockMaxDiff[tid] = diff;
    __syncthreads();
    
    // Reduction to find max difference in block
    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            blockMaxDiff[tid] = fmax(blockMaxDiff[tid], blockMaxDiff[tid + s]);
        }
        __syncthreads();
    }
    
    // Write the result for this block
    if (tid == 0) {
        maxDiffs[blockIdx.y * gridDim.x + blockIdx.x] = blockMaxDiff[0];
    }
}

// Function to save performance data
void savePerformanceData(const string& filename, size_t gridSize, double timeMs, int iterations) {
    ofstream outfile(filename, ios_base::app);
    
    if (!outfile.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }
    
    // Write header if file is empty
    outfile.seekp(0, ios::end);
    if (outfile.tellp() == 0) {
        outfile << "GridSize,Iterations,ExecutionTime_ms,Implementation\n";
    }
    
    outfile << gridSize << "," << iterations << "," << timeMs << ",CUDA\n";
    outfile.close();
}

int main(int argc, char* argv[]) {
    // Default parameters
    size_t gridSize = 256;
    float tolerance = 1e-4;
    int maxIterations = 10000;
    int blockSize = 16; // Default CUDA block size (16x16)
    
    // Parse command line arguments
    if (argc > 1) {
        gridSize = stoi(argv[1]);
    }
    
    if (argc > 2) {
        blockSize = stoi(argv[2]);
    }
    
    if (argc > 3) {
        tolerance = stof(argv[3]);
    }
    
    if (argc > 4) {
        maxIterations = stoi(argv[4]);
    }
    
    cout << "Running CUDA Laplace solver with grid size: " << gridSize << "x" << gridSize 
         << " using " << blockSize << "x" << blockSize << " block size" << endl;
    cout << "Tolerance: " << tolerance << ", Max iterations: " << maxIterations << endl;
    
    // Create host grid for input/output
    Grid hostGrid(gridSize, gridSize);
    hostGrid.applyBoundaryConditions();
    
    // Flatten the grid for CUDA
    vector<float> flatGrid(gridSize * gridSize);
    for (size_t i = 0; i < gridSize; ++i) {
        for (size_t j = 0; j < gridSize; ++j) {
            flatGrid[i * gridSize + j] = hostGrid.at(i, j);
        }
    }
    
    // Allocate device memory
    float *d_oldGrid, *d_newGrid;
    size_t gridBytes = gridSize * gridSize * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_oldGrid, gridBytes));
    CUDA_CHECK(cudaMalloc(&d_newGrid, gridBytes));
    
    // Copy initial grid to device
    CUDA_CHECK(cudaMemcpy(d_oldGrid, flatGrid.data(), gridBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_newGrid, flatGrid.data(), gridBytes, cudaMemcpyHostToDevice));
    
    // Define grid and block dimensions for main computation
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid((gridSize + blockSize - 1) / blockSize, 
                (gridSize + blockSize - 1) / blockSize);
    
    // Define grid and block dimensions for boundary conditions
    dim3 boundaryBlock(256);
    dim3 boundaryGrid((max(gridSize, gridSize) + 255) / 256);
    
    // Allocate memory for max differences
    int numBlocks = dimGrid.x * dimGrid.y;
    float *d_maxDiffs, *h_maxDiffs;
    h_maxDiffs = new float[numBlocks];
    CUDA_CHECK(cudaMalloc(&d_maxDiffs, numBlocks * sizeof(float)));
    
    // Apply boundary conditions on device
    setBoundaryConditionsKernel<<<boundaryGrid, boundaryBlock>>>(d_oldGrid, gridSize, gridSize);
    setBoundaryConditionsKernel<<<boundaryGrid, boundaryBlock>>>(d_newGrid, gridSize, gridSize);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Record start event
    CUDA_CHECK(cudaEventRecord(start));
    
    // Main iteration loop
    float maxDiff = tolerance + 1.0f;
    int iterations = 0;
    
    while (maxDiff > tolerance && iterations < maxIterations) {
        // Launch Jacobi kernel
        laplaceJacobiKernel<<<dimGrid, dimBlock>>>(d_oldGrid, d_newGrid, gridSize, gridSize);
        
        // Apply boundary conditions to ensure they remain fixed
        setBoundaryConditionsKernel<<<boundaryGrid, boundaryBlock>>>(d_newGrid, gridSize, gridSize);
        
        // Calculate maximum difference
        calcMaxDiffKernel<<<dimGrid, dimBlock>>>(d_oldGrid, d_newGrid, d_maxDiffs, gridSize, gridSize);
        
        // Copy max differences back to host
        CUDA_CHECK(cudaMemcpy(h_maxDiffs, d_maxDiffs, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Find global maximum on host
        maxDiff = 0.0f;
        for (int i = 0; i < numBlocks; ++i) {
            maxDiff = max(maxDiff, h_maxDiffs[i]);
        }
        
        // Swap old and new grids
        float* temp = d_oldGrid;
        d_oldGrid = d_newGrid;
        d_newGrid = temp;
        
        iterations++;
        
        // Print progress occasionally
        if (iterations % 100 == 0) {
            cout << "Iteration " << iterations << ", max diff: " << maxDiff << endl;
        }
    }
    
    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    cout << "Laplace equation solved in " << iterations << " iterations" << endl;
    cout << "Final diff: " << maxDiff << endl;
    cout << "Time: " << fixed << setprecision(2) << milliseconds << " ms" << endl;
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(flatGrid.data(), d_oldGrid, gridBytes, cudaMemcpyHostToDevice));
    
    // Transfer back to Grid format
    for (size_t i = 0; i < gridSize; ++i) {
        for (size_t j = 0; j < gridSize; ++j) {
            hostGrid.at(i, j) = flatGrid[i * gridSize + j];
        }
    }
    
    // Save results
    hostGrid.saveToFile("laplace_solution_cuda.csv");
    savePerformanceData("laplace_performance.csv", gridSize, milliseconds, iterations);
    
    // Calculate and save statistics for verification
    float sum = 0.0f, min = 1e10f, max = -1e10f;
    for (size_t i = 0; i < gridSize; ++i) {
        for (size_t j = 0; j < gridSize; ++j) {
            float val = hostGrid.at(i, j);
            sum += val;
            min = std::min(min, val);
            max = std::max(max, val);
        }
    }
    float mean = sum / (gridSize * gridSize);
    
    ofstream statsFile("laplace_cuda_stats.txt");
    statsFile << "Grid size: " << gridSize << "x" << gridSize << endl;
    statsFile << "Block size: " << blockSize << "x" << blockSize << endl;
    statsFile << "Iterations: " << iterations << endl;
    statsFile << "Min value: " << min << endl;
    statsFile << "Max value: " << max << endl;
    statsFile << "Mean value: " << mean << endl;
    statsFile << "Sum: " << sum << endl;
    statsFile.close();
    
    // Clean up
    CUDA_CHECK(cudaFree(d_oldGrid));
    CUDA_CHECK(cudaFree(d_newGrid));
    CUDA_CHECK(cudaFree(d_maxDiffs));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    delete[] h_maxDiffs;
    
    return 0;
}