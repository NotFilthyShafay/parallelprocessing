#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <random>
#include <cuda_runtime.h>
#include "../cpu/matrix.h"
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

// CUDA kernel for matrix multiplication
// Uses a 2D block of threads
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Block index
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread index
    int row = blockRow * blockDim.y + threadIdx.y;
    int col = blockCol * blockDim.x + threadIdx.x;
    
    // Each thread computes one element of C
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Optimized CUDA kernel using shared memory
__global__ void matrixMulSharedKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Block size (assumed to be BLOCK_SIZE x BLOCK_SIZE)
    const int BLOCK_SIZE = blockDim.x;
    
    // Shared memory for tiles of A and B
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];
    
    // Block index
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread index
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    
    // Index of the first element of sub-matrix of A processed by the block
    int aRow = blockRow * BLOCK_SIZE + threadRow;
    // Index of the first element of sub-matrix of B processed by the block
    int bCol = blockCol * BLOCK_SIZE + threadCol;
    
    // Initialize accumulator
    float sum = 0.0f;
    
    // Loop over all tiles
    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int tile = 0; tile < numTiles; ++tile) {
        // Load tiles into shared memory
        int aCol = tile * BLOCK_SIZE + threadCol;
        int bRow = tile * BLOCK_SIZE + threadRow;
        
        // Load tile of A
        if (aRow < M && aCol < K)
            As[threadRow][threadCol] = A[aRow * K + aCol];
        else
            As[threadRow][threadCol] = 0.0f;
        
        // Load tile of B
        if (bRow < K && bCol < N)
            Bs[threadRow][threadCol] = B[bRow * N + bCol];
        else
            Bs[threadRow][threadCol] = 0.0f;
        
        // Synchronize to ensure tiles are loaded before computation
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadRow][k] * Bs[k][threadCol];
        }
        
        // Synchronize to ensure shared memory is not overwritten 
        // before computation is complete
        __syncthreads();
    }
    
    // Write the result to C
    if (aRow < M && bCol < N)
        C[aRow * N + bCol] = sum;
}

// Function to save performance data
void savePerformanceData(const string& filename, size_t matrixSize, double timeMs, const string& kernelType) {
    ofstream outfile(filename, ios_base::app);
    
    if (!outfile.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }
    
    // Write header if file is empty
    outfile.seekp(0, ios::end);
    if (outfile.tellp() == 0) {
        outfile << "MatrixSize,KernelType,ExecutionTime_ms\n";
    }
    
    outfile << matrixSize << "," << kernelType << "," << timeMs << "\n";
    outfile.close();
}

int main(int argc, char* argv[]) {
    // Default matrix size and kernel type
    size_t matrixSize = 1000;
    bool useSharedMemory = true; // Default to shared memory kernel
    
    if (argc > 1) {
        matrixSize = stoi(argv[1]);
    }
    
    if (argc > 2) {
        useSharedMemory = (stoi(argv[2]) != 0);
    }
    
    string kernelType = useSharedMemory ? "shared_memory" : "naive";
    
    cout << "Running CUDA matrix multiplication with size: " << matrixSize << "x" << matrixSize
         << " using " << kernelType << " kernel" << endl;
    
    // Create and initialize matrices on host
    Matrix A(matrixSize, matrixSize);
    Matrix B(matrixSize, matrixSize);
    Matrix C(matrixSize, matrixSize); // For results
    
    A.randomize();
    B.randomize();
    
    // Allocate memory on device
    float *d_A, *d_B, *d_C;
    size_t size = matrixSize * matrixSize * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    
    // Copy data from host to device
    vector<float> hostA(matrixSize * matrixSize);
    vector<float> hostB(matrixSize * matrixSize);
    vector<float> hostC(matrixSize * matrixSize);
    
    // Flatten the matrices for CUDA
    for (size_t i = 0; i < matrixSize; ++i) {
        for (size_t j = 0; j < matrixSize; ++j) {
            hostA[i * matrixSize + j] = A(i, j);
            hostB[i * matrixSize + j] = B(i, j);
        }
    }
    
    CUDA_CHECK(cudaMemcpy(d_A, hostA.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, hostB.data(), size, cudaMemcpyHostToDevice));
    
    // Define grid and block dimensions
    int blockSize = 32; // 32x32 threads per block
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid((matrixSize + blockSize - 1) / blockSize, 
                 (matrixSize + blockSize - 1) / blockSize);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Record start event
    CUDA_CHECK(cudaEventRecord(start));
    
    // Launch kernel
    if (useSharedMemory) {
        matrixMulSharedKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, matrixSize, matrixSize, matrixSize);
    } else {
        matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, matrixSize, matrixSize, matrixSize);
    }
    
    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());
    
    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    cout << "Matrix multiplication completed in " << fixed << setprecision(2) 
         << milliseconds << " ms" << endl;
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(hostC.data(), d_C, size, cudaMemcpyDeviceToHost));
    
    // Transfer back to matrix format
    for (size_t i = 0; i < matrixSize; ++i) {
        for (size_t j = 0; j < matrixSize; ++j) {
            C(i, j) = hostC[i * matrixSize + j];
        }
    }
    
    // Save results
    savePerformanceData("cuda_performance.csv", matrixSize, milliseconds, kernelType);
    
    // Save checksum for verification
    float checksum = 0.0f;
    for (size_t i = 0; i < C.getRows(); ++i) {
        for (size_t j = 0; j < C.getCols(); ++j) {
            checksum += C(i, j);
        }
    }
    
    ofstream checksumFile("cuda_checksum.txt");
    checksumFile << "Matrix size: " << matrixSize << "x" << matrixSize << endl;
    checksumFile << "Kernel type: " << kernelType << endl;
    checksumFile << "Checksum: " << checksum << endl;
    checksumFile.close();
    
    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}