#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <random>
#include "matrix.h"
using namespace std;

// Single-threaded matrix multiplication
Matrix multiply(const Matrix& A, const Matrix& B) {
    if (A.getCols() != B.getRows()) {
        throw runtime_error("Matrix dimensions mismatch");
    }
    
    size_t M = A.getRows();
    size_t K = A.getCols();
    size_t N = B.getCols();
    
    Matrix C(M, N);
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    
    return C;
}

// Function to save performance data
void savePerformanceData(const string& filename, size_t matrixSize, double timeMs) {
    ofstream outfile(filename, ios_base::app);
    
    if (!outfile.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }
    
    // Write header if file is empty
    outfile.seekp(0, ios::end);
    if (outfile.tellp() == 0) {
        outfile << "MatrixSize,ExecutionTime_ms\n";
    }
    
    outfile << matrixSize << "," << timeMs << "\n";
    outfile.close();
}

int main(int argc, char* argv[]) {
    // Default matrix size if not provided
    size_t matrixSize = 1000;
    
    if (argc > 1) {
        matrixSize = stoi(argv[1]);
    }
    
    cout << "Running single-threaded matrix multiplication with size: " << matrixSize << "x" << matrixSize << endl;
    
    // Create and initialize matrices
    Matrix A(matrixSize, matrixSize);
    Matrix B(matrixSize, matrixSize);
    A.randomize();
    B.randomize();
    
    // Perform multiplication and measure time
    auto start = chrono::high_resolution_clock::now();
    Matrix C = multiply(A, B);
    auto end = chrono::high_resolution_clock::now();
    
    // Calculate duration
    chrono::duration<double, milli> duration = end - start;
    double timeMs = duration.count();
    
    cout << "Matrix multiplication completed in " << fixed << setprecision(2) 
              << timeMs << " ms" << endl;
    
    // Save results
    savePerformanceData("single_threaded_performance.csv", matrixSize, timeMs);
    
    // Save checksum for verification
    float checksum = 0.0f;
    for (size_t i = 0; i < C.getRows(); ++i) {
        for (size_t j = 0; j < C.getCols(); ++j) {
            checksum += C(i, j);
        }
    }
    
    ofstream checksumFile("single_threaded_checksum.txt");
    checksumFile << "Matrix size: " << matrixSize << "x" << matrixSize << endl;
    checksumFile << "Checksum: " << checksum << endl;
    checksumFile.close();
    
    return 0;
}