#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include "grid.h"

using namespace std;

// Function to save performance data
void savePerformanceData(const string& filename, size_t gridSize, double timeMs, int iterations, int numThreads) {
    ofstream outfile(filename, ios_base::app);
    
    if (!outfile.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }
    
    // Write header if file is empty
    outfile.seekp(0, ios::end);
    if (outfile.tellp() == 0) {
        outfile << "GridSize,NumThreads,Iterations,ExecutionTime_ms\n";
    }
    
    outfile << gridSize << "," << numThreads << "," << iterations << "," << timeMs << "\n";
    outfile.close();
}

// OpenMP Laplace equation solver using Jacobi iteration method
void solveLaplaceOmp(Grid& grid, float tolerance = 1e-4, int maxIterations = 10000) {
    size_t height = grid.getHeight();
    size_t width = grid.getWidth();
    
    // Create a temporary grid for the Jacobi method
    Grid newGrid(height, width);
    
    // Apply initial boundary conditions
    grid.applyBoundaryConditions();
    newGrid.applyBoundaryConditions();
    
    float maxDiff;
    int iterations = 0;
    
    do {
        maxDiff = 0.0f;
        
        // Update inner points using the Jacobi method - parallelized with OpenMP
        #pragma omp parallel
        {
            float localMaxDiff = 0.0f;
            
            #pragma omp for collapse(2)
            for (size_t i = 1; i < height - 1; ++i) {
                for (size_t j = 1; j < width - 1; ++j) {
                    // Calculate new value (average of neighbors)
                    newGrid.at(i, j) = 0.25f * (
                        grid.at(i+1, j) + grid.at(i-1, j) + 
                        grid.at(i, j+1) + grid.at(i, j-1)
                    );
                    
                    // Track the maximum difference for convergence check
                    float diff = fabs(newGrid.at(i, j) - grid.at(i, j));
                    localMaxDiff = max(localMaxDiff, diff);
                }
            }
            
            // Combine local maximums using a critical section
            #pragma omp critical
            {
                maxDiff = max(maxDiff, localMaxDiff);
            }
        }
        
        // Swap old and new grids (avoiding a full copy) - also parallelized
        #pragma omp parallel for collapse(2)
        for (size_t i = 1; i < height - 1; ++i) {
            for (size_t j = 1; j < width - 1; ++j) {
                grid.at(i, j) = newGrid.at(i, j);
            }
        }
        
        iterations++;
    } while (maxDiff > tolerance && iterations < maxIterations);
    
    cout << "Solved in " << iterations << " iterations with final diff: " << maxDiff << endl;
}

int main(int argc, char* argv[]) {
    // Default grid size and parameters
    size_t gridSize = 256;
    float tolerance = 1e-4;
    int maxIterations = 10000;
    int numThreads = omp_get_max_threads(); // Default to max available threads
    
    // Parse command line arguments
    if (argc > 1) {
        gridSize = stoi(argv[1]);
    }
    
    if (argc > 2) {
        numThreads = stoi(argv[2]);
        omp_set_num_threads(numThreads);
    }
    
    if (argc > 3) {
        tolerance = stof(argv[3]);
    }
    
    if (argc > 4) {
        maxIterations = stoi(argv[4]);
    }
    
    cout << "Running OpenMP Laplace solver with grid size: " << gridSize << "x" << gridSize 
         << " using " << numThreads << " threads" << endl;
    cout << "Tolerance: " << tolerance << ", Max iterations: " << maxIterations << endl;
    
    // Create grid with initial values
    Grid grid(gridSize, gridSize);
    
    // Solve and measure time
    auto start = chrono::high_resolution_clock::now();
    solveLaplaceOmp(grid, tolerance, maxIterations);
    auto end = chrono::high_resolution_clock::now();
    
    // Calculate duration
    chrono::duration<double, milli> duration = end - start;
    double timeMs = duration.count();
    
    cout << "Laplace equation solved in " << fixed << setprecision(2) 
         << timeMs << " ms" << endl;
    
    // Save results
    grid.saveToFile("laplace_solution_omp.csv");
    savePerformanceData("laplace_omp_performance.csv", gridSize, timeMs, maxIterations, numThreads);
    
    // Calculate and save statistics for verification
    float sum = 0.0f, min = 1e10f, max = -1e10f;
    
    #pragma omp parallel for reduction(+:sum) reduction(min:min) reduction(max:max) collapse(2)
    for (size_t i = 0; i < gridSize; ++i) {
        for (size_t j = 0; j < gridSize; ++j) {
            float val = grid.at(i, j);
            sum += val;
            min = std::min(min, val);
            max = std::max(max, val);
        }
    }
    float mean = sum / (gridSize * gridSize);
    
    ofstream statsFile("laplace_omp_stats.txt");
    statsFile << "Grid size: " << gridSize << "x" << gridSize << endl;
    statsFile << "Number of threads: " << numThreads << endl;
    statsFile << "Min value: " << min << endl;
    statsFile << "Max value: " << max << endl;
    statsFile << "Mean value: " << mean << endl;
    statsFile << "Sum: " << sum << endl;
    statsFile.close();
    
    return 0;
}