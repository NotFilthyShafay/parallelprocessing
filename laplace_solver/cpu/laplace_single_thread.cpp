#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cmath>
#include "grid.h"

using namespace std;

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
        outfile << "GridSize,Iterations,ExecutionTime_ms\n";
    }
    
    outfile << gridSize << "," << iterations << "," << timeMs << "\n";
    outfile.close();
}

// Single-threaded Laplace equation solver using Jacobi iteration method
void solveLaplace(Grid& grid, float tolerance = 1e-4, int maxIterations = 10000) {
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
        
        // Update inner points using the Jacobi method
        for (size_t i = 1; i < height - 1; ++i) {
            for (size_t j = 1; j < width - 1; ++j) {
                // Calculate new value (average of neighbors)
                newGrid.at(i, j) = 0.25f * (
                    grid.at(i+1, j) + grid.at(i-1, j) + 
                    grid.at(i, j+1) + grid.at(i, j-1)
                );
                
                // Track the maximum difference for convergence check
                float diff = fabs(newGrid.at(i, j) - grid.at(i, j));
                maxDiff = max(maxDiff, diff);
            }
        }
        
        // Swap old and new grids (avoiding a full copy)
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
    // Default grid size
    size_t gridSize = 256;
    float tolerance = 1e-4;
    int maxIterations = 10000;
    
    // Parse command line arguments
    if (argc > 1) {
        gridSize = stoi(argv[1]);
    }
    
    if (argc > 2) {
        tolerance = stof(argv[2]);
    }
    
    if (argc > 3) {
        maxIterations = stoi(argv[3]);
    }
    
    cout << "Running single-threaded Laplace solver with grid size: " << gridSize << "x" << gridSize 
         << " (tolerance=" << tolerance << ", max_iter=" << maxIterations << ")" << endl;
    
    // Create grid with initial values
    Grid grid(gridSize, gridSize);
    
    // Solve and measure time
    auto start = chrono::high_resolution_clock::now();
    solveLaplace(grid, tolerance, maxIterations);
    auto end = chrono::high_resolution_clock::now();
    
    // Calculate duration
    chrono::duration<double, milli> duration = end - start;
    double timeMs = duration.count();
    
    cout << "Laplace equation solved in " << fixed << setprecision(2) 
         << timeMs << " ms" << endl;
    
    // Save results
    grid.saveToFile("laplace_solution_cpu.csv");
    savePerformanceData("laplace_cpu_performance.csv", gridSize, timeMs, maxIterations);
    
    // Calculate and save statistics for verification
    float sum = 0.0f, min = 1e10f, max = -1e10f;
    for (size_t i = 0; i < gridSize; ++i) {
        for (size_t j = 0; j < gridSize; ++j) {
            float val = grid.at(i, j);
            sum += val;
            min = std::min(min, val);
            max = std::max(max, val);
        }
    }
    float mean = sum / (gridSize * gridSize);
    
    ofstream statsFile("laplace_cpu_stats.txt");
    statsFile << "Grid size: " << gridSize << "x" << gridSize << endl;
    statsFile << "Min value: " << min << endl;
    statsFile << "Max value: " << max << endl;
    statsFile << "Mean value: " << mean << endl;
    statsFile << "Sum: " << sum << endl;
    statsFile.close();
    
    return 0;
}