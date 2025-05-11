#ifndef GRID_H
#define GRID_H

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

using namespace std;

// 2D grid class for Laplace solver
class Grid {
private:
    vector<vector<float>> data;
    size_t width, height;

public:
    Grid(size_t h, size_t w, float initialValue = 0.0f) : 
        height(h), width(w), data(h, vector<float>(w, initialValue)) {}
    
    float& at(size_t i, size_t j) {
        return data[i][j];
    }
    
    const float& at(size_t i, size_t j) const {
        return data[i][j];
    }
    
    size_t getWidth() const { return width; }
    size_t getHeight() const { return height; }
    
    // Apply boundary conditions for a capacitor:
    // top = 5V, bottom = -5V, left = right = 0V
    void applyBoundaryConditions() {
        // Top boundary (5V)
        for (size_t j = 0; j < width; ++j) {
            data[0][j] = 5.0f;
        }
        
        // Bottom boundary (-5V)
        for (size_t j = 0; j < width; ++j) {
            data[height-1][j] = -5.0f;
        }
        
        // Left and right boundaries (0V)
        for (size_t i = 1; i < height-1; ++i) {
            data[i][0] = 0.0f;
            data[i][width-1] = 0.0f;
        }
    }
    
    // Save grid data to a CSV file for visualization
    void saveToFile(const string& filename) const {
        ofstream outfile(filename);
        if (!outfile.is_open()) {
            cerr << "Failed to open file: " << filename << endl;
            return;
        }
        
        for (size_t i = 0; i < height; ++i) {
            for (size_t j = 0; j < width; ++j) {
                outfile << data[i][j];
                if (j < width - 1) outfile << ",";
            }
            outfile << endl;
        }
        
        outfile.close();
    }
};

#endif // GRID_H