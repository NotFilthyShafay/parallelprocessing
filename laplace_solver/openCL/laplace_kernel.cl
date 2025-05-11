// OpenCL kernel for Laplace equation solver using Jacobi iteration
__kernel void laplaceJacobiKernel(__global const float* oldGrid, 
                                  __global float* newGrid,
                                  const int width, 
                                  const int height) {
    // Get global position in the grid
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    
    // Only update interior points (skip boundaries)
    if (row > 0 && row < height - 1 && col > 0 && col < width - 1) {
        // Calculate linear index
        const int idx = row * width + col;
        
        // Apply Jacobi iteration (average of neighbors)
        newGrid[idx] = 0.25f * (
            oldGrid[idx - width] +  // top neighbor
            oldGrid[idx + width] +  // bottom neighbor
            oldGrid[idx - 1] +      // left neighbor
            oldGrid[idx + 1]        // right neighbor
        );
    }
}

// OpenCL kernel for applying boundary conditions
__kernel void setBoundaryConditionsKernel(__global float* grid,
                                         const int width,
                                         const int height) {
    // Get global ID
    const int idx = get_global_id(0);
    
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

// OpenCL kernel to calculate maximum difference between two grids
// Using local memory for efficient reduction
__kernel void computeMaxDifferenceKernel(__global const float* oldGrid,
                                       __global const float* newGrid,
                                       __global float* blockMaxDiffs,
                                       __local float* localMaxDiffs,
                                       const int width,
                                       const int height) {
    // Get global and local indices
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    const int localIdx = get_local_id(1) * get_local_size(0) + get_local_id(0);
    const int blockIdx = get_group_id(1) * get_num_groups(0) + get_group_id(0);
    
    // Calculate grid index
    const int idx = row * width + col;
    
    // Initialize local difference value
    float localDiff = 0.0f;
    
    // Only compute difference for interior points
    if (row > 0 && row < height - 1 && col > 0 && col < width - 1) {
        localDiff = fabs(newGrid[idx] - oldGrid[idx]);
    }
    
    // Store in local memory
    localMaxDiffs[localIdx] = localDiff;
    
    // Wait for all local work items to store their values
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Perform reduction in local memory
    for (int stride = get_local_size(0) * get_local_size(1) / 2; stride > 0; stride >>= 1) {
        if (localIdx < stride) {
            localMaxDiffs[localIdx] = fmax(localMaxDiffs[localIdx], localMaxDiffs[localIdx + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result for this work-group to global memory
    if (localIdx == 0) {
        blockMaxDiffs[blockIdx] = localMaxDiffs[0];
    }
}

// Optimized Laplace solver with local memory for better memory access patterns
// This variant loads a block of the grid into local memory to reduce global memory accesses
__kernel void laplaceJacobiOptimizedKernel(__global const float* oldGrid,
                                         __global float* newGrid,
                                         const int width,
                                         const int height,
                                         __local float* localBlock) {
    // Get global position
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    
    // Get local position
    const int localCol = get_local_id(0);
    const int localRow = get_local_id(1);
    
    // Local block dimensions
    const int localWidth = get_local_size(0) + 2;  // +2 for halo cells
    
    // Load block into local memory including halo cells
    // Central part
    if (row < height && col < width) {
        localBlock[(localRow+1) * localWidth + (localCol+1)] = oldGrid[row * width + col];
    }
    
    // Top halo (if this work-item is in the first row of the block)
    if (localRow == 0 && row > 0 && col < width) {
        localBlock[localCol+1] = oldGrid[(row-1) * width + col];
    }
    
    // Bottom halo (if this work-item is in the last row of the block)
    if (localRow == get_local_size(1)-1 && row < height-1 && col < width) {
        localBlock[(localRow+2) * localWidth + (localCol+1)] = oldGrid[(row+1) * width + col];
    }
    
    // Left halo (if this work-item is in the first column of the block)
    if (localCol == 0 && row < height && col > 0) {
        localBlock[(localRow+1) * localWidth] = oldGrid[row * width + (col-1)];
    }
    
    // Right halo (if this work-item is in the last column of the block)
    if (localCol == get_local_size(0)-1 && row < height && col < width-1) {
        localBlock[(localRow+1) * localWidth + (localCol+2)] = oldGrid[row * width + (col+1)];
    }
    
    // Wait for all halo cells to be loaded
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Only update interior points of the grid
    if (row > 0 && row < height - 1 && col > 0 && col < width - 1) {
        // Apply Jacobi iteration using local memory
        newGrid[row * width + col] = 0.25f * (
            localBlock[(localRow) * localWidth + (localCol+1)] +     // top
            localBlock[(localRow+2) * localWidth + (localCol+1)] +   // bottom
            localBlock[(localRow+1) * localWidth + (localCol)] +     // left
            localBlock[(localRow+1) * localWidth + (localCol+2)]     // right
        );
    }
}