// Naive matrix multiplication kernel
__kernel void matrixMulNaive(__global const float* A,
                         __global const float* B,
                         __global float* C,
                         const int M, const int N, const int K) {
    // Get global position in the grid
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    // Check boundaries
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Optimized matrix multiplication kernel using local memory
__kernel void matrixMulOptimized(__global const float* A,
                             __global const float* B,
                             __global float* C,
                             const int M, const int N, const int K) {
    // Local memory for tiles of input matrices
    __local float Asub[16][16];
    __local float Bsub[16][16];
    
    // Block size
    const int bsize = 16;
    
    // Global position
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    // Local position within workgroup
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);
    
    // Initialize accumulator
    float sum = 0.0f;
    
    // Loop over all tiles needed to compute the element
    const int numTiles = (K + bsize - 1) / bsize;
    
    for (int t = 0; t < numTiles; ++t) {
        // Load one tile of A and B into local memory
        const int tileCol = t * bsize + localCol;
        const int tileRow = t * bsize + localRow;
        
        // Load elements into local memory with bounds checking
        if (row < M && tileCol < K)
            Asub[localRow][localCol] = A[row * K + tileCol];
        else
            Asub[localRow][localCol] = 0.0f;
            
        if (tileRow < K && col < N)
            Bsub[localRow][localCol] = B[tileRow * N + col];
        else
            Bsub[localRow][localCol] = 0.0f;
            
        // Synchronize to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Perform dot product for this tile
        for (int k = 0; k < bsize; ++k) {
            sum += Asub[localRow][k] * Bsub[k][localCol];
        }
        
        // Synchronize before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write the result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}