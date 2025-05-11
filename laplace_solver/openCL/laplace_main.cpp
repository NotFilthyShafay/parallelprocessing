#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <CL/cl.h>


using namespace std;

class Grid {
private:
    vector<vector<float>> data;
    size_t width, height;

public:
    Grid(size_t h, size_t w, float initialValue = 0.0f) :
        height(h), width(w), data(h, vector<float>(w, initialValue)) {
    }

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
            data[height - 1][j] = -5.0f;
        }

        // Left and right boundaries (0V)
        for (size_t i = 1; i < height - 1; ++i) {
            data[i][0] = 0.0f;
            data[i][width - 1] = 0.0f;
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


// Function to check OpenCL errors
void checkError(cl_int err, const string& operation) {
    if (err != CL_SUCCESS) {
        cerr << "Error during operation " << operation << ": " << err << endl;
        exit(EXIT_FAILURE);
    }
}

// Read kernel source code from file
string readKernelSource(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Could not open kernel file: " << filename << endl;
        exit(EXIT_FAILURE);
    }

    stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Function to save performance data
void savePerformanceData(const string& filename, size_t gridSize, double timeMs,
    int iterations, size_t workgroupSize) {
    ofstream outfile(filename, ios_base::app);

    if (!outfile.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }

    // Write header if file is empty
    outfile.seekp(0, ios::end);
    if (outfile.tellp() == 0) {
        outfile << "GridSize,WorkgroupSize,Iterations,ExecutionTime_ms,Implementation\n";
    }

    outfile << gridSize << "," << workgroupSize << "," << iterations << "," << timeMs << ",OpenCL\n";
    outfile.close();
}

int main(int argc, char* argv[]) {
    // Default parameters
    vector<size_t> gridSizes = { 256, 512, 1024 }; // Grid sizes to test
    float tolerance = 1e-4;
    int maxIterations = 10000;
    vector<size_t> workgroupSizes = { 8, 16, 32 }; // Workgroup sizes to test
    bool useOptimized = true; // Use optimized kernel by default
    bool runSingleConfig = false;

    // Parse command line arguments to allow running a specific configuration if desired
    if (argc > 1) {
        if (argc >= 4) { // If we have enough arguments for a specific configuration
            runSingleConfig = true;
            gridSizes = { static_cast<size_t>(stoi(argv[1])) };
            workgroupSizes = { static_cast<size_t>(stoi(argv[2])) };
            useOptimized = (stoi(argv[3]) != 0);

            if (argc > 4) {
                tolerance = stof(argv[4]);
            }

            if (argc > 5) {
                maxIterations = stoi(argv[5]);
            }
        }
    }

    cout << "OpenCL Laplace Solver Benchmark" << endl;
    cout << "===============================" << endl;
    cout << "Tolerance: " << tolerance << ", Max iterations: " << maxIterations << endl;
    cout << "Using " << (useOptimized ? "optimized" : "basic") << " kernel" << endl;
    cout << endl;

    // Create output directory if it doesn't exist
    system("mkdir -p results");

    // Iterate through all grid sizes
    for (size_t gridSize : gridSizes) {
        cout << "========== Grid Size: " << gridSize << "x" << gridSize << " ==========" << endl;

        // Iterate through all workgroup sizes
        for (size_t workgroupSize : workgroupSizes) {
            string kernelType = useOptimized ? "optimized" : "basic";

            cout << "Running with workgroup size: " << workgroupSize << "x" << workgroupSize << endl;

            // Create host grid
            Grid hostGrid(gridSize, gridSize);
            hostGrid.applyBoundaryConditions();

            // Flatten grid for OpenCL
            vector<float> flatGrid(gridSize * gridSize);
            for (size_t i = 0; i < gridSize; ++i) {
                for (size_t j = 0; j < gridSize; ++j) {
                    flatGrid[i * gridSize + j] = hostGrid.at(i, j);
                }
            }

            // OpenCL setup
            cl_int err;

            // Get platform
            cl_platform_id platform;
            err = clGetPlatformIDs(1, &platform, NULL);
            checkError(err, "getting platform");

            // Get device
            cl_device_id device;
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
            if (err != CL_SUCCESS) {
                cout << "GPU not found, trying CPU..." << endl;
                err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
                checkError(err, "getting device");
            }

            // Get device name for reporting
            char deviceName[256];
            clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
            cout << "Using device: " << deviceName << endl;

            // Create context
            cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
            checkError(err, "creating context");

            // Create command queue
#ifdef CL_VERSION_2_0
            cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
#else
            cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
#endif
            checkError(err, "creating command queue");

            // Create buffers for old and new grid
            cl_mem oldGridBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * flatGrid.size(), NULL, &err);
            checkError(err, "creating old grid buffer");

            cl_mem newGridBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * flatGrid.size(), NULL, &err);
            checkError(err, "creating new grid buffer");

            // Buffer for block-wise max differences
            size_t localSize[2] = { workgroupSize, workgroupSize };
            size_t globalSize[2] = {
                ((gridSize + localSize[0] - 1) / localSize[0]) * localSize[0],
                ((gridSize + localSize[1] - 1) / localSize[1]) * localSize[1]
            };

            size_t numBlocks = (globalSize[0] / localSize[0]) * (globalSize[1] / localSize[1]);
            vector<float> blockMaxDiffs(numBlocks);
            cl_mem maxDiffsBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * numBlocks, NULL, &err);
            checkError(err, "creating max diffs buffer");

            // Copy initial grid to device
            err = clEnqueueWriteBuffer(queue, oldGridBuffer, CL_TRUE, 0, sizeof(float) * flatGrid.size(),
                flatGrid.data(), 0, NULL, NULL);
            checkError(err, "writing initial grid");

            err = clEnqueueWriteBuffer(queue, newGridBuffer, CL_TRUE, 0, sizeof(float) * flatGrid.size(),
                flatGrid.data(), 0, NULL, NULL);
            checkError(err, "writing initial grid copy");

            // Load and compile kernel
            string kernelSource = readKernelSource("laplace_kernel.cl");
            const char* kernelSourcePtr = kernelSource.c_str();
            size_t kernelSourceSize = kernelSource.size();

            cl_program program = clCreateProgramWithSource(context, 1, &kernelSourcePtr, &kernelSourceSize, &err);
            checkError(err, "creating program");

            err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
            if (err != CL_SUCCESS) {
                // If there was a build error, get the build log
                size_t logSize;
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
                vector<char> log(logSize);
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), NULL);
                cerr << "Kernel build error: " << log.data() << endl;
                exit(EXIT_FAILURE);
            }

            // Create kernels
            cl_kernel jacobiKernel;
            if (useOptimized) {
                jacobiKernel = clCreateKernel(program, "laplaceJacobiOptimizedKernel", &err);
            }
            else {
                jacobiKernel = clCreateKernel(program, "laplaceJacobiKernel", &err);
            }
            checkError(err, "creating Jacobi kernel");

            cl_kernel boundaryKernel = clCreateKernel(program, "setBoundaryConditionsKernel", &err);
            checkError(err, "creating boundary conditions kernel");

            cl_kernel maxDiffKernel = clCreateKernel(program, "computeMaxDifferenceKernel", &err);
            checkError(err, "creating max diff kernel");

            // Set boundary kernel arguments
            err = clSetKernelArg(boundaryKernel, 0, sizeof(cl_mem), &oldGridBuffer);
            checkError(err, "setting boundary kernel arg 0");

            err = clSetKernelArg(boundaryKernel, 1, sizeof(int), &gridSize);
            checkError(err, "setting boundary kernel arg 1");

            err = clSetKernelArg(boundaryKernel, 2, sizeof(int), &gridSize);
            checkError(err, "setting boundary kernel arg 2");

            // Time the kernel execution
            auto startTime = chrono::high_resolution_clock::now();

            // Apply boundary conditions initially
            size_t boundaryGlobalSize = max(gridSize, gridSize);
            err = clEnqueueNDRangeKernel(queue, boundaryKernel, 1, NULL, &boundaryGlobalSize, NULL, 0, NULL, NULL);
            checkError(err, "enqueuing boundary kernel");

            // Wait for boundary initialization to complete
            err = clFinish(queue);
            checkError(err, "waiting for boundary kernel");

            // Main iteration loop
            float maxDiff = tolerance + 1.0f;
            int iterations = 0;

            while (maxDiff > tolerance && iterations < maxIterations) {
                // Set Jacobi kernel arguments
                if (useOptimized) {
                    size_t localMemSize = (workgroupSize + 2) * (workgroupSize + 2) * sizeof(float);

                    err = clSetKernelArg(jacobiKernel, 0, sizeof(cl_mem), &oldGridBuffer);
                    checkError(err, "setting Jacobi kernel arg 0");

                    err = clSetKernelArg(jacobiKernel, 1, sizeof(cl_mem), &newGridBuffer);
                    checkError(err, "setting Jacobi kernel arg 1");

                    err = clSetKernelArg(jacobiKernel, 2, sizeof(int), &gridSize);
                    checkError(err, "setting Jacobi kernel arg 2");

                    err = clSetKernelArg(jacobiKernel, 3, sizeof(int), &gridSize);
                    checkError(err, "setting Jacobi kernel arg 3");

                    err = clSetKernelArg(jacobiKernel, 4, localMemSize, NULL);
                    checkError(err, "setting Jacobi kernel arg 4 (local memory)");
                }
                else {
                    err = clSetKernelArg(jacobiKernel, 0, sizeof(cl_mem), &oldGridBuffer);
                    checkError(err, "setting Jacobi kernel arg 0");

                    err = clSetKernelArg(jacobiKernel, 1, sizeof(cl_mem), &newGridBuffer);
                    checkError(err, "setting Jacobi kernel arg 1");

                    err = clSetKernelArg(jacobiKernel, 2, sizeof(int), &gridSize);
                    checkError(err, "setting Jacobi kernel arg 2");

                    err = clSetKernelArg(jacobiKernel, 3, sizeof(int), &gridSize);
                    checkError(err, "setting Jacobi kernel arg 3");
                }

                // Execute Jacobi kernel
                err = clEnqueueNDRangeKernel(queue, jacobiKernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
                checkError(err, "enqueuing Jacobi kernel");

                // Apply boundary conditions to new grid
                err = clSetKernelArg(boundaryKernel, 0, sizeof(cl_mem), &newGridBuffer);
                checkError(err, "setting boundary kernel arg 0 (new grid)");

                err = clEnqueueNDRangeKernel(queue, boundaryKernel, 1, NULL, &boundaryGlobalSize, NULL, 0, NULL, NULL);
                checkError(err, "enqueuing boundary kernel");

                // Calculate maximum difference
                err = clSetKernelArg(maxDiffKernel, 0, sizeof(cl_mem), &oldGridBuffer);
                checkError(err, "setting max diff kernel arg 0");

                err = clSetKernelArg(maxDiffKernel, 1, sizeof(cl_mem), &newGridBuffer);
                checkError(err, "setting max diff kernel arg 1");

                err = clSetKernelArg(maxDiffKernel, 2, sizeof(cl_mem), &maxDiffsBuffer);
                checkError(err, "setting max diff kernel arg 2");

                err = clSetKernelArg(maxDiffKernel, 3, workgroupSize * workgroupSize * sizeof(float), NULL);
                checkError(err, "setting max diff kernel arg 3 (local memory)");

                err = clSetKernelArg(maxDiffKernel, 4, sizeof(int), &gridSize);
                checkError(err, "setting max diff kernel arg 4");

                err = clSetKernelArg(maxDiffKernel, 5, sizeof(int), &gridSize);
                checkError(err, "setting max diff kernel arg 5");

                err = clEnqueueNDRangeKernel(queue, maxDiffKernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
                checkError(err, "enqueuing max diff kernel");

                // Read back block differences
                err = clEnqueueReadBuffer(queue, maxDiffsBuffer, CL_TRUE, 0, sizeof(float) * numBlocks,
                    blockMaxDiffs.data(), 0, NULL, NULL);
                checkError(err, "reading max diff results");

                // Find global maximum
                maxDiff = 0.0f;
                for (float diff : blockMaxDiffs) {
                    maxDiff = max(maxDiff, diff);
                }

                // Swap old and new grid buffers
                cl_mem temp = oldGridBuffer;
                oldGridBuffer = newGridBuffer;
                newGridBuffer = temp;

                iterations++;

                // Print progress occasionally
                if (iterations % 500 == 0) {
                    cout << "Iteration " << iterations << ", max diff: " << maxDiff << endl;
                }
            }

            // Wait for all operations to complete
            err = clFinish(queue);
            checkError(err, "completing all operations");

            auto endTime = chrono::high_resolution_clock::now();
            chrono::duration<double, milli> duration = endTime - startTime;
            double timeMs = duration.count();

            cout << "Completed in " << iterations << " iterations, final diff: " << maxDiff << endl;
            cout << "Time: " << fixed << setprecision(2) << timeMs << " ms" << endl << endl;

            // Read back the final grid
            err = clEnqueueReadBuffer(queue, oldGridBuffer, CL_TRUE, 0, sizeof(float) * flatGrid.size(),
                flatGrid.data(), 0, NULL, NULL);
            checkError(err, "reading final grid");

            // Transfer back to Grid format
            for (size_t i = 0; i < gridSize; ++i) {
                for (size_t j = 0; j < gridSize; ++j) {
                    hostGrid.at(i, j) = flatGrid[i * gridSize + j];
                }
            }

            // Save solution to CSV file
            string outputFilename = "C:/Users/shaff/OneDrive/Desktop/results/laplace_opencl_" + to_string(gridSize) + "_" +
                to_string(workgroupSize) + "_" + kernelType + ".csv";
            hostGrid.saveToFile(outputFilename);

            // Save performance data
            savePerformanceData("C:/Users/shaff/OneDrive/Desktop/laplace_opencl_performance.csv", gridSize, timeMs, iterations, workgroupSize);

            // Calculate statistics for verification
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

            // Save statistics
            string statsFilename = "C:/Users/shaff/OneDrive/Desktop/results/laplace_opencl_stats_" + to_string(gridSize) + "_" +
                to_string(workgroupSize) + "_" + kernelType + ".txt";
            ofstream statsFile(statsFilename);
            statsFile << "Grid size: " << gridSize << "x" << gridSize << endl;
            statsFile << "Workgroup size: " << workgroupSize << "x" << workgroupSize << endl;
            statsFile << "Kernel type: " << kernelType << endl;
            statsFile << "Iterations: " << iterations << endl;
            statsFile << "Min value: " << min << endl;
            statsFile << "Max value: " << max << endl;
            statsFile << "Mean value: " << mean << endl;
            statsFile << "Sum: " << sum << endl;
            statsFile << "Execution time (ms): " << timeMs << endl;
            statsFile.close();

            // Clean up
            clReleaseMemObject(oldGridBuffer);
            clReleaseMemObject(newGridBuffer);
            clReleaseMemObject(maxDiffsBuffer);
            clReleaseKernel(jacobiKernel);
            clReleaseKernel(boundaryKernel);
            clReleaseKernel(maxDiffKernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }
    }

    cout << "All benchmarks completed successfully!" << endl;
    cout << "Results saved in the 'results' directory." << endl;

    return 0;
}