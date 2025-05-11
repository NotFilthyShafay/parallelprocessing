#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <CL/cl.h>

using namespace std;

class Matrix {
private:
    vector<float> data;
    size_t rows, cols;

public:
    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r* c, 0.0f) {}

    float& operator()(size_t i, size_t j) {
        return data[i * cols + j];
    }

    const float& operator()(size_t i, size_t j) const {
        return data[i * cols + j];
    }

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

    void randomize() {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<float> dis(0.0f, 1.0f);

        for (auto& val : data) {
            val = dis(gen);
        }
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

// Modify the savePerformanceData function to include workgroup size
void savePerformanceData(const string& filename, size_t matrixSize, double timeMs, 
                         const string& kernelType, size_t workgroupSize) {
    ofstream outfile(filename, ios_base::app);

    if (!outfile.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }

    // Write header if file is empty
    outfile.seekp(0, ios::end);
    if (outfile.tellp() == 0) {
        outfile << "MatrixSize,KernelType,WorkgroupSize,ExecutionTime_ms\n";
    }

    outfile << matrixSize << "," << kernelType << "," << workgroupSize << "," << timeMs << "\n";
    outfile.close();
}

int main(int argc, char* argv[]) {
    // Add a loop for workgroup sizes
    for (size_t workgroupSize : {8, 16, 32}) {
        for (size_t matrixSize = 1000; matrixSize < 4001; matrixSize += 1000) { 
            bool useOptimized = false; // Default to naive kernel

            if (argc > 1) {
                matrixSize = stoi(argv[1]);
            }

            if (argc > 2) {
                useOptimized = (stoi(argv[2]) != 0);
            }

            string kernelType = useOptimized ? "optimized" : "naive";

            cout << "Running OpenCL matrix multiplication with size: " << matrixSize << "x" << matrixSize
                 << " using " << kernelType << " kernel" 
                 << " with workgroup size: " << workgroupSize << "x" << workgroupSize << endl;

            // Create and initialize matrices on host
            Matrix A(matrixSize, matrixSize);
            Matrix B(matrixSize, matrixSize);
            Matrix C(matrixSize, matrixSize); // For results

            A.randomize();
            B.randomize();

            // Flatten matrices for OpenCL
            vector<float> hostA(matrixSize * matrixSize);
            vector<float> hostB(matrixSize * matrixSize);
            vector<float> hostC(matrixSize * matrixSize, 0.0f);

            for (size_t i = 0; i < matrixSize; ++i) {
                for (size_t j = 0; j < matrixSize; ++j) {
                    hostA[i * matrixSize + j] = A(i, j);
                    hostB[i * matrixSize + j] = B(i, j);
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
                cerr << "GPU not found, trying CPU..." << endl;
                err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
                checkError(err, "getting device");
            }

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

            // Create buffers
            cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * hostA.size(), NULL, &err);
            checkError(err, "creating buffer A");

            cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * hostB.size(), NULL, &err);
            checkError(err, "creating buffer B");

            cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * hostC.size(), NULL, &err);
            checkError(err, "creating buffer C");

            // Copy data to device
            err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, sizeof(float) * hostA.size(), hostA.data(), 0, NULL, NULL);
            checkError(err, "writing buffer A");

            err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, sizeof(float) * hostB.size(), hostB.data(), 0, NULL, NULL);
            checkError(err, "writing buffer B");

            // Load and compile kernel
            string kernelSource = readKernelSource("kernels.cl");
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

            // Create kernel
            string kernelName = useOptimized ? "matrixMulOptimized" : "matrixMulNaive";
            cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &err);
            checkError(err, "creating kernel");

            // Set kernel arguments
            err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
            checkError(err, "setting kernel arg 0");

            err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
            checkError(err, "setting kernel arg 1");

            err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
            checkError(err, "setting kernel arg 2");

            int M = matrixSize;
            err = clSetKernelArg(kernel, 3, sizeof(int), &M);
            checkError(err, "setting kernel arg 3");

            int N = matrixSize;
            err = clSetKernelArg(kernel, 4, sizeof(int), &N);
            checkError(err, "setting kernel arg 4");

            int K = matrixSize;
            err = clSetKernelArg(kernel, 5, sizeof(int), &K);
            checkError(err, "setting kernel arg 5");

            // Set work dimensions - use the workgroup size from the loop
            size_t localSize[2] = { workgroupSize, workgroupSize };
            size_t globalSize[2] = {
                ((matrixSize + localSize[0] - 1) / localSize[0]) * localSize[0],
                ((matrixSize + localSize[1] - 1) / localSize[1]) * localSize[1]
            };

            // Time the kernel execution
            auto startTime = chrono::high_resolution_clock::now();

            // Execute kernel
            err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
            checkError(err, "enqueuing kernel");

            // Wait for execution to complete
            err = clFinish(queue);
            checkError(err, "waiting for kernel");

            auto endTime = chrono::high_resolution_clock::now();
            chrono::duration<double, milli> duration = endTime - startTime;
            double timeMs = duration.count();

            cout << "Matrix multiplication completed in " << fixed << setprecision(2)
                 << timeMs << " ms" << endl;

            // Read back results
            err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float) * hostC.size(), hostC.data(), 0, NULL, NULL);
            checkError(err, "reading buffer C");

            // Transfer results to Matrix format
            for (size_t i = 0; i < matrixSize; ++i) {
                for (size_t j = 0; j < matrixSize; ++j) {
                    C(i, j) = hostC[i * matrixSize + j];
                }
            }

            // Use a single file for all results
            savePerformanceData("opencl_performance_all.csv", matrixSize, timeMs, kernelType, workgroupSize);

            // Calculate checksum as the sum of all elements in matrix C
            float checksum = 0.0f;
            for (size_t i = 0; i < matrixSize; ++i) {
                for (size_t j = 0; j < matrixSize; ++j) {
                    checksum += C(i, j);
                }
            }

            // Keep a single checksum file or append to an existing one
            ofstream checksumFile("opencl_checksums.txt", ios::app);
            checksumFile << "Matrix size: " << matrixSize << "x" << matrixSize 
                         << ", Kernel type: " << kernelType 
                         << ", Workgroup size: " << workgroupSize << "x" << workgroupSize
                         << ", Checksum: " << checksum << endl;
            checksumFile.close();

            // Clean up
            clReleaseMemObject(bufA);
            clReleaseMemObject(bufB);
            clReleaseMemObject(bufC);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }
    }
    return 0;
}