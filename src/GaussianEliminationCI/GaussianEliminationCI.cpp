#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <JC/util.h>
#include <JC/openCLUtil.hpp>

using namespace std;

// Optimized GPU Gaussian Elimination with Compute Intensity Control
class ComputeIntensiveGaussianElimination {
private:
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;

    cl::Kernel kernel_coalesced;
    cl::Kernel kernel_coalesced_intensive;
    cl::Kernel kernel_back;

    int n;
    size_t local_work_size;
    size_t max_local_memory;

public:
    ComputeIntensiveGaussianElimination(int matrix_size, cl::Device dev, cl::Context ctx, cl::CommandQueue q, cl::Program prog)
        : n(matrix_size), device(dev), context(ctx), queue(q), program(prog) {

        // Initialize kernels
        kernel_coalesced = cl::Kernel(program, "forwardEliminationCoalesced");
        kernel_coalesced_intensive = cl::Kernel(program, "forwardEliminationCoalescedIntensive");
        kernel_back = cl::Kernel(program, "backSubstitution");

        // Query device capabilities
        device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &local_work_size);
        device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &max_local_memory);

        local_work_size = min(local_work_size, (size_t)256);

        cout << "Compute Intensive solver initialized:" << endl;
        cout << "  Max work group size: " << local_work_size << endl;
        cout << "  Local memory: " << max_local_memory / 1024 << " KB" << endl;
    }

    // Original coalesced implementation
    _int64 solveCoalesced(cl::Buffer& matrix_buf, cl::Buffer& b_buf, cl::Buffer& x_buf) {
        chrono::time_point<chrono::system_clock> start = chrono::system_clock::now();

        for (int k = 0; k < n - 1; k++) {
            kernel_coalesced.setArg(0, matrix_buf);
            kernel_coalesced.setArg(1, b_buf);
            kernel_coalesced.setArg(2, n);
            kernel_coalesced.setArg(3, k);

            size_t remaining_rows = n - k - 1;
            size_t remaining_cols = n - k;

            if (remaining_rows == 0 || remaining_cols == 0) break;

            size_t total_elements = remaining_rows * remaining_cols;
            size_t global_size = ((total_elements + local_work_size - 1) / local_work_size) * local_work_size;

            queue.enqueueNDRangeKernel(kernel_coalesced, cl::NullRange,
                cl::NDRange(global_size),
                cl::NDRange(local_work_size));

            queue.finish();
        }

        solveBackSubstitution(matrix_buf, b_buf, x_buf);
        queue.finish();
        return chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - start).count();
    }

    // Compute intensive version with controllable loop count
    _int64 solveCoalescedIntensive(cl::Buffer& matrix_buf, cl::Buffer& b_buf, cl::Buffer& x_buf, int loop_count = 100) {
        chrono::time_point<chrono::system_clock> start = chrono::system_clock::now();

        for (int k = 0; k < n - 1; k++) {
            kernel_coalesced_intensive.setArg(0, matrix_buf);
            kernel_coalesced_intensive.setArg(1, b_buf);
            kernel_coalesced_intensive.setArg(2, n);
            kernel_coalesced_intensive.setArg(3, k);
            kernel_coalesced_intensive.setArg(4, loop_count); // Add loop count parameter

            size_t remaining_rows = n - k - 1;
            size_t remaining_cols = n - k;

            if (remaining_rows == 0 || remaining_cols == 0) break;

            size_t total_elements = remaining_rows * remaining_cols;
            size_t global_size = ((total_elements + local_work_size - 1) / local_work_size) * local_work_size;

            queue.enqueueNDRangeKernel(kernel_coalesced_intensive, cl::NullRange,
                cl::NDRange(global_size),
                cl::NDRange(local_work_size));

            queue.finish();
        }

        solveBackSubstitution(matrix_buf, b_buf, x_buf);
        queue.finish();
        return chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - start).count();
    }

    void solveBackSubstitution(cl::Buffer& matrix_buf, cl::Buffer& b_buf, cl::Buffer& x_buf) {
        kernel_back.setArg(0, matrix_buf);
        kernel_back.setArg(1, b_buf);
        kernel_back.setArg(2, x_buf);
        kernel_back.setArg(3, n);

        queue.enqueueNDRangeKernel(kernel_back, cl::NullRange,
            cl::NDRange(1),
            cl::NDRange(1));
    }
};

int main(int argc, char* argv[]) {
    try {
        string kernel_file("optimizedKernels.ocl");

        int PLATFORM_ID = defaultOrViaArgs(0, 'p', argc, argv);
        int DEVICE_ID = defaultOrViaArgs(0, 'd', argc, argv);
        int n = defaultOrViaArgs(512, 'n', argc, argv);

        // Parse loop count instead of compute intensity
        int loop_count = 100; // default
        for (int i = 1; i < argc - 1; i++) {
            if (strcmp(argv[i], "-c") == 0) {
                loop_count = atoi(argv[i + 1]);
                break;
            }
        }

        cout << "=== Compute Intensive Gaussian Elimination ===" << endl;
        cout << "Matrix size: " << n << "x" << n << endl;
        cout << "Loop count: " << loop_count << endl;

        // OpenCL initialization
        cl::Device device = jc::getDevice(PLATFORM_ID, DEVICE_ID, false);
        cl::Context context(device);
        cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
        cl::Program program = jc::buildProgram(kernel_file, context, device);

        ComputeIntensiveGaussianElimination solver(n, device, context, queue, program);

        // Allocate and initialize data
        float* matrix_original = new float[n * n];
        float* matrix_gpu = new float[n * n];
        float* matrix_intensive = new float[n * n];
        float* b_original = new float[n];
        float* b_gpu = new float[n];
        float* b_intensive = new float[n];
        float* x_cpu = new float[n];
        float* x_gpu = new float[n];
        float* x_intensive = new float[n];

        // Create test system
        srand(42);
        for (int i = 0; i < n; i++) {
            float rowSum = 0.0f;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    matrix_original[i * n + j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                    rowSum += abs(matrix_original[i * n + j]);
                }
            }
            matrix_original[i * n + i] = rowSum + 1.0f + ((float)rand() / RAND_MAX);
            b_original[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        }

        memcpy(matrix_gpu, matrix_original, n * n * sizeof(float));
        memcpy(matrix_intensive, matrix_original, n * n * sizeof(float));
        memcpy(b_gpu, b_original, n * sizeof(float));
        memcpy(b_intensive, b_original, n * sizeof(float));

        // CPU reference solution (simplified)
        cout << "\nRunning CPU reference..." << endl;
        auto cpu_start = chrono::system_clock::now();

        vector<vector<float>> aug(n, vector<float>(n + 1));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                aug[i][j] = matrix_original[i * n + j];
            }
            aug[i][n] = b_original[i];
        }

        for (int k = 0; k < n - 1; k++) {
            for (int i = k + 1; i < n; i++) {
                float factor = aug[i][k] / aug[k][k];
                for (int j = k; j <= n; j++) {
                    aug[i][j] -= factor * aug[k][j];
                }
            }
        }

        for (int i = n - 1; i >= 0; i--) {
            x_cpu[i] = aug[i][n];
            for (int j = i + 1; j < n; j++) {
                x_cpu[i] -= aug[i][j] * x_cpu[j];
            }
            x_cpu[i] /= aug[i][i];
        }

        auto cpu_time = chrono::duration_cast<chrono::microseconds>(
            chrono::system_clock::now() - cpu_start).count();
        cout << "CPU time: " << cpu_time << " us" << endl;

        // GPU solutions
        cl::Buffer matrix_buf(context, CL_MEM_READ_WRITE, n * n * sizeof(float));
        cl::Buffer b_buf(context, CL_MEM_READ_WRITE, n * sizeof(float));
        cl::Buffer x_buf(context, CL_MEM_WRITE_ONLY, n * sizeof(float));

        cl::Buffer matrix_intensive_buf(context, CL_MEM_READ_WRITE, n * n * sizeof(float));
        cl::Buffer b_intensive_buf(context, CL_MEM_READ_WRITE, n * sizeof(float));
        cl::Buffer x_intensive_buf(context, CL_MEM_WRITE_ONLY, n * sizeof(float));

        // Original coalesced version
        queue.enqueueWriteBuffer(matrix_buf, CL_TRUE, 0, n * n * sizeof(float), matrix_gpu);
        queue.enqueueWriteBuffer(b_buf, CL_TRUE, 0, n * sizeof(float), b_gpu);

        cout << "\nRunning original coalesced GPU solver..." << endl;
        _int64 gpu_time = solver.solveCoalesced(matrix_buf, b_buf, x_buf);
        queue.enqueueReadBuffer(x_buf, CL_TRUE, 0, n * sizeof(float), x_gpu);
        cout << "Original GPU time: " << gpu_time << " us" << endl;

        // Compute intensive version
        queue.enqueueWriteBuffer(matrix_intensive_buf, CL_TRUE, 0, n * n * sizeof(float), matrix_intensive);
        queue.enqueueWriteBuffer(b_intensive_buf, CL_TRUE, 0, n * sizeof(float), b_intensive);

        cout << "\nRunning compute intensive GPU solver..." << endl;
        _int64 intensive_time = solver.solveCoalescedIntensive(matrix_intensive_buf, b_intensive_buf, x_intensive_buf, loop_count);
        queue.enqueueReadBuffer(x_intensive_buf, CL_TRUE, 0, n * sizeof(float), x_intensive);
        cout << "Intensive GPU time: " << intensive_time << " us" << endl;

        // Verification
        bool verified_original = true;
        bool verified_intensive = true;
        float max_error_original = 0.0f;
        float max_error_intensive = 0.0f;

        for (int i = 0; i < n; i++) {
            float error_original = abs(x_cpu[i] - x_gpu[i]);
            float error_intensive = abs(x_cpu[i] - x_intensive[i]);

            max_error_original = max(max_error_original, error_original);
            max_error_intensive = max(max_error_intensive, error_intensive);

            if (error_original > 1e-3) verified_original = false;
            if (error_intensive > 1e-3) verified_intensive = false;
        }

        cout << "\n=== Results ===" << endl;
        cout << "Original verification: " << (verified_original ? "PASSED" : "FAILED") << endl;
        cout << "Intensive verification: " << (verified_intensive ? "PASSED" : "FAILED") << endl;
        cout << "Original max error: " << scientific << max_error_original << endl;
        cout << "Intensive max error: " << scientific << max_error_intensive << endl;

        // Performance analysis
        long long base_flops = (long long)n * n * n / 3 + n * n;
        long long intensive_flops = base_flops * loop_count; // Approximate

        float original_gflops = base_flops / (gpu_time / 1e6) / 1e9;
        float intensive_gflops = intensive_flops / (intensive_time / 1e6) / 1e9;

        cout << fixed << setprecision(2);
        cout << "\n=== Performance Analysis ===" << endl;
        cout << "Original GPU Performance: " << original_gflops << " GFLOPS" << endl;
        cout << "Intensive GPU Performance: " << intensive_gflops << " GFLOPS" << endl;
        cout << "Compute intensity effect: " << (float)intensive_time / gpu_time << "x time increase" << endl;

        // Memory bandwidth (original stays same, intensive should be similar memory but more compute)
        long long memory_ops = (long long)n * n * n * sizeof(float);
        float original_bandwidth = memory_ops / (gpu_time / 1e6) / 1e9;
        float intensive_bandwidth = memory_ops / (intensive_time / 1e6) / 1e9;

        cout << "Original Memory Bandwidth: " << original_bandwidth << " GB/s" << endl;
        cout << "Intensive Memory Bandwidth: " << intensive_bandwidth << " GB/s" << endl;
        cout << "Operational Intensity: " << (float)intensive_flops / memory_ops << " FLOPS/Byte" << endl;

        // Cleanup
        delete[] matrix_original;
        delete[] matrix_gpu;
        delete[] matrix_intensive;
        delete[] b_original;
        delete[] b_gpu;
        delete[] b_intensive;
        delete[] x_cpu;
        delete[] x_gpu;
        delete[] x_intensive;

        return 0;
    }
    catch (cl::Error& e) {
        cerr << "OpenCL Error: " << e.what() << ": " << jc::readableStatus(e.err()) << endl;
        return 3;
    }
    catch (exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 2;
    }
}