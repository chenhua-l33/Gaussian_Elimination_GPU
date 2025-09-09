#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <JC/util.h>
#include <JC/openCLUtil.hpp>

/**
 * @file gaussianElimination.cpp
 * @brief Basic implementation of Gaussian Elimination using OpenCL
 *
 * This file implements both sequential and parallel (GPU) versions of
 * Gaussian Elimination for solving systems of linear equations (Ax = b).
 *
 * Features:
 * - Sequential CPU implementation with partial pivoting
 * - Basic GPU implementation using OpenCL
 * - Solution verification
 * - Performance measurement and analysis
 * - Well-conditioned test matrix generation
 *
 * The GPU implementation uses a naive approach where:
 * - Each row elimination is handled by a separate work item
 * - Back substitution is done sequentially to avoid race conditions
 * - No optimization techniques are applied (baseline for comparison)
 */

using namespace std;
bool PRESS_KEY_TO_CLOSE_WINDOW = false;

/**
 * @brief Sequential Gaussian Elimination with Partial Pivoting
 *
 * Solves the system Ax = b using Gaussian Elimination with partial pivoting.
 * This is the reference CPU implementation for correctness verification
 * and performance comparison.
 *
 * Algorithm:
 * 1. Forward elimination with partial pivoting
 * 2. Back substitution
 *
 * @param matrix Input matrix A [n x n]
 * @param b      Input vector b [n]
 * @param x      Output solution vector x [n]
 * @param n      Matrix dimension
 * @return       Execution time in microseconds
 * @throws runtime_error if matrix is singular
 */
_int64 sequentialGaussianElimination(float *matrix, float *b, float *x, int n)
{
    chrono::time_point<chrono::system_clock> start = chrono::system_clock::now();

    // Create augmented matrix [A|b]
    vector<vector<float>> augmented(n, vector<float>(n + 1));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            augmented[i][j] = matrix[i * n + j];
        }
        augmented[i][n] = b[i];
    }

    // Forward elimination with partial pivoting
    for (int k = 0; k < n - 1; k++)
    {
        // Find pivot
        int pivotRow = k;
        for (int i = k + 1; i < n; i++)
        {
            if (abs(augmented[i][k]) > abs(augmented[pivotRow][k]))
            {
                pivotRow = i;
            }
        }

        // Swap rows if needed
        if (pivotRow != k)
        {
            swap(augmented[k], augmented[pivotRow]);
        }

        // Check for zero pivot
        if (abs(augmented[k][k]) < 1e-10)
        {
            throw runtime_error("Matrix is singular or nearly singular");
        }

        // Eliminate column k
        for (int i = k + 1; i < n; i++)
        {
            float factor = augmented[i][k] / augmented[k][k];
            for (int j = k; j <= n; j++)
            {
                augmented[i][j] -= factor * augmented[k][j];
            }
        }
    }

    // Back substitution
    for (int i = n - 1; i >= 0; i--)
    {
        x[i] = augmented[i][n];
        for (int j = i + 1; j < n; j++)
        {
            x[i] -= augmented[i][j] * x[j];
        }
        x[i] /= augmented[i][i];
    }

    return chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - start).count();
}

/**
 * @brief Verify solution accuracy for Gaussian Elimination
 *
 * Checks if the computed solution x satisfies the original system Ax = b
 * within a specified tolerance. For each equation i:
 * |sum(A[i,j] * x[j]) - b[i]| <= tolerance
 *
 * @param matrix    Original coefficient matrix A [n x n]
 * @param b         Original RHS vector b [n]
 * @param x         Computed solution vector x [n]
 * @param n         Matrix dimension
 * @param tolerance Maximum allowed error (default: 1e-4)
 * @return true if solution is valid, false otherwise
 */
bool verifySolution(float *matrix, float *b, float *x, int n, float tolerance = 1e-4)
{
    for (int i = 0; i < n; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < n; j++)
        {
            sum += matrix[i * n + j] * x[j];
        }
        if (abs(sum - b[i]) > tolerance)
        {
            cout << "Verification failed at row " << i << ": expected " << b[i]
                 << ", got " << sum << ", diff = " << abs(sum - b[i]) << endl;
            return false;
        }
    }
    return true;
}

/**
 * @brief Generate a well-conditioned test matrix and RHS vector
 *
 * Creates a diagonally dominant matrix A and random RHS vector b.
 * Properties of the generated system:
 * - Matrix is strictly diagonally dominant (guarantees solution exists)
 * - Off-diagonal elements are random in [-1, 1]
 * - Diagonal elements are larger than sum of absolute row values
 * - RHS vector elements are random in [-5, 5]
 * - Fixed random seed for reproducibility
 *
 * @param matrix Output matrix A [n x n]
 * @param b      Output RHS vector b [n]
 * @param n      Matrix dimension
 */
void createTestSystem(float *matrix, float *b, int n)
{
    srand(42); // Fixed seed for reproducibility

    // Create diagonally dominant matrix
    for (int i = 0; i < n; i++)
    {
        float rowSum = 0.0f;
        for (int j = 0; j < n; j++)
        {
            if (i != j)
            {
                matrix[i * n + j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // [-1, 1]
                rowSum += abs(matrix[i * n + j]);
            }
        }
        // Make diagonal element larger than sum of off-diagonal elements
        matrix[i * n + i] = rowSum + 1.0f + ((float)rand() / RAND_MAX);

        // Create RHS vector
        b[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f; // [-5, 5]
    }
}

/**
 * @brief Main function implementing basic Gaussian Elimination comparison
 *
 * This program compares sequential CPU and parallel GPU implementations:
 * 1. Generates a well-conditioned test system
 * 2. Solves using sequential CPU implementation
 * 3. Solves using basic GPU implementation
 * 4. Verifies solutions and measures performance
 *
 * GPU Implementation Details:
 * - Uses naive parallelization strategy
 * - One work item per row for elimination
 * - Sequential back substitution
 * - No memory or algorithmic optimizations
 *
 * Performance Metrics:
 * - Execution time (microseconds)
 * - Speedup ratio (CPU/GPU)
 * - GFLOPS for both implementations
 * - GPU memory bandwidth utilization
 *
 * Command Line Arguments:
 * -p <platform ID>  OpenCL platform ID
 * -d <device ID>    OpenCL device ID
 * -n <size>         Matrix dimension
 * -h                Show help
 */
int main(int argc, char *argv[])
{
    if (argsContainsOption('h', argc, argv) || argsContainsUnknownOption("dhpns", argc, argv))
    {
        cout << "Usage: " << argv[0] << " -p <platform ID> -d <device ID> -n <matrix size>" << endl;
        return 0;
    }
    if (argc > 1)
        PRESS_KEY_TO_CLOSE_WINDOW = false;

    try
    {
        // Configuration
        string kernel_file("basicKernels.ocl"); // Use existing kernel file temporarily
        string forward_kernel_name("forwardElimination");
        string back_kernel_name("backSubstitution");

        int PLATFORM_ID = defaultOrViaArgs(0, 'p', argc, argv);
        int DEVICE_ID = defaultOrViaArgs(0, 'd', argc, argv);
        int n = defaultOrViaArgs(512, 'n', argc, argv); // matrix size

        cout << "Gaussian Elimination with matrix size: " << n << "x" << n << endl;

        // OpenCL initialization
        cl::Device device = jc::getDevice(PLATFORM_ID, DEVICE_ID, PRESS_KEY_TO_CLOSE_WINDOW);
        cl::Context context(device);
        cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
        cl::Program program = jc::buildProgram(kernel_file, context, device);

        // Allocate host memory
        float *matrix_original = new float[n * n];
        float *matrix_gpu = new float[n * n];
        float *b_original = new float[n];
        float *b_gpu = new float[n];
        float *x_cpu = new float[n];
        float *x_gpu = new float[n];

        // Initialize test system
        createTestSystem(matrix_original, b_original, n);

        // Copy for GPU computation
        memcpy(matrix_gpu, matrix_original, n * n * sizeof(float));
        memcpy(b_gpu, b_original, n * sizeof(float));

        // Sequential computation
        cout << "Running sequential Gaussian elimination..." << endl;
        _int64 cpu_time = sequentialGaussianElimination(matrix_original, b_original, x_cpu, n);
        cout << "Sequential time: " << cpu_time << " us" << endl;

        // Verify sequential solution
        if (!verifySolution(matrix_gpu, b_gpu, x_cpu, n))
        {
            throw runtime_error("Sequential solution verification failed!");
        }
        cout << "Sequential solution verified successfully!" << endl;

        // GPU computation - Naive implementation
        cout << "Running naive GPU Gaussian elimination..." << endl;

        // Allocate device memory
        cl::Buffer matrix_buffer(context, CL_MEM_READ_WRITE, n * n * sizeof(float));
        cl::Buffer b_buffer(context, CL_MEM_READ_WRITE, n * sizeof(float));
        cl::Buffer x_buffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float));

        // Transfer data to device
        queue.enqueueWriteBuffer(matrix_buffer, CL_TRUE, 0, n * n * sizeof(float), matrix_gpu);
        queue.enqueueWriteBuffer(b_buffer, CL_TRUE, 0, n * sizeof(float), b_gpu);

        chrono::time_point<chrono::system_clock> gpu_start = chrono::system_clock::now();

        // Forward elimination kernel
        cl::Kernel forward_kernel(program, forward_kernel_name.c_str());
        forward_kernel.setArg<cl::Buffer>(0, matrix_buffer);
        forward_kernel.setArg<cl::Buffer>(1, b_buffer);
        forward_kernel.setArg<cl_int>(2, n);

        // Back substitution kernel
        cl::Kernel back_kernel(program, back_kernel_name.c_str());
        back_kernel.setArg<cl::Buffer>(0, matrix_buffer);
        back_kernel.setArg<cl::Buffer>(1, b_buffer);
        back_kernel.setArg<cl::Buffer>(2, x_buffer);
        back_kernel.setArg<cl_int>(3, n);

        // Execute forward elimination (step by step for each pivot)
        for (int k = 0; k < n - 1; k++)
        {
            forward_kernel.setArg<cl_int>(3, k); // current pivot
            cl::NDRange global(n - k - 1);       // work on remaining rows
            queue.enqueueNDRangeKernel(forward_kernel, cl::NDRange(k + 1), global);
            queue.finish();
        }

        // Execute back substitution (single work item to avoid race conditions)
        cl::NDRange back_global(1);
        queue.enqueueNDRangeKernel(back_kernel, cl::NullRange, back_global);
        queue.finish();

        _int64 gpu_time = chrono::duration_cast<chrono::microseconds>(
                              chrono::system_clock::now() - gpu_start)
                              .count();

        // Transfer result back
        queue.enqueueReadBuffer(x_buffer, CL_TRUE, 0, n * sizeof(float), x_gpu);

        cout << "GPU time: " << gpu_time << " us" << endl;

        // Verify GPU solution
        if (!verifySolution(matrix_original, b_original, x_gpu, n))
        {
            cout << "GPU solution verification failed!" << endl;

            // Print first few values for debugging
            cout << "First 5 CPU vs GPU results:" << endl;
            for (int i = 0; i < min(5, n); i++)
            {
                cout << "x[" << i << "]: CPU=" << x_cpu[i] << ", GPU=" << x_gpu[i]
                     << ", diff=" << abs(x_cpu[i] - x_gpu[i]) << endl;
            }
        }
        else
        {
            cout << "GPU solution verified successfully!" << endl;
        }

        // Performance analysis
        const int PRECISION = 3;
        float speedup = (float)cpu_time / gpu_time;
        cout << fixed << setprecision(PRECISION);
        cout << "Speedup: " << speedup << "x" << endl;

        // Estimate FLOPS (n^3/3 for elimination + n^2 for back substitution)
        long long flops = (long long)n * n * n / 3 + n * n;
        float cpu_gflops = flops / (cpu_time / 1e6) / 1e9;
        float gpu_gflops = flops / (gpu_time / 1e6) / 1e9;

        cout << "CPU Performance: " << cpu_gflops << " GFLOPS" << endl;
        cout << "GPU Performance: " << gpu_gflops << " GFLOPS" << endl;

        // Memory bandwidth estimation (rough)
        long long memory_ops = (long long)n * n * n * sizeof(float); // Rough estimate
        float gpu_bandwidth = memory_ops / (gpu_time / 1e6) / 1e9;
        cout << "GPU Memory Bandwidth: " << gpu_bandwidth << " GB/s" << endl;

        // Cleanup
        delete[] matrix_original;
        delete[] matrix_gpu;
        delete[] b_original;
        delete[] b_gpu;
        delete[] x_cpu;
        delete[] x_gpu;

        if (PRESS_KEY_TO_CLOSE_WINDOW)
        {
            cout << endl
                 << "Press ENTER to close window...";
            char c = cin.get();
        }

        return 0;
    }
    catch (cl::Error &e)
    {
        cerr << e.what() << ": " << jc::readableStatus(e.err()) << endl;
        if (PRESS_KEY_TO_CLOSE_WINDOW)
        {
            cout << endl
                 << "Press ENTER to close window...";
            char c = cin.get();
        }
        return 3;
    }
    catch (exception &e)
    {
        cerr << e.what() << endl;
        if (PRESS_KEY_TO_CLOSE_WINDOW)
        {
            cout << endl
                 << "Press ENTER to close window...";
            char c = cin.get();
        }
        return 2;
    }
    catch (...)
    {
        cerr << "Unexpected error. Aborting!" << endl;
        if (PRESS_KEY_TO_CLOSE_WINDOW)
        {
            cout << endl
                 << "Press ENTER to close window...";
            char c = cin.get();
        }
        return 1;
    }
}