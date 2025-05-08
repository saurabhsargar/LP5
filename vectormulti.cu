#include <iostream>
#include <cuda_runtime.h>
using namespace std;

_global_ void matrixMultiply(int *A, int *B, int *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Calculate row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Calculate column index

    if (row < N && col < N)
    {
        int sum = 0;
        for (int k = 0; k < N; k++)
        { // Multiply and accumulate
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum; // Store result
    }
}

int main()
{
    int N;
    cout << "Enter matrix size (N x N): ";
    cin >> N;

    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(int);

    // Allocate memory for host and device
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Input matrices
    cout << "Enter elements of matrix A:" << endl;
    for (int i = 0; i < N * N; i++)
        cin >> A[i];
    cout << "Enter elements of matrix B:" << endl;
    for (int i = 0; i < N * N; i++)
        cin >> B[i];

    // Copy matrices to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Output result
    cout << "Resultant matrix C:" << endl;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cout << C[i * N + j] << " ";
        }
        cout << endl;
    }

    // Free memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

// inputs

//     Enter matrix size(N x N) : 3
//     Enter elements of matrix A :
//      1 2 3
//      4 5 6
//      7 8 9

//     Enter elements of matrix B :
//      9 8 7
//      6 5 4
//      3 2 1

// output
//     Resultant matrix C:
//     30 24 18
//     84 69 54
//     138 114 90