#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel for vector addition
_global_ void vectorAdd(int *A, int *B, int *C, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x; // Calculate global thread index
    if (index < n)
    {                                   // Ensure thread index is within bounds
        C[index] = A[index] + B[index]; // Add corresponding elements of vectors A and B
    }
}

int main()
{
    int *A, *B, *C, *d_A, *d_B, *d_C; // Host and device pointers for vectors
    int N;                            // Size of the vectors

    // Input the size of the vectors
    cout << "Enter the size of the vectors: ";
    cin >> N;

    size_t size = N * sizeof(int); // Calculate the size of the vectors in bytes

    // Allocate memory for vectors on the host (CPU)
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);

    // Allocate memory for vectors on the device (GPU)
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Input elements for the first vector (A)
    cout << "Enter the elements of the first vector (A): ";
    for (int i = 0; i < N; i++)
    {
        cin >> A[i];
    }

    // Input elements for the second vector (B)
    cout << "Enter the elements of the second vector (B): ";
    for (int i = 0; i < N; i++)
    {
        cin >> B[i];
    }

    // Copy data from host memory to device memory
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Launch the CUDA kernel to perform vector addition
    // Using 256 threads per block and calculating the number of blocks dynamically
    vectorAdd<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);

    // Copy the result from device memory back to host memory
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Output the result of the vector addition (showing the first 10 elements)
    cout << "Result of vector addition (first 10 elements): ";
    for (int i = 0; i < 10 && i < N; i++)
    {
        cout << C[i] << " "; // Print each element
    }
    cout << endl;

    // Free memory allocated on the host and device
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

// inputs:
//     Enter the size of the vectors : 5 
//     Enter the elements of the first vector(A)  : 1 2 3 4 5 
//     Enter the elements of the second vector(B) : 5 4 3 2 1

// output :
//     Result of vector addition (first 10 elements): 6 6 6 6 6