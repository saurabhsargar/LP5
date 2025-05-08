#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

// Merge function
void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;

    while (i <= mid && j <= right)
        temp[k++] = (arr[i] < arr[j]) ? arr[i++] : arr[j++];

    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];

    for (int i = 0; i < temp.size(); ++i)
        arr[left + i] = temp[i];
}

// Sequential Merge Sort
void mergeSortSeq(vector<int>& arr, int left, int right) {
    if (left >= right) return;

    int mid = (left + right) / 2;
    mergeSortSeq(arr, left, mid);
    mergeSortSeq(arr, mid + 1, right);
    merge(arr, left, mid, right);
}

// Parallel Merge Sort using OpenMP
void mergeSortParallel(vector<int>& arr, int left, int right, int depth = 0) {
    if (left >= right) return;

    int mid = (left + right) / 2;

    if (depth <= 4) {
        #pragma omp task shared(arr)
        mergeSortParallel(arr, left, mid, depth + 1);

        #pragma omp task shared(arr)
        mergeSortParallel(arr, mid + 1, right, depth + 1);

        #pragma omp taskwait
    } else {
        mergeSortSeq(arr, left, mid);
        mergeSortSeq(arr, mid + 1, right);
    }

    merge(arr, left, mid, right);
}

void printArray(const vector<int>& arr) {
    for (int val : arr) cout << val << " ";
    cout << "\n";
}

int main() {
    int n;
    cout << "Enter size of array: ";
    cin >> n;

    vector<int> original(n);
    cout << "Enter " << n << " elements:\n";
    for (int i = 0; i < n; ++i)
        cin >> original[i];

    vector<int> arr1 = original;
    vector<int> arr2 = original;

    // Sequential Merge Sort
    double start_seq = omp_get_wtime();
    mergeSortSeq(arr1, 0, n - 1);
    double end_seq = omp_get_wtime();

    // Parallel Merge Sort
    double start_par = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        mergeSortParallel(arr2, 0, n - 1);
    }
    double end_par = omp_get_wtime();

    cout << "\nSorted Array (Sequential):\n";
    printArray(arr1);
    cout << "\nSorted Array (Parallel):\n";
    printArray(arr2);

    cout << "\nTime taken by Sequential Merge Sort: " << (end_seq - start_seq) << " seconds";
    cout << "\nTime taken by Parallel Merge Sort  : " << (end_par - start_par) << " seconds\n";

    return 0;
}
