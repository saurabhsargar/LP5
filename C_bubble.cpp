#include<iostream>
#include<omp.h>
using namespace std;

void bubble(int array[], int n){
    for (int i = 0; i < n - 1; i++){
        for (int j = 0; j < n - i - 1; j++){
            if (array[j] > array[j + 1])
                swap(array[j], array[j + 1]);
        }
    }
}

void pBubble(int array[], int n){
    for(int i = 0; i < n; ++i){    
        #pragma omp for
        for (int j = 1; j < n; j += 2){
            if (array[j] < array[j-1])
                swap(array[j], array[j - 1]);
        }

        #pragma omp barrier

        #pragma omp for
        for (int j = 2; j < n; j += 2){
            if (array[j] < array[j-1])
                swap(array[j], array[j - 1]);
        }

        #pragma omp barrier
    }
}

void printArray(int arr[], int n){
    for(int i = 0; i < n; i++)
        cout << arr[i] << " ";
    cout << "\n";
}

int main(){
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    int *arr = new int[n];
    int *brr = new int[n];
    double start_time, end_time;

    cout << "Enter " << n << " elements:\n";
    for(int i = 0; i < n; i++) {
        cin >> arr[i];
        brr[i] = arr[i]; // Copy to parallel array
    }

    // Sequential sort
    start_time = omp_get_wtime();
    bubble(arr, n);
    end_time = omp_get_wtime();     
    cout << "\nSequential Bubble Sort took: " << end_time - start_time << " seconds.\n";
    printArray(arr, n);
    
    // Parallel sort
    start_time = omp_get_wtime();
    #pragma omp parallel
    pBubble(brr, n);
    end_time = omp_get_wtime();     
    cout << "\nParallel Bubble Sort took: " << end_time - start_time << " seconds.\n";
    printArray(brr, n);

    delete[] arr;
    delete[] brr;
    return 0;
}
