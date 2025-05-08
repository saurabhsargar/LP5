#include <iostream>
#include <omp.h>
#include <climits> // for INT_MAX and INT_MIN

using namespace std;

int main()
{
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;
    int arr[n];
    cout << "Enter the elements: ";
    for (int i = 0; i < n; i++)
    {
        cin >> arr[i];
    }

    // Initialize variables
    int min_val = INT_MAX;
    int max_val = INT_MIN;
    int sum = 0;

// Parallel for loop with reduction
#pragma omp parallel for reduction(+ : sum) reduction(min : min_val) reduction(max : max_val)
    for (int i = 0; i < n; i++)
    {
        sum += arr[i]; // Local sum operation
        if (arr[i] < min_val)
        {
            min_val = arr[i]; // Local min operation
        }
        if (arr[i] > max_val)
        {
            max_val = arr[i]; // Local max operation
        }
    }

    double average = static_cast<double>(sum) / n; // Calculate average

    // Output the results
    cout << "Minimum value: " << min_val << endl;
    cout << "Maximum value: " << max_val << endl;
    cout << "Sum value: " << sum << endl;
    cout << "Average value: " << average << endl;

    return 0;
}



