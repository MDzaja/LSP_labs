#include <stdio.h>
#include <stdlib.h>

// Merge two sorted subarrays of the array
void merge(int arr[], int left, int mid, int right) {

    // Create L ← A[p..q] and M ← A[q+1..r]
    int n_left = mid - left + 1;
    int n_right = right - mid;

    int *left_arr = (int*)malloc(n_left * sizeof(int));
    int *right_arr = (int*)malloc(n_right * sizeof(int));

    for (int i = 0; i < n_left; i++)
        left_arr[i] = arr[left + i];
    for (int j = 0; j < n_right; j++)
        right_arr[j] = arr[mid + 1 + j];

    // Maintain current index of sub-arrays and main array
    int i, j, k;
    i = 0;
    j = 0;
    k = left;

    // Until we reach either end of either L or M, pick larger among
    // elements L and M and place them in the correct position at A[p..r]
    while (i < n_left && j < n_right) {
        if (left_arr[i] <= right_arr[j]) {
        arr[k++] = left_arr[i++];
        } else {
        arr[k++] = right_arr[j++];
        }
    }

    // When we run out of elements in either L or M,
    // pick up the remaining elements and put in A[p..r]
    while (i < n_left) {
        arr[k++] = left_arr[i++];
    }

    while (j < n_right) {
        arr[k++] = right_arr[j++];
    }

    free(left_arr);
    free(right_arr);

}

// Divide the array into two subarrays, sort them and merge them
void mergeSort(int arr[], int left, int right) {
    if (left < right) {

        // m is the point where the array is divided into two subarrays
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        // Merge the sorted subarrays
        merge(arr, left, mid, right);
    }
}

void parallelMergeSort(int arr[], int left, int right) {
    if (left < right) {

        // m is the point where the array is divided into two subarrays
        int mid = left + (right - left) / 2;

        //TODO

        // Merge the sorted subarrays
        merge(arr, left, mid, right);
    }
}

int main(int argc, char *argv[]) {
    FILE *fp;
    int n = 0, i, size;

    // Correcting the user of the wrong usage of the program
    if (argc < 2) {
        printf("Usage: %s file1 \n", argv[0]);
        return 1;
    }

    // Open the binary file for reading
    fp = fopen(argv[1], "rb");
    if (fp == NULL) {
        printf("Error opening file\n");
        exit(1);
    }

    fread(&size, sizeof(int), 1, fp);

    printf("Size: %d \n", size);

    //Allocating enough space to save all the numbers read from a file
    int *num = (int*)malloc(size * sizeof(int));

    // Read the numbers from the file
    while (fread(&num[n], sizeof(int), 1, fp) == 1) {
        n++;
    }

    // Close the file
    fclose(fp);

    //TODO

    // Sort the numbers using merge sort
    mergeSort(num, 0, size - 1);

    // Print the sorted numbers
    for (i = 0; i < n; i++) {
        printf("%d\n", num[i]);
    }

    return 0;
}
