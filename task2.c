#include <stdio.h>
#include <stdlib.h>

// Merge two subarrays L and M into arr
void merge(int arr[], int p, int q, int r) {

    // Create L ← A[p..q] and M ← A[q+1..r]
    int n1 = q - p + 1;
    int n2 = r - q;

    int *L = (int*)malloc(n1 * sizeof(int));
    int *M = (int*)malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++)
        L[i] = arr[p + i];
    for (int j = 0; j < n2; j++)
        M[j] = arr[q + 1 + j];

    // Maintain current index of sub-arrays and main array
    int i, j, k;
    i = 0;
    j = 0;
    k = p;

    // Until we reach either end of either L or M, pick larger among
    // elements L and M and place them in the correct position at A[p..r]
    while (i < n1 && j < n2) {
        if (L[i] <= M[j]) {
        arr[k] = L[i];
        i++;
        } else {
        arr[k] = M[j];
        j++;
        }
        k++;
    }

    // When we run out of elements in either L or M,
    // pick up the remaining elements and put in A[p..r]
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = M[j];
        j++;
        k++;
    }

    free(L);
    free(M);

}

// Divide the array into two subarrays, sort them and merge them
void mergeSort(int arr[], int l, int r) {
    if (l < r) {

        // m is the point where the array is divided into two subarrays
        int m = l + (r - l) / 2;

        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);

        // Merge the sorted subarrays
        merge(arr, l, m, r);
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

    // Sort the numbers using merge sort
    mergeSort(num, 0, size - 1);

    // Print the sorted numbers
    for (i = 0; i < n; i++) {
        printf("%d\n", num[i]);
    }

    return 0;
}
