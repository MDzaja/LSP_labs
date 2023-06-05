#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <ctype.h>

#define ARR_SIZE 1024 * 1024
#define COL_ROW_SIZE 1024

void readFile(char *filename, int *arr, int size);
__device__ void mergeSort(int *arr, int left, int right);
__device__ void merge(int *arr, int left, int mid, int right);

__global__ void mergeSort_cuda(int *array, int iter)
{
    // 2D grid and block
    /*int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockDim.x * gridDim.x * y + x;*/
    // 1D grid and block
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int subseq_size = COL_ROW_SIZE * pow(2, iter);
    int left = idx * subseq_size;
    int right = (idx + 1) * subseq_size - 1;

    if(iter == 0) {
        // Perform the merge sort operation.
        mergeSort(array, left, right);
    } else {
        int mid = left + (right - left) / 2;

        // Merge the two subarrays
        merge(array, left, mid, right);
    }
    
}

int main(int argc, char **argv)
{
    // Check if the correct number of command line arguments are provided
    struct timeval begin, end;

    gettimeofday(&begin, 0);

    if (argc != 2)
    {
        printf("Usage: %s file1\n", argv[0]);
    }

    int arr_size = ARR_SIZE;
    int *array;
    // Allocate memory for the array.
    cudaError_t err = cudaMallocManaged(&array, arr_size * sizeof(int));

    if (err != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    // Read the array from a file here.
    readFile(argv[1], array, arr_size);

    // Perform the sort.
    for (int iter = 0; iter <= 10; iter++)
    {
        int threads = 1024 / pow(2, iter);
        printf("Iteration %d, Threads: %d\n", iter+1, threads);

        mergeSort_cuda<<<2, threads>>>(array, iter);

        // Wait for GPU to finish before accessing on host.
        cudaDeviceSynchronize();
    }

    // Check if the sequence is properly sorted
    int i;
    for (i = 0; i < arr_size - 1; i++)
    {
        if (array[i] > array[i + 1])
        {
            printf("Error in position %d between element %d and %d\n", i, array[i], array[i + 1]);
            break;
        }
    }
    if (i == (arr_size - 1))
    {   
        printf("Everything is OK!\n");
    }

    // Free memory.
    cudaFree(array);

    gettimeofday(&end, 0);

    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds * 1e-6;
    printf ("\nElapsed time = %.6f s\n", elapsed);

    return 0;
}

void readFile(char *filename, int *arr, int size)
{
    FILE *fp;
    int n = 0;

    // Open the binary file for reading
    fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        printf("Error opening file\n");
        exit(1);
    }

    // Read the size of the array from the file
    int *true_size = (int *)malloc(sizeof(int));
    size_t bytes_read = fread(true_size, sizeof(int), 1, fp);

    if (size != *true_size)
    {
        printf("Error in size of the array. Array should have %d elements, but has %d elements.\n", size, *true_size);
        exit(1);
    }

    // Read the numbers from the file and store them in the array
    while ((bytes_read = fread(&arr[n], sizeof(int), 1, fp)) == 1)
    {
        n++;
    }

    if (ferror(fp)) {
        exit(1);
    }

    // Close the file
    fclose(fp);
}

// Merge two sorted subarrays of the array
__device__ void merge(int *arr, int left, int mid, int right)
{
    int n_left = mid - left + 1;
    int n_right = right - mid;

    int *left_arr = (int *)malloc(n_left * sizeof(int));
    int *right_arr = (int *)malloc(n_right * sizeof(int));

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
    while (i < n_left && j < n_right)
    {
        if (left_arr[i] <= right_arr[j])
        {
            arr[k++] = left_arr[i++];
        }
        else
        {
            arr[k++] = right_arr[j++];
        }
    }

    // When we run out of elements in either L or M,
    // pick up the remaining elements and put in A[p..r]
    while (i < n_left)
    {
        arr[k++] = left_arr[i++];
    }

    while (j < n_right)
    {
        arr[k++] = right_arr[j++];
    }

    free(left_arr);
    free(right_arr);
}

// Divide the array into two subarrays, sort them and merge them
__device__ void mergeSort(int *arr, int left, int right)
{
    if (left < right)
    {

        // m is the point where the array is divided into two subarrays
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        // Merge the sorted subarrays
        merge(arr, left, mid, right);
    }
}
