#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define ARR_SIZE 1024 * 1024
#define COL_ROW_SIZE 1024

typedef struct SeqWrapper {
    int startingColumnIndex;
    int *arr;
} SeqWrapper;

__host__ __device__ int getElementByIndex(SeqWrapper *wrapper, int i) {
    int rel_column_i = i / COL_ROW_SIZE + wrapper->startingColumnIndex;
    int i_in_column = i % COL_ROW_SIZE;

    int index = wrapper->startingColumnIndex + rel_column_i + i_in_column * COL_ROW_SIZE;
    if(index >= ARR_SIZE || index < 0) 
        printf("Index out of bounds: %d\n", index);

    return wrapper->arr[index];
}

__host__ __device__ void setElementByIndex(SeqWrapper *wrapper, int i, int value) {
    int rel_column_i = i / COL_ROW_SIZE - wrapper->startingColumnIndex;
    int i_in_column = i % COL_ROW_SIZE;

    int index = wrapper->startingColumnIndex + rel_column_i+ i_in_column * COL_ROW_SIZE;
    if(index >= ARR_SIZE || index < 0) 
        printf("Index out of bounds: %d\n", index);

    wrapper->arr[index] = value;
}

void readFile(char *filename, int *arr, int size);
void transpose(int* matrix, int* transposed);
__device__ void mergeSort(SeqWrapper wrapper, int left, int right);
__device__ void merge(SeqWrapper wrapper, int left, int mid, int right);

__global__ void mergeSort_cuda(int *array, int iter)
{
    // 2D grid and block
    /*int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockDim.x * gridDim.x * y + x;*/
    // 1D grid and block
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int subseq_size = COL_ROW_SIZE * pow(2, iter);

    SeqWrapper wrapper;
    int columnPerThread = pow(2, iter);
    wrapper.startingColumnIndex = idx * columnPerThread;
    wrapper.arr = array;

    // Perform the merge sort operation.
    mergeSort(wrapper, 0, subseq_size - 1);

    if (idx == 0) {
        printf("bla bla\n");
        printf("First element after merge sort: %d\n", array[0]);
    }
}

__global__ void merge_cuda(int *array, int iter)
{
    // 1D grid and block
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // m is the point where the array is divided into two subarrays
    int subseq_size = COL_ROW_SIZE * pow(2, iter);
    int mid = (subseq_size - 1) / 2;

    SeqWrapper wrapper;
    int column_per_thread = pow(2, iter);
    wrapper.startingColumnIndex = idx * column_per_thread;
    wrapper.arr = array;

    // Merge the two subarrays
    merge(wrapper, 0, mid, subseq_size-1);
}

int main(int argc, char **argv)
{
    // Check if the correct number of command line arguments are provided
    if (argc != 2)
    {
        printf("Usage: %s file1\n", argv[0]);
    }

    int *array, *transposed;
    // Allocate memory for the array.
    cudaMallocManaged(&array, ARR_SIZE * sizeof(int));
    cudaMallocManaged(&transposed, ARR_SIZE * sizeof(int));
    // Read the array from a file here.
    readFile(argv[1], array, ARR_SIZE);
    printf("First element: %d\n", array[0]);

    // Perform the sort.
    for (int iter = 0; iter <= 10; iter++)
    {
        int threads = 1024 / pow(2, iter);
        printf("Iteration %d, Threads: %d\n", iter+1, threads);
        if (iter == 0)
        {
            mergeSort_cuda<<<1, threads>>>(array, iter);
        }
        else
        {
            merge_cuda<<<1, threads>>>(array, iter);
        }
        //printf("First element after iteration %d: %d\n", iter+1, array[0]);
        // Wait for GPU to finish before accessing on host.
        cudaDeviceSynchronize();
    }

    // Transpose the array
    /*printf("Transposing\n");
    transpose(array, transposed);
    printf("Transposed\n");*/
    printf("bzvz\n");
    printf("%d prvi clan\n", array[0]);
    SeqWrapper wrapper;
    wrapper.startingColumnIndex = 0;
    wrapper.arr = array;

    // Check if the sequence is properly sorted
    int i;
    for (i = 0; i < ARR_SIZE - 1; i++)
    {
        printf("prvi if\n");
        if (getElementByIndex(&wrapper, i) > getElementByIndex(&wrapper, i + 1))
        {
            printf("unutra\n");
            printf("Error in position %d between element %d and %d\n", i, getElementByIndex(&wrapper, i), getElementByIndex(&wrapper, i + 1));
            break;
        }
        printf("drugi\n");
    }
    if (i == (ARR_SIZE - 1))
    {
        printf("Everything is OK!\n");
    }

    // Free memory.
    cudaFree(array);
    cudaFree(transposed);

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
    while (fread(&arr[n], sizeof(int), 1, fp) == 1)
    {
        n++;
    }

    // Close the file
    fclose(fp);
}

void transpose(int* matrix, int* transposed) {
    for (int i = 0; i < COL_ROW_SIZE; i++) {
        for (int j = 0; j < COL_ROW_SIZE; j++) {
            transposed[j*COL_ROW_SIZE + i] = matrix[i*COL_ROW_SIZE + j];
        }
    }
}

// Merge two sorted subarrays of the array
__device__ void merge(SeqWrapper wrapper, int left, int mid, int right)
{
    int n_left = mid - left + 1;
    int n_right = right - mid;

    int *left_arr = (int *)malloc(n_left * sizeof(int));
    int *right_arr = (int *)malloc(n_right * sizeof(int));

    for (int i = 0; i < n_left; i++)
        left_arr[i] = getElementByIndex(&wrapper, left + i);
    for (int j = 0; j < n_right; j++)
        right_arr[j] = getElementByIndex(&wrapper, mid + 1 + j);

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
            setElementByIndex(&wrapper, k++, left_arr[i++]);
        }
        else
        {
            setElementByIndex(&wrapper, k++, right_arr[j++]);
        }
    }

    // When we run out of elements in either L or M,
    // pick up the remaining elements and put in A[p..r]
    while (i < n_left)
    {
        setElementByIndex(&wrapper, k++, left_arr[i++]);
    }

    while (j < n_right)
    {
        setElementByIndex(&wrapper, k++, right_arr[j++]);
    }

    free(left_arr);
    free(right_arr);
}

// Divide the array into two subarrays, sort them and merge them
__device__ void mergeSort(SeqWrapper wrapper, int left, int right)
{
    if (left < right)
    {

        // m is the point where the array is divided into two subarrays
        int mid = left + (right - left) / 2;

        mergeSort(wrapper, left, mid);
        mergeSort(wrapper, mid + 1, right);

        // Merge the sorted subarrays
        merge(wrapper, left, mid, right);
    }
}

/*int calcIndex(int i, int idx, int iter) {
    int column_per_thread = pow(2, iter);
    int start_col_i = idx * column_per_thread;

    int rel_column_i = i / COL_ROW_SIZE - start_col_i;
    int i_in_column = i % COL_ROW_SIZE;

    return start_col_i + rel_column_i + i_in_column * COL_ROW_SIZE ;
}*/