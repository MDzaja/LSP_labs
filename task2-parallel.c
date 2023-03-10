#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>

int *arr;
pthread_mutex_t mutex;
int tasksIndex = 0;
int tasksSize = 0;

typedef struct Task
{
    int left;
    int mid;
    int right;
    bool done;
    struct Task *parent1;
    struct Task *parent2;
} Task;
Task *tasks;

// Merge two sorted subarrays of the array
void merge(int left, int mid, int right)
{

    // Create L ← A[p..q] and M ← A[q+1..r]
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
void mergeSort(int left, int right)
{
    if (left < right)
    {

        // m is the point where the array is divided into two subarrays
        int mid = left + (right - left) / 2;

        mergeSort(left, mid);
        mergeSort(mid + 1, right);

        // Merge the sorted subarrays
        merge(left, mid, right);
    }
}

void distributor(int left, int right, int num_workers)
{
    if (left < right)
    {
        if (num_workers == 1)
        {
            tasks = (Task *)malloc(sizeof(Task));
            tasksSize = 1;
            tasks[0].left = left;
            tasks[0].right = right;
            tasks[0].done = false;
            tasks[0].parent1 = NULL;
            tasks[0].parent2 = NULL;
            return;
        }

        int num_size = right - left + 1;
        int subarray_size = num_size / num_workers;
        if (subarray_size < 2)
        {
            subarray_size = 2;
        }

        int leafTasks = ceil((double)num_size / subarray_size);
        int treeHeight = ceil(log2(leafTasks));
        int maxTasks = pow(2, treeHeight + 1) - 1;
        tasks = (Task *)malloc(maxTasks * sizeof(Task));

        int index = 0;
        for (int step = subarray_size; step < num_size; step *= 2)
        {
            for (int i = left; i < right; i += step)
            {
                int r = i + step - 1;
                if (r > num_size)
                {
                    r = num_size;
                }
                tasks[index].left = i;
                tasks[index].right = r;
                tasks[index].done = false;
                tasks[index].parent1 = NULL;
                tasks[index].parent2 = NULL;
                for (int j = index - 1; j >= 0; j--)
                {
                    if (tasks[j].left == i)
                    {
                        tasks[index].parent1 = &tasks[j];
                        tasks[index].mid = tasks[j].right;
                        break;
                    }
                }
                for (int j = index - 1; j >= 0; j--)
                {
                    if (tasks[j].right == r)
                    {
                        tasks[index].parent2 = &tasks[j];
                        break;
                    }
                }
                index++;
            }
        }
        tasks[index].left = left;
        tasks[index].right = right;
        tasks[index].done = false;
        tasks[index].parent1 = &tasks[index - 2];
        tasks[index].parent2 = &tasks[index - 1];
        tasks[index].mid = tasks[index].parent1->right;

        tasksSize = index + 1;
    }
}

void *worker(void *arg)
{
    while (true)
    {
        pthread_mutex_lock(&mutex);
        if (tasksIndex >= tasksSize)
        {
            pthread_mutex_unlock(&mutex);
            return NULL;
        }
        Task *task = &tasks[tasksIndex];
        tasksIndex++;
        pthread_mutex_unlock(&mutex);

        if (task->parent1 != NULL)
        {
            while (!task->parent1->done)
            {
                // wait for parent1 task to finish
            }
        }
        if (task->parent2 != NULL)
        {
            while (!task->parent2->done)
            {
                // wait for parent2 task to finish
            }
        }

        if(task->parent1 == NULL) {
            mergeSort(task->left, task->right);
        } else {
            merge(task->left, task->mid, task->right);
        }

        task->done = true;
    }
}

bool is_sorted(int arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[])
{
    struct timeval begin, end;
    FILE *fp;
    int n = 0, size, i;

    // Correcting the user of the wrong usage of the program
    if (argc < 3)
    {
        printf("Usage: %s file1 worker_number\n", argv[0]);
        return 1;
    }

    // Open the binary file for reading
    fp = fopen(argv[1], "rb");
    if (fp == NULL)
    {
        printf("Error opening file\n");
        exit(1);
    }

    fread(&size, sizeof(int), 1, fp);

    printf("Size: %d \n", size);

    // Allocating enough space to save all the numbers read from a file
    arr = (int *)malloc(size * sizeof(int));

    // Read the numbers from the file
    while (fread(&arr[n], sizeof(int), 1, fp) == 1)
    {
        n++;
    }

    // Close the file
    fclose(fp);

    // Get the number of workers
    int num_workers = atoi(argv[2]);

    gettimeofday(&begin, 0);
    // Sort the numbers using parallel merge sort
    distributor(0, size - 1, num_workers);
    printf("tasksSize: %d\n", tasksSize);
    pthread_t threads[num_workers];
    pthread_mutex_init(&mutex, NULL);
    for (i = 0; i < num_workers; i++)
    {
        pthread_create(&threads[i], NULL, worker, NULL);
    }
    for (i = 0; i < num_workers; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // Calculate the execution time
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds*1e-6;
    printf("Execution time: %.3f seconds\n", elapsed);

    // Print the sorted numbers
    /*for (int i = 0; i < size; i++)
    {
        printf("%d\n", arr[i]);
    }*/

    // Check if the numbers are sorted
    for (i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            printf ("Error in position %d between element %d and %d\n", i, arr[i], arr[i+1]);
            break;
        }
    }
    if (i == (size - 1)) {
        printf("Everything is OK!\n");
    }
        

    return 0;
}
