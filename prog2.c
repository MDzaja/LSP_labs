#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include "prog2monitor.h"
#include "prog2const.h"

typedef struct Task
{
    int left;
    int mid;
    int right;
    bool done;
    struct Task *leftParent;
    struct Task *rightParent;
} Task;
typedef struct ThreadArgs
{
    int index;
    int num_workers;
} ThreadArgs;

int *statusWorker;
int *statusDistributor;
Monitor m;
int *arr, arr_size;
char *filename;

void *thread_work(void *arg);
void distributor(int num_workers);
void worker(int wIndex);
int readFile(char *filename);
Task* createTasks(int *taskArrSize, int left, int right, int num_workers);
bool isSorted(int arr[], int size);
void merge(int left, int mid, int right);
void mergeSort(int left, int right);

int main(int argc, char *argv[])
{
    struct timeval begin, end;
    int i;

    if (argc != 3)
    {
        printf("Usage: %s file1 thread_number\n", argv[0]);
        return 1;
    }
    int num_workers = atoi(argv[2]);
    /*
    MAIN THREAD TASK 1
    to get the binary file name by processing the command line and storing it in the shared region
    */
    filename = argv[1];

    statusWorker = malloc(num_workers * sizeof(int));
    statusDistributor = malloc(sizeof(int));
    m.workerStateArr = malloc(num_workers * sizeof(WorkerState));
    m.workerTaskArr = malloc(num_workers * sizeof(WorkerTask));
    initialization(num_workers, &m);

    gettimeofday(&begin, 0);
    /*
    MAIN THREAD TASK 2
    to create the distributor and the worker threads and wait for their termination
    */
    pthread_t threads[num_workers + 1];
    ThreadArgs thread_args[num_workers + 1];
    for (i = 0; i < num_workers + 1; i++)
    {
        thread_args[i].index = i;
        if (i == 0)
        {
            thread_args[i].num_workers = num_workers;
        }
        pthread_create(&threads[i], NULL, thread_work, (void *)&thread_args[i]);
    }
    for (i = 0; i < num_workers + 1; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // Calculate the execution time
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds * 1e-6;
    printf("Execution time: %.3f seconds\n", elapsed);

    // Print the sorted numbers
    /*for (int i = 0; i < arr_size; i++)
    {
        printf("%d\n", arr[i]);
    }*/

    /*
    MAIN THREAD TASK 3
    to check if the sequence is properly sorted
    */
    for (i = 0; i < arr_size - 1; i++)
    {
        if (arr[i] > arr[i + 1])
        {
            printf("Error in position %d between element %d and %d\n", i, arr[i], arr[i + 1]);
            break;
        }
    }
    if (i == (arr_size - 1))
    {
        printf("Everything is OK!\n");
    }

    return 0;
}

void *thread_work(void *arg)
{
    ThreadArgs *args = (ThreadArgs *)arg;
    if (args->index == 0)
    {
        distributor(args->num_workers);
    }
    else
    {
        worker(args->index - 1);
    }
    pthread_exit(NULL);
}

void distributor(int num_workers)
{

    /*
    DISTRIBUTOR THREAD TASK 1
    to read the sequence of integers from the binary file
    */
    arr_size = readFile(filename);
    /*
    DISTRIBUTOR THREAD TASK 2
    to distribute sub-sequences of it to the worker threads
    */
    int taskArrSize;
    Task *taskArr = createTasks(&taskArrSize, 0, arr_size - 1, num_workers);
    int workerIdToTaskIndexMap[num_workers];
    for (int i = 0; i < num_workers; i++)
    {
        workerIdToTaskIndexMap[i] = -1;
    }
    int currentTaskIndex = 0;
    int availableWorkers[num_workers];

    while (currentTaskIndex < taskArrSize)
    {
        waitForAtLeastOneAvailableWorker(availableWorkers, num_workers, &m);
        for (int i = 0; i < num_workers; i++)
        {
            if (availableWorkers[i] == AVAILABLE)
            {
                if (workerIdToTaskIndexMap[i] != -1)
                {
                    taskArr[workerIdToTaskIndexMap[i]].done = true;
                }

                Task task = taskArr[currentTaskIndex];
                if ((task.leftParent == NULL || task.leftParent->done) && (task.rightParent == NULL || task.rightParent->done))
                {
                    workerIdToTaskIndexMap[i] = currentTaskIndex++;
                    bool leafTask = task.leftParent == NULL;
                    setWorkerTask(i, leafTask, task.left, task.mid, task.right, &m);
                    if (currentTaskIndex >= taskArrSize)
                    {
                        break;
                    }
                }
            }
        }
    }

    commandAllWorkersToTerminate(num_workers, &m);
    // Wait for worker to terminate
    waitForAllWorkersToTerminate(num_workers, &m);
}

void worker(int wIndex)
{
    requestNewTask(wIndex, &m);
    while (true)
    {
        /*
        WORKER THREAD TASK 1
        to request a sub-sequence of the sequence of integers
        */
        WorkerTask wTask = waitForNewTask(wIndex, &m);
        if (wTask.left == -1)
        {
            break;
        }
        /*
        WORKER THREAD TASK 2
        to sort it
        */
        if (wTask.leafTask)
        {
            mergeSort(wTask.left, wTask.right);
        }
        else
        {
            merge(wTask.left, wTask.mid, wTask.right);
        }
        /*
        WORKER THREAD TASK 3
        to let the distributor thread know the work is done
        */
        requestNewTask(wIndex, &m);
    }

    notifyWorkerTermination(wIndex, &m);
}

int readFile(char *filename)
{
    FILE *fp;
    int n = 0, size;

    // Open the binary file for reading
    fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        printf("Error opening file\n");
        exit(1);
    }

    __attribute__((unused)) size_t bytes_read = fread(&size, sizeof(int), 1, fp);

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

    return size;
}

Task* createTasks(int *taskArrSize, int left, int right, int num_workers)
{
    Task *taskArr;
    if (left < right)
    {
        if (num_workers == 1)
        {
            taskArr = (Task *)malloc(sizeof(Task));
            taskArr[0].left = left;
            taskArr[0].right = right;
            taskArr[0].done = false;
            taskArr[0].leftParent = NULL;
            taskArr[0].rightParent = NULL;

            *taskArrSize = 1;
            return taskArr;
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
        taskArr = (Task *)malloc(maxTasks * sizeof(Task));
        int index = 0;
        for (int step = subarray_size; step < num_size; step *= 2)
        {
            for (int l = left; l <= right; l += step)
            {
                int r = l + step - 1;
                if (r > num_size)
                {
                    r = num_size;
                }
                taskArr[index].left = l;
                taskArr[index].right = r;
                taskArr[index].done = false;
                taskArr[index].leftParent = NULL;
                taskArr[index].rightParent = NULL;
                for (int j = index - 1; j >= 0; j--)
                {
                    if (taskArr[j].left == l)
                    {
                        taskArr[index].leftParent = &taskArr[j];
                        taskArr[index].mid = taskArr[j].right;
                        break;
                    }
                }
                for (int j = index - 1; j >= 0; j--)
                {
                    if (taskArr[j].right == r)
                    {
                        taskArr[index].rightParent = &taskArr[j];
                        break;
                    }
                }
                index++;
            }
        }
        taskArr[index].left = left;
        taskArr[index].right = right;
        taskArr[index].done = false;
        taskArr[index].leftParent = &taskArr[index - 2];
        taskArr[index].rightParent = &taskArr[index - 1];
        taskArr[index].mid = taskArr[index].leftParent->right;

        *taskArrSize = index + 1;
        return taskArr;
    }
    *taskArrSize = 0;
    return NULL;
}

bool isSorted(int arr[], int size)
{
    for (int i = 0; i < size - 1; i++)
    {
        if (arr[i] > arr[i + 1])
        {
            return false;
        }
    }
    return true;
}

// Merge two sorted subarrays of the array
void merge(int left, int mid, int right)
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