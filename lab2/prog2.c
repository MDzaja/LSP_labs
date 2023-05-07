#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <mpi.h>

// Define constants for task types and termination
#define TASK_META_OR_TERMINATE 1
#define TASK_DATA 2
#define TASK_DONE 3
#define TERMINATED 4

// Task struct definition
typedef struct Task
{
    int left;
    int mid;
    int right;
    bool done;
    struct Task *leftParent;
    struct Task *rightParent;
} Task;

// Distribute tasks among the workers, receive results, and return the sorted array.
int *distributor(char *filename, int num_workers, int *arr_size);
// Worker process that receives tasks, performs merge sort or merge operation, and sends back results.
void worker(int wIndex);
// Read integers from a binary file and return them as an array.
int *readFile(char *filename, int *size);
// Create an array of tasks to perform merge sort on the given array.
Task *createTasks(int *taskArrSize, int left, int right, int num_workers);
// Check if the given array is sorted.
bool isSorted(int arr[], int size);
// Merge two sorted subarrays of the array.
void merge(int *arr, int left, int mid, int right);
// Divide the array into two subarrays, sort them and merge them.
void mergeSort(int *arr, int left, int right);
// Receive task results from workers, update the main array and the task array, and update worker availability.
void receiveTaskResultAndUpdateArray(int mpi_source, bool *available_workers, int *available_workers_number, int *done_tasks, int *arr, Task *task_arr);

int main(int argc, char *argv[])
{
    int rank, nProc;
    int *seqValMin, *seqValMax, *seqValParcMin, *seqValParcMax;
    bool goOn;
    int i, n, nNorm;

    // Initialize MPI environment and get the rank and size
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);

    // Check if the number of processes is greater than 1
    if (nProc <= 1)
    {
        if (rank == 0)
            printf("Wrong number of processes! It must be greater than 1.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Check if the correct number of command line arguments are provided
    if (argc != 2)
    {
        if (rank == 0)
            printf("Usage: %s file1\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Distributor process (rank 0)
    if (rank == 0)
    {
        struct timeval begin, end;
        int i, arr_size;

        // Get the start time
        gettimeofday(&begin, 0);

        // Call the distributor function and receive the sorted array
        int *arr = distributor(argv[1], nProc - 1, &arr_size);

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

        // Check if the sequence is properly sorted
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
    }
    // Worker process (rank != 0)
    else
    {
        // Call the worker function, as long as the distributor
        // process has not signaled termination, receive a new task
        worker(rank);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

int *distributor(char *filename, int num_workers, int *arr_size)
{
    MPI_Status status;
    MPI_Request request;

    // Read the input file and store the data in an array
    int *arr = readFile(filename, arr_size);
    // Create an array of tasks to be distributed among the workers
    int taskArrSize;
    Task *taskArr = createTasks(&taskArrSize, 0, *arr_size - 1, num_workers);

    int currentTaskIndex = 0;
    int doneTasks = 0;
    int numberOfAvailableWorkers = 0;
    bool availableWorkers[num_workers];

    // Initialize all workers as unavailable
    for (int i = 0; i < num_workers; i++)
    {
        availableWorkers[i] = false;
    }

    // Loop until all tasks are completed
    while (doneTasks < taskArrSize)
    {
        // If no workers are available, wait for a worker to become available
        if (numberOfAvailableWorkers == 0)
        {
            // Update the task and worker status
            receiveTaskResultAndUpdateArray(MPI_ANY_SOURCE, availableWorkers, &numberOfAvailableWorkers, &doneTasks, arr, taskArr);
        }

        // Check the status of each worker
        for (int i = 1; i <= num_workers; i++)
        {
            // If the worker is not available, check if it has completed its task
            if (!availableWorkers[i - 1])
            {
                int flag = 0;
                MPI_Iprobe(i, TASK_DONE, MPI_COMM_WORLD, &flag, &status);
                if (flag)
                {
                    // Update the task and worker status
                    receiveTaskResultAndUpdateArray(i, availableWorkers, &numberOfAvailableWorkers, &doneTasks, arr, taskArr);
                }
            }
        }

        //  Distribute tasks to available workers
        for (int i = 1; i <= num_workers && currentTaskIndex < taskArrSize; i++)
        {
            Task task = taskArr[currentTaskIndex];
            if (availableWorkers[i - 1])
            {
                // Ensure the task's dependencies are completed before sending it
                if ((task.leftParent == NULL || task.leftParent->done) && (task.rightParent == NULL || task.rightParent->done))
                {
                    int seqSize = task.right - task.left + 1;
                    int task_meta[] = {
                        currentTaskIndex,
                        task.leftParent == NULL ? 1 : 0,
                        task.mid - task.left,
                        seqSize};
                    // Send the task metadata to the worker: 
                    // task index, is task a leaf, index of mid element, size of sequence
                    MPI_Isend(task_meta, 4, MPI_INT, i, TASK_META_OR_TERMINATE, MPI_COMM_WORLD, &request);

                    // Send the task data to the worker
                    int *seq = (int *)malloc(seqSize * sizeof(int));
                    for (int j = 0; j < seqSize; j++)
                    {
                        seq[j] = arr[task.left + j];
                    }
                    MPI_Send(seq, seqSize, MPI_INT, i, TASK_DATA, MPI_COMM_WORLD);

                    // Mark the worker as busy
                    availableWorkers[i - 1] = false;
                    numberOfAvailableWorkers--;
                    currentTaskIndex++;
                }
                else
                {
                    break;
                }
            }
        }
    }

    // Broadcast termination message
    int dummy[4] = {-1};
    for (int i = 1; i <= num_workers; i++)
    {
        MPI_Isend(dummy, 4, MPI_INT, i, TASK_META_OR_TERMINATE, MPI_COMM_WORLD, &request);
    }
    // Wait for termination messages from workers
    for (int i = 1; i <= num_workers; i++)
    {
        int tmp;
        MPI_Recv(&tmp, 1, MPI_INT, i, TERMINATED, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    return arr;
}

void worker(int rank)
{
    MPI_Status status;
    MPI_Request request;

    // Notify distributor that the worker is ready to receive tasks
    int dummy = -1;
    MPI_Send(&dummy, 1, MPI_INT, 0, TASK_DONE, MPI_COMM_WORLD);

    // Main loop for processing tasks
    while (1)
    {
        int task_meta[4];
        // Receive task metadata or termination signal from the distributor
        MPI_Recv(task_meta, 4, MPI_INT, 0, TASK_META_OR_TERMINATE, MPI_COMM_WORLD, &status);

        // Check if the received message is not a termination signal
        if (task_meta[0] != -1)
        {
            // Extract the task metadata
            int taskIndex = task_meta[0];
            bool leafTask = task_meta[1] == 1;
            int mid = task_meta[2];
            int seqSize = task_meta[3];

            // Receive the task data from the distributor
            int *seq = (int *)malloc(seqSize * sizeof(int));
            MPI_Recv(seq, seqSize, MPI_INT, 0, TASK_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Process the received task
            if (leafTask)
            {
                // Sort the sequence using merge sort
                mergeSort(seq, 0, seqSize - 1);
            }
            else
            {
                // Merge the two sorted subsequences
                merge(seq, 0, mid, seqSize - 1);
            }

            // Notify the distributor that the task is done and send the task index
            MPI_Isend(&taskIndex, 1, MPI_INT, 0, TASK_DONE, MPI_COMM_WORLD, &request);
            // Send the processed data back to the distributor
            MPI_Send(seq, seqSize, MPI_INT, 0, TASK_DATA, MPI_COMM_WORLD);
        }
        else
        {
            // If the termination signal is received, notify the distributor and exit the loop
            int dummy;
            MPI_Send(&dummy, 1, MPI_INT, 0, TERMINATED, MPI_COMM_WORLD);
            break;
        }
    }
}

int *readFile(char *filename, int *size)
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
    __attribute__((unused)) size_t bytes_read = fread(size, sizeof(int), 1, fp);

    // Print the size of the array
    printf("Size: %d \n", *size);

    // Allocating enough space to save all the numbers read from a file
    int *arr = (int *)malloc(*size * sizeof(int));
    // Read the numbers from the file and store them in the array
    while (fread(&arr[n], sizeof(int), 1, fp) == 1)
    {
        n++;
    }

    // Close the file
    fclose(fp);

    // Return the pointer to the array with the numbers read from the file
    return arr;
}

Task *createTasks(int *taskArrSize, int left, int right, int num_workers)
{
    Task *taskArr;
    // Check if the input range is valid
    if (left < right)
    {
        // If there is only one worker, create a single task
        if (num_workers == 1)
        {
            taskArr = (Task *)malloc(sizeof(Task));
            taskArr[0].left = left;
            taskArr[0].right = right;
            taskArr[0].done = false;
            taskArr[0].mid = -1;
            taskArr[0].leftParent = NULL;
            taskArr[0].rightParent = NULL;

            *taskArrSize = 1;
            return taskArr;
        }

        // Calculate the subarray size based on the number of workers
        int num_size = right - left + 1;
        int subarray_size = num_size / num_workers;
        if (subarray_size < 2)
        {
            subarray_size = 2;
        }

        // Determine the number of leaf tasks and the height of the task tree
        int leafTasks = ceil((double)num_size / subarray_size);
        int treeHeight = ceil(log2(leafTasks));
        int maxTasks = pow(2, treeHeight + 1) - 1;

        // Allocate memory for the task array
        taskArr = (Task *)malloc(maxTasks * sizeof(Task));
        int index = 0;

        // Create tasks for each level of the task tree
        for (int step = subarray_size; step < num_size; step *= 2)
        {
            for (int l = left; l <= right; l += step)
            {
                int r = l + step - 1;
                if (r > right)
                {
                    r = right;
                }

                // Initialize task with the calculated range
                taskArr[index].left = l;
                taskArr[index].right = r;
                taskArr[index].done = false;
                taskArr[index].mid = -1;
                taskArr[index].leftParent = NULL;
                taskArr[index].rightParent = NULL;

                // Set the task's left parent
                for (int j = index - 1; j >= 0; j--)
                {
                    if (taskArr[j].left == l)
                    {
                        taskArr[index].leftParent = &taskArr[j];
                        taskArr[index].mid = taskArr[j].right;
                        break;
                    }
                }

                // Set the task's right parent
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

        // Create the root task
        taskArr[index].left = left;
        taskArr[index].right = right;
        taskArr[index].done = false;
        taskArr[index].leftParent = &taskArr[index - 2];
        taskArr[index].rightParent = &taskArr[index - 1];
        taskArr[index].mid = taskArr[index].leftParent->right;

        // Set the task array size and return the task array
        *taskArrSize = index + 1;
        return taskArr;
    }

    // If the input range is invalid, return NULL and set the task array size to 0
    *taskArrSize = 0;
    return NULL;
}

bool isSorted(int arr[], int size)
{
    // Iterate through the array elements and check if they are sorted
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
void merge(int *arr, int left, int mid, int right)
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
void mergeSort(int *arr, int left, int right)
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

void receiveTaskResultAndUpdateArray(int mpi_source, bool *available_workers, int *available_workers_number, int *done_tasks, int *arr, Task *task_arr)
{
    MPI_Status status;
    int taskIndex;
    // Receive the task index from the worker
    MPI_Recv(&taskIndex, 1, MPI_INT, mpi_source, TASK_DONE, MPI_COMM_WORLD, &status);
    // Mark the worker as available
    available_workers[status.MPI_SOURCE - 1] = true;
    // Increment the number of available workers
    (*available_workers_number)++;
    
    if (taskIndex != -1)
    {
        // Increment the number of done tasks
        (*done_tasks)++;
        // Mark the task as done
        task_arr[taskIndex].done = true;
        // Calculate the size of the sequence in the task
        int seqSize = task_arr[taskIndex].right - task_arr[taskIndex].left + 1;
        // Allocate memory for the received sequence
        int *seq = (int *)malloc(seqSize * sizeof(int));
        // Receive the sorted sequence from the worker
        MPI_Recv(seq, seqSize, MPI_INT, status.MPI_SOURCE, TASK_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Update the main array with the received sorted sequence
        for (int i = 0; i < seqSize; i++)
        {
            arr[task_arr[taskIndex].left + i] = seq[i];
        }
    }
}
