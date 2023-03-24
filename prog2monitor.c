#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <pthread.h>
#include "prog2monitor.h"
#include "prog2const.h"

extern int *statusWorker;
extern int *statusDistributor;

static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t *waitingForTask;
pthread_cond_t *waitingForWorkerTermination;
static pthread_cond_t waitingForAnyWorker;

// Initialization of the monitor
void initialization(int workersNumber, Monitor *m)
{
    waitingForTask = malloc(workersNumber * sizeof(pthread_cond_t));
    waitingForWorkerTermination = malloc(workersNumber * sizeof(pthread_cond_t));

    for (int i = 0; i < workersNumber; i++)
    {
        m->workerStateArr[i].terminate = false;
        m->workerStateArr[i].state = AVAILABLE;
        pthread_cond_init(&waitingForTask[i], NULL);
        pthread_cond_init(&waitingForWorkerTermination[i], NULL);
    }
    pthread_cond_init(&waitingForAnyWorker, NULL);
}

void setWorkerTask(int workerId, bool leafTask, int left, int mid, int right, Monitor *m)
{
    if ((statusDistributor[0] = pthread_mutex_lock(&accessCR)) != 0)
    {
        printf("Error entering monitor.\n");
        statusDistributor[0] = EXIT_FAILURE;
        pthread_exit(&statusDistributor[0]);
    }

    m->workerTaskArr[workerId].leafTask = leafTask;
    m->workerTaskArr[workerId].left = left;
    m->workerTaskArr[workerId].mid = mid;
    m->workerTaskArr[workerId].right = right;
    m->workerStateArr[workerId].state = BUSY;

    if ((statusDistributor[0] = pthread_cond_signal(&waitingForTask[workerId])) != 0)
    {
        printf("Error signaling worker.\n");
        statusDistributor[0] = EXIT_FAILURE;
        pthread_exit(&statusDistributor[0]);
    }

    if ((statusDistributor[0] = pthread_mutex_unlock(&accessCR)) != 0)
    {
        printf("Error exiting monitor.\n");
        statusDistributor[0] = EXIT_FAILURE;
        pthread_exit(&statusDistributor[0]);
    }
}

void commandAllWorkersToTerminate(int workersNumber, Monitor *m)
{
    if ((statusDistributor[0] = pthread_mutex_lock(&accessCR)) != 0)
    {
        printf("Error entering monitor.\n");
        statusDistributor[0] = EXIT_FAILURE;
        pthread_exit(&statusDistributor[0]);
    }

    for (int i = 0; i < workersNumber; i++)
    {
        m->workerStateArr[i].terminate = true;
        if ((statusDistributor[0] = pthread_cond_signal(&waitingForTask[i])) != 0)
        {
            printf("Error signaling worker.\n");
            statusDistributor[0] = EXIT_FAILURE;
            pthread_exit(&statusDistributor[0]);
        }
    }

    if ((statusDistributor[0] = pthread_mutex_unlock(&accessCR)) != 0)
    {
        printf("Error exiting monitor.\n");
        statusDistributor[0] = EXIT_FAILURE;
        pthread_exit(&statusDistributor[0]);
    }
}

void requestNewTask(int workerId, Monitor *m)
{
    if ((statusWorker[workerId] = pthread_mutex_lock(&accessCR)) != 0)
    {
        printf("Error entering monitor.\n");
        statusWorker[workerId] = EXIT_FAILURE;
        pthread_exit(&statusWorker[workerId]);
    }

    m->workerTaskArr[workerId].oldLeft = m->workerTaskArr[workerId].left;
    m->workerTaskArr[workerId].oldMid = m->workerTaskArr[workerId].mid;
    m->workerTaskArr[workerId].oldRight = m->workerTaskArr[workerId].right;
    m->workerStateArr[workerId].state = AVAILABLE;

    if ((statusWorker[workerId] = pthread_cond_signal(&waitingForAnyWorker)) != 0)
    {
        printf("Error signaling distributor.\n");
        statusWorker[workerId] = EXIT_FAILURE;
        pthread_exit(&statusWorker[workerId]);
    }

    if ((statusWorker[workerId] = pthread_mutex_unlock(&accessCR)) != 0)
    {
        printf("Error exiting monitor.\n");
        statusWorker[workerId] = EXIT_FAILURE;
        pthread_exit(&statusWorker[workerId]);
    }
}

void notifyWorkerTermination(int workerId, Monitor *m)
{
    if ((statusWorker[workerId] = pthread_mutex_lock(&accessCR)) != 0)
    {
        printf("Error entering monitor.\n");
        statusWorker[workerId] = EXIT_FAILURE;
        pthread_exit(&statusWorker[workerId]);
    }

    m->workerStateArr[workerId].state = TERMINATED;

    if ((statusWorker[workerId] = pthread_cond_signal(&waitingForWorkerTermination[workerId])) != 0)
    {
        printf("Error signaling worker.\n");
        statusWorker[workerId] = EXIT_FAILURE;
        pthread_exit(&statusWorker[workerId]);
    }

    if ((statusWorker[workerId] = pthread_mutex_unlock(&accessCR)) != 0)
    {
        printf("Error exiting monitor.\n");
        statusWorker[workerId] = EXIT_FAILURE;
        pthread_exit(&statusWorker[workerId]);
    }
}

WorkerTask waitForNewTask(int workerId, Monitor *m)
{
    WorkerTask task;
    if ((statusWorker[workerId] = pthread_mutex_lock(&accessCR)) != 0)
    {
        printf("Error entering monitor.\n");
        statusWorker[workerId] = EXIT_FAILURE;
        pthread_exit(&statusWorker[workerId]);
    }

    if (m->workerStateArr[workerId].state == AVAILABLE && m->workerStateArr[workerId].terminate == false)//TODO
    {
        if ((statusWorker[workerId] = pthread_cond_wait(&waitingForTask[workerId], &accessCR)) != 0)
        {
            printf("Error waiting for task.\n");
            statusWorker[workerId] = EXIT_FAILURE;
            pthread_exit(&statusWorker[workerId]);
        }
    }
    if (m->workerStateArr[workerId].terminate == true
        && m->workerTaskArr[workerId].left == m->workerTaskArr[workerId].oldLeft
        && m->workerTaskArr[workerId].right == m->workerTaskArr[workerId].oldRight
        && m->workerTaskArr[workerId].mid == m->workerTaskArr[workerId].oldMid)
    {
        task.leafTask = false;
        task.left = -1;
        task.mid = -1;
        task.right = -1;
    }
    else
    {
        task.leafTask = m->workerTaskArr[workerId].leafTask;
        task.left = m->workerTaskArr[workerId].left;
        task.mid = m->workerTaskArr[workerId].mid;
        task.right = m->workerTaskArr[workerId].right;
    }

    if ((statusWorker[workerId] = pthread_mutex_unlock(&accessCR)) != 0)
    {
        printf("Error exiting monitor.\n");
        statusWorker[workerId] = EXIT_FAILURE;
        pthread_exit(&statusWorker[workerId]);
    }
    return task;
}

int* waitForAtLeastOneAvailableWorker(int *availableWorkers, int workersNumber, Monitor *m)
{
    for (int i = 0; i < workersNumber; i++)
    {
        availableWorkers[i] = BUSY;
    }
    int availableWorkersNumber = 0;

    if ((statusDistributor[0] = pthread_mutex_lock(&accessCR)) != 0)
    {
        printf("Error entering monitor.\n");
        statusDistributor[0] = EXIT_FAILURE;
        pthread_exit(&statusDistributor[0]);
    }

    for (int i = 0; i < workersNumber; i++)
    {
        if (m->workerStateArr[i].state == AVAILABLE)
        {
            availableWorkers[i] = AVAILABLE;
            availableWorkersNumber++;
        }
    }

    if (availableWorkersNumber == 0)
    {
        if ((statusDistributor[0] = pthread_cond_wait(&waitingForAnyWorker, &accessCR)) != 0)
        {
            printf("Error waiting for task.\n");
            statusDistributor[0] = EXIT_FAILURE;
            pthread_exit(&statusDistributor[0]);
        }
    }

    for (int i = 0; i < workersNumber; i++)
    {
        if (m->workerStateArr[i].state == AVAILABLE)
        {
            availableWorkers[i] = AVAILABLE;
        }
    }

    if ((statusDistributor[0] = pthread_mutex_unlock(&accessCR)) != 0)
    {
        printf("Error exiting monitor.\n");
        statusDistributor[0] = EXIT_FAILURE;
        pthread_exit(&statusDistributor[0]);
    }

    return availableWorkers;
}

void waitForAllWorkersToTerminate(int workersNumber, Monitor *m)
{
    if ((statusDistributor[0] = pthread_mutex_lock(&accessCR)) != 0)
    {
        printf("Error entering monitor.\n");
        statusDistributor[0] = EXIT_FAILURE;
        pthread_exit(&statusDistributor[0]);
    }

    for (int i = 0; i < workersNumber; i++)
    {
        if (m->workerStateArr[i].state != TERMINATED)
        {
            if ((statusDistributor[0] = pthread_cond_wait(&waitingForWorkerTermination[i], &accessCR)) != 0)
            {
                printf("Error waiting for task.\n");
                statusDistributor[0] = EXIT_FAILURE;
                pthread_exit(&statusDistributor[0]);
            }
        }
    }

    if ((statusDistributor[0] = pthread_mutex_unlock(&accessCR)) != 0)
    {
        printf("Error exiting monitor.\n");
        statusDistributor[0] = EXIT_FAILURE;
        pthread_exit(&statusDistributor[0]);
    }
}