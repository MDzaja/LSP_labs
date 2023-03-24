#ifndef SORT_H
#define SORT_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <pthread.h>

typedef struct WorkerState
{
    bool terminate;
    int state;
} WorkerState;
typedef struct WorkerTask
{
    bool leafTask;
    int left, mid, right;
    int oldLeft, oldMid, oldRight;
} WorkerTask;
typedef struct {
    WorkerState *workerStateArr;
    WorkerTask *workerTaskArr;
} Monitor;

extern void initialization(int workersNumber, Monitor *m);
extern void setWorkerTask(int workerId, bool leafTask, int left, int mid, int right, Monitor *m);
extern void commandAllWorkersToTerminate(int workersNumber, Monitor *m);
extern void requestNewTask(int workerId, Monitor *m);
extern void notifyWorkerTermination(int workerId, Monitor *m);

extern WorkerTask waitForNewTask(int workerId, Monitor *m);
extern int* waitForAtLeastOneAvailableWorker(int *availableWorkers, int workersNumber, Monitor *m);
extern void waitForAllWorkersToTerminate(int workersNumber, Monitor *m);


#endif /* SORT_H */
