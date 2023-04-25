#ifndef COUNTERS_H
#define COUNTERS_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <pthread.h>

typedef struct {
    int word_counter;
    int *vowel_counter;
    pthread_mutex_t accessCR;
    int *thread_arr;
    int thread_counter;
} Monitor;

void initialization(Monitor *m);
void update_counters(unsigned int workerId, int* counter_arr, Monitor *m);
void print_counters(Monitor *m);

#endif /* COUNTERS_H */
