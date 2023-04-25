#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <pthread.h>

extern int *statusWorker;

//Struct of a monitor with shared memory region and a mutex lock
typedef struct {
    int word_counter;
    int *vowel_counter;
    pthread_mutex_t accessCR;
} Monitor;

//Initialization of the monitor
void initialization(Monitor *m){
    m->word_counter = 0;
    for(int i = 0; i < 6; i++){
        m->vowel_counter[i] = 0;
    }
    pthread_mutex_init(&m->accessCR, NULL);
}

//Function for updating the counters in the monitor
void update_counters(unsigned int workerId, int* counter_arr, Monitor *m){
    if((statusWorker[workerId] = pthread_mutex_lock(&m->accessCR)) != 0){
        printf("Error entering monitor.\n");
        statusWorker[workerId] = EXIT_FAILURE;
        pthread_exit(&statusWorker[workerId]);
    }

    m->word_counter += counter_arr[0];
    int counter_arr_count = 1;
    for(int i = 0; i < 6; i++){
        m->vowel_counter[i] += counter_arr[counter_arr_count];
        counter_arr_count++;
    }

    if((statusWorker[workerId] = pthread_mutex_unlock(&m->accessCR)) != 0){
        printf("Error exiting monitor.\n");
        statusWorker[workerId] = EXIT_FAILURE;
        pthread_exit(&statusWorker[workerId]);
    }
}

//Function for printing the counter values
void print_counters(Monitor *m){
    if(pthread_mutex_lock(&m->accessCR) != 0){
        printf("Error entering printing monitor.\n");
        pthread_exit(EXIT_FAILURE);
    }

    char vowels[] = {'A', 'E', 'I', 'O', 'U', 'Y'};

    printf("Total number of words: %d\n", m->word_counter);
    for(int i = 0; i < 6; i++){
        printf("%c: %d\n", vowels[i], m->vowel_counter[i]);
    }
    printf("\n");

    if(pthread_mutex_unlock(&m->accessCR) != 0){
        printf("Error exiting printing monitor.\n");
        pthread_exit(EXIT_FAILURE);
    }    
}