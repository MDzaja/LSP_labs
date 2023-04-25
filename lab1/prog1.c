#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <pthread.h>
#include <sys/time.h>
#include "prog1monitor.h"


#define MAX_WORD_LENGTH 100
#define MAX_WORDS 20000
#define MAX_CHUNK_SIZE 3000

int *statusWorker;
Monitor m;

struct thread_data {
    int id;
    char *file_name;
    long start_pointer;
    long end_pointer;
};

//Function for spliting file in to chunks so that each thread can process a part of the file
long* split_file(char* filename, int num_chunks){
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: could not open file %s\n", filename);
        return -1;
    }

    num_chunks++;
    char chunk_delimiters[] = " \t\n-\"[]().,?!;:";
    size_t num_delimiters = strlen(chunk_delimiters);
    
    long file_size;
    fseek(fp, 0L, SEEK_END);
    file_size = ftell(fp);
    rewind(fp);
    
    long* chunk_start_positions = malloc(sizeof(long) * num_chunks);
    if (!chunk_start_positions) {
        printf("Error: could not allocate memory for chunk_start_positions\n");
        return 1;
    }
    
    chunk_start_positions[0] = ftell(fp);
    rewind(fp);

    long chunk_size = file_size / num_chunks;
    long current_chunk_size = 0;
    long current_chunk_start_position = 0;
    int current_chunk_index = 1;
    
    for (long i = 0; i < file_size; i++) {
        char current_byte;
        fread(&current_byte, 1, 1, fp);
        current_chunk_size++;
        
        if (current_chunk_size >= chunk_size && strchr(chunk_delimiters, current_byte) != NULL) {
            // Found a chunk delimiter, save the current position as the start of a new chunk
            chunk_start_positions[current_chunk_index] = current_chunk_start_position;
            current_chunk_start_position = ftell(fp);
            current_chunk_index++;
            current_chunk_size = 0;
        }
        
    }
    
    // Print each chunk to stdout
    char* buffer = malloc(sizeof(char) * MAX_CHUNK_SIZE);
    if (!buffer) {
        printf("Error: could not allocate memory for buffer\n");
        return 1;
    }

    long* return_arr = malloc(sizeof(long) * num_chunks);
    int return_arr_count = 0;

    for(int i = 1; i < num_chunks; i++){
        return_arr[return_arr_count] = chunk_start_positions[i];
        return_arr_count++;
    }
    return_arr[return_arr_count] = file_size;



    return return_arr;
}

//Function for replacing the portuguese characters with ascii letters
void replace_portuguese_chars(char *str) {
    int i, j;
    int len = strlen(str);
    char new_str[len+1];

    // iterate over each character in the string
    for (i = 0, j = 0; i < len; i++) {
        if ((unsigned char)str[i] == 0xC3) {  // check for Portuguese UTF-8 character
            if ((unsigned char)str[i+1] == 0xA1 || (unsigned char)str[i+1] == 0xA0 || (unsigned char)str[i+1] == 0xA2 || (unsigned char)str[i+1] == 0xA3) {  
                new_str[j] = 'a';
                i++;
            } else if ((unsigned char)str[i+1] == 0xA9 || (unsigned char)str[i+1] == 0xA8 || (unsigned char)str[i+1] == 0xAA) {  
                new_str[j] = 'e';
                i++;
            } else if ((unsigned char)str[i+1] == 0xAD || (unsigned char)str[i+1] == 0xAC) {  // í, ì
                new_str[j] = 'i';
                i++;
            } else if ((unsigned char)str[i+1] == 0xB3 || (unsigned char)str[i+1] == 0xB2 || (unsigned char)str[i+1] == 0xB4 || (unsigned char)str[i+1] == 0xB5) {  
                new_str[j] = 'o';
                i++;
            } else if ((unsigned char)str[i+1] == 0xBA || (unsigned char)str[i+1] == 0xB9) {  // ú, ù
                new_str[j] = 'u';
                i++;
            } else if ((unsigned char)str[i+1] == 0xA7) {  // ç
                new_str[j] = 'c';
                i++;
            } else if ((unsigned char)str[i+1] == 0x81 || (unsigned char)str[i+1] == 0x80 || (unsigned char)str[i+1] == 0x82 || (unsigned char)str[i+1] == 0x83){
                new_str[j] = 'A';
                i++;
            } else if ((unsigned char)str[i+1] == 0x89 || (unsigned char)str[i+1] == 0x88 || (unsigned char)str[i+1] == 0x8A) {  
                new_str[j] = 'E';
                i++;
            } else if ((unsigned char)str[i+1] == 0x8D || (unsigned char)str[i+1] == 0x8C) {  // í, ì
                new_str[j] = 'I';
                i++;
            } else if ((unsigned char)str[i+1] == 0x93 || (unsigned char)str[i+1] == 0x92 || (unsigned char)str[i+1] == 0x94 || (unsigned char)str[i+1] == 0x95) {  
                new_str[j] = 'O';
                i++;
            } else if ((unsigned char)str[i+1] == 0x9A || (unsigned char)str[i+1] == 0x99) {  // ú, ù
                new_str[j] = 'U';
                i++;
            } else if ((unsigned char)str[i+1] == 0x87) {  // Ç
                new_str[j] = 'C';
                i++;
            } else {  // not a Portuguese UTF-8 character
                new_str[j] = str[i];
            }
        } else {  // not a Portuguese UTF-8 character
            new_str[j] = str[i];
        }
        j++;
    }

    new_str[j] = '\0';  // add null terminator to new string
    strcpy(str, new_str);  // copy new string back to original string
}

//Function for counting the words in a chunk
int* word_count(int id, char *file_name, long start_pos, long end_pos){
    FILE *file = fopen(file_name, "r"); 
    if (!file) {
        printf("Error: Failed to open file.\n");
        return 1;
    }

    char words[MAX_WORDS][MAX_WORD_LENGTH];
    size_t line_len = 0;
    size_t read_len;
    int word_counter = 0;
    bool is_word_in_apostrophe = false;
    int letter_counter = 0;
    bool is_new_word_started = false;
    char vowel;
    int* return_values = malloc(7 * sizeof(int));

    int block_size = end_pos - start_pos;
    char* line = malloc(block_size + 1);
    fseek(file, start_pos, SEEK_SET);
    fread(line, block_size, 1, file);
    line[block_size] = '\0';

    //Replacing the portuguese characters and counting the words
    replace_portuguese_chars(line);
    for (int i = 0; i < strlen(line); i++){
        if (line[i] == ' ' || line[i] == '\t' || line[i] == '\n' || line[i] == '-' || line[i] == '"' || line[i] == '[' || line[i] == ']' || line[i] == '(' || line[i] == ')' || line[i] == '.' || line[i] == ',' || line[i] == ':' || line[i] == ';' || line[i] == '!' || line[i] == '?'){
            if(is_new_word_started){
                word_counter++;
                letter_counter = 0;
                is_new_word_started = false;
            }
        } else if ((unsigned char)line[i] == 0xE2 && (unsigned char)line[i+1] == 0x80 && (unsigned char)line[i+2] == 0x93){
            if(is_new_word_started){
                word_counter++;
                letter_counter = 0;
                is_new_word_started = false;
                i += 2;
            }
        } else if((unsigned char)line[i] == 0xE2 && (unsigned char)line[i+1] == 0x80 && (unsigned char)line[i+2] == 0xA6){
            if(is_new_word_started){
                word_counter++;
                letter_counter = 0;
                is_new_word_started = false;
                i += 2;
            }
        } else{
            if(isalnum(line[i]) != 0 || line[i] == '_'){
                is_new_word_started = true;
                words[word_counter][letter_counter] = line[i];
                letter_counter++;
            }
        }
    }


    free(line);
    fclose(file);

    return_values[0] = word_counter;
    int return_counter = 1;

    //Counting the words with each vowel
    char vowels[] = {'a', 'e', 'i', 'o', 'u', 'y'};

    for(int k = 0; k < 6; k++){
        vowel = vowels[k];
        int vowel_counter = 0;
        for(int i = 0; i < word_counter; i++){
            for(int j = 0; j < strlen(words[i]); j++){
                if(words[i][j] == vowel || words[i][j] == vowel - 'a' + 'A'){
                    vowel_counter++;
                    break;
                }
            }
        }
        return_values[return_counter] = vowel_counter;
        return_counter++;
    }

    return return_values;
}

//Worker thread job
static void *worker(void *arg){
    struct thread_data *data = (struct thread_data *) arg;
    int* counter_arr = word_count(data->id, data->file_name, data->start_pointer, data->end_pointer);
    update_counters(data->id, counter_arr, &m);
    statusWorker[data->id] = EXIT_SUCCESS;
    pthread_exit(&statusWorker[data->id]);
}

int main(int argc, char *argv[]) {
    struct timeval begin, end;

    if (argc < 2) {
        printf("Usage: %s thread_number file1 [file2 ...]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int nThreads = atoi(argv[1]);

    pthread_t  *idWorkers;
    unsigned int *workers;
    int *pStatus;

    //Initializing the memory for thread arrays
    if((statusWorker = malloc(nThreads * sizeof(int))) == NULL){
        printf("Error on allocating space to the return status arrays of worker threads.\n");
        return EXIT_FAILURE;
    }

    
    if((idWorkers = malloc(nThreads * sizeof(pthread_t))) == NULL){
        printf("Error allocating space for worker id arrays.\n");
        return EXIT_FAILURE;
    }

    if((workers = malloc(nThreads * sizeof(unsigned int))) == NULL){
        printf("Error allocating space for worker id array.\n");
        return EXIT_FAILURE;
    }
    

    for(int i = 0; i < nThreads; i++){
        workers[i] = i;
    }

    //Starting the thread work
    gettimeofday(&begin, 0);

    m.vowel_counter = malloc(6 * sizeof(int));

    for (int i = 2; i < argc; i++) {
        initialization(&m);
        long* pointer_arr = split_file(argv[i], nThreads);
        struct thread_data data_arr[nThreads];
        for(int j = 0; j < nThreads; j++){
            data_arr[j].id = workers[j];
            data_arr[j].file_name = argv[i];
            data_arr[j].start_pointer = pointer_arr[j];
            data_arr[j].end_pointer = pointer_arr[j+1];
            if(pthread_create(&idWorkers[j], NULL, worker, (void *) &data_arr[j]) != 0){
                printf("Error creating worker thread");
                return EXIT_FAILURE;
            }
        }

        for(int j = 0; j < nThreads; j++){
            pthread_join(idWorkers[j], (void *) &pStatus);
        }
        print_counters(&m);
    }

    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds * 1e-6;
    printf ("\nElapsed time = %.6f s\n", elapsed);

    return 0;
}