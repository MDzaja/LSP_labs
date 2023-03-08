#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>


#define MAX_WORD_LENGTH 100
#define MAX_WORDS 20000

void replace_portuguese_chars(char *str) {
    //printf("Entering replace_portuguese_chars\n");
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

int word_count(char *file_name){
    //printf("Entering word_count\n");
    FILE *file = fopen(file_name, "r"); 
    if (!file) {
        printf("Error: Failed to open file.\n");
        return 1;
    }

    char words[MAX_WORDS][MAX_WORD_LENGTH];
    char *line = NULL;
    size_t line_len = 0;
    size_t read_len;
    int word_counter = 0;
    bool is_word_in_apostrophe = false;
    int letter_counter = 0;
    bool is_new_word_started = false;
    char vowel;

    //printf("Entering while getlines\n");
    while ((read_len = getline(&line, &line_len, file)) != -1) {
        replace_portuguese_chars(line);
        for (int i = 0; i < strlen(line); i++){
            if (line[i] == ' ' || line[i] == '\t' || line[i] == '\n' || line[i] == '-' || line[i] == '"' || line[i] == '[' || line[i] == ']' || line[i] == '(' || line[i] == ')' || line[i] == '.' || line[i] == ',' || line[i] == ':' || line[i] == ';' || line[i] == '!' || line[i] == '?'){
                //printf("if 1\n");
                if(is_new_word_started){
                    word_counter++;
                    letter_counter = 0;
                    is_new_word_started = false;
                }
            } else if ((unsigned char)line[i] == 0xE2 && (unsigned char)line[i+1] == 0x80 && (unsigned char)line[i+2] == 0x93){
                //printf("if 2\n");
                if(is_new_word_started){
                    word_counter++;
                    letter_counter = 0;
                    is_new_word_started = false;
                    i += 2;
                }
            } else if((unsigned char)line[i] == 0xE2 && (unsigned char)line[i+1] == 0x80 && (unsigned char)line[i+2] == 0xA6){
                //printf("if 3\n");
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
    }

    //printf("Exiting while\n");

    free(line);
    fclose(file);


    printf("File name: %s\n", file_name);
    printf("Total number of words in the file: %d\n", word_counter);

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
        printf("%c\t%d\n", vowel, vowel_counter);
    }
    
    printf("\n");
    return 0;
}

int main(int argc, char *argv[]) {
    //printf("Beginning program\n");
    if (argc < 2) {
        printf("Usage: %s file1 [file2 ...]\n", argv[0]);
        return 1;
    }

    //printf("Entering for loop\n");
    for (int i = 1; i < argc; i++) {
        word_count(argv[i]);
    }
    return 0;
}
