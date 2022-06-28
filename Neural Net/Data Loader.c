//
//  Data Loader.c
//  Neural Net
//
//  Created by Shaunte Mellor on 6/23/22.
//

#include "Data Loader.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

#define MIN(x, y) x < y ? x : y


uint32_t num_rows_of_csv(const char* path){
    FILE* file_ptr;
    file_ptr = fopen(path, "r");
    
    if (file_ptr == NULL){
        printf("ERROR: Could not open file in num_rows_of_csv. Exiting...\n");
        fclose(file_ptr);
        exit(-1);
    }
    
    
    uint32_t row_count = 0;
    char c;
    do{
        c = fgetc(file_ptr);
        if (c == '\n' || c == EOF)
            row_count++;
    }while(c != EOF);
    
    fclose(file_ptr);

    return row_count - 1; //to omit the informatiion row
}

uint32_t num_cols_of_csv(const char* path){
    FILE* file_ptr;
    file_ptr = fopen(path, "r");
    
    if (file_ptr == NULL){
        printf("ERROR: Could not open file in num_rows_of_csv. Exiting...\n");
        fclose(file_ptr);
        exit(-1);
    }
    
    char line_buffer[8000];
    char* token;
    
    
    fgets(line_buffer, 8000, file_ptr);
    token = strtok(line_buffer, ",");
    uint32_t token_count = 1;
    while(token != NULL){
        token = strtok(NULL, ",");
        token_count += token == NULL ? 0 : 1;
    }
    
    fclose(file_ptr);
    
    return token_count;
}


Matrix** read_csv(const char* path, uint32_t num_rows, uint32_t num_cols, uint32_t target_column, uint32_t num_targets){
    
    Matrix* inputs = (Matrix*) calloc(sizeof(Matrix), num_rows);
    Matrix* outputs = (Matrix*) calloc(sizeof(Matrix), num_rows);
    
    FILE* file_ptr;
    file_ptr = fopen(path, "r");
    
    if (file_ptr == NULL){
        printf("ERROR: Could not open file in num_rows_of_csv. Exiting...\n");
        fclose(file_ptr);
        exit(-1);
    }
    
    char line_buffer[8000];
    char* token;
    fgets(line_buffer, 8000, file_ptr); //move past the first line
    
    uint32_t token_count = 0;
    uint32_t matrix_count = 0;
    while (!feof(file_ptr)){
        fgets(line_buffer, 8000, file_ptr);
        token = strtok(line_buffer, ",");
        
        inputs[matrix_count] = create_matrix(num_cols - 1, 1); //column vector
        outputs[matrix_count] = create_matrix(num_targets, 1); //one hot encoded column vector
        token_count = 0;
        
        do {
            
            if (token_count == target_column){
                
                //one hot encoding...
                if (num_targets > 1){
                    set_values_with(outputs + matrix_count, 0.0f);
                    uint32_t idx = atoi(token);
                    outputs[matrix_count].values[idx] = 1.0f;
                }
                else
                    outputs[matrix_count].values[0] = atof(token);
            }
            else{
                uint32_t index = token_count > target_column ? token_count - 1 : token_count;
                inputs[matrix_count].values[index] = atof(token);
            }
            
            token_count++;
            token = strtok(NULL, ",");
        }while(token != NULL);
        
        matrix_count++;
    }
    
    
    
    Matrix** mats = (Matrix**) calloc(sizeof(Matrix*), 2);
    mats[0] = inputs;
    mats[1] = outputs;
    
    fclose(file_ptr);

    return mats;
}


void delete_data(Matrix** data, uint32_t num_data_points){
    for (uint32_t i = 0; i < num_data_points; i++){
        delete_matrix(data[0] + i); //inputs
        delete_matrix(data[1] + i); //outputs
    }
    
    free(data[0]);
    free(data[1]);
    free(data);
}

Data train_test_split(Matrix** data, uint32_t train_size, uint32_t num_data_points){
    Data dat;
    
    dat.train = (Matrix**) calloc(sizeof(Matrix*), 2);
    dat.train[0] = (Matrix*) calloc(sizeof(Matrix), train_size); //inputs
    dat.train[1] = (Matrix*) calloc(sizeof(Matrix), train_size); //outputs
    
    uint32_t test_size = num_data_points - train_size;
    dat.test = (Matrix**) calloc(sizeof(Matrix*), 2);
    dat.test[0] = (Matrix*) calloc(sizeof(Matrix), test_size); //inputs
    dat.test[1] = (Matrix*) calloc(sizeof(Matrix), test_size); //outputs
    
    dat.train_size = train_size;
    dat.test_size = test_size;
    
    Vector indices = create_vector(num_data_points);
    for (uint32_t i = 0; i < num_data_points; i++)
        push(&indices, i);
    
    srand((unsigned int) time(0)); //cast to get rid of warning
    
    for (uint32_t i = 0; i < train_size; i++){
        uint32_t rand_index = (uint32_t) ( ( rand() / (float)RAND_MAX ) * indices.size );
        
        rand_index = remove_at(&indices, rand_index);
        
        dat.train[0][i] = data[0][rand_index]; //train inputs
        dat.train[1][i] = data[1][rand_index]; //train outputs
    }
    
    uint32_t i = 0;
    while (indices.size > 0){
        uint32_t index = remove_at(&indices, indices.size - 1);
        
        dat.test[0][i] = data[0][index]; //test inputs
        dat.test[1][i] = data[1][index]; //test outputs
        i++;
    }
    
    delete_vector(&indices);
    //free the allocated arrays that were holding the data
    free(data[0]);
    free(data[1]);
    free(data);
    
    return dat;
}

void delete_split_data(Data* data){
    for (uint32_t i = 0; i < data->train_size; i++){
            //deleting the inputs x and outputs y of the train and test sets
            delete_matrix(data->train[0] + i);
            delete_matrix(data->train[1] + i);
    }
    
    for (uint32_t i = 0; i < data->test_size; i++){
        //deleting the inputs x outputs y of the train and test sets
        delete_matrix(data->test[0] + i);
        delete_matrix(data->test[1] + i);
    }


    //free the inputs x of both sets
    free(data->train[0]);
    free(data->test[0]);
    
    //free the outputs of both sets
    free(data->train[1]);
    free(data->test[1]);
    
    //free the double pointers themselves
    free(data->train);
    free(data->test);
}
