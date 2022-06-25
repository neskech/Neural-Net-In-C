//
//  Data Loader.c
//  Neural Net
//
//  Created by Shaunte Mellor on 6/23/22.
//

#include "Data Loader.h"
#include "Float Vector.h"
#include <stdio.h>
#include <string.h>

#define MIN(x, y) x < y ? x : y


uint32_t num_rows_of_csv(const char* path){
    FILE* file_ptr;
    file_ptr = fopen(path, "r");
    
    char dummy_buffer[1];
    char* token;
    
    uint32_t row_count = 0;
    while (!feof(file_ptr)){
        fgets(dummy_buffer, 1, file_ptr);
        row_count++;
    }
    return row_count;
}
uint32_t num_cols_of_csv(const char* path){
    FILE* file_ptr;
    file_ptr = fopen(path, "r");
}


Matrix** read_csv(const char* path, uint32_t num_rows, uint32_t num_cols, uint32_t target_column, uint32_t num_targets){
    
    Matrix* inputs = (Matrix*) calloc(sizeof(Matrix), num_rows);
    Matrix* outputs = (Matrix*) calloc(sizeof(Matrix), num_rows);
    
    FILE* file_ptr;
    file_ptr = fopen(path, "r");
    
    char line_buffer[200];
    char* token;
    fgets(line_buffer, 1, file_ptr); //move past the first line
    
    uint32_t token_count;
    uint32_t matrix_count = 0;
    while (!feof(file_ptr)){
        fgets(line_buffer, 200, file_ptr);
        token = strtok(line_buffer, ",");
        
        
        inputs[matrix_count] = create_matrix(num_cols - 1, 1); //column vector
        outputs[matrix_count] = create_matrix(num_targets, 1); //one hot encoded column vector
        token_count = 0;
        
        while(token != NULL){
            token = strtok(NULL, ",");
            
            if (token_count == target_column){
                set_values_with(inputs + matrix_count, 0.0f);
                inputs[matrix_count].values[target_column] = 1.0f;
            }
            else{
                uint32_t index = token_count > target_column ? token_count - 1 : token_count;
                outputs[matrix_count].values[index] = atof(token);
            }
            
            token_count++;
        }
        
        matrix_count++;
    }
    
    
    
    Matrix** mats = (Matrix**) calloc(sizeof(Matrix*), 2);
    mats[0] = inputs;
    mats[1] = outputs;
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

void delete_split_data(Matrix*** data, uint32_t num_data_points){
    for (uint32_t i = 0; i < num_data_points; i++){
        
        for (uint32_t j = 0; j < 2; j++){
            //deleting the inputs x and outputs of y of the train and test sets
            delete_matrix(data[0][j] + i); //train
            delete_matrix(data[1][j] + i); //test
        }
    }
    
    for (uint32_t j = 0; j < 2; j++){
        free(data[0][j]); //train
        free(data[1][j]); //test
    }
    free(data[0]);
    free(data[1]);
    free(data);
}
