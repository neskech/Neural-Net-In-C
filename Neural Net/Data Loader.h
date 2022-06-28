//
//  Data Loader.h
//  Neural Net
//
//  Created by Shaunte Mellor on 6/23/22.
//

#ifndef Data_Loader_h
#define Data_Loader_h

#include <stdio.h>
#include "Vector.h"
#include "Matrix.h"

typedef struct Data{
    uint32_t num_data_points;
    uint32_t train_size;
    uint32_t test_size;
    
    Matrix** train;
    Matrix** test;
} Data;

uint32_t num_rows_of_csv(const char* path);
uint32_t num_cols_of_csv(const char* path);

Matrix** read_csv(const char* path, uint32_t num_rows, uint32_t num_cols, uint32_t target_column, uint32_t num_targets);

Data train_test_split(Matrix** data, uint32_t train_size, uint32_t num_data_points);

void delete_data(Matrix** data, uint32_t num_data_points);
void delete_split_data(Data* data);

#endif /* Data_Loader_h */
