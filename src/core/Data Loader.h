//
//  Data Loader.h
//  Neural Net
//
//  Created by Shaunte Mellor on 6/23/22.
//

#ifndef Data_Loader_h
#define Data_Loader_h

#include "pch.h"
#include "Data Structure/Vector.h"
#include "Model/Matrix.h"

typedef struct Data{
    uint32_t num_data_points;
    Matrix* inputs;
    Matrix* outputs;
} Data;

typedef struct DataSplit{
    uint32_t total_data_points;
    Data train;
    Data test;
} DataSplit;

uint32_t num_datapoints_of_csv(const char* path);
//includes the target feature + input feature(s)
uint32_t num_features_of_csv(const char* path);

Data read_csv(const char* path, uint32_t num_rows, uint32_t num_cols, uint32_t target_column, uint32_t num_targets);

DataSplit train_test_split(Data* data, uint32_t train_size);

void delete_data(Data* data);
void delete_split_data(DataSplit* data);

#endif /* Data_Loader_h */
