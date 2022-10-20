//
//  Data Loader.c
//  Neural Net
//
//  Created by Shaunte Mellor on 6/23/22.
//

#include "core/Data Loader.h"
#include "pch.h"


#define CHAR_BUFF_SIZE 8000



uint32_t num_datapoints_of_csv(const char* path){
    FILE* file_ptr;
    file_ptr = fopen(path, "r");
    
    if (file_ptr == NULL){
        fprintf(stderr, "ERROR: Could not open file in num_rows_of_csv. Exiting...\n");
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

    return row_count - 1; //to omit the information row
}

uint32_t num_features_of_csv(const char* path){
    FILE* file_ptr;
    file_ptr = fopen(path, "r");
    
    if (file_ptr == NULL){
        fprintf(stderr, "ERROR: Could not open file in num_rows_of_csv. Exiting...\n");
        fclose(file_ptr);
        exit(-1);
    }
    
    char line_buffer[CHAR_BUFF_SIZE];
    char* token;
    
    
    fgets(line_buffer, CHAR_BUFF_SIZE, file_ptr);
    token = strtok(line_buffer, ",");
    uint32_t token_count = 1;
    while(token != NULL){
        token = strtok(NULL, ",");
        token_count += token == NULL ? 0 : 1;
    }
    
    fclose(file_ptr);
    
    return token_count;
}


//num targets is how many targets we have. So say for a digit dataset like MNIST, we would have 10 targets because there are 10 digits to choose from
Data read_csv(const char* path, uint32_t num_rows, uint32_t num_cols, uint32_t target_column, uint32_t num_targets){
    
    Data data;
    //TODO make it clear that num_rows is the number of desired data points and not the number of file rows
    //IF num rows doesnt equal file rows then bad shit happens, hence the matrix count < num_rows on the loop
    data.inputs = (Matrix*) calloc(sizeof(Matrix), num_rows);
    data.outputs = (Matrix*) calloc(sizeof(Matrix), num_rows);
    data.num_data_points = num_rows;
    
    FILE* file_ptr;
    file_ptr = fopen(path, "r");
    
    if (file_ptr == NULL){
        fprintf(stderr, "ERROR: Could not open file in num_rows_of_csv. Exiting...\n");
        fclose(file_ptr);
        exit(-1);
    }
    
    char line_buffer[CHAR_BUFF_SIZE];
    char* token;
    fgets(line_buffer, CHAR_BUFF_SIZE, file_ptr); //move past the first line
    
    uint32_t token_count = 0;
    uint32_t matrix_count = 0;
    while (!feof(file_ptr) && matrix_count < num_rows){
        fgets(line_buffer, CHAR_BUFF_SIZE, file_ptr);
        token = strtok(line_buffer, ",");
        
        data.inputs[matrix_count] = create_matrix(num_cols - 1, 1); //column vector. -1 to omit the target feature
        data.outputs[matrix_count] = create_matrix(num_targets, 1); //one hot encoded column vector
        token_count = 0;
        
        do {
            
            if (token_count == target_column){
                
                //one hot encoding...
                if (num_targets > 1){
                    set_values_with(data.outputs + matrix_count, 0.0f);
                    uint32_t idx = atoi(token);
                    data.outputs[matrix_count].values[idx] = 1.0f;
                }
                else
                    data.outputs[matrix_count].values[0] = atof(token);
            }
            else{
                uint32_t index = token_count > target_column ? token_count - 1 : token_count;
                data.inputs[matrix_count].values[index] = atof(token);
            }
            
            token_count++;
            token = strtok(NULL, ",");
        }while(token != NULL);

        matrix_count++;
    }
    
    fclose(file_ptr);
    return data;
}


void delete_data(Data* data){
    for (uint32_t i = 0; i < data->num_data_points; i++){
        delete_matrix(data->inputs + i);
        delete_matrix(data->outputs + i);
    }
    
    free(data->inputs);
    free(data->outputs);
}

DataSplit train_test_split(Data* data, uint32_t train_size){
    uint32_t total_data_points = data->num_data_points;
    uint32_t test_size = total_data_points - train_size;
    
    if (train_size > total_data_points){
        fprintf(stderr, "ERROR: Training dataset size of %u is greater than total dataset size of %u. Returning empty struct...\n", train_size, total_data_points);
        DataSplit split;
        return split;
    }
    
    DataSplit split = {
        .total_data_points = total_data_points,
        
        .train = {
            .inputs = (Matrix*) calloc(sizeof(Matrix), train_size),
            .outputs = (Matrix*) calloc(sizeof(Matrix), train_size),
            .num_data_points = train_size,
        },
        
        .test = {
            .inputs = (Matrix*) calloc(sizeof(Matrix), test_size),
            .outputs = (Matrix*) calloc(sizeof(Matrix), test_size),
            .num_data_points = test_size,
        },
    };
  
    
    //make a vector of indices
    Vector indices = create_vector(total_data_points);
    for (uint32_t i = 0; i < total_data_points; i++)
        push(&indices, i);
    
    //initialize the random seed for the rand() function
    srand((unsigned int) time(0)); //cast to get rid of warning
    
    //remove random elements from the index vector and use those elements to add random matrices to the train test split
    //this is the 'shuffling' portion of the split
    uint32_t rand_index;
    for (uint32_t i = 0; i < train_size; i++){
        rand_index = (uint32_t) ( ( rand() / (float)RAND_MAX ) * indices.size );
        
        rand_index = remove_at(&indices, rand_index);
        
        //shallow copies
        split.train.inputs[i] = data->inputs[rand_index];
        split.train.outputs[i] = data->outputs[rand_index];
    }
    
    uint32_t i = 0;
    while (indices.size > 0){
        rand_index = remove_at(&indices, indices.size - 1);
        
        //shallow copies
        split.test.inputs[i] = data->inputs[rand_index];
        split.test.outputs[i] = data->outputs[rand_index];
        i++;
    }
    
    
    //free the allocated arrays that were holding the data
    free(data->inputs);
    free(data->outputs);
    delete_vector(&indices);
    
    return split;
}

void delete_split_data(DataSplit* split){
    for (size_t i = 0; i < split->train.num_data_points; i++){
            //deleting the inputs x and outputs y of the train and test sets
            delete_matrix(split->train.inputs + i);
            delete_matrix(split->train.outputs + i);
    }
    
    for (size_t i = 0; i < split->test.num_data_points; i++){
        //deleting the inputs x outputs y of the train and test sets
        delete_matrix(split->test.inputs + i);
        delete_matrix(split->test.outputs + i);
    }


    //free the inputs x and y of the training set
    free(split->train.inputs);
    free(split->train.outputs);
    
    //free the inputs x and y of the test set
    free(split->test.inputs);
    free(split->test.outputs);
}
