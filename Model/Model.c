//
//  Model.c
//  Neural Net
//
//  Created by Shaunte Mellor on 5/2/22.
//

#include "Model.h"
#include <math.h>
#include <time.h>
#include <string.h>

Model* create_model(ModelParams* params, LearningRateTuning* tuning){
    Model* m = (Model*) malloc(sizeof(Model));
    
    //intialize random seed for parmeter intitialization and training
    srand((unsigned int) time(0)); //cast to get rid of warning
    
    //preallocate some space in the vectors
    m->layer_sizes = create_vector(5);
    m->activations = create_vector(5);
    
    m->params = *params;
    if (tuning == NULL)
        m->use_tuning = 0;
    else{
        m->use_tuning = 1;
        m->tuning = *tuning;
    }
    
    return m;
}

Model* load_model(const char* path){
    FILE* f;
    f = fopen(path, "r");
    if (f == NULL){
        fprintf(stderr, "ERROR: Could not open the path %s in the save_model function. Exiting...\n", path);
        exit(-1); 
    }
    
    Model* m = (Model*) malloc(sizeof(Model));
    uint32_t offset = 0;
    char* token;
    char line[100];
    
    fgets(line, 100, f);
    offset = strlen("num_layers: ");
    m->num_layers = atoi(&line[offset]);
    
    fgets(line, 100, f);
    offset = strlen("Layer Sizes: ");
    token = strtok(line + offset, ",");
    m->layer_sizes = create_vector(3);
    do {
        push(&m->layer_sizes, atoi(token));
        token = strtok(NULL, ",");
    }while(token != NULL);
    
    
    fgets(line, 100, f);
    offset = strlen("Activations: ");
    token = strtok(line + offset, ",");
    m->activations = create_vector(3);
    do {
        push(&m->activations, atoi(token));
        token = strtok(NULL, ",");
    }while(token != NULL);
    
    
    fgets(line, 100, f);
    offset = strlen("Loss Function: ");
    m->loss_func = atoi(&line[offset]);
    
    //skip the empty line...
    fgets(line, 100, f);
    //skip the "weights:" line...
    fgets(line, 100, f);
    
    //allocate the space for the weights and biases...
    m->weights = (Matrix*) calloc(sizeof(Matrix), m->num_layers - 1);
    m->biases = (Matrix*) calloc(sizeof(Matrix), m->num_layers - 1);
    
    m->expwa_weights = (Matrix*) calloc(sizeof(Matrix), m->num_layers - 1);
    m->expwa_biases = (Matrix*) calloc(sizeof(Matrix), m->num_layers - 1);
    
    m->expwa_weights_squared = (Matrix*) calloc(sizeof(Matrix), m->num_layers - 1);
    m->expwa_biases_squared = (Matrix*) calloc(sizeof(Matrix), m->num_layers - 1);
    for (uint32_t i = 0; i < m->num_layers - 1; i++){
        m->weights[i] = create_matrix(get(&m->layer_sizes, i + 1), get(&m->layer_sizes, i));
        m->biases[i] = create_matrix(get(&m->layer_sizes, i + 1), 1);
        
        m->expwa_weights[i] = create_matrix(get(&m->layer_sizes, i + 1), get(&m->layer_sizes, i));
        set_values_with(m->expwa_weights + i, 0.0f);
        m->expwa_biases[i] = create_matrix(get(&m->layer_sizes, i + 1), 1);
        set_values_with(m->expwa_biases + i, 0.0f);
        
        m->expwa_weights_squared[i] = create_matrix(get(&m->layer_sizes, i + 1), get(&m->layer_sizes, i));
        set_values_with(m->expwa_weights_squared + i, 0.0f);
        m->expwa_biases_squared[i] = create_matrix(get(&m->layer_sizes, i + 1), 1);
        set_values_with(m->expwa_biases_squared + i, 0.0f);
    }
    
    uint16_t ind = 0;
    uint16_t mat_index = 0;
    fgets(line, 100, f); //get to the "["
    fgets(line, 100, f); //now at the first element of the matrix
    while (strncmp(line, "Biases", 6) != 0){
        while (strncmp(line, "]", 1) != 0){
            m->weights[mat_index].values[ind] = atof(line);
            ind++;
            fgets(line, 100, f); //go to the next line, aka the next matrix
        }
        
        ind = 0;
        mat_index++;
        fgets(line, 100, f); //go to the "["
        fgets(line, 100, f); //go to the first element of the new matrix
    }
    
    //skip the empty line...
    fgets(line, 100, f);
    
    ind = 0;
    mat_index = 0;
    fgets(line, 100, f); //get to the "["
    fgets(line, 100, f); //now at the first element of the matrix
    while (!feof(f)){
        while (strncmp(line, "]", 1) != 0){
            m->biases[mat_index].values[ind] = atof(line);
            ind++;
            fgets(line, 100, f); //go to the next line, aka the next matrix
        }
        
        ind = 0;
        mat_index++;
        fgets(line, 100, f); //go to the "["
        fgets(line, 100, f); //go to the first element of the new matrix
    }
    
    return m;
}

uint8_t save_model(Model* m, const char* path){
    FILE* f;
    f = fopen(path, "w");
    if (f == NULL){
        fprintf(stderr, "ERROR: Could not open the path %s in the save_model function. Exiting...\n", path);
        return 0;
    }
    
    fprintf(f, "num_layers: %u\n", m->num_layers);
    fprintf(f, "Layer Sizes: ");
    
    for (uint32_t i = 0; i < m->num_layers; i++){
        if (i != m->num_layers - 1)
            fprintf(f, "%d, ", get(&m->layer_sizes, i));
        else
            fprintf(f, "%d\n", get(&m->layer_sizes, i));
    }
    
    fprintf(f, "Activations: ");
    for (uint32_t i = 0; i < m->num_layers - 1; i++){
        if (i != m->num_layers - 2)
            fprintf(f, "%d, ", get(&m->activations, i));
        else
            fprintf(f, "%d\n", get(&m->activations, i));
    }
    
    fprintf(f, "Loss Function: %d\n\n", m->loss_func);
    fprintf(f, "Weights:\n");
    for (uint32_t i = 0; i < m->num_layers - 1; i++){
        fprintf(f, "[\n");
        for (size_t j  = 0; j < size(m->weights + i); j++){
            if (j != size(m->weights + i) - 1)
                fprintf(f, "%f\n", m->weights[i].values[j]);
            else
                fprintf(f, "%f\n]\n", m->weights[i].values[j]);
        }
    }
    
    fprintf(f, "\n");
    
    fprintf(f, "Biases:\n");
    for (uint32_t i = 0; i < m->num_layers - 1; i++){
        fprintf(f, "[\n");
        for (size_t j  = 0; j < size(m->biases + i); j++){
            if (j != size(m->biases + i) - 1)
                fprintf(f, "%f\n", m->biases[i].values[j]);
            else
                fprintf(f, "%f\n]\n", m->biases[i].values[j]);
        }
    }
    
    return 1;
}

void delete_model(Model* m){
    for (size_t i = 0; i < m->num_layers - 1; i++){
        delete_matrix(m->weights + i);
        delete_matrix(m->biases + i);

        delete_matrix(m->expwa_weights + i);
        delete_matrix(m->expwa_biases + i);
        
        delete_matrix(m->expwa_weights_squared + i);
        delete_matrix(m->expwa_biases_squared + i);
    }
    
    free(m->weights);
    free(m->biases);
    
    free(m->expwa_weights);
    free(m->expwa_biases);
    
    free(m->expwa_weights_squared);
    free(m->expwa_biases_squared);
    
    delete_vector(&m->layer_sizes);
    delete_vector(&m->activations);
    
    free(m);
}

void add_layer(Model* m, int elem, Activation act){
    push(&m->layer_sizes, elem);
    //don't add the activation of the first layer, as it doesn't have one
    if (m->layer_sizes.size != 1)
        push(&m->activations, act);
}

void set_loss_func(Model* m, Loss loss_func_){
    m->loss_func = loss_func_;
}

uint8_t compile(Model* m){
    m->num_layers = m->layer_sizes.size;
    if (m->num_layers <= 0){
        fprintf(stderr, "ERROR: Model compilation failed. You must have layers in your model!\n");
        return 0;
    }
    
    for (size_t i = 0; i < m->num_layers; i++){
        if (get(&m->activations, i) == NONE){
            fprintf(stderr, "ERROR: Model compilation failed. Cannot have a 'NONE' layer after the input layer\n");
            return 0;
        }
    }
    
    m->weights = (Matrix*) calloc(sizeof(Matrix), m->num_layers - 1);
    m->biases = (Matrix*) calloc(sizeof(Matrix), m->num_layers - 1);
    
    m->expwa_weights = (Matrix*) calloc(sizeof(Matrix), m->num_layers - 1);
    m->expwa_biases = (Matrix*) calloc(sizeof(Matrix), m->num_layers - 1);
    
    m->expwa_weights_squared = (Matrix*) calloc(sizeof(Matrix), m->num_layers - 1);
    m->expwa_biases_squared = (Matrix*) calloc(sizeof(Matrix), m->num_layers - 1);
    
    for (size_t i = 0; i < m->num_layers - 1; i++){
        m->weights[i] = create_matrix(get(&m->layer_sizes, i + 1), get(&m->layer_sizes, i));
        m->biases[i] = create_matrix(get(&m->layer_sizes, i + 1), 1);
        
        m->expwa_weights[i] = create_matrix(get(&m->layer_sizes, i + 1), get(&m->layer_sizes, i));
        set_values_with(m->expwa_weights + i, 0.0f);
        m->expwa_biases[i] = create_matrix(get(&m->layer_sizes, i + 1), 1);
        set_values_with(m->expwa_biases + i, 0.0f);
        
        m->expwa_weights_squared[i] = create_matrix(get(&m->layer_sizes, i + 1), get(&m->layer_sizes, i));
        set_values_with(m->expwa_weights_squared + i, 0.0f);
        m->expwa_biases_squared[i] = create_matrix(get(&m->layer_sizes, i + 1), 1);
        set_values_with(m->expwa_biases_squared + i, 0.0f);
    }
    
    return 1;
}

void init_weights_and_biases(Model* m, float mean, float standard_dev){
    for (size_t i = 0; i < m->num_layers - 1; i++){
        
        float val;
        //Box–Muller transform for generating numbers on a guassian distribution
        for (size_t r = 0; r < (m->weights + i)->rows * (m->weights + i)->cols; r++){
            val = rand() / (float)RAND_MAX;
            
            if (r % 2 == 0)
                val = sqrtf(-2.0f * logf(val)) * cosf(2.0f * M_PI * val);
            else
                val = sqrtf(-2.0f * logf(val)) * sinf(2.0f * M_PI * val);
            
            val = val * standard_dev + mean;
            m->weights[i].values[r] = val;
        }
        
        for (size_t r = 0; r < (m->biases + i)->rows; r++){
            val = rand() / (float)RAND_MAX;
            
            if (r % 2 == 0)
                val = sqrtf(-2.0f * logf(val)) * cosf(2.0f * M_PI * val);
            else
                val = sqrtf(-2.0f * logf(val)) * sinf(2.0f * M_PI * val);
            
            val = val * standard_dev + mean;
            m->biases[i].values[r] = val;
        }
            
       
    }
}

Matrix eval(Model* m, Matrix* x){
    //running matrix will propogate through the layers. Make a copy of x so it isn't deleted
    Matrix running = matrix_copy(x);
    
    for (size_t i = 0; i < m->num_layers - 1; i++){
        //when we copy a new value to running, the previous value's memory is lost. Be sure to delete it
        Matrix before = running;
        running = mult(m->weights + i, &running);
        delete_matrix(&before);
        
        //add the biases
        add_in_place(&running, m->biases + i);
        //apply the activation function
        act_func(&running, get(&m->activations, i));
        
    }
    
    return running; //will have to clean up the return value...
}


float loss_on_dataset(Model* m, Matrix* x, Matrix* y, uint32_t num_data_points){
    float summed_loss = 0.0f;
    for (size_t i = 0; i < num_data_points; i++){
        Matrix eval_ = eval(m, x + i);
        summed_loss += loss_func(&eval_, y + i, m->loss_func);
        delete_matrix(&eval_);
    }
    return summed_loss / num_data_points;
}

static size_t total_params(Model* m){
    size_t total_params = 0;
    for (size_t i = 0; i < m->num_layers - 1; i++){
        total_params += size(m->weights + i);
        total_params += size(m->biases + i);
    }
    return total_params;
}

void summary(Model* m, uint8_t print_matrices){
    //Print Layer Sizes
    printf("Layer Sizes: ");
    for (size_t i = 0; i < m->num_layers; i++){
        printf("%d", get(&m->layer_sizes, i));
        if (i != m->num_layers - 1)
            printf(", ");
    }
    
    const char* act_names[6] = { "reLu", "sigmoid", "hyperbolic tangent", "soft plus", "linear", "sotfmax" };
    const char* loss_names[2] = { "least squares", "cross entropy" };
    
    printf("\n------------------------------------------\nActivation Functions: ");
    for (size_t i = 0; i < m->num_layers - 1; i++){
        printf("%s", act_names[ get(&m->activations, i) ] );
        if (i != m->num_layers - 1)
            printf(", ");
    }
    
    printf("\n");
    
    printf("------------------------------------------\nLoss Function: %s\n", loss_names[m->loss_func]);
    
    if (print_matrices){
        
        
            printf("------------------------------------------\nWeights:\n\n");
            for (size_t i = 0; i < m->num_layers - 1; i++){
                print_matrix(m->weights + i);
                printf("\n");
            }
            
            printf("------------------------------------------\nBiases:\n\n");
            for (size_t i = 0; i < m->num_layers - 1; i++){
                print_matrix(m->biases + i);
                printf("\n");
            }
        
    }
    
    size_t params = total_params(m);
    printf("------------------------------------------\n\nTotal Parameters: %zu\n\n", params);
}
