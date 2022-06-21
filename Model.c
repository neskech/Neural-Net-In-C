//
//  Model.c
//  Neural Net
//
//  Created by Shaunte Mellor on 5/2/22.
//

#include "Model.h"
#include <math.h>


static void for_each_with(Matrix* mat, Activation act){
    switch (act){
        case RELU:
            matrix_for_each(mat, reLu);
            return;
        case SIGMOID:
            matrix_for_each(mat, sigmoid);
            return;
        case HYPERBOLIC_TANGENT:
            matrix_for_each(mat, hyperbolic_tangent);
            return;
        case LEAKY_RELU:
            matrix_for_each(mat, leakyReLU);
            return;
        default:
            exit(-1);
    }
}

Model* create_model(void){
    Model* m = (Model*) malloc(sizeof(Model));
    
    //preallocate some space in the vectors
    m->layerSizes = create_vector(5);
    m->activations = create_vector(5);
    return m;
}

void free_cache(Model* m){
    if (m->cache.activations == NULL || m->cache.outputs == NULL)
        return;
    
    for (size_t i = 0; i < m->numLayers; i++){
        delete_matrix(m->cache.activations + i);
        
        if (i < m->numLayers - 1)
            delete_matrix(m->cache.outputs + i);
    }
    
    free(m->cache.activations);
    free(m->cache.outputs);
        
}

void delete_model(Model* m){
    for (size_t i = 0; i < m->numLayers - 1; i++){
        delete_matrix(m->biases + i);
        if (i < m->numLayers - 1)
            delete_matrix(m->weights + i);
    }
    
    free_cache(m);

    delete_vector(&m->layerSizes);
    delete_vector(&m->activations);
    free(m);
}

void add_layer(Model* m, int elem, Activation act){
    push(&m->layerSizes, elem);
    push(&m->activations, act);
}

uint8_t compile(Model* m){
    m->numLayers = m->layerSizes.size;
    if (m->numLayers <= 0)
        return 0;
    
    m->weights = (Matrix*) calloc(sizeof(Matrix), m->numLayers - 1);
    m->biases = (Matrix*) calloc(sizeof(Matrix), m->numLayers - 1);
    for (size_t i = 0; i < m->numLayers - 1; i++){
        *(m->biases + i) = create_matrix(get(&m->layerSizes, i + 1), 1);
        *(m->weights + i) = create_matrix(get(&m->layerSizes, i + 1), get(&m->layerSizes, i));
    }
    
    m->cache.activations = (Matrix*) calloc(sizeof(Matrix), m->numLayers);
    m->cache.outputs = (Matrix*) calloc(sizeof(Matrix), m->numLayers - 1);
    
    return 1;
}

void init_weights_and_biases(Model* m, float mean, float standard_dev){
    for (size_t i = 0; i < m->numLayers - 1; i++){
        //Box–Muller transform for generating numbers on a guassian distribution
        
        float val;
        for (size_t r = 0; r < (m->weights + i)->rows * (m->weights + i)->cols; r++){
            val = rand() / (float)RAND_MAX;
            
            if (r % 2 == 0)
                val = sqrt(-2.0f * log(val)) * cos(2.0f * M_PI * val);
            else
                val = sqrt(-2.0f * log(val)) * sin(2.0f * M_PI * val);
            
            val = val * standard_dev + mean;
            *((m->weights + i)->values + r) = val;
        }
        
        for (size_t r = 0; r < (m->biases + i)->rows; r++){
            val = rand() / (float)RAND_MAX;
            
            if (r % 2 == 0)
                val = sqrt(-2.0f * log(val)) * cos(2.0f * M_PI * val);
            else
                val = sqrt(-2.0f * log(val)) * sin(2.0f * M_PI * val);
            
            val = val * standard_dev + mean;
            *((m->biases + i)->values + r) = val;
        }
            
       
    }
}

Matrix eval(Model* m, Matrix* x){
    Matrix running = *x;
    
    *(m->cache.activations + 0) = matrix_copy(x);
    
    for (size_t i = 0; i < m->numLayers - 1; i++){

            
        //when we copy a new value to running, the previous value's memory is lost. Be sure to delete it
        Matrix before = running;
        running = mult(m->weights + i, &running);
        delete_matrix(&before);
        
        add_in_place(&running, m->biases + i);
        
        *(m->cache.outputs + i) = matrix_copy(&running);
        for_each_with(&running, get(&m->activations, i));
        
        *(m->cache.activations + i + 1) = matrix_copy(&running);
        
    }
    
    return running;
}

void forwardProp(Model* m, Matrix* x){
    Matrix running = *x;
    
    *(m->cache.activations + 0) = matrix_copy(x);
    
    for (size_t i = 0; i < m->numLayers - 1; i++){

            
        //when we copy a new value to running, the previous value's memory is lost. Be sure to delete it
        Matrix before = running;
        running = mult(m->weights + i, &running);
        delete_matrix(&before);
        
        add_in_place(&running, m->biases + i);
        
        *(m->cache.outputs + i) = matrix_copy(&running);
        for_each_with(&running, get(&m->activations, i));
        
        *(m->cache.activations + i + 1) = matrix_copy(&running);
        
    }
    
    delete_matrix(&running);
}

void train(Model* m,Matrix* x, Matrix* y, uint32_t num_iterations){
}

void summary(Model* m){
    //Print Layer Sizes
    printf("Layer Sizes: ");
    for (size_t i = 0; i < m->numLayers; i++){
        printf("%d", get(&m->layerSizes, i));
        if (i != m->numLayers - 1)
            printf(", ");
    }
    
    printf("\n------------------------------------------\nActivation Functions: ");
    for (size_t i = 0; i < m->numLayers; i++){
        printf("%d", get(&m->activations, i));
        if (i != m->numLayers - 1)
            printf(", ");
    }
    
    int total_params = 0;
    printf("\n------------------------------------------\nWeights:\n\n");
    for (size_t i = 0; i < m->numLayers - 1; i++){
        print_matrix(m->weights + i);
        total_params += size(m->weights + i);
        printf("\n\n");
    }
    
    printf("------------------------------------------\nABiases:\n\n");
    for (size_t i = 0; i < m->numLayers - 1; i++){
        print_matrix(m->biases + i);
        total_params += size(m->biases + i);
        printf("\n\n");
    }
    
    printf("------------------------------------------\n\nTotal Parameters: %d\n", total_params);
}
