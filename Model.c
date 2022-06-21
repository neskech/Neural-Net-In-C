//
//  Model.c
//  Neural Net
//
//  Created by Shaunte Mellor on 5/2/22.
//

#include "Model.h"

Model* new_model(void){
    Model* m = (Model*) malloc(sizeof(Model));
    
    //preallocate some space in the vectors
    m->layerSizes = create_vector(5);
    m->activations = create_vector(5);
    return m;
}

void delete_model(Model* m){
    for (size_t i = 0; i < m->numLayers; i++){
        delete_matrix(m->biases + i);
        if (i < m->numLayers - 1)
            delete_matrix(m->weights + i);
    }

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
    m->biases = (Matrix*) calloc(sizeof(Matrix), m->numLayers);
    for (size_t i = 0; i < m->numLayers; i++){
        *(m->biases + i) = create_matrix(get(&m->layerSizes, i), 1);
        if (i < m->numLayers - 1)
            *(m->weights + i) = create_matrix(get(&m->layerSizes, i + 1), get(&m->layerSizes, i));
    }
    
    m->cache.activations = (Matrix*) calloc(sizeof(Matrix), m->numLayers);
    
    return 1;
}

Matrix eval(Model* m, Matrix x){
    Matrix running = x;
    
    for (size_t i = 0; i < m->numLayers; i++){

        if (i < m->numLayers - 1){
            Matrix before = running;
            running = mult(m->weights + i, &running);
            delete_matrix(&before);
        }
        
        Matrix before = running;
        running = add(m->biases + i, &running);
        delete_matrix(&before);
    }
    
    return running;
}

void train(Model* m, Matrix x, Matrix y){
    
}
