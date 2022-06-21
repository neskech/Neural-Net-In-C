//
//  Model.h
//  Neural Net
//
//  Created by Shaunte Mellor on 5/2/22.
//

#ifndef Model_h
#define Model_h

#include <stdlib.h>
#include "Activations.h"
#include "Matrix.h"
#include "Vector.h"

typedef struct ForwardPassCache{
    Matrix* activations; //array of activations
    Matrix* outputs;
} ForwardPassCache;

typedef struct Model{
    uint8_t numLayers;
    Vector layerSizes;
    Vector activations;
    
    Matrix* weights;
    Matrix* biases;
    
    ForwardPassCache cache;
} Model;


Model* create_model(void);

void free_cache(Model* model);

void delete_model(Model* model);



void add_layer(Model* m, int elem, Activation act);

uint8_t compile(Model* m);

void init_weights_and_biases(Model* m, float mean, float standard_deviation);


void backProp(Model* m);

void forwardProp(Model* m, Matrix* x);

void train(Model* m, Matrix* x, Matrix* y, uint32_t num_iterations);


Matrix eval(Model* m, Matrix* x);

void summary(Model* m);

#endif /* Model_h */
