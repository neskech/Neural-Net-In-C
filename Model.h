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
    //Matrix*
} ForwardPassCache;

typedef struct Model{
    uint8_t numLayers;
    Vector layerSizes;
    Vector activations;
    
    Matrix* weights;
    Matrix* biases;
    
    ForwardPassCache cache;
} Model;


Model* new_model(void);

void delete_model(Model* model);

void add_layer(Model* m, int elem, Activation act);

uint8_t compile(Model* m);

Matrix eval(Model* m, Matrix x);

void train(Model* m, Matrix x, Matrix y);

#endif /* Model_h */
