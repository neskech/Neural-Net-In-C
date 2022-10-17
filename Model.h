//
//  Model.h
//  Neural Net
//
//  Created by Shaunte Mellor on 5/2/22.
//

#ifndef Model_h
#define Model_h
#include <stdlib.h>
#include "Layer.h"

typedef struct Model{
    Vector layers;
    uint8_t frozen;
} Model;

void delete_model(Model* model);

void add_layer(void* layer);

void compile(void);

void train(float** x, float** y);

float** eval(float* x);

#endif /* Model_h */
