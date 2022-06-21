//
//  Activations.h
//  Neural Net
//
//  Created by Shaunte Mellor on 5/1/22.
//

#ifndef Activations_h
#define Activations_h
#include "Matrix.h"

typedef enum Activation{
    RELU = 0,
    SIGMOID,
    HYPERBOLIC_TANGENT,
    LEAKY_RELU,
    SOFT_MAX,
    ARG_MAX
} Activation;


float reLu(float input);

float sigmoid(float input);

float hyperbolic_tangent(float input);

float leakyReLU(float input);

void softmax(Matrix* mat);

void argmax(Matrix* mat);

#endif /* Activations_h */
