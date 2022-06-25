//
//  Activations.h
//  Neural Net
//
//  Created by Shaunte Mellor on 5/1/22.
//

#ifndef Activations_h
#define Activations_h
#include "Matrix.h"

#define MAX(x, y) x > y ? x : y
#define MIN(x, y) x < y ? x : y

typedef enum Activation{
    RELU = 0,
    SIGMOID,
    HYPERBOLIC_TANGENT,
    SOFT_MAX,
    ARG_MAX
} Activation;


void reLu(Matrix* mat);

void sigmoid(Matrix* mat);

void hyperbolic_tangent(Matrix* mat);

void softmax(Matrix* mat);

void argmax(Matrix* mat);



void reLu_deriv(Matrix* mat);

void sigmoid_deriv(Matrix* mat);

void hyperbolic_tangent_deriv(Matrix* mat);

void softmax_deriv(Matrix* mat);



void act_func(Matrix* mat, Activation act);

void act_func_deriv(Matrix* mat, Activation act);


#endif /* Activations_h */
