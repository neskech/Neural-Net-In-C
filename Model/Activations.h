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
    SOFT_PLUS,
    LINEAR,
    SOFT_MAX,
    ARG_MAX,
    NONE,
} Activation;


void reLu(Matrix* mat);

void sigmoid(Matrix* mat);

void hyperbolic_tangent(Matrix* mat);

void soft_plus(Matrix* mat);

void softmax(Matrix* mat);

uint32_t argmax(Matrix* mat);



void reLu_deriv(Matrix* mat, Matrix* observ);

void sigmoid_deriv(Matrix* mat, Matrix* observ);

void hyperbolic_tangent_deriv(Matrix* mat, Matrix* observ);

void soft_plus_deriv(Matrix* mat, Matrix* observ);

void softmax_deriv(Matrix* mat, Matrix* observ);



void act_func(Matrix* mat, Activation act);

void act_func_deriv(Matrix* mat, Matrix* observ, Activation act);


#endif /* Activations_h */
