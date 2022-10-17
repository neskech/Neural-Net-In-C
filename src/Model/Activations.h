//
//  Activations.h
//  Neural Net
//
//  Created by Shaunte Mellor on 5/1/22.
//

#ifndef Activations_h
#define Activations_h
#include "Model/Matrix.h"

#define MAX(x, y) x > y ? x : y
#define MIN(x, y) x < y ? x : y

typedef enum Activation{
    RELU = 0,
    SIGMOID,
    HYPERBOLIC_TANGENT,
    SOFT_PLUS,
    SOFT_MAX,
    LINEAR,
    NONE,
} Activation;


void reLu(Matrix* mat);

void sigmoid(Matrix* mat);

void hyperbolic_tangent(Matrix* mat);

void soft_plus(Matrix* mat);

void softmax(Matrix* mat);

uint32_t argmax(Matrix* mat);



void reLu_deriv(Matrix* mat);

void sigmoid_deriv(Matrix* mat);

void hyperbolic_tangent_deriv(Matrix* mat);

void soft_plus_deriv(Matrix* mat);

void softmax_deriv(Matrix* mat, Matrix* observ);



void act_func(Matrix* mat, Activation act);

void act_func_deriv(Matrix* mat, Activation act, Matrix* observ);


#endif /* Activations_h */
