//
//  Activations.c
//  Neural Net
//
//  Created by Shaunte Mellor on 5/1/22.
//

#include "Activations.h"
#include <math.h>

#define MAX(x, y) x > y ? x : y
#define MIN(x, y) x < y ? x : y

inline static float clamp(float x, float lower, float upper){
    return MAX( (MIN(x, upper)), lower );
}

void reLu(Matrix* mat){
    for (size_t i = 0; i < mat->rows * mat->cols; i++)
        *(mat->values + i) = MAX(*(mat->values + i), 0.0f);
}

void sigmoid(Matrix* mat){
    for (size_t i = 0; i < mat->rows * mat->cols; i++)
        *(mat->values + i) = 1.0f / ( 1.0f + expf(*(mat->values + i) ) );
                                    
}

void hyperbolic_tangent(Matrix* mat){
    for (size_t i = 0; i < mat->rows * mat->cols; i++)
        *(mat->values + i) = tanhf(*(mat->values + i));
}

void softmax(Matrix* mat){
}

void argmax(Matrix* mat){
}




void reLu_deriv(Matrix* mat){
    for (size_t i = 0; i < mat->rows * mat->cols; i++)
        *(mat->values + i) = clamp(*(mat->values + i), 0.0f, 1.0f);
}

void sigmoid_deriv(Matrix* mat){
    for (size_t i = 0; i < mat->rows * mat->cols; i++)
        *(mat->values + i) = expf(-*(mat->values + i)) / ( (1.0f + expf(*(mat->values + i)) * 2.0f ) );
                                    
}

void hyperbolic_tangent_deriv(Matrix* mat){
    for (size_t i = 0; i < mat->rows * mat->cols; i++)
        mat->values[i] = 1.0f - (  ( expf(mat->values[i]) - expf(-mat->values[i]) ) / powf( expf(mat->values[i]) + expf(-mat->values[i]), 2.0f) );
}



void act_func(Matrix* mat, Activation act){
    switch (act){
        case RELU:
            reLu(mat);
            return;
        case SIGMOID:
            sigmoid(mat);
            return;
        case HYPERBOLIC_TANGENT:
            hyperbolic_tangent(mat);
            return;
        default:
            exit(-1);
    }
}

void act_func_deriv(Matrix* mat, Activation act){
    switch (act){
        case RELU:
            reLu_deriv(mat);
            return;
        case SIGMOID:
            sigmoid_deriv(mat);
            return;
        case HYPERBOLIC_TANGENT:
            hyperbolic_tangent_deriv(mat);
            return;
        default:
            exit(-1);
    }
}

