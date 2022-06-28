//
//  Activations.c
//  Neural Net
//
//  Created by Shaunte Mellor on 5/1/22.
//

#include "Activations.h"
#include <stdio.h>
#include <math.h>

#define CLIP_RANGE 15

inline static float clamp(float x, float lower, float upper){
    float a = x < upper ? x : upper;
    float b = a > lower ? a : lower;
    return b;
}

void reLu(Matrix* mat){
    for (size_t i = 0; i < mat->rows * mat->cols; i++){
        mat->values[i] = clamp(mat->values[i], -CLIP_RANGE, CLIP_RANGE);
        mat->values[i] = MAX(mat->values[i], 0.0f);
    }
}

void sigmoid(Matrix* mat){
    for (size_t i = 0; i < mat->rows * mat->cols; i++){
        mat->values[i] = clamp(mat->values[i], -CLIP_RANGE, CLIP_RANGE);
        mat->values[i]  = 1.0f / ( 1.0f + expf(mat->values[i]) );
    }
                                    
}

void hyperbolic_tangent(Matrix* mat){
    for (size_t i = 0; i < mat->rows * mat->cols; i++){
        mat->values[i] = clamp(mat->values[i], -CLIP_RANGE, CLIP_RANGE);
        mat->values[i]  = tanhf(mat->values[i]);
    }
}

void soft_plus(Matrix* mat){
    for (size_t i = 0; i < mat->rows * mat->cols; i++){
        mat->values[i] = clamp(mat->values[i], -CLIP_RANGE, CLIP_RANGE);
        mat->values[i]  = logf(1.0f + expf(mat->values[i]));
    }
}


uint32_t argmax(Matrix* mat){
    float max = mat->values[0];
    uint32_t max_dex = 0;
    for (uint32_t i = 1; i < mat->rows * mat->cols; i++){
        if (mat->values[i] > max){
            max = mat->values[i];
            max_dex = i;
        }
    }
    return max_dex;
}


void reLu_deriv(Matrix* mat){
    for (size_t i = 0; i < mat->rows * mat->cols; i++){
        mat->values[i] = clamp(mat->values[i], -CLIP_RANGE, CLIP_RANGE);
        mat->values[i] = clamp(mat->values[i], 0.0f, 1.0f);
    }
}

void sigmoid_deriv(Matrix* mat){
    for (size_t i = 0; i < mat->rows * mat->cols; i++){
        mat->values[i] = clamp(mat->values[i], -CLIP_RANGE, CLIP_RANGE);
        mat->values[i]  = expf(-mat->values[i] ) / ( (1.0f + expf(mat->values[i]) * 2.0f ) );
    }
                                    
}

void hyperbolic_tangent_deriv(Matrix* mat){
    for (size_t i = 0; i < mat->rows * mat->cols; i++){
        mat->values[i] = clamp(mat->values[i], -CLIP_RANGE, CLIP_RANGE);
        mat->values[i] = 1.0f - (  ( expf(mat->values[i]) - expf(-mat->values[i]) ) / powf( expf(mat->values[i]) + expf(-mat->values[i]), 2.0f) );
    }
}

void soft_plus_deriv(Matrix* mat){
    for (size_t i = 0; i < mat->rows * mat->cols; i++){
        mat->values[i] = clamp(mat->values[i], -CLIP_RANGE, CLIP_RANGE);
        mat->values[i]  = expf(mat->values[i]) / (1.0f + expf(mat->values[i]));
    }
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
        case SOFT_PLUS:
            soft_plus(mat);
            return;
        case LINEAR:
            return;
        default:
            fprintf(stderr, "ERROR: Invalid activation function of %d. Exiting...\n", act);
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
        case SOFT_PLUS:
            soft_plus_deriv(mat);
            return;
        case LINEAR:
            set_values_with(mat, 1.0f);
            return;
        default:
            fprintf(stderr, "ERROR: Invalid activation function of %d. Exiting...\n", act);
            exit(-1);
    }
}

