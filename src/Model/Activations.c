//
//  Activations.c
//  Neural Net
//
//  Created by Shaunte Mellor on 5/1/22.
//

#include "Model/Matrix.h"
#include "Model/Activations.h"
#include "pch.h"


#define CLIP_RANGE 15

inline static float clamp(float x, float lower, float upper){
    float a = x < upper ? x : upper;
    float b = a > lower ? a : lower;
    return b;
}

void reLu(Matrix* mat){
    for (size_t i = 0; i < mat->rows * mat->cols; i++){
        //xmat->values[i] = clamp(mat->values[i], -CLIP_RANGE, CLIP_RANGE);
        mat->values[i] = MAX(mat->values[i], 0.0f);
    }
}

void leaky_reLu(Matrix* mat){
     for (size_t i = 0; i < mat->rows * mat->cols; i++){
        //mat->values[i] = clamp(mat->values[i], -CLIP_RANGE, CLIP_RANGE);
        mat->values[i] *= mat->values[i] < 0.0f ? 0.01f : 1.0f;
    }
}

void sigmoid(Matrix* mat){
    for (size_t i = 0; i < mat->rows * mat->cols; i++){
        mat->values[i] = clamp(mat->values[i], -CLIP_RANGE, CLIP_RANGE);
        mat->values[i]  = 1.0f / ( 1.0f + expf(-mat->values[i]) );
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

void softmax(Matrix* mat){
    mat->values[0] = clamp(mat->values[0], -CLIP_RANGE, CLIP_RANGE);

    float max = mat->values[0];
    for (size_t i = 1; i < mat->rows * mat->cols; i++){
        mat->values[i] = clamp(mat->values[i], -CLIP_RANGE, CLIP_RANGE);
        if (mat->values[i] > max)
            max = mat->values[i];
    }

    float denom = 0.0f;
    for (size_t i = 0; i < mat->rows * mat->cols; i++){
        //storing the e^x so we don't have to recalculate it
        float val = exp(mat->values[i] - max);
        denom += val;
        mat->values[i] = val;
    }

    
    for (size_t i = 0; i < mat->rows * mat->cols; i++){
        mat->values[i] /= denom;
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

//TODO format these the same as loss file

void reLu_deriv(Matrix* mat){
    for (size_t i = 0; i < mat->rows * mat->cols; i++){
        mat->values[i] = mat->values[i] < 0.0f ? 0.0f : 1.0f;
    }
}

void leaky_reLu_deriv(Matrix* mat){
     for (size_t i = 0; i < mat->rows * mat->cols; i++){
        mat->values[i] = mat->values[i] < 0.0f ? 0.01f : 1.0f;
    }
}

void sigmoid_deriv(Matrix* mat){
    for (size_t i = 0; i < mat->rows * mat->cols; i++){
        mat->values[i] = clamp(mat->values[i], -CLIP_RANGE, CLIP_RANGE);
        mat->values[i] = ( 1.0f / (1.0f + exp(-mat->values[i])) ) * (1.0f - ( 1.0f / (1.0f + exp(-mat->values[i])) ));
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

void softmax_deriv(Matrix* mat, Matrix* observ){
    softmax(mat);
    
    size_t j;
    for (size_t i = 0; i < observ->rows * observ->cols; i++){
        if (observ->values[i] == 1.0f){
            j = i;
            break;
        }
    }

    float targ = mat->values[j];

    for (size_t i = 0; i < mat->rows * mat->cols; i++){
        if (i == j){
            mat->values[i] = targ * (1.0f - targ);
            //mat->values[i] = targ - 1;
        }
        else{
            mat->values[i] = -mat->values[i] * targ;
             //mat->values[i] = mat->values[i];
        }
    }
}


void act_func(Matrix* mat, Activation act){
    switch (act){
        case RELU:
            reLu(mat);
            return;
        case LEAKY_RELU:
            leaky_reLu(mat);
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
        case SOFT_MAX:
            softmax(mat);
            return;
        case LINEAR:
            return;
        default:
            fprintf(stderr, "ERROR: Invalid activation function of %d. Exiting...\n", act);
            exit(-1);
    }
}

void act_func_deriv(Matrix* mat, Activation act, Matrix* observ){
    switch (act){
        case RELU:
            reLu_deriv(mat);
            return;
        case LEAKY_RELU:
            leaky_reLu_deriv(mat);
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
        case SOFT_MAX:
            softmax_deriv(mat, observ);
            return;
        case LINEAR:
            set_values_with(mat, 1.0f);
            return;
        default:
            fprintf(stderr, "ERROR: Invalid activation function of %d. Exiting...\n", act);
            exit(-1);
    }
}

