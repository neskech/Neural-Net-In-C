//
//  Activations.c
//  Neural Net
//
//  Created by Shaunte Mellor on 5/1/22.
//

#include "Activations.h"

#define MAX(x, y) x > y ? x : y

float reLu(float input){
    return MAX(input, 0.0f);
}

float sigmoid(float input){
    return 0.0f;
}

inline float hyperbolic_tangent(float input){
    return 0.0f;
}

inline float leakyReLU(float input){
    return 0.0f;
}

inline void softmax(Matrix* mat){
}

inline void argmax(Matrix* mat){
}
