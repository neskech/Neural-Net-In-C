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

//inline float reLu(float input);

inline float sigmoid(float input);

inline float hyperbolic_tangent(float input);

inline float leakyReLU(float input);

inline void softmax(Matrix* mat);

inline void argmax(Matrix* mat);

#endif /* Activations_h */