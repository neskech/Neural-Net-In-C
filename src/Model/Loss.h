//
//  Loss.h
//  Neural Net
//
//  Created by Shaunte Mellor on 6/22/22.
//

#ifndef Loss_h
#define Loss_h

#include "Model/Matrix.h"

typedef enum Loss{
    LEAST_SQUARES,
    CROSS_ENTROPY,
    BINARY_CROSS_ENTROPY
} Loss;

//inputs are column vectors
float least_squares(Matrix* pred, Matrix* observ);
Matrix least_squares_deriv(Matrix* pred, Matrix* observ);

float cross_entropy(Matrix* pred, Matrix* observ);
Matrix cross_entropy_deriv(Matrix* pred, Matrix* observ);

float bin_cross_entropy(Matrix* pred, Matrix* observ);
Matrix bin_cross_entropy_deriv(Matrix* pred, Matrix* observ);


float loss_func(Matrix* pred, Matrix* observ, Loss loss);
Matrix loss_func_deriv(Matrix* pred, Matrix* observ, Loss loss);

#endif /* Loss_h */
