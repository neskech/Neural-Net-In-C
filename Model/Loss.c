//
//  Loss.c
//  Neural Net
//
//  Created by Shaunte Mellor on 6/22/22.
//

#include "Loss.h"
#include <math.h>

float least_squares(Matrix* pred, Matrix* observ){
    float sum = 0.0f;
    for (size_t i = 0; i < pred->rows * pred->cols; i++){
        sum += powf(observ->values[i] - pred->values[i], 2.0f);
    }
    return sum;
            
}


Matrix least_squares_deriv(Matrix* pred, Matrix* observ){
    Matrix m = create_matrix(pred->rows, pred->cols);
    for (size_t i = 0; i < pred->rows * pred->cols; i++){
        m.values[i] = -2.0f * (observ->values[i] - pred->values[i]);
    }
    return m;
}


float loss_func(Matrix* pred, Matrix* observ, Loss loss){
    switch (loss){
        case LEAST_SQUARES:
            return least_squares(pred, observ);
        default:
            fprintf(stderr, "ERROR: Unkown loss function of %d in loss_func() dispatch function. Exiting...\n", loss);
            exit(-1);
    }
}


Matrix loss_func_deriv(Matrix* pred, Matrix* observ, Loss loss){
    switch (loss){
        case LEAST_SQUARES:
            return least_squares_deriv(pred, observ);
        default:
            fprintf(stderr, "ERROR: Unkown loss function of %d in loss_func() dispatch function. Exiting...\n", loss);
            exit(-1);
    }
}



