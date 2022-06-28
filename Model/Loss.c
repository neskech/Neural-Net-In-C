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

float cross_entropy(Matrix* pred, Matrix* observ){
    //find the observed label from the one-hot encoded vector
    size_t label = 0;
    for (size_t i = 0; i < observ->rows * observ->cols; i++){
        if (observ->values[i] == 1.0f){
            label = i;
            break;
        }
            
    }

    return -logf(pred->values[label] + 1e-8);
}

Matrix least_squares_deriv(Matrix* pred, Matrix* observ){
    Matrix m = create_matrix(pred->rows, pred->cols);
    for (size_t i = 0; i < pred->rows * pred->cols; i++){
        m.values[i] = -2.0f * (observ->values[i] - pred->values[i]);
    }
    return m;
}

Matrix cross_entropy_deriv(Matrix* pred, Matrix* observ){
    //find the observed label from the one-hot encoded vector
    size_t label = 0;
    for (size_t i = 0; i < observ->rows * observ->cols; i++){
        if (observ->values[i] == 1.0f){
            label = i;
            break;
        }
//
   }
//
    Matrix m = create_matrix(pred->rows, pred->cols);
    set_values_with(&m, -1.0f / (pred->values[label] + 1e-8));
//    for (size_t i = 0; i < pred->rows * pred->cols; i++){
//        m.values[i] = -1.0f / pred->values[i];
//    }
    return m;
}

float loss_func(Matrix* pred, Matrix* observ, Loss loss){
    switch (loss){
        case LEAST_SQUARES:
            return least_squares(pred, observ);
        case CROSS_ENTROPY:
            return cross_entropy(pred, observ);
        default:
            exit(-1);
    }
}


Matrix loss_func_deriv(Matrix* pred, Matrix* observ, Loss loss){
    switch (loss){
        case LEAST_SQUARES:
            return least_squares_deriv(pred, observ);
        case CROSS_ENTROPY:
            return cross_entropy_deriv(pred, observ);
        default:
            exit(-1);
    }
}



