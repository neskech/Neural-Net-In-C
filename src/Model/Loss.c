//
//  Loss.c
//  Neural Net
//
//  Created by Shaunte Mellor on 6/22/22.
//

#include "Model/Loss.h"
#include "pch.h"

float least_squares(Matrix* pred, Matrix* observ){
    float sum = 0.0f;
    for (size_t i = 0; i < pred->rows * pred->cols; i++){
        sum += powf(observ->values[i] - pred->values[i], 2.0f);
    }
    return sum / (pred->rows * pred->cols);
            
}


Matrix least_squares_deriv(Matrix* pred, Matrix* observ){
    Matrix m = create_matrix(pred->rows, pred->cols);
    for (size_t i = 0; i < pred->rows * pred->cols; i++){
        m.values[i] = -2.0f * (observ->values[i] - pred->values[i]);
    }
    return m;
}

float cross_entropy(Matrix* pred, Matrix* observ){
    //The observed vector will be one hot encoded
    float sum = 0.0f;
    for (size_t i = 0; i < pred->rows * pred->cols; i++){
         //anything thats not the desired classification will
         //have an observed value of 0, cancelling out the log
         sum += observ->values[i] * -log(pred->values[i]);
    }
    return sum / (pred->cols * pred->rows);
}

Matrix cross_entropy_deriv(Matrix* pred, Matrix* observ){
    Matrix m = create_matrix(pred->rows, pred->cols); //Vector

    for (size_t i = 0; i < pred->rows * pred->cols; i++){
       if (observ->values[i] == 1.0f){        
            m.values[i] = - 1.0f / (pred->values[i]);
            break;
       }
    }
    return m;
}

float bin_cross_entropy(Matrix* pred, Matrix* observ){
    //OBSERV and PRED vector should be size 1
    float o = observ->values[0];
    float p = pred->values[0];
    float loss = o * -log(p) + (1.0f - o) * -log(1.0f - p);
    return loss;
}

Matrix bin_cross_entropy_deriv(Matrix* pred, Matrix* observ){
    Matrix m = create_matrix(pred->rows, pred->cols); //Vector
    float o = observ->values[0];
    float p = pred->values[0];
    m.values[0] = -o / p + (1.0f - o) / (1.0f - p);
    return m;
}


float loss_func(Matrix* pred, Matrix* observ, Loss loss){
    switch (loss){
        case LEAST_SQUARES:
            return least_squares(pred, observ);
        case CROSS_ENTROPY:
             return cross_entropy(pred, observ);
        case BINARY_CROSS_ENTROPY:
             return bin_cross_entropy(pred, observ);
        default:
            fprintf(stderr, "ERROR: Unkown loss function of %d in loss_func() dispatch function. Exiting...\n", loss);
            exit(-1);
    }
}


Matrix loss_func_deriv(Matrix* pred, Matrix* observ, Loss loss){
    switch (loss){
        case LEAST_SQUARES:
            return least_squares_deriv(pred, observ);
        case CROSS_ENTROPY:
             return cross_entropy_deriv(pred, observ);
        case BINARY_CROSS_ENTROPY:
             return bin_cross_entropy_deriv(pred, observ);
        default:
            fprintf(stderr, "ERROR: Unkown loss function of %d in loss_func() dispatch function. Exiting...\n", loss);
            exit(-1);
    }
}



