//
//  Matrix.h
//  Neural Net
//
//  Created by Shaunte Mellor on 5/1/22.
//

#ifndef Matrix_h
#define Matrix_h

#include <stdlib.h>



typedef struct matrix{
    size_t rows;
    size_t cols;
    float* values;
} Matrix;

Matrix new_matrix(uint16_t rows, uint16_t cols);

Matrix new_matrix_from_values(uint16_t rows, uint16_t cols, float* values);

Matrix dot(Matrix* mat_one, Matrix* mat_two);

Matrix mult(Matrix* mat_one, Matrix* mat_two);

Matrix add(Matrix* mat_one, Matrix* mat_two);

Matrix sub(Matrix* mat_one, Matrix* mat_two);

Matrix copy(Matrix* mat);

void mut_add(Matrix* mat_one, Matrix* mat_two);

void mut_sub(Matrix* mat_one, Matrix* mat_two);

void tranpose(Matrix* mat);

void scalar_mult(Matrix* mat_one, float scalar);

void scalar_div(Matrix* mat_one, float scalar);

void forEach(Matrix* mat, int (*func)(float));

float dot_prod(float* sub_one, float* sub_two, size_t length);

void delete(Matrix* mat);

void print_matrix(Matrix* mat);




#endif /* Matrix_h */
