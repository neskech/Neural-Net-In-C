//
//  Matrix.h
//  Neural Net
//
//  Created by Shaunte Mellor on 5/1/22.
//

#ifndef Matrix_h
#define Matrix_h

#include <stdlib.h>



typedef struct Matrix{
    size_t rows;
    size_t cols;
    float* values;
} Matrix;

Matrix create_matrix(uint16_t rows, uint16_t cols);

Matrix create_matrix_from_values(uint16_t rows, uint16_t cols, float* values);

void set_values_with(Matrix* mat, float val);

Matrix dot(Matrix* mat_one, Matrix* mat_two);

Matrix mult(Matrix* mat_one, Matrix* mat_two);

Matrix add(Matrix* mat_one, Matrix* mat_two);

Matrix sub(Matrix* mat_one, Matrix* mat_two);


Matrix matrix_copy(Matrix* mat);


void dot_in_place(Matrix* mat_one, Matrix* mat_two);

void add_in_place(Matrix* mat_one, Matrix* mat_two);

void sub_in_place(Matrix* mat_one, Matrix* mat_two);


void tranpose(Matrix* mat);

void scalar_mult(Matrix* mat_one, float scalar);

void scalar_div(Matrix* mat_one, float scalar);

void matrix_for_each(Matrix* mat, float (*func)(float));

float dot_prod(float* sub_one, float* sub_two, size_t length);

void delete_matrix(Matrix* mat);

void print_matrix(Matrix* mat);

size_t size(Matrix* mat);




#endif /* Matrix_h */
