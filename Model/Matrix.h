//
//  Matrix.h
//  Neural Net
//
//
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

void set_values_with(Matrix* mat, float val); //fills matrix with a value

void delete_matrix(Matrix* mat);

void move_matrix(Matrix* from, Matrix* to); //like std::move()

Matrix matrix_copy(Matrix* mat);




Matrix dot(Matrix* mat_one, Matrix* mat_two); //element wise multiplication

Matrix matrix_div(Matrix* mat_one, Matrix* mat_two); //element wise division. Naming conflict, preface with 'matrix_'

Matrix mult(Matrix* mat_one, Matrix* mat_two);

Matrix add(Matrix* mat_one, Matrix* mat_two);

Matrix sub(Matrix* mat_one, Matrix* mat_two);




//in place functions to prevent allocating new memory for a new matrix

void dot_in_place(Matrix* mat_one, Matrix* mat_two);

void div_in_place(Matrix* mat_one, Matrix* mat_two);

void add_in_place(Matrix* mat_one, Matrix* mat_two);

void sub_in_place(Matrix* mat_one, Matrix* mat_two);



Matrix transpose(Matrix* mat);



void scalar_mult(Matrix* mat, float scalar);

void scalar_div(Matrix* mat, float scalar);

void scalar_add(Matrix* mat, float scalar);

void matrix_for_each(Matrix* mat, float (*func)(float));

float dot_prod(float* sub_one, float* sub_two, size_t length);



void matrix_square(Matrix* mat);

void matrix_sqrt(Matrix* mat);

float magnitude(Matrix* mat);

void reciprocal(Matrix* mat);



size_t size(Matrix* mat);

void print_matrix(Matrix* mat);



#endif /* Matrix_h */
