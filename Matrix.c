//
//  Matrix.c
//  Neural Net
//
//  Created by Shaunte Mellor on 5/1/22.
//

#include "Matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

//returning by value simply copys the address of the pointer, so no memory leak
Matrix create_matrix(uint16_t rows, uint16_t cols){
    Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.values = (float*) calloc(rows * cols, sizeof(float));
    return mat;
}

Matrix create_matrix_from_values(uint16_t rows, uint16_t cols, float* values){
    Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.values = values;
    return mat;
}

Matrix dot(Matrix* mat_one, Matrix* mat_two){
    if (mat_one->rows != mat_two->rows && mat_one->cols != mat_two->cols)
        return create_matrix(0, 0);
    
    Matrix new_mat = create_matrix(mat_one->rows, mat_one->cols);
    
    for (size_t r = 0; r < mat_one->rows; ++r){
        for (size_t c = 0; c < mat_one->cols; ++c){
            uint16_t index = r * mat_one->cols + c;
            *(new_mat.values + index) = (*(mat_one->values + index)) * (*(mat_two->values + index));
        }
    }
    return new_mat;
}

Matrix mult(Matrix* mat_one, Matrix* mat_two){
    if (mat_one->cols != mat_two->rows)
        return create_matrix(0, 0);
    
    Matrix new_mat = create_matrix(mat_one->rows, mat_two->cols);
    for (size_t r = 0; r < mat_one->rows; ++r){
            for (size_t step = 0; step < mat_two->cols; ++step){
                
                
                float sum = 0.0f;
                for (size_t c = 0; c < mat_one->cols; ++c){
                    size_t index_first = r * mat_one->cols + c;
                    size_t index_second = c * mat_two->cols + step;
                    sum += (*(mat_one->values + index_first)) * (*(mat_two->values + index_second));
                }
                size_t index = r * mat_two->cols + step;
                *(new_mat.values +  index) = sum;
            }
    }
    return new_mat;
    
}

Matrix add(Matrix* mat_one, Matrix* mat_two){
    if (mat_one->rows != mat_two->rows && mat_one->cols != mat_two->cols)
        return create_matrix(0, 0);
    
    Matrix new_mat = create_matrix(mat_one->rows, mat_one->cols);
    for (size_t r = 0; r < mat_one->rows; ++r){
        for (size_t c = 0; c < mat_one->cols; ++c){
            uint16_t index = r * mat_one->cols + c;
            *(new_mat.values + index) = (*(mat_one->values + index)) + (*(mat_two->values + index));
        }
    }
    return new_mat;
}

Matrix sub(Matrix* mat_one, Matrix* mat_two){
    if (mat_one->rows != mat_two->rows && mat_one->cols != mat_two->cols)
        return create_matrix(0, 0);
    
    Matrix new_mat = create_matrix(mat_one->rows, mat_one->cols);
    for (size_t r = 0; r < mat_one->rows; ++r){
        for (size_t c = 0; c < mat_one->cols; ++c){
            uint16_t index = r * mat_one->cols + c;
            *(new_mat.values + index) = (*(mat_one->values + index)) - (*(mat_two->values + index));
        }
    }
    return new_mat;
}


void mut_add(Matrix* mat_one, Matrix* mat_two){
    if (mat_one->rows != mat_two->rows && mat_one->cols != mat_two->cols)
        return;
    
    for (size_t r = 0; r < mat_one->rows; ++r){
        for (size_t c = 0; c < mat_one->cols; ++c){
            uint16_t index = r * mat_one->cols + c;
            *(mat_one->values + index) = (*(mat_one->values + index)) + (*(mat_two->values + index));
        }
    }
}

void mut_sub(Matrix* mat_one, Matrix* mat_two){
    if (mat_one->rows != mat_two->rows && mat_one->cols != mat_two->cols)
        return;
    
    for (size_t r = 0; r < mat_one->rows; ++r){
        for (size_t c = 0; c < mat_one->cols; ++c){
            uint16_t index = r * mat_one->cols + c;
            *(mat_one->values + index) = (*(mat_one->values + index)) - (*(mat_two->values + index));
        }
    }
}

void scalar_mult(Matrix* mat_one, float scalar){
    for (size_t r = 0; r < mat_one->rows; ++r){
        for (size_t c = 0; c < mat_one->cols; ++c){
            uint16_t index = r * mat_one->cols + c;
            *(mat_one->values + index) *= scalar;
        }
    }
}

void scalar_div(Matrix* mat_one, float scalar){
    for (size_t r = 0; r < mat_one->rows; ++r){
        for (size_t c = 0; c < mat_one->cols; ++c){
            uint16_t index = r * mat_one->cols + c;
            *(mat_one->values + index) /= scalar;
        }
    }
}

void forEach(Matrix* mat, int (*func)(float)){
    for (size_t r = 0; r < mat->rows; ++r){
        for (size_t c = 0; c < mat->cols; ++c){
            uint16_t index = r * mat->cols + c;
            *(mat->values + index) = func(*(mat->values + index));
        }
    }
}

void tranpose(Matrix* mat){
    size_t temp = mat->cols;
    mat->cols = mat->rows;
    mat->rows = temp;
}

Matrix copy(Matrix* mat){
    Matrix m;
    m.rows = mat->rows;
    m.cols = mat->cols;
    memcpy(m.values, mat->values, mat->rows * mat->cols);
    return m;
}

void delete_matrix(Matrix* mat){
    free(mat->values);
}

void print_matrix(Matrix* mat){
    printf("[");
    for (size_t r = 0; r < mat->rows; ++r){
        for (size_t c = 0; c < mat->cols; ++c){
            uint16_t index = r * mat->cols + c;
            printf("%.2f", *(mat->values + index));
            
            if (c != mat->cols - 1)
                printf(", ");
        }
        if (r != mat->rows - 1)
            printf("\n");
    }
    printf("]");
    
}
