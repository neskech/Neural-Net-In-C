//
//  Matrix.c
//  Neural Net
//
//  Created by Shaunte Mellor on 5/1/22.
//

#include "Model/Matrix.h"
#include "pch.h"

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

void set_values_with(Matrix* mat, float val){
    for (size_t i = 0; i < mat->rows * mat->cols; i++){
        *(mat->values + i) = val;
    }
}

void delete_matrix(Matrix* mat){
    free(mat->values);
}

void move_matrix(Matrix* from, Matrix* to){
    if (to->values != NULL)
        free(to->values);
    
    to->values = from->values;
    to->rows = from->rows;
    to->cols = from->cols;
    from->values = NULL;
}

Matrix matrix_copy(Matrix* mat){
    Matrix m = create_matrix(mat->rows, mat->cols);
    
    memcpy(m.values, mat->values, mat->rows * mat->cols * sizeof(float));
    return m;
}

Matrix transpose(Matrix* mat){
    Matrix trans = create_matrix(mat->cols, mat->rows);
    
    for (size_t r = 0; r < mat->rows; ++r){
        for (size_t c = 0; c < mat->cols; ++c){
            uint16_t index_1 = r * mat->cols + c;
            uint16_t index_2 = c * mat->rows + r;
            trans.values[index_2] = mat->values[index_1];
        }
    }
    
    delete_matrix(mat);
    return trans;
}

Matrix dot(Matrix* mat_one, Matrix* mat_two){
    if (mat_one->rows != mat_two->rows && mat_one->cols != mat_two->cols){
        fprintf(stderr, "ERROR: Matrix dimensions of (%d x %d) and (%d x %d) unfit for element wise multiplication. Returning...",
        mat_one->rows, mat_one->cols, mat_two->rows, mat_two->cols);
        return create_matrix(0, 0);
    }
    
    Matrix new_mat = create_matrix(mat_one->rows, mat_one->cols);
    
    for (size_t r = 0; r < mat_one->rows; ++r){
        for (size_t c = 0; c < mat_one->cols; ++c){
            uint16_t index = r * mat_one->cols + c;
            *(new_mat.values + index) = (*(mat_one->values + index)) * (*(mat_two->values + index));
        }
    }
    return new_mat;
}

Matrix matrix_div(Matrix* mat_one, Matrix* mat_two){
    if (mat_one->rows != mat_two->rows && mat_one->cols != mat_two->cols){
        fprintf(stderr, "ERROR: Matrix dimensions unfit for element wise division. Returning 0-sized matrix...");
        return create_matrix(0, 0);
    }
    
    Matrix new_mat = create_matrix(mat_one->rows, mat_one->cols);
    
    for (size_t r = 0; r < mat_one->rows; ++r){
        for (size_t c = 0; c < mat_one->cols; ++c){
            uint16_t index = r * mat_one->cols + c;
            new_mat.values[index] = mat_one->values[index] / mat_two->values[index];
        }
    }
    return new_mat;
}

Matrix mult(Matrix* mat_one, Matrix* mat_two){
    if (mat_one->cols != mat_two->rows){
        fprintf(stderr, "ERROR: Matrix dimensions unfit for multiplication. Returning 0-sized matrix...");
        exit(-1);
    }
    
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
    if (mat_one->rows != mat_two->rows && mat_one->cols != mat_two->cols){
        fprintf(stderr, "ERROR: Matrix dimensions unfit for addition. Returning...");
        return create_matrix(0, 0);
    }
    
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
    if (mat_one->rows != mat_two->rows && mat_one->cols != mat_two->cols){
        fprintf(stderr, "ERROR: Matrix dimensions unfit for subtraction. Returning...");
        return create_matrix(0, 0);
    }
    
    Matrix new_mat = create_matrix(mat_one->rows, mat_one->cols);
    for (size_t r = 0; r < mat_one->rows; ++r){
        for (size_t c = 0; c < mat_one->cols; ++c){
            uint16_t index = r * mat_one->cols + c;
            *(new_mat.values + index) = (*(mat_one->values + index)) - (*(mat_two->values + index));
        }
    }
    return new_mat;
}










void dot_in_place(Matrix* mat_one, Matrix* mat_two){
    if (mat_one->rows != mat_two->rows && mat_one->cols != mat_two->cols){
        fprintf(stderr, "ERROR: Matrix dimensions unfit for element wise multiplication (in place). Returning...");
        return;
    };
    for (size_t r = 0; r < mat_one->rows; ++r){
        for (size_t c = 0; c < mat_one->cols; ++c){
            uint16_t index = r * mat_one->cols + c;
            mat_one->values[index] *= mat_two->values[index];
        }
    }
}

void div_in_place(Matrix* mat_one, Matrix* mat_two){
    if (mat_one->rows != mat_two->rows && mat_one->cols != mat_two->cols){
        fprintf(stderr, "ERROR: Matrix dimensions unfit for element wise multiplication (in place). Returning...");
        return;
    };
    for (size_t r = 0; r < mat_one->rows; ++r){
        for (size_t c = 0; c < mat_one->cols; ++c){
            uint16_t index = r * mat_one->cols + c;
            mat_one->values[index] /= mat_two->values[index];
        }
    }
}

void add_in_place(Matrix* mat_one, Matrix* mat_two){
    if (mat_one->rows != mat_two->rows && mat_one->cols != mat_two->cols){
        fprintf(stderr, "ERROR: Matrix dimensions unfit for addition (in place). Returning...");
        return;
    }
    
    for (size_t r = 0; r < mat_one->rows; ++r){
        for (size_t c = 0; c < mat_one->cols; ++c){
            uint16_t index = r * mat_one->cols + c;
            mat_one->values[index] += mat_two->values[index];
        }
    }
}

void sub_in_place(Matrix* mat_one, Matrix* mat_two){
    if (mat_one->rows != mat_two->rows && mat_one->cols != mat_two->cols){
        fprintf(stderr, "ERROR: Matrix dimensions unfit for subtraction (in place). Returning...");
        return;
    }
    
    for (size_t r = 0; r < mat_one->rows; ++r){
        for (size_t c = 0; c < mat_one->cols; ++c){
            uint16_t index = r * mat_one->cols + c;
            mat_one->values[index] -= mat_two->values[index];
        }
    }
}








void scalar_mult(Matrix* mat, float scalar){
    for (size_t r = 0; r < mat->rows; ++r){
        for (size_t c = 0; c < mat->cols; ++c){
            uint16_t index = r * mat->cols + c;
            mat->values[index] *= scalar;
        }
    }
}

void scalar_div(Matrix* mat, float scalar){
    for (size_t r = 0; r < mat->rows; ++r){
        for (size_t c = 0; c < mat->cols; ++c){
            uint16_t index = r * mat->cols + c;
            mat->values[index] /= scalar;
        }
    }
}

void scalar_add(Matrix* mat, float scalar){
    for (size_t r = 0; r < mat->rows; ++r){
        for (size_t c = 0; c < mat->cols; ++c){
            uint16_t index = r * mat->cols + c;
            mat->values[index] += scalar;
        }
    }
}









void matrix_square(Matrix* mat){
    for (size_t r = 0; r < mat->rows; ++r){
        for (size_t c = 0; c < mat->cols; ++c){
            uint16_t index = r * mat->cols + c;
            mat->values[index] *= mat->values[index];
        }
    }
}

void matrix_sqrt(Matrix* mat){
    for (size_t r = 0; r < mat->rows; ++r){
        for (size_t c = 0; c < mat->cols; ++c){
            uint16_t index = r * mat->cols + c;
            mat->values[index] = sqrtf(mat->values[index]);
        }
    }
}

float magnitude(Matrix* mat){
    float mag = 0.0f;
    for (size_t r = 0; r < mat->rows; ++r){
        for (size_t c = 0; c < mat->cols; ++c){
            uint16_t index = r * mat->cols + c;
            mag += mat->values[index] * mat->values[index];
        }
    }
    return mag;
}

void reciprocal(Matrix* mat){
    for (size_t r = 0; r < mat->rows; ++r){
        for (size_t c = 0; c < mat->cols; ++c){
            uint16_t index = r * mat->cols + c;
            mat->values[index] = 1.0f / mat->values[index];
        }
    }
}






size_t size(Matrix* mat){ return mat->rows * mat->cols; }

void print_matrix(Matrix* mat){
    
    for (size_t r = 0; r < mat->rows; ++r){
        printf("[");
        for (size_t c = 0; c < mat->cols; ++c){
            uint16_t index = r * mat->cols + c;
            printf("%.6f", *(mat->values + index));
            
            if (c != mat->cols - 1)
                printf(", ");
        }
        printf("]");
        if (r != mat->rows - 1)
            printf("\n");
    }
    printf("\n\n");
    
}
