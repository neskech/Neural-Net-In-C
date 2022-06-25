//
//  Float Vector.h
//  Neural Net
//
//  Created by Shaunte Mellor on 6/23/22.
//

#ifndef Float_Vector_h
#define Float_Vector_h

#include <stdio.h>


typedef struct FloatVector{
    int* elements;
    size_t size;
    size_t capacity;
} FloatVector;

FloatVector create_FloatVector(size_t capacity);

void push_int_vec(FloatVector* vec, int elem);

int int_vec_pop(FloatVector* vec);

int int_vec_get(FloatVector* vec, size_t index);

int int_vec_remove_at(FloatVector* vec, size_t index);

void int_vec_add_at(FloatVector* vec, size_t index, int elem);

void int_vec_set_element(FloatVector* vec, size_t index, int elem);

void delete_FloatVector(FloatVector* vec);

#endif /* Float_Vector_h */
