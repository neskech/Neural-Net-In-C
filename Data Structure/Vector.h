
//  Vector.h
//  Neural Net
//
//  Created by Shaunte Mellor on 6/20/22.
//

#ifndef Vector_h
#define Vector_h

#include <stdio.h>


typedef struct Vector{
    int* elements;
    size_t size;
    size_t capacity;
} Vector;

Vector create_vector(size_t capacity);

void delete_vector(Vector* vec);



void push(Vector* vec, int elem);

int pop(Vector* vec);

int get(Vector* vec, size_t index);

int remove_at(Vector* vec, size_t index);

void add_at(Vector* vec, size_t index, int elem);

void set_element(Vector* vec, size_t index, int elem);



void print_vector(Vector* vec);


#endif /* Vector_h */
