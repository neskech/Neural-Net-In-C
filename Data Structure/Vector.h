//
//  IntVector.h
//  Neural Net
//
//  Created by Shaunte Mellor on 6/20/22.
//

#ifndef IntVector_h
#define IntVector_h

#include <stdio.h>


typedef struct IntVector{
    int* elements;
    size_t size;
    size_t capacity;
} IntVector;

IntVector create_IntVector(size_t capacity);

void push(IntVector* vec, int elem);

int pop(IntVector* vec);

int get(IntVector* vec, size_t index);

int remove_at(IntVector* vec, size_t index);

void add_at(IntVector* vec, size_t index, int elem);

void set_element(IntVector* vec, size_t index, int elem);

void delete_IntVector(IntVector* vec);


#endif /* IntVector_h */
