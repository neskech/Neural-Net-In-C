
//
//  IntVector.c
//  Neural Net
//
//  Created by Shaunte Mellor on 5/7/22.
//

#include "Int Vector.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

//can return by value, as the address of the pointer to the inner array is copied, and as such, not lost
IntVector create_IntVector(size_t capacity){
    IntVector vec;
    vec.size = 0;
    vec.capacity = capacity;

    vec.elements = (int*) calloc(capacity, sizeof(int));
    return vec;
}


static void resize(IntVector* vec){
    vec->capacity *= 2;
    vec->elements = (int*) realloc(vec->elements, sizeof(int) * (vec->capacity));
}

static void shrink(IntVector* vec){
    vec->capacity /= 2;
    vec->elements = (int*) realloc(vec->elements, sizeof(int) * (vec->capacity));
}

void push_int_vec(IntVector* vec, int elem){
    if (vec->size + 1 >= vec->capacity)
        resize(vec);

    *(vec->elements + vec->size) = elem;

    vec->size += 1;
}

int pop_int_vec(IntVector* vec){
    int hold = *(vec->elements + vec->size);

    vec->size -= 1;
    if (vec->size <= vec->capacity / 2)
        shrink(vec);

    return hold;
}

int get_int_vec(IntVector* vec, size_t index){
    return *(vec->elements + index);
}

int int_vec_remove_at(IntVector* vec, size_t index){
    int hold = *(vec->elements + index);
    for (size_t i = index; i < vec->size - 1; i++){
        *(vec->elements + i) = *(vec->elements + i + 1);
    }

    vec->size -= 1;
    if (vec->size <= vec->capacity / 2)
        shrink(vec);
    
    return hold;

}

void int_vec_add_at(IntVector* vec, size_t index, int elem){
    if (vec->size + 1 >= vec->capacity)
        resize(vec);

    for (size_t i = vec->size; i > index; i++){
        *(vec->elements + i) = *(vec->elements + i - 1);
    }

    *(vec->elements + index) = elem;

    vec->size += 1;
}

void int_vec_set_element(IntVector* vec, size_t index, int elem){
    *(vec->elements + index) = elem;
}

void delete_IntVector(IntVector* vec){
    free(vec->elements);
}
