
//
//  Vector.c
//  Neural Net
//
//  Created by Shaunte Mellor on 5/7/22.
//

#include "Vector.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

//can return by value, as the address of the pointer to the inner array is copied, and as such, not lost
Vector create_vector(size_t capacity){
    Vector vec;
    vec.size = 0;
    vec.capacity = capacity;

    vec.elements = (int*) calloc(capacity, sizeof(int));
    return vec;
}


static void resize(Vector* vec){
    vec->capacity *= 2;
    vec->elements = (int*) realloc(vec->elements, sizeof(int) * (vec->capacity));
}

static void shrink(Vector* vec){
    vec->capacity /= 2;
    vec->elements = (int*) realloc(vec->elements, sizeof(int) * (vec->capacity));
}

void push(Vector* vec, int elem){
    if (vec->size + 1 >= vec->capacity)
        resize(vec);

    *(vec->elements + vec->size) = elem;

    vec->size += 1;
}

int pop(Vector* vec){
    int hold = *(vec->elements + vec->size);

    vec->size -= 1;
    if (vec->size <= vec->capacity / 2)
        shrink(vec);

    return hold;
}

int get(Vector* vec, size_t index){
    return *(vec->elements + index);
}

int remove_at(Vector* vec, size_t index){
    int hold = *(vec->elements + index);
    for (size_t i = index; i < vec->size - 1; i++){
        vec->elements[i] = vec->elements[i + 1];
    }

    vec->size -= 1;
    if (vec->size <= vec->capacity / 2)
        shrink(vec);
    
    return hold;

}

void add_at(Vector* vec, size_t index, int elem){
    if (vec->size + 1 >= vec->capacity)
        resize(vec);

    for (size_t i = vec->size; i > index; i++){
        *(vec->elements + i) = *(vec->elements + i - 1);
    }

    *(vec->elements + index) = elem;

    vec->size += 1;
}

void set_element(Vector* vec, size_t index, int elem){
    *(vec->elements + index) = elem;
}

void delete_vector(Vector* vec){
    free(vec->elements);
}

void print_vector(Vector* vec){
    printf("[");
    for (size_t i = 0; i < vec->size; i++){
        printf("%d", vec->elements[i]);
        if (i != vec->size - 1)
            printf(", ");
    }
    printf("]\n");
}
