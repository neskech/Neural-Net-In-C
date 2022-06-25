//
//  Set.h
//  Neural Net
//
//  Created by Shaunte Mellor on 6/22/22.
//

#ifndef Set_h
#define Set_h

#include <stdio.h>
#include "Vector.h"
#include "Linked List.h"

typedef struct Set{
    LinkedList* lists;
} Set;

uint32_t index_from_hash(int num);

void add_to_set(int elem);
void remove_from_set(int elem);

#endif /* Set_h */
