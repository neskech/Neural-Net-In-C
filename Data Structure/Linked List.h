//
//  Linked List.h
//  Neural Net
//
//  Created by Shaunte Mellor on 6/23/22.
//

#ifndef Linked_List_h
#define Linked_List_h

#include <stdio.h>

typedef struct Node{
    struct Node* next;
    int data;
} Node;

typedef struct LinkedList{
    size_t size;
    Node* head;
} LinkedList;

//can return by value because the address of the pointer is copied, so no data is lost...
LinkedList new_linked(void);
void destroy_linked(LinkedList* list);
void append_to_linked(LinkedList* list, int elem);
void delete_from_linked(LinkedList* list, int elem);
uint8_t linked_contains(LinkedList* list, int elem);

#endif /* Linked_List_h */
