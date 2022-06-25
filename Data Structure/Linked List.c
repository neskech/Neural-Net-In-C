//
//  Linked List.c
//  Neural Net
//
//  Created by Shaunte Mellor on 6/23/22.
//

#include "Linked List.h"
#include <stdlib.h>

LinkedList new_linked(void){
    LinkedList list;
    list.head = (Node*) malloc(sizeof(Node));
    list.size = 0;
    return list;
}

void destroy_linked(LinkedList* list){
    Node* curr = list->head;
    while (curr != NULL){
        Node* next_node = curr->next;
        free(curr);
        curr = next_node;
    }
}

void append_to_linked(LinkedList* list, int elem){
    Node* curr = list->head;
    while (curr->next != NULL)
        curr = curr->next;
    
    curr->next = (Node*) malloc(sizeof(Node));
    curr->next->data = elem;
}

void delete_from_linked(LinkedList* list, int elem){
    //if the size is 0...
    if (list->size == 0)
        return;
    
    //if the node to be deleted is the first node...
    if (list->head->data == elem){
        Node* new_head = list->head->next;
        free(list->head);
        list->head = new_head;
        
        list->size--;
        return;
    }
        
    Node* curr = list->head;
    while (curr->next != NULL){
        
        if (curr->next->data == elem){
            Node* next_next = curr->next->next;
            free(curr->next);
            curr->next = next_next;
            
            list->size--;
            return;
        }
            
        curr = curr->next;
    }
}

uint8_t linked_contains(LinkedList* list, int elem){
    Node* curr = list->head;
    while (curr->next != NULL){
        
        if (curr->data == elem)
            return 1;
            
        curr = curr->next;
    }
    
    return 0;
}
