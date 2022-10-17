//
//  Model.c
//  Neural Net
//
//  Created by Shaunte Mellor on 5/2/22.
//

#include "Model.h"
#include "Vector.h"

void addLayer(Model* model, void* layer){
    push(&model->layers, layer);
}

