//
//  main.c
//  Neural Net
//
//  Created by Shaunte Mellor on 5/1/22.
//

#include <stdio.h>
#include "Model.h"

int main(int argc, const char * argv[]) {
    Model* m = create_model();
    
    add_layer(m, 3, RELU);
    add_layer(m, 3, RELU);
    add_layer(m, 9, RELU);
    
    compile(m);
    init_weights_and_biases(m, 0, 0);
    
    summary(m);
    
    delete_model(m);
    
    
    return 0;
}


