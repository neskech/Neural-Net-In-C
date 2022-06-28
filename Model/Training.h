//
//  Training.h
//  Neural Net
//
//  Created by Shaunte Mellor on 6/25/22.
//

#ifndef Training_h
#define Training_h

#include <stdio.h>
#include "Model.h"

//trains the model. Optionally writes training data to a file. If NULL is passed in for the string, no such data will be written
uint8_t train(Model* m, Matrix* inputs, Matrix* observ, uint32_t num_data_points, uint32_t num_epochs, const char* file_name);


#endif /* Training_h */
