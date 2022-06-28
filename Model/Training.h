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

void train(Model* m, Matrix* x, Matrix* y, uint32_t num_data_points, uint32_t num_epochs, uint8_t write_to_file, const char* file_name);

void MAX_THREADS(size_t num_threads);
void ENABLE_THREADING(void);

#endif /* Training_h */
