//
//  Model.h
//  Neural Net
//
//  Created by Shaunte Mellor on 5/2/22.
//

#ifndef Model_h
#define Model_h

#include <stdlib.h>
#include "Activations.h"
#include "Loss.h"
#include "Matrix.h"
#include "Vector.h"

typedef struct Gradients{
    Matrix* weights;
    Matrix* biases;
} Gradients;

typedef struct ForwardPassCache{
    Matrix* activations; //array of activations
    Matrix* outputs;
} ForwardPassCache;



typedef struct ModelParams{
    float learning_rate;
    uint32_t batch_size;
    uint8_t verbose;
    
    float momentum;
    float EXPWA;
    float epsillon;
    
} ModelParams;

typedef struct LearningRateTuning{
    uint8_t use_tuning;
    uint32_t patience;
    float decrease;
    float min;
} LearningRateTuning;


typedef struct Model{
    uint8_t numLayers;
    Vector layerSizes;
    Vector activations;
    
    Matrix* weights;
    Matrix* biases;
    
    Loss loss_func;
    
    ModelParams params;
    LearningRateTuning tuning;

} Model;



Model* create_model(void);

void delete_model(Model* model);



float loss_on_dataset(Model* m, Matrix* x, Matrix* y, uint32_t num_data_points);
float estimated_loss_on_dataset(Model* m, Matrix* x, Matrix* y, uint32_t desired_data_points);



void add_layer(Model* m, int elem, Activation act);

void add_loss_func(Model* m, Loss loss_func_);

uint8_t compile(Model* m);

void init_weights_and_biases(Model* m, float mean, float standard_deviation);

void train(Model* m, Matrix* x, Matrix* y, uint32_t num_data_points, uint32_t num_iterations);


Matrix eval(Model* m, Matrix* x);

void summary(Model* m, uint8_t print_matrices);

#endif /* Model_h */
