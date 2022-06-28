//
//  Model.h
//  Neural Net
//
//  Created by Shaunte Mellor on 5/2/22.
//

#ifndef Model_h
#define Model_h

#include "pch.h"
#include "Model/Activations.h"
#include "Model/Loss.h"
#include "Model/Matrix.h"
#include "Data Structure/Vector.h"

typedef struct ModelParams{
    float learning_rate;
    uint32_t batch_size;
    uint8_t verbose;
    
    float momentum;
    float momentum2;
    float epsillon;
    
} ModelParams;

typedef struct LearningRateTuning{
    uint32_t patience;
    float decrease;
    float min;
} LearningRateTuning;


typedef struct Model{
    uint8_t num_layers; //never going to exceed more than 255 layers (hopefully)
    Vector layer_sizes;
    Vector activations;
    
    Matrix* weights;
    Matrix* biases;
    
    //expwa = exponentially weighted averged
    Matrix* expwa_weights;
    Matrix* expwa_biases;
    
    Matrix* expwa_weights_squared;
    Matrix* expwa_biases_squared;
    
    Loss loss_func;
    
    ModelParams params;
    uint8_t use_tuning;
    LearningRateTuning tuning;
} Model;



Model* create_model(ModelParams* params, LearningRateTuning* tuning);

Model* load_model(const char* path);

uint8_t save_model(Model* m, const char* path);

void delete_model(Model* model);



float loss_on_dataset(Model* m, Matrix* x, Matrix* y, uint32_t num_data_points);

float accuracy_on_dataset(Model* m, Matrix* x, Matrix* y, uint32_t num_data_points);

void add_layer(Model* m, int elem, Activation act);

void set_loss_func(Model* m, Loss loss_func_);

void init_weights_and_biases(Model* m, float mean, float standard_deviation); //to be used after compile...

uint8_t compile(Model* m);



Matrix eval(Model* m, Matrix* x);

void summary(Model* m, uint8_t print_matrices);

#endif /* Model_h */
