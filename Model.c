//
//  Model.c
//  Neural Net
//
//  Created by Shaunte Mellor on 5/2/22.
//

#include "Model.h"
#include <math.h>

Model* create_model(void){
    Model* m = (Model*) malloc(sizeof(Model));
    
    //preallocate some space in the vectors
    m->layerSizes = create_vector(5);
    m->activations = create_vector(5);
    return m;
}

void delete_model(Model* m){
    for (size_t i = 0; i < m->numLayers - 1; i++){
        delete_matrix(m->biases + i);
        if (i < m->numLayers - 1)
            delete_matrix(m->weights + i);
    }

    delete_vector(&m->layerSizes);
    delete_vector(&m->activations);
    free(m);
}




void add_layer(Model* m, int elem, Activation act){
    push(&m->layerSizes, elem);
    push(&m->activations, act);
}

void add_loss_func(Model* m, Loss loss_func_){
    m->loss_func = loss_func_;
}

uint8_t compile(Model* m){
    m->numLayers = m->layerSizes.size;
    if (m->numLayers <= 0)
        return 0;
    
    m->weights = (Matrix*) calloc(sizeof(Matrix), m->numLayers - 1);
    m->biases = (Matrix*) calloc(sizeof(Matrix), m->numLayers - 1);
    for (size_t i = 0; i < m->numLayers - 1; i++){
        *(m->biases + i) = create_matrix(get(&m->layerSizes, i + 1), 1);
        *(m->weights + i) = create_matrix(get(&m->layerSizes, i + 1), get(&m->layerSizes, i));
    }
    
    return 1;
}

void init_weights_and_biases(Model* m, float mean, float standard_dev){
    for (size_t i = 0; i < m->numLayers - 1; i++){
        //Box–Muller transform for generating numbers on a guassian distribution
        
        float val;
        for (size_t r = 0; r < (m->weights + i)->rows * (m->weights + i)->cols; r++){
            val = rand() / (float)RAND_MAX;
            
            if (r % 2 == 0)
                val = sqrt(-2.0f * log(val)) * cos(2.0f * M_PI * val);
            else
                val = sqrt(-2.0f * log(val)) * sin(2.0f * M_PI * val);
            
            val = val * standard_dev + mean;
            *((m->weights + i)->values + r) = val;
        }
        
        for (size_t r = 0; r < (m->biases + i)->rows; r++){
            val = rand() / (float)RAND_MAX;
            
            if (r % 2 == 0)
                val = sqrt(-2.0f * log(val)) * cos(2.0f * M_PI * val);
            else
                val = sqrt(-2.0f * log(val)) * sin(2.0f * M_PI * val);
            
            val = val * standard_dev + mean;
            *((m->biases + i)->values + r) = val;
        }
            
       
    }
}

Matrix eval(Model* m, Matrix* x){
    Matrix running = *x;
    
    
    for (size_t i = 0; i < m->numLayers - 1; i++){

            
        //when we copy a new value to running, the previous value's memory is lost. Be sure to delete it
        Matrix before = running;
        running = mult(m->weights + i, &running);
        delete_matrix(&before);
        
        add_in_place(&running, m->biases + i);
        
        act_func(&running, get(&m->activations, i));
        
    }
    
    return running;
}



float loss_on_dataset(Model* m, Matrix* x, Matrix* y, uint32_t num_data_points){
    float summed_loss = 0.0f;
    for (uint32_t i = 0; i < num_data_points; i++){
        Matrix eval_ = eval(m, x + i);
        summed_loss += loss_func(&eval_, y + i, m->loss_func);
        delete_matrix(&eval_);
    }
    return summed_loss
}

float estimated_loss_on_dataset(Model* m, Matrix* x, Matrix* y, uint32_t num_data_points){
    //Only sample a portion of the dataset. To avoid bias, make this sampling randomized
    
}




static void forwardProp(Model* m, Matrix* x, ForwardPassCache* cache){
    Matrix running = *x;
    
    *(cache->activations + 0) = matrix_copy(x);
    
    for (size_t i = 0; i < m->numLayers - 1; i++){

            
        //when we copy a new value to running, the previous value's memory is lost. Be sure to delete it
        Matrix before = running;
        running = mult(m->weights + i, &running);
        delete_matrix(&before);
        
        add_in_place(&running, m->biases + i);
        
        *(cache->outputs + i) = matrix_copy(&running);
        
        act_func(&running, get(&m->activations, i));
        
        *(cache->activations + i + 1) = matrix_copy(&running);
        
    }
    
    delete_matrix(&running);
}

static void backProp(Model* m, Matrix* observ, ForwardPassCache* cache, Gradients* grads){
    //Last value of the activations array is the final output of the network
    Matrix* pred = cache->activations + (m->numLayers - 1);
    Matrix running_deriv = loss_func_deriv(pred, observ, m->loss_func);
    
    //the weight and bias matrices correspond to the last two layers (the input layer has neither weights nor biases)
    //the weights belonging to layer 2 are at index 1, the weights to layer 3 are at index 2, etc....
    for (int32_t i = m->numLayers - 2; i >= 0; i--){
        
        //Get the activation functions derivative...
        Activation act = get(&m->activations, i + 1);
        act_func_deriv(cache->outputs + i, act); //Stores the derivative in outputs[i]
        
        //element-wise-multiply the activation derivatives by running_deriv...
        dot_in_place(&running_deriv, cache->outputs + i);
        
        //Get the derivative of the weights and biases and store them...
        grads->biases[i] = running_deriv;
        //matrix multiply the outputs of layer - 1 (represented by activations[i]) and the running_deriv
        grads->weights[i] = mult(cache->activations + i, &running_deriv);
        
        //Continue on with the chain rule by multiplying the weights transpose by the running_deriv
        if (i != 0){
            transpose(m->weights + i);
            
            //since matrix multiplication creates a new matrix, we must delete the previous value of
            //running_deriv to prevent a memory leak...
            Matrix before = running_deriv;
            running_deriv = mult(m->weights, &running_deriv);
            delete_matrix(&before);
            
            transpose(m->weights + i); //undo transpose
        }
    }
}


static void free_cache(ForwardPassCache* cache, uint16_t numLayers){
    if (cache->activations == NULL || cache->outputs == NULL)
        return;
    
    for (size_t i = 0; i < numLayers; i++){
        delete_matrix(cache->activations + i);
        
        if (i < numLayers - 1){
            delete_matrix(cache->outputs + i);
        }
    }
        
}

static void free_gradients(Gradients* grads, uint16_t numLayers){
    for (size_t i = 0; i < numLayers - 1; i++){
        delete_matrix(grads->biases + i);
        delete_matrix(grads->weights + i);
    }
}

static void apply_batching(Model* m, Matrix* inputs, Matrix* observ, ForwardPassCache* cache, Gradients* avg_grads){
    //allocate the gradients that will be used from datapoint to datapoint
    Gradients grads;
    grads.weights = (Matrix*) calloc(sizeof(Matrix), m->numLayers - 1);
    grads.biases = (Matrix*) calloc(sizeof(Matrix), m->numLayers - 1);
    
    //fill an array of random indices to choose from the datatset
    uint32_t* indices = (uint32_t*) calloc(sizeof(uint32_t), m->params.batch_size);
    
    for (uint32_t i = 0; i < m->params.batch_size; i++){
        indices[i] = (uint32_t) (rand() / (float) RAND_MAX);
    }
    
    //now go through the random data-indices and generate gradients
    for (uint32_t i = 0; i < m->params.batch_size; i++){
        forwardProp(m, inputs + i, cache);
        backProp(m, observ, cache, &grads);
        
        for (size_t j = 0; j < m->numLayers - 1; j++){
            scalar_div(grads.weights + j, m->params.batch_size);
            scalar_div(grads.biases + j, m->params.batch_size);
            
            add_in_place(avg_grads->weights + j, grads.weights + j);
            add_in_place(avg_grads->biases + j, grads.biases + j);
        }
        
        free_cache(cache, m->numLayers);
        free_gradients(&grads, m->numLayers);
    }
    
    free(indices);
    free(grads.weights);
    free(grads.biases);
}

static void apply_gradients(Model* m, Gradients* grads){
    for (size_t i = 0; i < m->numLayers - 1; i++){
        scalar_mult(grads->weights + i, m->params.learning_rate);
        scalar_mult(grads->biases + i, m->params.learning_rate);
        
        sub_in_place(m->weights + i, grads->weights + i);
        sub_in_place(m->biases + i, grads->biases + i);
    }
}


void train(Model* m, Matrix* x, Matrix* y, uint32_t num_iterations){
    
    ForwardPassCache cache;
    cache.activations = (Matrix*) calloc(sizeof(Matrix), m->numLayers);
    cache.outputs = (Matrix*) calloc(sizeof(Matrix), m->numLayers - 1);
    
    Gradients grads;
    grads.weights = (Matrix*) calloc(sizeof(Matrix), m->numLayers - 1);
    grads.biases = (Matrix*) calloc(sizeof(Matrix), m->numLayers - 1);
    
    float loss = 0.0f;
    for (uint32_t i = 0; i < num_iterations; i++){
        
        
        
        apply_batching(m, x, y, &cache, &grads);
        apply_gradients(m, &grads);
        free_gradients(&grads, m->numLayers);
    }
    
    free(cache.activations);
    free(cache.outputs);
    
    free(grads.weights);
    free(grads.biases);
}





static size_t total_params(Model* m){
    size_t total_params = 0;
    for (size_t i = 0; i < m->numLayers - 1; i++){
        total_params += size(m->weights + i);
        total_params += size(m->biases + i);
    }
    return total_params;
}

void summary(Model* m, uint8_t print_matrices){
    //Print Layer Sizes
    printf("Layer Sizes: ");
    for (size_t i = 0; i < m->numLayers; i++){
        printf("%d", get(&m->layerSizes, i));
        if (i != m->numLayers - 1)
            printf(", ");
    }
    
    printf("\n------------------------------------------\nActivation Functions: ");
    for (size_t i = 0; i < m->numLayers; i++){
        printf("%d", get(&m->activations, i));
        if (i != m->numLayers - 1)
            printf(", ");
    }
    
    
    if (print_matrices){
        
        
            printf("\n------------------------------------------\nWeights:\n\n");
            for (size_t i = 0; i < m->numLayers - 1; i++){
                print_matrix(m->weights + i);
                printf("\n\n");
            }
            
            printf("------------------------------------------\nABiases:\n\n");
            for (size_t i = 0; i < m->numLayers - 1; i++){
                print_matrix(m->biases + i);
                printf("\n\n");
            }
        
    }
    
    size_t params = total_params(m);
    printf("------------------------------------------\n\nTotal Parameters: %zu\n", params);
}
