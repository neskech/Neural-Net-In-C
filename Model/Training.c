//
//  Training.c
//  Neural Net
//
//  Created by Shaunte Mellor on 6/25/22.
//

#include "Training.h"
#include <time.h>
#include <math.h>
#include <string.h>
#include <pthread.h>

static uint8_t IS_THREADING_ENABLED = 0;
static size_t NUM_AVAILABLE_THREADS = -1;

void MAX_THREADS(size_t num_threads){
    NUM_AVAILABLE_THREADS = num_threads;
}
void ENABLE_THREADING(void){
    IS_THREADING_ENABLED = 1;
}

static void write_meta_data(const char* path, float* loss_data, float* gradient_mag_data, uint32_t num_epochs){
    FILE* f;
    f = fopen(path, "w");
    
    //if f == NULL or path is not a json file
    unsigned long len = strlen(path);
    if (f == NULL || strcmp("json", path + len - 4) != 0){
        printf("Error writing to file path of %s. Exiting...\n", path);
        fclose(f);
        return;
    }
    
    fprintf(f, "{\n\n");
    
    fprintf(f, "  \"Loss\": [\n");
    for (uint32_t i = 0; i < num_epochs; i++){
        if (i != num_epochs - 1)
            fprintf(f, "    %f,\n", loss_data[i]);
        else
            fprintf(f, "    %f\n", loss_data[i]);
    }
    fprintf(f, "   ],\n\n");
    
    fprintf(f, "  \"GradientMag\": [\n");
    for (uint32_t i = 0; i < num_epochs; i++){
        if (i != num_epochs - 1)
            fprintf(f, "    %f,\n", gradient_mag_data[i]);
        else
            fprintf(f, "    %f\n", gradient_mag_data[i]);
    }
    fprintf(f, "   ]\n\n");
    
    fprintf(f, "}\n");
    
    fclose(f);
    
}

static Vector randomize_dataset(uint32_t num_data_points){
    //generate a list of random indices from which to sample from
    Vector vec = create_vector(num_data_points);
    for (uint32_t i = 0; i < num_data_points; i++)
        push(&vec, i);
    
    Vector indices = create_vector(num_data_points);
    for (uint32_t i = 0; i < num_data_points; i++){
        uint32_t idx = (uint32_t) ( vec.size * (rand() / (float) RAND_MAX) );
        push(&indices, get(&vec, idx));
        remove_at(&vec, idx);
    }
        
    delete_vector(&vec);
    return indices;
}


static void forwardProp(Model* m, Matrix* x, ForwardPassCache* cache){
    Matrix running = matrix_copy(x);

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
        Activation act = get(&m->activations, i);
        act_func_deriv(cache->outputs + i, observ, act); //Stores the derivative in outputs[i]
        
        //element-wise-multiply the activation derivatives by running_deriv...
        dot_in_place(&running_deriv, cache->outputs + i);
        
        //Get the derivative of the weights and biases and store them...
        grads->biases[i] = matrix_copy(&running_deriv);
        
        //matrix multiply the outputs of layer - 1 (represented by activations[i]) and the running_deriv
        //Transpose the activations so the resulting matrix has the same shape as the weights matrix
        cache->activations[i]  = transpose(cache->activations + i);
        grads->weights[i] = mult(&running_deriv, cache->activations + i);
        
        
        //Continue on with the chain rule by multiplying the weights transpose by the running_deriv
        if (i != 0){
           Matrix trans_weights = transpose_copy(m->weights + i);
            //since matrix multiplication creates a new matrix, we must delete the previous value of
            //running_deriv to prevent a memory leak...
            Matrix before = running_deriv;
            running_deriv = mult(&trans_weights, &running_deriv);
            delete_matrix(&before);
            
            delete_matrix(&trans_weights);
           // m->weights[i] = transpose(m->weights + i); //undo transpose
        }
    }
}


static void free_cache(ForwardPassCache* cache, uint16_t numLayers){
//    if (cache->activations == NULL || cache->outputs == NULL)
//        return;
    
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

static void reset_gradients(Gradients* grads, uint16_t numLayers){
    for (size_t i = 0; i < numLayers - 1; i++){
        set_values_with(grads->weights + i, 0.0f);
        set_values_with(grads->biases + i, 0.0f);
    }
}

static void* get_gradients(Model* m, Matrix* inputs, Matrix* observ, Vector* indices, uint32_t offset, uint32_t num_data_points, Gradients* collective_grads, float* cumulative_loss){
    
    //allocate the gradients that will be used from datapoint to datapoint
    Gradients grads;
    grads.weights = (Matrix*) calloc(sizeof(Matrix), m->numLayers - 1);
    grads.biases = (Matrix*) calloc(sizeof(Matrix), m->numLayers - 1);
    
    //allocate the cache used to store data from the forward pass
    ForwardPassCache cache;
    cache.activations = (Matrix*) calloc(sizeof(Matrix), m->numLayers);
    cache.outputs = (Matrix*) calloc(sizeof(Matrix), m->numLayers - 1);
    

    //now go through the random data-indices and generate gradients
    uint32_t size = MIN(num_data_points, offset + m->params.batch_size); //do not exceed data set size
    for (uint32_t i = offset; i < size; i++){
        uint32_t idx = get(indices, i);
        forwardProp(m, inputs + idx, &cache);
        backProp(m, observ + idx, &cache, &grads);
        printf("PREDICTED %u AND OBSERVED %u \n", argmax(cache.activations + m->numLayers - 1), argmax(observ + idx));
        
        
        for (size_t j = 0; j < m->numLayers - 1; j++){
            scalar_div(grads.weights + j, m->params.batch_size);
            scalar_div(grads.biases + j, m->params.batch_size);
            
            add_in_place(collective_grads->weights + j, grads.weights + j);
            add_in_place(collective_grads->biases + j, grads.biases + j);
        }
        
        *cumulative_loss += loss_func(cache.activations + m->numLayers - 1, observ + idx, m->loss_func);
        
        free_cache(&cache, m->numLayers);
        free_gradients(&grads, m->numLayers);
    }
    
    //free the containers
    free(cache.activations);
    free(cache.outputs);
    
    free(grads.weights);
    free(grads.biases);
    
    return NULL;
}

typedef struct Args{
    Model* model;
    Matrix* inputs;
    Matrix* observ;
    Vector* indices;
    uint32_t offset;
    uint32_t num_data_points;
    Gradients* collective_grads;
    float* cumulative_loss;
} Args;

static void* get_gradients_threaded(void* args_){
    Args* args = (Args*) args_;
    //allocate the gradients that will be used from datapoint to datapoint
    Gradients grads;
    grads.weights = (Matrix*) calloc(sizeof(Matrix), args->model->numLayers - 1);
    grads.biases = (Matrix*) calloc(sizeof(Matrix), args->model->numLayers - 1);
    
    //allocate the cache used to store data from the forward pass
    ForwardPassCache cache;
    cache.activations = (Matrix*) calloc(sizeof(Matrix), args->model->numLayers);
    cache.outputs = (Matrix*) calloc(sizeof(Matrix), args->model->numLayers - 1);
    

    //now go through the random data-indices and generate gradients
    uint32_t size = MIN(args->num_data_points, args->offset + args->model->params.batch_size); //do not exceed data set size
    for (uint32_t i = args->offset; i < size; i++){
        uint32_t idx = get(args->indices, i);
        forwardProp(args->model, args->inputs + idx, &cache);
        backProp(args->model, args->observ + idx, &cache, &grads);
        
        for (size_t j = 0; j < args->model->numLayers - 1; j++){
            scalar_div(grads.weights + j, args->model->params.batch_size);
            scalar_div(grads.biases + j, args->model->params.batch_size);
            
            add_in_place(args->collective_grads->weights + j, grads.weights + j);
            add_in_place(args->collective_grads->biases + j, grads.biases + j);
        }
        
        *args->cumulative_loss += loss_func(cache.activations + args->model->numLayers - 1, args->observ + idx, args->model->loss_func);
        
        free_cache(&cache, args->model->numLayers);
        free_gradients(&grads, args->model->numLayers);
    }
    
    //free the containers
    free(cache.activations);
    free(cache.outputs);
    
    free(grads.weights);
    free(grads.biases);
    
    return NULL;
}

static void apply_gradients(Model* m, Gradients* grads, float* gradient_mag, uint32_t time_step){
    for (size_t i = 0; i < m->numLayers - 1; i++){
        //Get the Mt and Vt of the current timestep...

        //Mt Weights...
        //Mt = B1 * Mt-1 + (1 - B1)Gt^1
        scalar_mult(m->expwa_weights + i, m->params.momentum);
        scalar_mult(grads->weights + i, 1.0f - m->params.momentum);
        add_in_place(m->expwa_weights + i, grads->weights + i);
        scalar_mult(grads->weights + i, 1.0f / (1.0f - m->params.momentum)); // undo multiplication

        //Mt Biases...
        //Mt = B1 * Mt-1 + (1 - B1)Gt^1
        scalar_mult(m->expwa_biases + i, m->params.momentum);
        scalar_mult(grads->biases + i, 1.0f - m->params.momentum);
        add_in_place(m->expwa_biases + i, grads->biases + i);
        scalar_mult(grads->biases + i, 1.0f / (1.0f - m->params.momentum)); // undo multiplication

        //Vt Weights...
        //Vt = B2 * Vt-1 + (1 - B2)Gt^2
        scalar_mult(m->expwa_weights_squared + i, m->params.momentum2);
        matrix_square(grads->weights + i);
        scalar_mult(grads->weights + i, 1.0f - m->params.momentum2);
        add_in_place(m->expwa_weights_squared + i, grads->weights + i);
        //No need to undo multiplication, as the gradients will be reset after this

        //Vt Biases...
        //Vt = B2 * Vt-1 + (B2 - 1)Gt^2
        scalar_mult(m->expwa_biases_squared + i, m->params.momentum2);
        matrix_square(grads->biases + i);
        scalar_mult(grads->biases + i, 1.0f - m->params.momentum2);
        add_in_place(m->expwa_biases_squared + i, grads->biases + i);
        //No need to undo multiplication, as the gradients will be reset after this

        //We can't modify Vt and Mt, so make copies...
        Matrix Vt_copy_weights = matrix_copy(m->expwa_weights_squared + i);
        Matrix Vt_copy_biases = matrix_copy(m->expwa_biases_squared + i);
//        Matrix Mt_copy_weights = matrix_copy(m->expwa_weights + i);
//        Matrix Mt_copy_biases = matrix_copy(m->expwa_biases + i);

        //Bias correct...
//        scalar_div(&Vt_copy_weights, 1.0f - powf(m->params.momentum, time_step));
//        scalar_div(&Vt_copy_biases, 1.0f - powf(m->params.momentum, time_step));
//        scalar_div(&Mt_copy_weights, 1.0f - powf(m->params.momentum2, time_step));
//        scalar_div(&Mt_copy_biases, 1.0f - powf(m->params.momentum2, time_step));

        //Sqrt(Vt + Epsillon)
        matrix_sqrt(&Vt_copy_weights);
        scalar_add(&Vt_copy_weights, m->params.epsillon);
        matrix_sqrt(&Vt_copy_biases);
        scalar_add(&Vt_copy_biases, m->params.epsillon);
        
        reciprocal(&Vt_copy_weights);
        reciprocal(&Vt_copy_biases);
        scalar_mult(&Vt_copy_weights, m->params.learning_rate);
        scalar_mult(&Vt_copy_biases, m->params.learning_rate);
        
        dot_in_place(&Vt_copy_weights, m->expwa_weights + i);
        dot_in_place(&Vt_copy_biases, m->expwa_biases + i);


        //a * Mt /  Sqrt(Vt) + Epsillon)
        scalar_mult(&Vt_copy_weights, m->params.learning_rate);
        scalar_mult(&Vt_copy_biases, m->params.learning_rate);

        //Gt+1 = Gt - a * Mt / Sqrt(Vt + Epsillon)
        sub_in_place(m->weights + i, &Vt_copy_weights);
        sub_in_place(m->biases + i, &Vt_copy_biases);
        
        if (gradient_mag != NULL){
            *gradient_mag += magnitude(&Vt_copy_weights);
            *gradient_mag += magnitude(&Vt_copy_biases);
        }

        //cleanup
//        delete_matrix(&result_weights);
//        delete_matrix(&result_biases);
//        delete_matrix(&Mt_copy_weights);
//        delete_matrix(&Mt_copy_biases);
        delete_matrix(&Vt_copy_weights);
        delete_matrix(&Vt_copy_biases);


    }
}

static void perform_epoch(Model* m, Matrix* x, Matrix* y, uint32_t num_data_points, uint32_t epoch, float* cumulative_loss, float* gradient_mag){
    
    uint32_t num_mini_batches = num_data_points / m->params.batch_size + (num_data_points % m->params.batch_size != 0);
    
    Gradients collective_grads;
    collective_grads.weights = (Matrix*) calloc(sizeof(Matrix), m->numLayers - 1);
    collective_grads.biases = (Matrix*) calloc(sizeof(Matrix), m->numLayers - 1);
    
    for (size_t i = 0; i < m->numLayers - 1; i++){
        collective_grads.weights[i] = create_matrix(m->weights[i].rows, m->weights[i].cols);
        collective_grads.biases[i] = create_matrix(m->biases[i].rows, m->biases[i].cols);
    }
    
    Vector indices = randomize_dataset(num_data_points);
    uint32_t offset = 0;
    for (uint32_t i = 0; i < num_mini_batches; i++){
        
        //if we need to utilize learning rate scheduling or want to print the loss, get the loss
        get_gradients(m, x, y, &indices, offset, num_data_points, &collective_grads, cumulative_loss);
        apply_gradients(m, &collective_grads, gradient_mag, epoch + 1);
        reset_gradients(&collective_grads, m->numLayers); //resets to 0
    
        offset += m->params.batch_size;
    }
    
    delete_vector(&indices);
    free_gradients(&collective_grads, m->numLayers);
    free(collective_grads.weights);
    free(collective_grads.biases);
}

static void perform_threaded_epoch(Model* m, Matrix* x, Matrix* y, uint32_t num_data_points, uint32_t epoch, float* cumulative_loss, float* gradient_mag){
    
    uint32_t num_mini_batches = num_data_points / m->params.batch_size + (num_data_points % m->params.batch_size != 0);
    uint32_t offset = 0;
    
    Vector random_indices = randomize_dataset(num_data_points);
    for (uint32_t a = 0; a < num_mini_batches; a += NUM_AVAILABLE_THREADS){
        size_t size = MIN(a + NUM_AVAILABLE_THREADS, num_mini_batches) - a;
        
        pthread_t* threads = (pthread_t*) calloc(sizeof(pthread_t), size);
        Gradients* grads = (Gradients*) calloc(sizeof(Gradients), size);
        float* losses = (float*) calloc(sizeof(float), size);
        
        //allocate the gradient matrices and vectors, then dispatch the threads
        for (size_t i = 0; i < size; i++){
            losses[i] = 0.0f;
            
            grads[i].weights = (Matrix*) calloc(sizeof(Matrix), m->numLayers - 1);
            grads[i].biases = (Matrix*) calloc(sizeof(Matrix), m->numLayers - 1);
            for (size_t j = 0; j < m->numLayers - 1; j++){
                grads[i].weights[j] = create_matrix(m->weights[j].rows, m->weights[j].cols);
                grads[i].biases[j] = create_matrix(m->biases[j].rows, m->biases[j].cols);
            }
            
            Args args = {
                .model = m,
                .inputs = x,
                .observ = y,
                .indices = &random_indices,
                .offset = offset,
                .num_data_points = num_data_points,
                .collective_grads = grads + i,
                .cumulative_loss = losses + i,
            };
                 
            pthread_create(threads + i, NULL, get_gradients_threaded, (void*)&args);
        }

        
        //join the threads
        for (size_t i = 0; i < size; i++){
            pthread_join(threads[i], NULL);
        }
        
        //apply the gradients, then release them
        for (size_t i = 0; i < size; i++){
            apply_gradients(m, grads + i, gradient_mag, epoch);
            free_gradients(grads + i, m->numLayers);
            free((grads + i)->weights);
            free((grads + i)->biases);
            
            *cumulative_loss += losses[i];
        }

        offset += m->params.batch_size;
        free(grads);
        free(losses);
        free(threads);
        
    }
    delete_vector(&random_indices);

}


void train(Model* m, Matrix* x, Matrix* y, uint32_t num_data_points, uint32_t num_epochs, uint8_t write_to_file, const char* file_name){
    
    srand((unsigned int) time(0)); //cast to get rid of warning
    //do some validation...
    //TODO have train return a bool so no memory is leaked on the main function
    if (num_data_points < m->params.batch_size || num_data_points <= 0){
        delete_model(m);
        printf("ERROR: Bad parameters for train function. Exiting...\n");
        exit(-1);
    }
    
    void (*epoch_func)(Model*, Matrix*, Matrix*, uint32_t, uint32_t, float*, float*);
    if (IS_THREADING_ENABLED)
        epoch_func = &perform_threaded_epoch;
    else
        epoch_func = &perform_epoch;
        
        
    float* loss_data = NULL;
    float* gradient_mag_data = NULL;
    if (write_to_file){
        loss_data = (float*) calloc(sizeof(float), num_epochs);
        gradient_mag_data = (float*) calloc(sizeof(float), num_epochs);
    }

    uint32_t num_loss_increases = 0; //number of times loss has increased (bad)
    float loss = 0.0f, gradient_mag = 0.0f;
    for (uint32_t i = 0; i < num_epochs; i++){
        
        float curr_loss = 0.0f;
        float* grad_p = m->params.verbose == 2 || write_to_file ? &gradient_mag : NULL;
        float* loss_p = m->params.verbose >= 1 || m->use_tuning || write_to_file ? &curr_loss : NULL;
        
        clock_t begin = clock();
        epoch_func(m, x, y, num_data_points, i, loss_p, grad_p);
        clock_t end = clock();
        printf("TIME FOR EPOCH:: %f\n\n", (float)(end - begin) / CLOCKS_PER_SEC);
        
        if (m->params.verbose >= 1){
            printf("Epoch #%d, Loss: %f", i, loss);
            if (m->params.verbose == 2)
                printf(", Gradient Magnitude: %f\n", gradient_mag);
            else
                printf("\n");
        }
        
        if (write_to_file){
            loss_data[i] = curr_loss;
            gradient_mag_data[i] = gradient_mag;
        }
        
        if (i != 0 && m->use_tuning && curr_loss < loss){
            num_loss_increases++;
            
            if (num_loss_increases == m->tuning.patience){
                m->params.learning_rate = MAX(m->params.learning_rate * m->tuning.decrease, m->tuning.min);
                num_loss_increases = 0;
            }
        }
        
        loss = curr_loss;
    }
    
    
    
    if (write_to_file){
        write_meta_data(file_name, loss_data, gradient_mag_data, num_epochs);
        free(loss_data);
        free(gradient_mag_data);
    }
    
}
