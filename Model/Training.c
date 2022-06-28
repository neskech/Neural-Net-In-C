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


typedef struct Gradients{
    Matrix* weights;
    Matrix* biases;
} Gradients;

typedef struct ForwardPassCache{
    Matrix* activations;
    Matrix* outputs;
} ForwardPassCache;


//write loss and gradient magnitude data to a file so it can later be plotted by a python script
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

static void free_gradient_matrices(Gradients* grads, uint16_t num_layers){
    for (size_t i = 0; i < num_layers - 1; i++){
        delete_matrix(grads->biases + i);
        delete_matrix(grads->weights + i);
    }
}

static void reset_gradient_matrices(Gradients* grads, uint16_t num_layers){
    for (size_t i = 0; i < num_layers - 1; i++){
        set_values_with(grads->weights + i, 0.0f);
        set_values_with(grads->biases + i, 0.0f);
    }
}

static void free_cache_matrices(ForwardPassCache* cache, uint16_t num_layers){
    //free data from a forward pass cache
    for (size_t i = 0; i < num_layers; i++){
        delete_matrix(cache->activations + i);
        
        if (i < num_layers - 1){
            delete_matrix(cache->outputs + i);
        }
    }
}







static Vector randomize_dataset(uint32_t num_data_points){
    //generate a list of indices from which to sample from
    Vector vec = create_vector(num_data_points);
    for (uint32_t i = 0; i < num_data_points; i++)
        push(&vec, i);
    
    Vector indices = create_vector(num_data_points);
    //generare a random index, and then remove an element from the index array at that index
    for (uint32_t i = 0; i < num_data_points; i++){
        uint32_t idx = (uint32_t) ( vec.size * (rand() / (float) RAND_MAX) );
        push(&indices, get(&vec, idx));
        remove_at(&vec, idx);
    }
        
    delete_vector(&vec);
    return indices;
}


static void forward_prop(Model* m, Matrix* x, ForwardPassCache* cache){
    //running value propogated through the network
    Matrix running = matrix_copy(x);
    //first activations stores the input for convience's sake in backProp
    *(cache->activations + 0) = matrix_copy(x);
    
    for (size_t i = 0; i < m->num_layers - 1; i++){
        //when we copy a new value to running, the previous value's memory is lost. Be sure to delete it
        Matrix before = running; //shallow copy
        running = mult(m->weights + i, &running); //new matrix allocated by mult() function
        delete_matrix(&before);
        
        //add the biases
        add_in_place(&running, m->biases + i);
        //store the raw output before the activation function
        *(cache->outputs + i) = matrix_copy(&running);
        //apply the activation function to the running matrix
        act_func(&running, get(&m->activations, i));
        //store the result of the activation function
        *(cache->activations + i + 1) = matrix_copy(&running);
    }
    
    //be sure to cleanup running. The output of the network is stored in activations[num_layers - 1]
    delete_matrix(&running);
}

static void back_prop(Model* m, Matrix* observ, ForwardPassCache* cache, Gradients* grads){
    //Last value of the activations array is the final output of the network
    Matrix* pred = cache->activations + (m->num_layers - 1);
    //running derivative to be propogated down the network
    Matrix running_deriv = loss_func_deriv(pred, observ, m->loss_func);
 
    
    //the weight and bias matrices correspond to the last two layers (the input layer has neither weights nor biases)
    //the weights belonging to layer 2 are at index 1, the weights to layer 3 are at index 2, etc....
    for (int32_t i = m->num_layers - 2; i >= 0; i--){ //a signed integer b/c unsigned int going backwards is inf loop

        //Get the activation functions derivative...
        Activation act = get(&m->activations, i);
        act_func_deriv(cache->outputs + i, act); //Stores the derivative in outputs[i]
        
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
            Matrix trans_weights = transpose(m->weights + i);
            
            //since matrix multiplication creates a new matrix, we must delete the previous value of
            //running_deriv to prevent a memory leak...
            Matrix before = running_deriv;
            running_deriv = mult(&trans_weights, &running_deriv);
            delete_matrix(&before);
            
            delete_matrix(&trans_weights);
            m->weights[i] = transpose(m->weights + i); //undo transpose
        }
    }
}


static void apply_gradients(Model* m, Gradients* grads, float* gradient_mag, uint32_t time_step){
    for (size_t i = 0; i < m->num_layers - 1; i++){
        //Get the Mt and Vt of the current timestep...

        //Mt Weights...
        //Mt = B1 * Mt-1 + (1 - B1)Gt^1
        scalar_mult(m->expwa_weights + i, m->params.momentum);
        scalar_mult(grads->weights + i, 1.0f - m->params.momentum);
        add_in_place(m->expwa_weights + i, grads->weights + i);
        scalar_div(grads->weights + i, 1.0f - m->params.momentum); // undo multiplication

        //Mt Biases...
        //Mt = B1 * Mt-1 + (1 - B1)Gt^1
        scalar_mult(m->expwa_biases + i, m->params.momentum);
        scalar_mult(grads->biases + i, 1.0f - m->params.momentum);
        add_in_place(m->expwa_biases + i, grads->biases + i);
        scalar_div(grads->biases + i, 1.0f - m->params.momentum); // undo multiplication

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
        Matrix Mt_copy_weights = matrix_copy(m->expwa_weights + i);
        Matrix Mt_copy_biases = matrix_copy(m->expwa_biases + i);

        //Bias correct...
        scalar_div(&Vt_copy_weights, 1.0f - powf(m->params.momentum, time_step));
        scalar_div(&Vt_copy_biases, 1.0f - powf(m->params.momentum, time_step));
        scalar_div(&Mt_copy_weights, 1.0f - powf(m->params.momentum2, time_step));
        scalar_div(&Mt_copy_biases, 1.0f - powf(m->params.momentum2, time_step));

        //Sqrt(Vt) + Epsillon
        matrix_sqrt(&Vt_copy_weights);
        scalar_add(&Vt_copy_weights, m->params.epsillon);
        matrix_sqrt(&Vt_copy_biases);
        scalar_add(&Vt_copy_biases, m->params.epsillon);
        
        //a * ( 1 / Sqrt(Vt) + Epsillon )
        reciprocal(&Vt_copy_weights);
        reciprocal(&Vt_copy_biases);
        scalar_mult(&Vt_copy_weights, m->params.learning_rate);
        scalar_mult(&Vt_copy_biases, m->params.learning_rate);
        
        dot_in_place(&Vt_copy_weights, &Mt_copy_weights);
        dot_in_place(&Vt_copy_biases, &Mt_copy_biases);

        //calculation isnt done in the same order as the original equation, but still remains equivalent
        //Mt * a * ( 1 / Sqrt(Vt) + Epsillon )
        scalar_mult(&Vt_copy_weights, m->params.learning_rate);
        scalar_mult(&Vt_copy_biases, m->params.learning_rate);

        //Gt+1 = Gt - a * Mt / Sqrt(Vt + Epsillon)
        sub_in_place(m->weights + i, &Vt_copy_weights);
        sub_in_place(m->biases + i, &Vt_copy_biases);
        
        //really tried avoiding making new matrices to avoid extra memory allocation
        
        if (gradient_mag != NULL){
            *gradient_mag += magnitude(&Vt_copy_weights);
            *gradient_mag += magnitude(&Vt_copy_biases);
        }

        //cleanup
        delete_matrix(&Mt_copy_weights);
        delete_matrix(&Mt_copy_biases);
        delete_matrix(&Vt_copy_weights);
        delete_matrix(&Vt_copy_biases);


    }
}

static void retrieve_gradients(Model* m, Matrix* inputs, Matrix* observ, Vector* indices, uint32_t offset, uint32_t num_data_points, Gradients* collective_grads, float* cumulative_loss){
    
    //allocate the gradients that will be used from datapoint to datapoint
    Gradients grads;
    grads.weights = (Matrix*) calloc(sizeof(Matrix), m->num_layers - 1);
    grads.biases = (Matrix*) calloc(sizeof(Matrix), m->num_layers - 1);
    
    //allocate the cache used to store data from the forward pass
    ForwardPassCache cache;
    cache.activations = (Matrix*) calloc(sizeof(Matrix), m->num_layers);
    cache.outputs = (Matrix*) calloc(sizeof(Matrix), m->num_layers - 1);
    

    //now go through the random data-indices and generate gradients
    uint32_t size = MIN(num_data_points, offset + m->params.batch_size); //do not exceed data set size
    for (uint32_t i = offset; i < size; i++){
        uint32_t idx = get(indices, i); //data index
        forward_prop(m, inputs + idx, &cache);
        back_prop(m, observ + idx, &cache, &grads);
        
        for (size_t j = 0; j < m->num_layers - 1; j++){
            //average the weights and biases collected from the batch
            // 1/batch_size * (grads1 + grads2 + gradsn...) = 1/batch_size * grads1 + 1/batch_size * grads2 +...
            scalar_div(grads.weights + j, m->params.batch_size);
            scalar_div(grads.biases + j, m->params.batch_size);
            
            //add them to a collective gradient for the entire batch
            add_in_place(collective_grads->weights + j, grads.weights + j);
            add_in_place(collective_grads->biases + j, grads.biases + j);
        }
        
        //add to the cumulative loss
        if (cumulative_loss != NULL)
            *cumulative_loss += loss_func(cache.activations + m->num_layers - 1, observ + idx, m->loss_func);
        
        //free the matrices, as they will be replaced in the next iteration and memory will be leaked
        free_cache_matrices(&cache, m->num_layers);
        free_gradient_matrices(&grads, m->num_layers);
    }
    
    //free the containers for the cache...
    free(cache.activations);
    free(cache.outputs);
    
    //and gradients...
    free(grads.weights);
    free(grads.biases);
}


static void perform_epoch(Model* m, Matrix* inputs, Matrix* observ, uint32_t num_data_points, uint32_t epoch, float* cumulative_loss, float* gradient_mag){
    
    //add +1 to the number of batches if num_data_points doesn't divide evenly by the batch size (we have some data points left over)
    uint32_t num_mini_batches = num_data_points / m->params.batch_size + (num_data_points % m->params.batch_size != 0);
    
    Gradients collective_grads;
    collective_grads.weights = (Matrix*) calloc(sizeof(Matrix), m->num_layers - 1);
    collective_grads.biases = (Matrix*) calloc(sizeof(Matrix), m->num_layers - 1);
    
    for (size_t i = 0; i < m->num_layers - 1; i++){
        collective_grads.weights[i] = create_matrix(m->weights[i].rows, m->weights[i].cols);
        collective_grads.biases[i] = create_matrix(m->biases[i].rows, m->biases[i].cols);
    }
    
    //randomize the order of the dataset
    Vector indices = randomize_dataset(num_data_points);
    //data offset to be used by the retrieve_gradients() function
    uint32_t offset = 0;
    for (uint32_t i = 0; i < num_mini_batches; i++){
        retrieve_gradients(m, inputs, observ, &indices, offset, num_data_points, &collective_grads, cumulative_loss);
        apply_gradients(m, &collective_grads, gradient_mag, epoch + 1 + i);
        //reset gradients to 0-matrices. No need to delete their matrices
        reset_gradient_matrices(&collective_grads, m->num_layers);
        offset += m->params.batch_size;
    }
    
    //cleanup
    delete_vector(&indices);
    free_gradient_matrices(&collective_grads, m->num_layers);
    free(collective_grads.weights);
    free(collective_grads.biases);
}

uint8_t train(Model* m, Matrix* inputs, Matrix* observ, uint32_t num_data_points, uint32_t num_epochs, const char* file_name){
    
    //do some validation...
    if (num_data_points < m->params.batch_size || num_data_points <= 0){
        delete_model(m);
        printf("ERROR: Bad parameters for train function. Exiting...\n");
        return 0;
    }
       
    //initialize rand function with a seed
    srand((unsigned int) time(0)); //cast to get rid of warning...
    
    //prepare data arrays if we're writing the loss and gradient magnitude data to a file
    float* loss_data = NULL;
    float* gradient_mag_data = NULL;
    uint8_t write_to_file = file_name != NULL;
    if (write_to_file){
        loss_data = (float*) calloc(sizeof(float), num_epochs);
        gradient_mag_data = (float*) calloc(sizeof(float), num_epochs);
    }

    uint32_t num_loss_increases = 0; //number of times loss has increased
    float loss = 0.0f, gradient_mag = 0.0f;
    float cumulative_time = 0.0f;
    
    for (uint32_t i = 0; i < num_epochs; i++){
        
        float curr_loss = 0.0f;
        float* grad_p = m->params.verbose == 2 || write_to_file ? &gradient_mag : NULL;
        float* loss_p = m->params.verbose >= 1 || m->use_tuning || write_to_file ? &curr_loss : NULL;
        
        //time how long each epoch takes and add it to a total
        clock_t begin = clock();
        perform_epoch(m, inputs, observ, num_data_points, i, loss_p, grad_p);
        clock_t end = clock();
        cumulative_time += (float)(end - begin) / CLOCKS_PER_SEC;
        
        //printing information
        if (m->params.verbose >= 1){
            printf("Epoch #%d, Loss: %f", i, loss);
            if (m->params.verbose >= 2){
                printf(", Gradient Magnitude: %f", gradient_mag);
                if (m->params.verbose == 3)
                    printf(", Average time per epoch: %f\n, ", cumulative_time / i);
                else
                    printf("\n");
            }
            else
                printf("\n");
        }
        
        //adding data to data arrays
        if (write_to_file){
            loss_data[i] = curr_loss;
            gradient_mag_data[i] = gradient_mag;
        }
        
        
        //learning rate scheduler
        if (i != 0 && m->use_tuning && curr_loss < loss){
            num_loss_increases++;
            
            if (num_loss_increases == m->tuning.patience){
                m->params.learning_rate = MAX(m->params.learning_rate * m->tuning.decrease, m->tuning.min);
                num_loss_increases = 0;
            }
        }
        
        loss = curr_loss;
    }
    
    //finally writing data to a file, then freeing it
    if (write_to_file){
        write_meta_data(file_name, loss_data, gradient_mag_data, num_epochs);
        free(loss_data);
        free(gradient_mag_data);
    }
    
    return 1;
    
}
