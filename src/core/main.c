//
//  main.c
//  Neural Net
//
//  Created by Shaunte Mellor on 5/1/22.
//

#include "pch.h"
#include "Model/Model.h"
#include "Model/Training.h"
#include "core/Data Loader.h"

DataSplit get_data(const char* path){
    
    uint32_t num_data_points = num_datapoints_of_csv(path);
    uint32_t num_features = num_features_of_csv(path);
    
    //the paremeter after num_features
    Data m_data = read_csv(path, num_data_points, num_features, 1, 1);
    
    float train_percent = 0.80f;
    uint32_t train_size = (uint32_t) (train_percent * num_data_points);
    DataSplit split = train_test_split(&m_data, train_size);

    return split; //pointers are copied, so no memory leak
}

Model* get_model(void){
    ModelParams params = {
        .learning_rate = 0.01f,
        .batch_size = 2,
        .verbose = 3,
        .momentum = 0.9f,
        .momentum2 = 0.999f,
        .epsillon = 1e-8,
    };
    
    //verbose level 1 : prints loss
    //verbose level 2: prints gradient magnitude
    //verbose level 3: prints average time per epoch
    
    Model* m = create_model(&params, NULL);
    
    //params: Model*, size of layer, activation function
    //first layer has no activation, so pass 'NONE'
    add_layer(m, 1, NONE);
    add_layer(m, 10, HYPERBOLIC_TANGENT);
    add_layer(m, 10, HYPERBOLIC_TANGENT);
    add_layer(m, 1, HYPERBOLIC_TANGENT);
   
    set_loss_func(m, LEAST_SQUARES);
    
    //compile creates the weights and biases along with other things such as the matrices needed for ADAM optimization
    uint8_t success = compile(m);
    if (!success){
        delete_model(m);
        exit(-1);
    }

    init_weights_and_biases(m, 0, 1);
    
    //the second parameter controls whether the weights and biases are printed. 1 = true, 0 = false
    summary(m, 0);
    
    return m;
}


int main(int argc, const char* argv[]) {
    //workspace directory is only one path out...
    Model* model = get_model();
    DataSplit split = get_data("../data/test.csv");

    uint32_t epochs = 200;
    const char* training_data_file_ = "../training data/example.json";
    uint8_t success = train(model, split.train.inputs, split.train.outputs, split.train.num_data_points, epochs, training_data_file_);

    //can save and load models with save_model and load_model
    if (success)
         save_model(model, "../saved models/example.txt");

    float loss = loss_on_dataset(model, split.test.inputs, split.test.outputs, split.test.num_data_points);
    printf("Loss on test set: %f\n", loss);

    //cleanup...
    delete_split_data(&split);
    delete_model(model);
    
    return 0;
}


