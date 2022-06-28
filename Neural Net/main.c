//
//  main.c
//  Neural Net
//
//  Created by Shaunte Mellor on 5/1/22.
//

#include <stdio.h>
#include "Model.h"
#include "Training.h"
#include "Data Loader.h"

Data get_data(const char* path){
    
    uint32_t num_data_points = num_rows_of_csv(path);
    uint32_t num_features = num_cols_of_csv(path);
    printf("NUM DATA POINTS %u AND NUM FEATURS %u\n", num_data_points, num_features);
    
    Matrix** m_data = read_csv(path, num_data_points, num_features, 0, 10);
    for (size_t i = 0; i < num_features; i++){
        for (size_t j = 0; j < (m_data[0] + i)->rows * (m_data[0] + i)->cols; j++){
            (m_data[0] + i)->values[j] /= 255.0f;
        }
    }
    
    for (size_t i = 0; i < 20; i++)
        printf("Observ %u \n", argmax(m_data[1] + i));
    
    float train_percent = 1.00f;
    uint32_t train_size = (uint32_t) (train_percent * num_data_points);
    Data dat = train_test_split(m_data, train_size, num_data_points);

    return dat;
}

Model* get_model(void){
    ModelParams params = {
        .learning_rate = 0.01f,
        .batch_size = 20,
        .verbose = 2,
        .momentum = 0.9f,
        .momentum2 = 0.999f,
        .epsillon = 1e-8,
    };
    
    
    Model* m = create_model(&params, NULL);
    
    add_layer(m, 784, NONE);
    add_layer(m, 200, HYPERBOLIC_TANGENT);
    add_layer(m, 200, HYPERBOLIC_TANGENT);
    add_layer(m, 10, SOFT_MAX);
    
    set_loss_func(m, CROSS_ENTROPY);
    
    compile(m);
    init_weights_and_biases(m, 0, 1);
    
    summary(m, 0);
    
    return m;
}


int main(int argc, const char * argv[]) {
    
//    Model* model = load_model("/Users/shauntemellor/Documents/CS/comsci/Projects/Neural Net/models/stuff.txt");
//    summary(model, 1);
//    delete_model(model);
//
    
//    ENABLE_THREADING();
//    MAX_THREADS(4);
    Data dat = get_data("/Users/shauntemellor/Documents/CS/comsci/Projects/Neural Net/train.csv");
    Model* model = get_model();


    train(model, dat.train[0], dat.train[1], 40, 200, 1, "/Users/shauntemellor/Documents/CS/comsci/vs_code/Python/Jonny/test.json");
   // summary(model, 0);




    save_model(model, "/Users/shauntemellor/Documents/CS/comsci/Projects/Neural Net/models/stuff.txt");

    delete_split_data(&dat);
    delete_model(model);

    
    return 0;
}


