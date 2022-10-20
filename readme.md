# A Forward Neural Network

**Made From Scratch in C**

This is my first project in c, and what better way to learn the ropes than to make my own neural net!



## Building the Project

- Clone the repository
```shell
git clone --recursive https://github.com/neskech/ANN-in-C.git
```
- Build the project either using 'Release' or 'Debug' in substitution for <BUILD_TYPE>
```shell
cd ANN-in-C
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=<BUILD_TYPE> ..
cmake --build .
```

- Run the executable on...

  - Mac / Linux 
  ```shell
  ./main
  ```
  - Windows
  ```shell
  main.exe
  ```

  -*Note* This project was made using the gcc compiler
  
## Features

- Activation Functions
  - 
   - Relu
   - Sigmoid
   - Soft Plus
   - Hyperbolic Tangent
   - Linear


- Loss Functions
  -
    - Least Squares

- Other
  -
     -  Model Saving and Loading
     -  Display of Training Statistics Using a Python Script
     -  Reading of CSV files

- Get Started
  -
     - `src/core/main.c` contains helpful example code, along with `dataPlot.py`
     - Once you train a model, plot the loss using the `dataPlot.py` python script
     - inside the `data` directory, there's an example CSV for training 
     - There is also an example json file in the `training data` directory that can be used to plot loss with the python script
  