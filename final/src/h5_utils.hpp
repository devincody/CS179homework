#include "H5Cpp.h"
#include <iostream>
#include <cstring>	// malloc
#include <cmath> 	// std::abs
// #include <cstddef> 	// nullptr

const H5std_string FILE_NAME("/home/dcody/CS179homework/final/src/weights.h5");
const H5std_string kernel_NAME( "kernel:0" );
const H5std_string bias_NAME( "bias:0" );

float* get_weights(std::string name, int n, int c, int h, int w);
float* get_bias(std::string name, int n);