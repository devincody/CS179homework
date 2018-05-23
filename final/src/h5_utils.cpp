// #include "H5Cpp.h"
// #include <iostream>

// const H5std_string FILE_NAME("weights.h5");
// const H5std_string kernel_NAME( "kernel:0" );
// const H5std_string bias_NAME( "bias:0" );

#include "h5_utils.hpp"

// little test script
int test(){
	std::cout << "***********************************" <<std::endl;
	std::cout << "*         Reading Weights         *" <<std::endl;
	std::cout << "***********************************" <<std::endl;
	float *data = get_weights("block1_conv1", 64, 3, 3, 3);
	float ans_weight[10] = {0.341195, 0.339992, -0.0448442, 0.232159, 0.0897821, -0.303314, -0.0726092, -0.246675, -0.336833, 0.464184};
	float ans_bias[10] = {0.7301776,0.06493629,0.03428847,0.8260386,0.2578029,0.54867655,-0.012438543,0.34789944,0.5510871,0.06297145};
	float tolerance = 1E-5;

	int check = 0;
	for (int i = 0; i < 10; i ++){
		if (std::abs(data[i] - ans_weight[i]) > tolerance){
			check = 1;
		}
		std::cout << "weights["<<i<<"] = " << data[i] << std::endl;
	}

	if (check){
		std::cout << "error reading file" << std::endl;
	} else {
		std::cout << "File read correctly" << std::endl;
	}

	delete[] data;

	std::cout << "\n***********************************" <<std::endl;
	std::cout << "*         Reading Biases          *" <<std::endl;
	std::cout << "***********************************" <<std::endl;

	data = get_bias("block1_conv1", 64);
	
	check = 0;
	for (int i = 0; i < 10; i ++){
		if (std::abs(data[i] - ans_bias[i]) > tolerance){
			check = 1;
		}
		std::cout << "bias["<<i<<"] = " << data[i] << std::endl;
	}

	if (check){
		std::cout << "error reading file" << std::endl;
	} else {
		std::cout << "File read correctly" << std::endl;
	}

	delete[] data;
	return 0;
}

float* get_weights(std::string name, int n, int c, int h, int w){
	int len = n*c*h*w;

	float data[len];
	float* odata = new float[len]();

	memset(data, 0, len*sizeof(float));
	memset(odata, 0, len*sizeof(float));

	try{
		H5::Exception::dontPrint();
		H5::H5File file(FILE_NAME, H5F_ACC_RDONLY); // open read only file
		H5::Group group = file.openGroup(name);		// open file group
		group = group.openGroup(name);
		H5::DataSet dataset = group.openDataSet(kernel_NAME); // open dataset

	    H5T_class_t type_class = dataset.getTypeClass(); 	  // check datatype

	    if (type_class != H5T_FLOAT){
	    	std::cout <<"ERROR: Not float type in " << name << std::endl;
	    }		

	    dataset.read(data, H5::PredType::NATIVE_FLOAT);

	    // Format data into NCHW from HWCN 
	    if (data){
			int z = 0;
			int loc = 0;
			for(int i = 0; i < n; ++i){
				for(int j = 0; j < c; ++j){
					for(int k = 0; k < h; ++k){
						for(int l = 0; l < w; ++l){
							loc = i + j*n + l*n*c + k*n*c*w;
							odata[z] = data[loc];
							z++;
						}
					}
				}
			}
		} else {
			std::cout << "ERROR: Failed to read weight data" << std::endl;
		}



	}
	// catch failure caused by the H5File operations
	catch( H5::FileIException error )
	{
		error.printError();
		return nullptr;
	}

	// catch failure caused by the DataSet operations
	catch( H5::DataSetIException error )
	{
		error.printError();
		return nullptr;
	}

	// catch failure caused by the DataSpace operations
	catch( H5::DataSpaceIException error )
	{
		error.printError();
		return nullptr;
	}

	// catch failure caused by the DataSpace operations
	catch( H5::DataTypeIException error )
	{
		error.printError();
		return nullptr;
	}

	return odata;
}





float* get_bias(std::string name, int n){
	float* odata = new float[n]();

	memset(odata, 0, n*sizeof(float));

	try{
		H5::Exception::dontPrint();
		H5::H5File file(FILE_NAME, H5F_ACC_RDONLY); // open read only file
		H5::Group group = file.openGroup(name);		// open file group
		group = group.openGroup(name);
		H5::DataSet dataset = group.openDataSet(bias_NAME); // open dataset

	    H5T_class_t type_class = dataset.getTypeClass(); 	  // check datatype
	    // std::cout << "type: " << type_class << std::endl;
	    if (type_class != H5T_FLOAT){
	    	std::cout <<"ERROR: Not float type in " << name << std::endl;
	    }		

	    dataset.read(odata, H5::PredType::NATIVE_FLOAT);
	    if (!odata){
			std::cout << "ERROR: Failed to read bias data" << std::endl;
		}

	}
	// catch failure caused by the H5File operations
	catch( H5::FileIException error )
	{
		error.printError();
		return nullptr;
	}

	// catch failure caused by the DataSet operations
	catch( H5::DataSetIException error )
	{
		error.printError();
		return nullptr;
	}

	// catch failure caused by the DataSpace operations
	catch( H5::DataSpaceIException error )
	{
		error.printError();
		return nullptr;
	}

	// catch failure caused by the DataSpace operations
	catch( H5::DataTypeIException error )
	{
		error.printError();
		return nullptr;
	}

	return odata;
}





