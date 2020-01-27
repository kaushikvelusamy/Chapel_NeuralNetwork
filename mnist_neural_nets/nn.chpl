/*
Author: Kaushik Velusamy
Org: UMBC
About: Neural network from scratch in chapel language
To Compile:  chpl -I/usr/local/opt/openblas/include -L/usr/local/opt/openblas/lib -lblas nn.chpl --fast  -M ../lib/
To Run: ./nn --train_file --test_file 
 */

use Random;
use IO, CSV;
use LinearAlgebra, Norm;
use Math;
use Time;
var watch: Timer;
var file_read_time : real;
var training_time : real;
var testing_time : real;

config const train_file = "mnist_dataset/mnist_train.csv";
config const test_file  = "mnist_dataset/mnist_test.csv";
config const learn_rate : real = 0.08;
config const training_epochs_iterations : int = 800;
const digits_range      = 0..9;
const pixels_per_line   = 1..784; 		// 28 X 28 pixels
// values in the labels are from 0 to 9, but the indexes/domain is from 1 to 10

config const layer1_neurons: int 	 = 784;
config const layer2_neurons: int 	 = 128;
config const layer3_neurons: int 	 = 128;

//*********File Reading Started*********
watch.start(); 
var trainReader = if train_file == "" then stdin else openreader(train_file);
var testReader 	= if test_file   == "" then stdin else openreader(test_file);
var r_train 	= new CSVIO(trainReader, hasHeader=false, sep =",");
var r_test  	= new CSVIO(testReader, hasHeader=false, sep =",");
var A_train 	= r_train.read(string):real;
var A_test  	= r_test.read(string):real;

var training_sample_size: int; 			//num lines in trainfile
var testing_sample_size : int; 			//num lines in testfile
training_sample_size = A_train.domain.dim(1).length:int;
testing_sample_size  = A_test.domain.dim(1).length:int;

var training_inputs  : [1..training_sample_size, pixels_per_line] real;
var training_outputs : [1..training_sample_size] int;
var train_out_hot    : [1..training_sample_size, 1..10] real = 0;
var testing_inputs   : [1..testing_sample_size, pixels_per_line] real;
var testing_outputs  : [1..testing_sample_size] real;


for (i,j) in A_train.domain do 
{
	if (j == 1) 
	{ 
		training_outputs[i]     = A_train(i,1):int;
		train_out_hot[i , training_outputs[i]:int + 1] = 1;
		// because train_out_hot domain is 1..10 not 0..9 hence adjusting with +1
	}
	else 	    
	{ 
		 //Range normalization 
		training_inputs[i, j-1] = (A_train(i,j)) / 255; 
        }
}

for (i,j) in A_test.domain do 
{
	if (j == 1) { testing_outputs[i]     = A_test(i,1); }
	else        { testing_inputs[i, j-1] = A_test(i,j); }
}

trainReader.close();
testReader.close();

/*
   writeln("training_sample_size \n"	, training_sample_size);
   writeln("training_inputs \n"		, training_inputs);
   writeln("training_outputs \n"	, training_outputs);
   writeln("train_out_hot \n"		, train_out_hot);
   writeln("testing_sample_size \n"	, testing_sample_size);
   writeln("testing_inputs \n"		, testing_inputs);
   writeln("testing_outputs \n"		, testing_outputs);
 */
watch.stop();
// *********File Reading Ended********** 
file_read_time = watch.elapsed();
writeln('\n File reading time took ',file_read_time ,' seconds');
watch.clear();


watch.start();
// *********Random Matrix Function started********** 

const globalRandomSeed:int = 1;

proc fillNormallyDistributed(array)
{
	const arrayDomain 	= array.domain;
	fillRandom(array, globalRandomSeed);
	const mean:real 	= 0;
	const precision:int 	= 2;
	const precisionByRootTwoPi:real = (precision:real/((2.0 * Math.pi) ** 0.5));
	const minusPrecisionSquaredByTwo:real = (-1.0 * ((precision:real ** 2.0)/2.0));
	forall i in arrayDomain
	{
		const x:real 		= array[i];
		const power:real 	= (minusPrecisionSquaredByTwo
						* ((x - mean) ** 2.0));
		const translatedX:real 	= (precisionByRootTwoPi * (Math.e ** power));
		array[i] 		= translatedX;
	}
	return array;
}

proc create_matrix_random(rows_cols_dom)
{
	var newMatrix : [rows_cols_dom] real;
	newMatrix = fillNormallyDistributed(newMatrix);
	for idxy in newMatrix.domain 
	{	
		// Setting values between -2 to +2 instead of 0 to 1
		newMatrix[idxy] = (newMatrix[idxy] * 4 ) -2 ;
	}
	return newMatrix;
}
var rows_cols_domain1: domain(2) = {1..layer1_neurons, 1..layer2_neurons};
var rows_cols_domain2: domain(2) = {1..layer2_neurons, 1..layer3_neurons};
var rows_cols_domain3: domain(2) = {1..layer3_neurons, 1..10};
var synaptic_weights_mat1 	 = create_matrix_random(rows_cols_domain1);
var synaptic_weights_mat2 	 = create_matrix_random(rows_cols_domain2);
var synaptic_weights_mat3 	 = create_matrix_random(rows_cols_domain3);

//writeln("Synaptic Weight Matrix \n",synaptic_weights_mat1);

// *********Random Matrix Function Ended********** 



// *********Matrix Vector Multiplication started********** 

var layer1_result_dom	: domain(2) 	= {1..training_sample_size, 1..layer2_neurons};
var layer1_result	: [layer1_result_dom] real;
var layer1_temp_vec 	: [1..layer2_neurons] real;

var layer2_result_dom	: domain(2) 	= {1..training_sample_size, 1..layer3_neurons};
var layer2_result	: [layer2_result_dom] real;
var layer2_temp_vec 	: [1..layer3_neurons] real;

var layer3_result_dom	: domain(2) 	= {1..training_sample_size, 1..10};
var layer3_result	: [layer3_result_dom] real;
var layer3_temp_vec 	: [1..10] real;


proc feedforward() 
{

	// Matrix Vector Multiplication at layer 1
	
	for i in layer1_result_dom.dim(1)  
	{
		layer1_temp_vec = dot(training_inputs[i,1..layer1_neurons], synaptic_weights_mat1);
		for j in layer1_result_dom.dim(2)  
		{
			layer1_result[i,j] = layer1_temp_vec[j];
		}
	}
	//writeln(" layer1_result before sig \n", layer1_result);	
	layer1_result = sigmoid(layer1_result);
	//writeln(" layer1_result after sig \n", layer1_result);	

	// Matrix Vector Multiplication at layer 2

	for i in layer2_result_dom.dim(1)  
	{
		layer2_temp_vec = dot(layer1_result[i,1..layer2_neurons], synaptic_weights_mat2);
		for j in layer2_result_dom.dim(2)  
		{
			layer2_result[i,j] = layer2_temp_vec[j];
		}
	}
	layer2_result = sigmoid(layer2_result);
	//writeln(" layer2_result after sigmoid \n", layer2_result);	

	// Matrix Vector Multiplication at layer 3

	for i in layer3_result_dom.dim(1)  
	{
		layer3_temp_vec = dot(layer2_result[i,1..layer3_neurons], synaptic_weights_mat3);
		for j in layer3_result_dom.dim(2)  
		{
			layer3_result[i,j] = layer3_temp_vec[j];
		}

	}
	layer3_result = softmax(layer3_result);
	//writeln(" layer3_result  after softmax \n", layer3_result);	
}
// *********Matrix Vector Multiplication Ended********** 

// *********Activation Function 1: Sigmoid Part started********** 

proc sigmoid(layer2_result) 
{
	for idxy in layer2_result.domain  do 
	{
		layer2_result[idxy]  = 1 / ( 1 + exp(-1 * layer2_result[idxy]));

	}
	//writeln(layer2_result);
	return layer2_result;
}
//writeln("Result from Sigmoid : ", sigmoid(layer2_result));

// *********Activation Function 1: Sigmoid Part ended********** 

// *********Activation Function 2: Soft Max Part started********** 
proc softmax(layer3_result) 
{
	for idxy in layer3_result.domain  do 
	{
		layer3_result[idxy] = exp(layer3_result[idxy]);
	}
	var sum_vec : [1..training_sample_size] real;
	for i in layer3_result.domain.dim(1)  
	{
		for j in layer3_result.domain.dim(2)
		{       
			sum_vec[i]  = sum_vec[i] + layer3_result[i,j];
		}
	}
	for idxy in layer3_result.domain do 
	{
		layer3_result[idxy] = layer3_result[idxy]/sum_vec[idxy[1]];
	}
	// To check if all values in a row sum up to 1
	var total_check : [1..training_sample_size] real;
	for idxy in layer3_result.domain do 
	{
		total_check[idxy[1]] = total_check[idxy[1]] + layer3_result[idxy];
	}
	//	writeln(total_check);
	return layer3_result;
}
// *********Activation Function 2: Soft Max Part Ended**********

// *********Back Propogation and derivatives part started********
proc backprop()
{
	var a3_delta_domain:domain(2) = {1..training_sample_size, 1..10};
	var a3_delta: [a3_delta_domain] real;
	var a2_delta_domain:domain(2) = {1..training_sample_size, 1..layer3_neurons};
	var a2_delta: [a2_delta_domain] real;
	var a1_delta_domain:domain(2) = {1..training_sample_size, 1..layer2_neurons};
	var a1_delta: [a1_delta_domain] real;
	var z2_delta_domain:domain(2) = {1..training_sample_size, 1..layer3_neurons};
	var z2_delta: [z2_delta_domain] real;
	var z1_delta_domain:domain(2) = {1..training_sample_size, 1..layer2_neurons};
	var z1_delta: [z1_delta_domain] real;
	
	var error_res = error(layer3_result, training_outputs);
	writeln("error in backprop ", error_res);
	a3_delta = cross_entropy(layer3_result, train_out_hot);
	z2_delta = dot(a3_delta, transpose(synaptic_weights_mat3));
	a2_delta = z2_delta * sigmoid_deriv(layer2_result) ; // expected 100 X layer3_neurons here 
	z1_delta = dot(a2_delta, transpose(synaptic_weights_mat2)); 
	a1_delta = z1_delta * sigmoid_deriv(layer1_result);
	
	var adju_w3_domain:domain(2) = {1..layer3_neurons, 1..10};
	var adju_w3: [adju_w3_domain] real;
	var adju_w2_domain:domain(2) = {1..layer2_neurons, 1..layer3_neurons};
	var adju_w2: [adju_w2_domain] real;
	var adju_w1_domain:domain(2) = {1..784, 1..layer2_neurons};
	var adju_w1: [adju_w1_domain] real;

	adju_w3 = dot(transpose(layer2_result), a3_delta);
	adju_w3 = adju_w3 * learn_rate; 
	for idxy in synaptic_weights_mat3.domain 
	{
		synaptic_weights_mat3[idxy] = synaptic_weights_mat3[idxy] - adju_w3[idxy];
	}

	adju_w2 = dot(transpose(layer1_result), a2_delta);
	adju_w2 = adju_w2 * learn_rate;
	for idxy in synaptic_weights_mat2.domain 
	{
		synaptic_weights_mat2[idxy] = synaptic_weights_mat2[idxy] - adju_w2[idxy]; 
	}

	adju_w1 = dot(transpose(training_inputs), a1_delta);
	adju_w1 = adju_w1 * learn_rate;
	for idxy in synaptic_weights_mat1.domain 
	{
		synaptic_weights_mat1[idxy] = synaptic_weights_mat1[idxy] - adju_w1[idxy]; 
	}

}

proc error(pred, training_outputs)
{
	var sum : real = 0;
	for i in 1..training_sample_size
	{
		sum  += -log(pred[i, training_outputs[i]+1]);
		// values in the labels are from 0 to 9, but the indexes/domain is from 1 to 10
	}
	return sum/training_sample_size;
}

proc cross_entropy(pred, train_out_hot_label)
{
	var result_entropy_domain : domain(2) 	= {1..training_sample_size, 1..10};
	var result_entropy: [result_entropy_domain] real;
	for i in result_entropy_domain.dim(1)
	{       
		for j in result_entropy_domain.dim(2)
		{       
			result_entropy[i,j] = (pred[i,j] - train_out_hot_label [i,j])  / training_sample_size;
		}
	}
	return result_entropy;
}

proc sigmoid_deriv(layer2_result_temp) 
{
	// expected 100 X layer3_neurons here 
	for idxy in layer2_result_temp.domain do
	{
		layer2_result_temp[idxy]  =  layer2_result_temp[idxy] * (1 -  layer2_result_temp[idxy]);
	}
	return layer2_result_temp;
}
// *********Back Propogation and derivatives part Ended**********


// ********* Training started**********

for idxy in 1..training_epochs_iterations 
{	
	write("training_epochs_iterations \t ",idxy, " \t");
	feedforward();
	backprop();	
}

// ********* Training ended**********

watch.stop();
training_time = watch.elapsed();
writeln('\n Training time took ',training_time ,' seconds');
watch.clear();


// ********* Testing started**********
watch.start();
var tlayer1_result_dom   : domain(2)     = {1..testing_sample_size, 1..layer2_neurons};
var tlayer1_result       : [tlayer1_result_dom] real;
var tlayer1_temp_vec     : [1..layer2_neurons] real;

var tlayer2_result_dom   : domain(2)     = {1..testing_sample_size, 1..layer3_neurons};
var tlayer2_result       : [tlayer2_result_dom] real;
var tlayer2_temp_vec     : [1..layer3_neurons] real;

var tlayer3_result_dom   : domain(2)     = {1..testing_sample_size, 1..10};
var tlayer3_result       : [tlayer3_result_dom] real;
var tlayer3_temp_vec     : [1..10] real;

get_accuracy();

watch.stop();
testing_time = watch.elapsed();
writeln('\n Testing time took ',testing_time ,' seconds');

proc get_accuracy()
{
	var g_a_prediction_domain:domain(2) = {1..testing_sample_size, 1..10};
	var g_a_prediction: [g_a_prediction_domain] real;
	g_a_prediction = test_feedforward();

	// To find the maximum value and index in the layer3_result
	var o_out_dom	: domain(2) 	= {1..testing_sample_size, 1..2};
	var o_out	: [o_out_dom] real;
	for i in 1..testing_sample_size
	{
		var (theMaxValue, idxOfMax) = maxloc reduce zip(g_a_prediction[i,1..10], 1..10);
		o_out[i,1] = theMaxValue;
		o_out[i,2] = idxOfMax;
		//writeln(i, "\t", theMaxValue, "\t", idxOfMax);
	}   

	var count:int = 0;
	for i in 1..testing_sample_size
	{
		if(o_out[i,2] - 1 == testing_outputs[i]) then
		{
			writeln(o_out[i,2] - 1, "\t",testing_outputs[i]); 
			count += 1;
			writeln("Count is ", count);
		} 
	}
	writeln("Testing Accuracy \t ", count/testing_sample_size:real);
}

proc test_feedforward()
{
	// Test Matrix Vector Multiplication at layer 1
	for i in 1..testing_sample_size  
	{
		tlayer1_temp_vec = dot(testing_inputs[i,1..layer1_neurons], synaptic_weights_mat1);
		for j in tlayer1_result_dom.dim(2)  
		{
			tlayer1_result[i,j] = tlayer1_temp_vec[j];
		}

	}
	//writeln(" tlayer1_result before sig \n", tlayer1_result);	
	tlayer1_result = sigmoid(tlayer1_result);
	//writeln(" tlayer1_result after sig \n", tlayer1_result);	

	// Test Matrix Vector Multiplication at layer 2
	for i in 1..testing_sample_size
	{
		tlayer2_temp_vec = dot(tlayer1_result[i,1..layer2_neurons], synaptic_weights_mat2);
		for j in tlayer2_result_dom.dim(2)  
		{
			tlayer2_result[i,j] = tlayer2_temp_vec[j];
		}
	}
	tlayer2_result = sigmoid(tlayer2_result);
	//writeln(" tlayer2_result after sigmoid \n", tlayer2_result);	

	// Test Matrix Vector Multiplication at layer 3
	for i in 1..testing_sample_size
	{
		tlayer3_temp_vec = dot(tlayer2_result[i,1..layer3_neurons], synaptic_weights_mat3);
		for j in tlayer3_result_dom.dim(2)  
		{
			tlayer3_result[i,j] = tlayer3_temp_vec[j];
		}
	}
	tlayer3_result = softmax(tlayer3_result);
	//writeln(" tlayer3_result after softmax \n", tlayer3_result);	
	return tlayer3_result;
}

// ********* Testing Ended**********


writeln('\n File reading time took ',file_read_time ,' seconds');
writeln('\n Training time took ',training_time ,' seconds');
writeln('\n Testing time took ',testing_time ,' seconds');
