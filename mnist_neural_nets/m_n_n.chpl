/*
Author: Kaushik Velusamy
Org: UMBC
About: Neural network from scratch in chapel language
To Compile:  chpl -I/usr/local/opt/openblas/include -L/usr/local/opt/openblas/lib -lblas m_n_n.chpl --fast  -M ../lib/
To Run: ./m_n_n --iterations 100

 */

use Random;
use IO, CSV;
use LinearAlgebra, Norm;

config const infile  = "mnist_dataset/mnist_test_10.csv";
const label_range = 1..10;
const pixels_range = 1..785;

var label_arr : [1..10] real;
var pixels_arr : [1..10, 1..784] real;

try {
	var myReader = if infile == "" then stdin else openreader(infile);
	var r = new CSVIO(myReader, hasHeader=false, sep =",");
	var A = r.read(string):real;

	for i in label_range do{
		for j in pixels_range do{
			if (j == 1) { label_arr[i]= A(i,1); }
			else { pixels_arr[i,j-1]= A(i,j); }
		}
	}
} catch error { writeln("Error: ", error); }


var training_inputs = Matrix(pixels_arr);
//writeln("training_inputs \n",training_inputs);
writeln();

var training_outputs =  Vector(label_arr);
//writeln("training_outputs \n",training_outputs);
writeln();


var synaptic_weights: [1..784] real;
var seed = 1;
fillRandom(synaptic_weights, seed);
synaptic_weights = 2 * synaptic_weights -1;
//writeln("synaptic weights before training \n", synaptic_weights); 


proc sigmoid(x) {
	var y=exp(-x);
	return 1 / (1 + y);
}

proc sigmoid_derivative(x) {
	return x * (1 - x);
}


var result_sigmoid: [1..10] real; 

config const iterations: int = 5;

proc train_training(training_inputs, training_outputs, iterations){

	for iteration in 1..iterations do {

		result_sigmoid = think_testing(training_inputs);

		var error = training_outputs - result_sigmoid;
		//	writeln("Error \n",error);

		var result_sig_derivative = sigmoid_derivative(result_sigmoid);
		var adjustments = error * result_sig_derivative; 
		//	writeln("Adjustment after sigmoid derivative \n",adjustments);

		var trans_input_layer = transpose(training_inputs);

		var temp_result = dot(trans_input_layer, adjustments);
		//	writeln("Result of 2nd dot op \n",temp_result);

		synaptic_weights = synaptic_weights + temp_result;
		// 	writeln("From Train :synaptic weights at loop \t",iteration,"\t", synaptic_weights);

	}
}

proc think_testing(input)
{
	var input_layer = input;
	var res_dot = dot(input_layer, synaptic_weights);
	//writeln("After dot product \n",res_dot);

	result_sigmoid = sigmoid(res_dot); 
	//	writeln("From Test : After sigmoid function \n", result_sigmoid);

	return result_sigmoid;	
}


train_training(training_inputs, training_outputs, iterations);
writeln("synaptic weights after training \n", synaptic_weights);
writeln("Output After Training \n", result_sigmoid);
writeln("\nIf we can see this, everything works!");

/*
   var new_testing_input = Vector([1,0,0], eltType=real);
   writeln("New result of 1,0,0 is \t ",think_testing(new_testing_input)[1]);
   writeln("\nIf we can see this, everything works!");
 */
