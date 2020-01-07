/*
Author: Kaushik Velusamy
Org: UMBC
About: Neural network from scratch in chapel language
To compile:  chpl -I/usr/local/opt/openblas/include -L/usr/local/opt/openblas/lib -lblas hello.chpl --fast
To run : ./hello --iterations 100
 */

use Random;
use LinearAlgebra, Norm;


var training_inputs = Matrix([0,0,1], [1,1,1], [1,0,1], [0,1,1], eltType=real);
writeln("training_inputs \n",training_inputs);
writeln();

var training_outputs =  Vector([0, 1, 1, 0], eltType=real);
writeln("training_outputs \n",training_outputs);
writeln();

var synaptic_weights: [0..2] real;
var seed = 1;
fillRandom(synaptic_weights, seed);
synaptic_weights = 2 * synaptic_weights -1;
writeln("synaptic weights before training \n", synaptic_weights); 


proc sigmoid(x) {
	var y=exp(-x);
	return 1 / (1 + y);
}

proc sigmoid_derivative(x) {
	return x * (1 - x);
}


var result_sigmoid: [0..2] real; 

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
		writeln("After dot product \n",res_dot);

	result_sigmoid = sigmoid(res_dot); 
		writeln("From Test : After sigmoid function \n", result_sigmoid);

	return result_sigmoid;	
}


train_training(training_inputs, training_outputs, iterations);
writeln("synaptic weights after training \n", synaptic_weights);
writeln("Output After Training \n", result_sigmoid);
writeln("\nIf we can see this, everything works!");
var new_testing_input = Vector([1,0,0], eltType=real);
writeln("New result of 1,0,0 is \t ",think_testing(new_testing_input)[1]);
writeln("\nIf we can see this, everything works!");

