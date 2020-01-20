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

config const train_file = "mnist_dataset/mnist_train_100.csv";
config const test_file  = "mnist_dataset/mnist_test_10.csv";
const digits_range      = 0..9;
const pixels_per_line   = 1..784; 		// 28 X 28 pixels

//*********File Reading Started********* 
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
var training_outputs : [1..training_sample_size] real;
var testing_inputs   : [1..testing_sample_size, pixels_per_line] real;
var testing_outputs  : [1..testing_sample_size] real;

for (i,j) in A_train.domain do 
{
	if (j == 1) { training_outputs[i]     = A_train(i,1); }
	else 	    { training_inputs[i, j-1] = (A_train(i,j) - 127.50) / 255; } //Range normalization 
}

for (i,j) in A_test.domain do 
{
	if (j == 1) { testing_outputs[i]     = A_test(i,1); }
	else        { testing_inputs[i, j-1] = A_test(i,j); }
}

trainReader.close();
testReader.close();


/*
   writeln("training_sample_size ", training_sample_size);
   writeln("training_inputs \n"   , training_inputs);
   writeln("training_outputs \n"  , training_outputs);
   writeln("testing_sample_size " , testing_sample_size);
   writeln("testing_inputs \n"    , testing_inputs);
   writeln("testing_outputs \n"   , testing_outputs);
 */

// *********File Reading Ended********** 



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


proc create_matrix_random(rows_cols_dom){
	var newMatrix : [rows_cols_dom] real;
	newMatrix = fillNormallyDistributed(newMatrix);
	return newMatrix;
}

config const rows: int 		= 784;
config const cols: int 		= 10;
var rows_cols_domain: domain(2) = {1..rows,1..cols};
var synaptic_weights_mat 	= create_matrix_random(rows_cols_domain);

//writeln("Synaptic Weight Matrix \n",synaptic_weights_mat);

// *********Random Matrix Function Ended********** 



// *********Matrix Vector Multiplication started********** 


var realDomain: domain(2) = {1..100,1..10};
var dot_result_mat: [realDomain] real;
var temp_result_vec : [1..10] real;


for i in realDomain.dim(1)  
{
	//writeln(i);
	temp_result_vec = dot(training_inputs[i,1..784], synaptic_weights_mat);

	for j in realDomain.dim(2)  
	{
		dot_result_mat[i,j] = temp_result_vec[j];
	}

}
writeln(" Result of MxM product \n",dot_result_mat);	


// *********Matrix Vector Multiplication Ended********** 


// ********* Soft Max Part started********** 


proc softmax(dot_result_mat) {
   for idxy in realDomain  do 
   {
   	dot_result_mat[idxy] = exp(dot_result_mat[idxy]);
   }
   var sum_vec : [1..100] real;
   for i in realDomain.dim(1)  
   {
	for j in realDomain.dim(2)
        {       
        	sum_vec[i]  = sum_vec[i] + dot_result_mat[i,j];
   	}
   }
   for idxy in realDomain  do 
   {
   	dot_result_mat[idxy] = dot_result_mat[idxy]/sum_vec[idxy[1]];
   }

   // To check if all values in a row sum up to 1
   /*
   var total_check : [1..100] real;
   for idxy in realDomain  do 
   {
   	total_check[idxy[1]] = total_check[idxy[1]] + dot_result_mat[idxy];
   }
   writeln(total_check);
   */

   return dot_result_mat;
}

//writeln("Result from Softmax : ", softmax(dot_result_mat));



// ********* Soft Max Part Ended**********


