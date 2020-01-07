// chpl readcsv.chpl --fast -M ../lib/

use IO, CSV;
config const infile  = "../mnist_neural_nets/mnist_dataset/mnist_test_10.csv";
const label_range = 1..10;
const pixels_range = 1..785;

var label_arr : [1..10] int;
var pixels_arr : [1..10, 1..784] int;

try {
	var myReader = if infile == "" then stdin else openreader(infile);
	var r = new CSVIO(myReader, hasHeader=false, sep =",");
	var A = r.read(string):int;

	for i in label_range do{
		for j in pixels_range do{
			if (j == 1) { label_arr[i]= A(i,1); }
			else { pixels_arr[i,j-1]= A(i,j); }	
		}
	}
} catch error { writeln("Error: ", error); }

writeln(label_arr);
writeln(pixels_arr);

