use Random;



proc create_matrix_random(rows_cols_dom){

	var seed = 1;
	var randStreamSeeded = new owned RandomStream(real, seed);

	var newMatrix : [rows_cols_dom] real;

	for idxy in newMatrix.domain do newMatrix[idxy] = randStreamSeeded.getNext();

	return newMatrix;
}


config const rows: int = 4;
config const cols: int = 4;
var rows_cols_domain: domain(2) = {1..rows,1..cols};



var synaptic_weights_mat = create_matrix_random(rows_cols_domain);

/*
writeln("Printing the entire Matrix \n",synaptic_weights_mat);
writeln("\n\n");
writeln("Printing just the 2nd Vector in the matrix \n",synaptic_weights_mat[2,1..cols]);
*/
